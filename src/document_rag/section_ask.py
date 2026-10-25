"""Ask the current RAG model about the tracker's section text."""

import hashlib
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from document_rag.answer import generate_answer
from document_rag.config import EMBED_MODEL, RagError
from document_rag.retrieve import lexical_tokens

CACHE_NAME = "section-vectors.json"
CHANGE = re.compile(r"\b(chang\w*|differ\w*|revis\w*|what was (?:added|removed|updated))\b", re.I)
_TITLE_STOP = frozenset(
    {
        "what", "do", "does", "did", "for", "the", "and", "or", "a", "an", "of", "to",
        "in", "on", "is", "are", "was", "were", "how", "this", "that", "with", "from",
        "by", "about", "when", "where", "which", "who", "into", "their", "they",
    }
)
NUMBER = re.compile(r"\b(?:BHIN[-\s]+)?(\d{2}-\d{3})\b", re.I)


def load_library(data_dir):
    path = Path(data_dir) / "library.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    return [item for item in payload.get("documents") or [] if item.get("id")]


def _history(data_dir, document_id):
    path = Path(data_dir) / document_id / "history.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _judgments(data_dir, document_id):
    path = Path(data_dir) / document_id / "judgments.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _slot(values, index):
    """The value for one version, or None when that list is missing or short."""
    if not isinstance(values, list) or index < 0 or index >= len(values):
        return None
    return values[index]


def comparison_href(document_id, versions, version, section_id):
    """A tracker link whose right-hand side is the cited version.

    The page ignores a link that compares a version with itself, so an older
    notice is shown against its successor and every later notice against its predecessor.
    """
    versions = [item for item in (versions or []) if item]
    section = (section_id or "").replace("~", "!")
    if version in versions and len(versions) > 1:
        index = versions.index(version)
        if index == 0:
            left, right = versions[0], versions[1]
        else:
            left, right = versions[index - 1], versions[index]
    else:
        left = right = version or ""
    return f"#{document_id}/{left}~{right}~{section}"


def _version_index(history, question):
    versions = history.get("versions") or []
    if not versions:
        return None
    named = NUMBER.search(question or "")
    if named:
        token = named.group(1)
        for index, version in enumerate(versions):
            if version.endswith(token):
                return index
    return len(versions) - 1


def passages_for(data_dir, document_id, question):
    """Citable section passages for one document, using the version the question names."""
    history = _history(data_dir, document_id)
    if not history:
        return []
    index = _version_index(history, question)
    if index is None:
        return []
    version = history["versions"][index]
    texts = history.get("texts") or []
    found = []
    for section in history.get("sections") or []:
        slot = _slot(section.get("texts"), index)
        if slot is None or not isinstance(slot, int) or slot >= len(texts):
            continue
        body = texts[slot]
        prose = body.split("\n", 1)[1].strip() if isinstance(body, str) and "\n" in body else (body or "")
        if not isinstance(prose, str) or not prose.strip():
            continue
        label = section.get("label") or ""
        found.append(
            {
                "document_id": document_id,
                "title": section.get("title") or label,
                "filename": f"{version}.md",
                "version": version,
                "section_id": f"{section.get('chapter')}/{label}",
                "page": _slot(section.get("pages"), index) or None,
                "text": prose[:6000],
                "kind": "content",
            }
        )
    if CHANGE.search(question or ""):
        for key, note in _judgments(data_dir, document_id).items():
            if not isinstance(note, dict) or not note.get("why"):
                continue
            if version not in key and f"|{version}|" not in f"{key}|":
                # Keep notes that mention this version on either side of the comparison.
                parts = key.split("|")
                if version not in parts:
                    continue
            found.append(
                {
                    "document_id": document_id,
                    "title": "Change note",
                    "filename": "judgments.json",
                    "version": version,
                    "section_id": key.split("|", 1)[0],
                    "page": 1,
                    "text": f"{note.get('level', '')}: {note.get('why', '')}",
                    "kind": "change",
                }
            )
    return found


def _title_overlap(question, title):
    """How much of the question's distinctive wording is in the section title."""
    terms = [word for word in lexical_tokens(question) if word not in _TITLE_STOP and len(word) > 2]
    if not terms:
        return 0.0
    title_terms = lexical_tokens(title or "")
    hits = sum(
        1
        for word in terms
        if any(token == word or token.startswith(word) for token in title_terms)
    )
    return hits / len(terms)


def _blend(question, passages, models, cache_path):
    """BM25 plus title overlap, then embeddings over that shortlist when Ollama is up."""
    if not passages:
        return []
    corpus = [lexical_tokens((item.get("title") or "") + "\n" + item["text"]) for item in passages]
    scores = BM25Okapi(corpus).get_scores(lexical_tokens(question))
    top_lex = max(scores) or 1
    ranked = []
    for index, passage in enumerate(passages):
        lexical = scores[index] / top_lex
        bonus = _title_overlap(question, passage.get("title"))
        ranked.append((0.75 * lexical + 0.25 * bonus, index))
    ranked.sort(reverse=True)
    shortlist = [index for _score, index in ranked[:24]]
    vectors = _embed_shortlist(question, [passages[index] for index in shortlist], models, cache_path)
    if vectors is None:
        return [passages[index] for index in shortlist]
    query, matrix = vectors
    blended = []
    for offset, index in enumerate(shortlist):
        semantic = _cosine(query, matrix[offset])
        lexical = scores[index] / top_lex
        bonus = _title_overlap(question, passages[index].get("title"))
        blended.append((0.35 * lexical + 0.45 * semantic + 0.20 * bonus, index))
    blended.sort(reverse=True)
    return [passages[index] for _score, index in blended]


def _cosine(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _embed_shortlist(question, passages, models, cache_path):
    try:
        embeddings = models.embeddings()
    except RagError:
        return None
    cache = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            cache = {}
    needed = []
    keys = []
    for passage in passages:
        key = hashlib.sha256(passage["text"].encode()).hexdigest()
        keys.append(key)
        if key not in cache:
            needed.append((key, passage["text"][:1500]))
    query_key = "q:" + hashlib.sha256(question.encode()).hexdigest()
    try:
        if needed:
            vectors = embeddings.embed_documents([text for _key, text in needed])
            for (key, _text), vector in zip(needed, vectors):
                cache[key] = vector
        if query_key not in cache:
            cache[query_key] = embeddings.embed_query(question)
    except (RagError, OSError, ValueError):
        return None
    stored = {}
    if cache_path.is_file():
        try:
            stored = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            stored = {}
    stored.update({key: cache[key] for key in keys})
    stored[query_key] = cache[query_key]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(stored))
    return cache[query_key], [cache[key] for key in keys]


def ask_document(data_dir, models, question, document_id=None, cache_dir=None):
    """Ground an answer in the open document. The generator is the same client as the PDF app."""
    question = (question or "").strip()
    if not question:
        raise RagError("Enter a question.")
    data_dir = Path(data_dir)
    documents = load_library(data_dir)
    if document_id:
        documents = [item for item in documents if item["id"] == document_id] or [{"id": document_id}]
    collected = []
    for document in documents:
        collected.extend(passages_for(data_dir, document["id"], question))
    if not collected:
        raise RagError("This document has no section text to search yet.")
    cache_path = Path(cache_dir or data_dir.parent.parent / "index") / CACHE_NAME
    ordered = _blend(question, collected, models, cache_path)
    sources = []
    for passage in ordered[:6]:
        item = dict(passage)
        item["source_id"] = f"S{len(sources) + 1}"
        item["rank"] = len(sources) + 1
        sources.append(item)
    result = generate_answer(question, sources, models)
    result["chat_model"] = models.chat_model
    result["chat_provider"] = models.provider
    result["embedding_model"] = EMBED_MODEL
    result["retrieved_sources"] = sources
    histories = {}
    for source in result.get("sources") or []:
        document = source.get("document_id") or ""
        if document not in histories:
            histories[document] = _history(data_dir, document) or {}
        source["href"] = comparison_href(
            document,
            histories[document].get("versions"),
            source.get("version") or "",
            source.get("section_id") or "",
        )
    return result

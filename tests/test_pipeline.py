import pytest

from document_rag.answer import generate_answer, select_evidence
from document_rag.config import (
    EMBED_INPUT_BYTES,
    EVIDENCE_CHAR_BUDGET,
    SUMMARY_CHAR_BUDGET,
    RagError,
    configured_provider,
)
from document_rag.models import ApiModels, parse_generated_json
from document_rag.ingest import chunks_for_page
from document_rag.retrieve import overlaps
from document_rag.sections import compare_sections, split_sections
from document_rag.summarize import pack_pages
from document_rag.versions import annotate_versions, parse_filename, scope_for_question


def test_versioned_filenames_mark_only_the_newest_as_latest():
    parsed = parse_filename(
        "Behavioral Health Services Act County Policy Manual-V.1.5.0-20260630.pdf"
    )
    assert parsed["version"] == "1.5.0"
    assert parsed["version_date"] == "20260630"
    documents = annotate_versions(
        [
            {"id": "a", "title": "a", "filename": "Manual-V.1.3.0-202506.pdf"},
            {"id": "b", "title": "b", "filename": "Manual-V.1.5.0-20260630.pdf"},
            {"id": "c", "title": "c", "filename": "Manual-V.1.4.0-20260403.pdf"},
        ]
    )
    latest = {doc["version"]: doc["is_latest"] for doc in documents}
    assert latest == {"1.3.0": 0, "1.4.0": 0, "1.5.0": 1}
    assert documents[0]["family"] == "Manual"


def test_questions_default_to_the_latest_version_or_change_notes():
    assert scope_for_question("Who is eligible?")[1] == {
        "$and": [{"kind": "content"}, {"is_latest": 1}]
    }
    assert scope_for_question("What did version 1.3.0 require?")[1] == {
        "$and": [{"kind": "content"}, {"version": "1.3.0"}]
    }
    assert scope_for_question("What changed between 1.4.0 and 1.5.0?")[1] == {
        "$and": [
            {"kind": "change"},
            {"from_version": "1.4.0"},
            {"to_version": "1.5.0"},
        ]
    }


def test_section_diff_ignores_unchanged_text_and_page_shifts():
    left = split_sections(
        [
            {
                "page": 7,
                "text": "1. Policy Manual Introduction\nCounties must file a plan.\nA.1 Bond\nThe bond is $6 billion.",
            }
        ]
    )
    right = split_sections(
        [
            {
                "page": 9,
                "text": "1. Policy Manual Introduction\nCounties must file a plan.\nA.1 Bond\nThe bond is $6.38 billion.\nB.3 Eligible Populations\nAdults and children are eligible.",
            }
        ]
    )
    changes = {item["section_id"]: item["change"] for item in compare_sections(left, right)}
    assert changes == {"A.1": "modified", "B.3": "added"}
    years = split_sections(
        [{"page": 12, "text": "2029 County Integrated Plan\nDue in June.\nB.3 Eligible Populations\nAdults are eligible."}]
    )
    assert [section["section_id"] for section in years] == ["B.3"]


def test_chunks_stay_within_embedding_budget():
    doc = {"id": "notes", "title": "notes", "filename": "notes.pdf"}
    text = ("A method for retrieval. " * 400).strip()
    chunks = list(chunks_for_page(doc, 1, text))
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content.encode("utf-8")) <= EMBED_INPUT_BYTES
        assert chunk.metadata["document_id"] == "notes"
        assert chunk.metadata["page"] == 1
        assert chunk.page_content.startswith("title: notes\n")


def test_overlap_dedupes_mostly_shared_spans():
    left = {
        "id": "a",
        "document_id": "notes",
        "page": 1,
        "start_char": 0,
        "end_char": 100,
    }
    right = {
        "id": "b",
        "document_id": "notes",
        "page": 1,
        "start_char": 40,
        "end_char": 140,
    }
    other_page = {**right, "id": "c", "page": 2}
    assert overlaps(left, right)
    assert not overlaps(left, other_page)


def test_select_evidence_respects_character_budget():
    passages = [
        {"text": "x" * 100},
        {"text": "y" * (EVIDENCE_CHAR_BUDGET)},
    ]
    chosen = select_evidence("q", passages)
    assert chosen == [passages[0]]


def test_pack_pages_splits_long_documents():
    pages = [{"page": 1, "text": "a" * (SUMMARY_CHAR_BUDGET + 10)}, {"page": 2, "text": "tail"}]
    batches = pack_pages(pages, budget=SUMMARY_CHAR_BUDGET)
    assert len(batches) >= 2
    assert all(len(batch) <= SUMMARY_CHAR_BUDGET for batch in batches)
    assert batches[-1] == "Page 2\ntail"


class _Scripted:
    def __init__(self, responses):
        self.responses = list(responses)
        self.chat_model = "fake"

    def chat(self, system, question, schema=None, max_tokens=900):
        return self.responses.pop(0)


def _source():
    return {
        "source_id": "S1",
        "document_id": "notes",
        "title": "notes",
        "filename": "notes.pdf",
        "page": 1,
        "text": "The study used BM25.",
    }


def test_generate_answer_cites_a_supported_claim():
    models = _Scripted(
        [
            {"reason": "The excerpt states the method.", "supported": True},
            {
                "supported": True,
                "claims": [{"text": "The study used BM25. [S1]", "sources": ["S1"]}],
            },
            {"reason": "The claim matches the excerpt.", "supported": True, "unsupported_sources": []},
        ]
    )
    result = generate_answer("Which method was used?", [_source()], models)
    assert result["supported"] is True
    assert result["answer"] == "The study used BM25. [S1]"
    assert result["sources"][0]["source_id"] == "S1"


def test_parse_generated_json_reads_fenced_objects():
    assert parse_generated_json('```json\n{"supported": false}\n```') == {
        "supported": False
    }
    with pytest.raises(ValueError):
        parse_generated_json("no object here")


def test_local_generation_skips_the_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    models = ApiModels("org/model", provider="huggingface")

    def complete(messages, max_tokens):
        assert "JSON schema" in messages[0]["content"]
        assert max_tokens == 50
        return '{"supported": true, "reason": "ok"}'

    models._complete = complete
    assert models.chat("system", "question", schema={"type": "object"}, max_tokens=50) == {
        "supported": True,
        "reason": "ok",
    }


def test_anthropic_generation_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    models = ApiModels(provider="anthropic")
    with pytest.raises(RagError, match="ANTHROPIC_API_KEY"):
        models.chat("system", "question")


def test_unknown_generation_provider_is_rejected():
    with pytest.raises(RagError, match="provider"):
        ApiModels(provider="openai")


def test_configured_provider_falls_back_to_anthropic(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "other")
    assert configured_provider() == "anthropic"
    monkeypatch.setenv("CHAT_PROVIDER", "huggingface")
    assert configured_provider() == "huggingface"


def test_generate_answer_abstains_when_evidence_is_insufficient():
    models = _Scripted([{"reason": "The year is missing.", "supported": False}])
    result = generate_answer("What happened in 2024?", [_source()], models)
    assert result["supported"] is False
    assert result["sources"] == []

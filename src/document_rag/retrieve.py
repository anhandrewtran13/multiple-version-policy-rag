"""Hybrid retrieval over Chroma and a BM25 retriever."""

import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langsmith import tracing_context

from document_rag.config import RagError
from document_rag.versions import is_change_question


def lexical_tokens(text):
    return re.findall(r"\w+", text.casefold())


def source_record(document):
    text = document.page_content
    if "\n" in text:
        text = text.split("\n", 1)[1]
    return {
        **document.metadata,
        "id": document.metadata["chunk_id"],
        "text": text,
    }


def overlaps(left, right):
    if left["id"] == right["id"]:
        return True
    if left["document_id"] != right["document_id"] or left["page"] != right["page"]:
        return False
    intersection = max(
        0,
        min(left["end_char"], right["end_char"])
        - max(left["start_char"], right["start_char"]),
    )
    smaller = min(
        left["end_char"] - left["start_char"], right["end_char"] - right["start_char"]
    )
    return smaller > 0 and intersection > 0.5 * smaller


class Retriever:
    def __init__(self, index, models):
        self.index, self.models = index, models

    def search(self, question, *, file_ids=None, where=None, top_k=6):
        store = self.index.vectorstore(self.models)
        chunks = self.index.documents(file_ids, where)
        if not chunks:
            if is_change_question(question):
                raise RagError(
                    "No change notes are indexed. Run: document-rag diff"
                )
            if file_ids:
                raise RagError("No indexed text matches the selected files.")
            return []
        selected = []
        for document in self._rank(question, store, chunks):
            item = source_record(document)
            if not any(overlaps(item, previous) for previous in selected):
                item["source_id"] = f"S{len(selected) + 1}"
                item["rank"] = len(selected) + 1
                selected.append(item)
            if len(selected) == top_k:
                break
        return selected

    def _rank(self, question, store, documents):
        if not documents:
            return []
        ids = sorted({d.metadata["document_id"] for d in documents})
        semantic = store.as_retriever(
            search_kwargs={
                "k": min(20, len(documents)),
                "filter": {"document_id": {"$in": ids}},
            }
        )
        lexical = BM25Retriever.from_documents(
            documents, k=min(20, len(documents)), preprocess_func=lexical_tokens
        )
        ensemble = EnsembleRetriever(
            retrievers=[semantic, lexical], weights=[0.5, 0.5], c=60, id_key="chunk_id"
        )
        try:
            with tracing_context(enabled=False):
                return ensemble.invoke(question)
        except Exception as exc:
            raise RagError(f"Retrieval failed: {exc}") from exc

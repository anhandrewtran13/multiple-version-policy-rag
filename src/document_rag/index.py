"""Persistent Chroma vector stores and immutable index generations."""

import json
import sqlite3
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document

from document_rag.config import EMBED_MODEL, RagError

COLLECTION = "documents"
INDEX_SCHEMA_VERSION = 2


def chroma_client(directory):
    return chromadb.PersistentClient(
        path=str(directory.resolve()), settings=Settings(anonymized_telemetry=False)
    )


def embedding_namespace(models):
    # LocalFileStore rejects ":" in cache keys, and the Ollama tag contains one.
    model = EMBED_MODEL.replace(":", "-")
    return f"{model}-{models.digest(EMBED_MODEL)}-"


def cached_embeddings(directory, models):
    return CacheBackedEmbeddings.from_bytes_store(
        models.embeddings(),
        LocalFileStore(directory / "embedding-cache"),
        namespace=embedding_namespace(models),
        key_encoder="sha256",
        query_embedding_cache=LocalFileStore(directory / "query-cache"),
    )


def as_documents(result):
    return [
        Document(id=key, page_content=text, metadata=metadata)
        for key, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"], strict=True
        )
    ]


class Index:
    def __init__(self, directory):
        self.db = self.client = None
        self.directory = Path(directory)
        try:
            generation = json.loads((self.directory / "current.json").read_text())[
                "generation"
            ]
            if (
                not generation.startswith("build-")
                or "/" in generation
                or "\\" in generation
            ):
                raise ValueError("invalid generation")
            root = self.directory / generation
            self.manifest = json.loads((root / "manifest.json").read_text())
            if self.manifest["schema_version"] != INDEX_SCHEMA_VERSION:
                raise ValueError("unsupported index format; rebuild with ingest")
            if self.manifest["embedding_model"] != EMBED_MODEL:
                raise ValueError("unsupported embedding model")
            self.db = sqlite3.connect(
                (root / "catalog.sqlite").resolve().as_uri() + "?mode=ro",
                uri=True,
                # The demo reuses this read-only connection across Streamlit reruns.
                check_same_thread=False,
            )
            self.db.row_factory = sqlite3.Row
            if not (root / "chroma" / "chroma.sqlite3").is_file():
                raise ValueError("Chroma database is missing")
            self.client = chroma_client(root / "chroma")
            self.collection = self.client.get_collection(
                COLLECTION, embedding_function=None
            )
            if self.collection.count() != self.manifest["chunks"]:
                raise ValueError("Chroma document count does not match the manifest")
        except (OSError, ValueError, KeyError, sqlite3.Error, ChromaError) as exc:
            self.close()
            raise RagError(
                f"Cannot open index at {directory}: {exc}. Run: document-rag ingest"
            ) from exc

    def vectorstore(self, models):
        if (
            self.manifest["embedding_model"] != EMBED_MODEL
            or models.digest(EMBED_MODEL) != self.manifest["embedding_digest"]
        ):
            raise RagError(
                "The embedding model changed since ingestion. Run document-rag ingest to rebuild the index."
            )
        return Chroma(
            client=self.client,
            collection_name=COLLECTION,
            embedding_function=cached_embeddings(self.directory, models),
            create_collection_if_not_exists=False,
        )

    def documents(self, file_ids=None, where=None):
        filters = []
        if file_ids:
            filters.append({"document_id": {"$in": list(file_ids)}})
        if where:
            filters.append(where)
        if not filters:
            query = None
        elif len(filters) == 1:
            query = filters[0]
        else:
            query = {"$and": filters}
        return as_documents(
            self.collection.get(where=query, include=["documents", "metadatas"])
        )

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.client is not None:
            self.client.close()
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def resolve_files(db, refs):
    """Match a filename, path, or filename stem to exactly one indexed document."""
    if not refs:
        return None
    ids = []
    for ref in refs:
        needle = ref.strip()
        name = Path(needle).name
        stem = Path(name).stem
        rows = db.execute(
            """
            SELECT id FROM documents
            WHERE id = ? OR filename = ? OR title = ?
            """,
            (needle, name, stem),
        ).fetchall()
        matched, seen = [], set()
        for row in rows:
            if row["id"] not in seen:
                seen.add(row["id"])
                matched.append(row["id"])
        if len(matched) != 1:
            available = ", ".join(
                row["filename"]
                for row in db.execute(
                    "SELECT filename FROM documents ORDER BY filename LIMIT 12"
                )
            )
            raise RagError(
                f"Could not uniquely match file {ref!r}. "
                f"Indexed files include: {available or 'none'}"
            )
        ids.append(matched[0])
    return ids

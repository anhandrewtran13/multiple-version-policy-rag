"""PDF extraction, splitting, and caching into persistent Chroma."""

import hashlib
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_rag.config import (
    CHUNK_BYTES,
    EMBED_INPUT_BYTES,
    EMBED_MODEL,
    OVERLAP_BYTES,
    RagError,
)
from document_rag.index import (
    COLLECTION,
    INDEX_SCHEMA_VERSION,
    cached_embeddings,
    chroma_client,
    embedding_namespace,
)
from document_rag.versions import annotate_versions

EXTRACTION_VERSION = "langchain-pypdf-pages-v1"
SCHEMA = """
CREATE TABLE documents (
 id TEXT PRIMARY KEY,
 title TEXT,
 filename TEXT,
 family TEXT,
 version TEXT,
 version_date TEXT,
 is_latest INTEGER,
 sha256 TEXT,
 status TEXT,
 error TEXT,
 page_count INTEGER,
 warnings TEXT
);
"""


def content_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def documents_from_dir(data_dir):
    paths = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not paths:
        raise RagError(f"No PDF files found in {data_dir}.")
    documents, seen = [], set()
    for path in paths:
        doc_id = path.stem
        if not doc_id or doc_id in seen:
            raise RagError(f"PDF filename stem is missing or duplicated: {path.name}")
        seen.add(doc_id)
        documents.append({"id": doc_id, "title": doc_id, "filename": path.name})
    return annotate_versions(documents)


def chunks_for_page(doc, page, text):
    if not text or not text.strip():
        return
    label = doc.get("family") or doc["title"]
    version = doc.get("version") or ""
    prefix = f"title: {label}" + (f" | v{version}" if version else "") + "\n"
    capacity = min(CHUNK_BYTES, EMBED_INPUT_BYTES - len(prefix.encode("utf-8")))
    if capacity < 80:
        raise RagError(f"Filename is too long to embed: {doc['filename']}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=capacity,
        chunk_overlap=min(OVERLAP_BYTES, capacity // 5),
        length_function=lambda value: len(value.encode("utf-8")),
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    for chunk in splitter.create_documents([text]):
        if not chunk.page_content.strip():
            continue
        start = chunk.metadata["start_index"]
        key = content_hash(f"{doc['id']}:{page}:{start}:{chunk.page_content}")
        yield Document(
            id=key,
            page_content=prefix + chunk.page_content,
            metadata={
                "chunk_id": key,
                "document_id": doc["id"],
                "page": page,
                "start_char": start,
                "end_char": start + len(chunk.page_content),
                "title": doc.get("family") or doc["title"],
                "filename": doc["filename"],
                "kind": "content",
                "family": doc.get("family") or doc["title"],
                "version": doc.get("version") or "",
                "version_date": doc.get("version_date") or "",
                "is_latest": int(doc.get("is_latest") or 0),
                "section_id": "",
                "from_version": "",
                "to_version": "",
                "change_type": "",
            },
        )


def extract_pdf(path, *, preserve_lines=False):
    pages, warnings, error = [], [], ""
    try:
        for document in PyPDFLoader(
            path, mode="page", extract_images=False
        ).lazy_load():
            number = document.metadata["page"] + 1
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", document.page_content)
            text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
            if preserve_lines:
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
            else:
                text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 80:
                warnings.append(f"PDF page {number}: little or no extractable text")
            pages.append({"page": number, "text": text})
    except Exception as exc:  # noqa: BLE001 - retain readable pages and record malformed PDFs
        error = str(exc)
        warnings.append(f"PDF loading stopped after {len(pages)} pages: {exc}")
    return {"pages": pages, "warnings": warnings, "error": error}


@contextmanager
def build_lock(index_dir):
    import fcntl

    with (index_dir / "build.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RagError("Another ingestion is running for this index.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def ingest(data_dir, index_dir, models, *, batch_size=16, progress=print):
    if not data_dir.is_dir():
        raise RagError(f"PDF directory does not exist: {data_dir}.")
    index_dir.mkdir(parents=True, exist_ok=True)
    with build_lock(index_dir):
        return _ingest(data_dir, index_dir, models, batch_size, progress)


def _ingest(data_dir, index_dir, models, batch_size, progress):
    started = time.perf_counter()
    documents = documents_from_dir(data_dir)
    embeddings = cached_embeddings(index_dir, models)
    namespace = embedding_namespace(models)
    before = sum(
        1
        for key in LocalFileStore(index_dir / "embedding-cache").yield_keys()
        if key.startswith(namespace)
    )
    generation = f"build-{uuid.uuid4().hex}"
    build_dir = index_dir / generation
    build_dir.mkdir()
    extraction_cache = LocalFileStore(index_dir / "extraction-cache")
    client = chroma_client(build_dir / "chroma")
    store = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    db = sqlite3.connect(build_dir / "catalog.sqlite")
    try:
        db.executescript(SCHEMA)
        extracted = reused = chunks_total = 0
        for number, doc in enumerate(documents, 1):
            path = data_dir / doc["filename"]
            with path.open("rb") as stream:
                file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            key = content_hash(EXTRACTION_VERSION + file_hash)
            cached = extraction_cache.mget([key])[0]
            if cached:
                result = json.loads(cached)
                reused += 1
            else:
                result = extract_pdf(path)
                if not result["error"]:
                    extraction_cache.mset([(key, json.dumps(result).encode())])
                extracted += 1
            status = "error" if result["error"] else "no_text"
            chunks = [
                chunk
                for page in result["pages"]
                for chunk in chunks_for_page(doc, page["page"], page["text"])
            ]
            if chunks:
                status = "partial" if result["warnings"] else "indexed"
            db.execute(
                """
                INSERT INTO documents (
                    id, title, filename, family, version, version_date, is_latest,
                    sha256, status, error, page_count, warnings
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    doc["id"],
                    doc.get("family") or doc["title"],
                    doc["filename"],
                    doc.get("family") or doc["title"],
                    doc.get("version") or "",
                    doc.get("version_date") or "",
                    int(doc.get("is_latest") or 0),
                    file_hash,
                    status,
                    result["error"],
                    len(result["pages"]),
                    json.dumps(result["warnings"]),
                ),
            )
            for offset in range(0, len(chunks), batch_size):
                store.add_documents(chunks[offset : offset + batch_size])
            chunks_total += len(chunks)
            db.commit()
            progress(
                f"[{number}/{len(documents)}] {doc['filename']}: {status}, "
                f"{len(chunks)} passages; {chunks_total} total"
            )
        if not chunks_total:
            raise RagError(
                "No passages were extracted; the previous index was preserved."
            )
        after = sum(
            1
            for key in LocalFileStore(index_dir / "embedding-cache").yield_keys()
            if key.startswith(namespace)
        )
        collection = client.get_collection(COLLECTION, embedding_function=None)
        if collection.count() != chunks_total:
            raise RagError(
                "Chroma passage count differs from ingestion; the previous index was preserved."
            )
        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "generation": generation,
            "created_at": datetime.now(UTC).isoformat(),
            "data_dir": str(data_dir.resolve()),
            "embedding_model": EMBED_MODEL,
            "embedding_digest": models.digest(EMBED_MODEL),
            "vector_store": "chromadb",
            "collection": COLLECTION,
            "chunks": chunks_total,
            "documents": len(documents),
            "extraction_version": EXTRACTION_VERSION,
            "splitter": "RecursiveCharacterTextSplitter",
            "chunk_bytes": CHUNK_BYTES,
            "overlap_bytes": OVERLAP_BYTES,
            "extracted_documents": extracted,
            "reused_extractions": reused,
            "new_embeddings": after - before,
            "reused_embeddings": chunks_total - (after - before),
            "ingestion_seconds": round(time.perf_counter() - started, 2),
        }
        (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        pointer = index_dir / "current.tmp"
        pointer.write_text(json.dumps({"generation": generation}) + "\n")
        pointer.replace(index_dir / "current.json")
        return manifest
    finally:
        db.close()
        client.close()

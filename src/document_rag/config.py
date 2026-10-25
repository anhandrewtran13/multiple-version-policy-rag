"""Model names, indexing defaults, and local environment loading."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

CHAT_MODEL = "claude-sonnet-4-5"
HF_CHAT_MODEL = os.environ.get("HF_CHAT_MODEL", "Qwen/Qwen2.5-3B-Instruct").strip()
EMBED_MODEL = "embeddinggemma:300m"
OLLAMA_HOST = "http://localhost:11434"
# EmbeddingGemma's context is about 2K tokens. Stay under a 1,900-byte input.
EMBED_INPUT_BYTES = 1900
CHUNK_BYTES = 1400
OVERLAP_BYTES = 250
EVIDENCE_CHAR_BUDGET = 80_000
SUMMARY_CHAR_BUDGET = 80_000


class RagError(Exception):
    """An actionable error suitable for display without a traceback."""


def require_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RagError(f"Missing {name}. Set it in document-rag/.env.")
    return value


def configured_provider():
    """Generation backend from CHAT_PROVIDER. Unknown values use the API model."""
    value = os.environ.get("CHAT_PROVIDER", "anthropic").strip().lower()
    if value in {"anthropic", "huggingface"}:
        return value
    return "anthropic"

"""Serve the Change Tracker site and the RAG ask API."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from document_rag.config import RagError, configured_provider
from document_rag.models import ApiModels
from document_rag.section_ask import ask_document

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
INDEX = ROOT / "index"


def _models(provider):
    if provider not in {"anthropic", "huggingface"}:
        provider = configured_provider()
    return ApiModels(provider=provider)


class Handler(BaseHTTPRequestHandler):
    server_version = "ChangeTracker/1.0"

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _send(self, status, body, content_type):
        data = body if isinstance(body, bytes) else body.encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in {"", "/"}:
            path = "/index.html"
        target = (SITE / path.lstrip("/")).resolve()
        try:
            target.relative_to(SITE.resolve())
        except ValueError:
            self._send(404, "Not found", "text/plain; charset=utf-8")
            return
        if not target.is_file():
            self._send(404, "Not found", "text/plain; charset=utf-8")
            return
        kind = "text/html; charset=utf-8"
        if target.suffix == ".json":
            kind = "application/json"
        elif target.suffix == ".js":
            kind = "text/javascript; charset=utf-8"
        self._send(200, target.read_bytes(), kind)

    def do_POST(self):
        if urlparse(self.path).path != "/api/ask":
            self._send(404, "Not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "The request was not valid JSON."}), "application/json")
            return
        question = str(payload.get("question") or "")
        document_id = str(payload.get("document_id") or "")
        provider = str(payload.get("provider") or configured_provider())
        try:
            result = ask_document(
                SITE / "data",
                _models(provider),
                question,
                document_id=document_id or None,
                cache_dir=INDEX,
            )
        except (RagError, OSError, ValueError) as exc:
            self._send(400, json.dumps({"error": str(exc)}), "application/json")
            return
        except Exception as exc:
            self._send(500, json.dumps({"error": f"Ask failed: {exc}"}), "application/json")
            return
        body = {
            "answer": result.get("answer") or "",
            "supported": bool(result.get("supported")),
            "chat_model": result.get("chat_model"),
            "chat_provider": result.get("chat_provider"),
            "sources": [
                {
                    "source_id": source.get("source_id"),
                    "title": source.get("title"),
                    "version": source.get("version"),
                    "section_id": source.get("section_id"),
                    "page": source.get("page"),
                    "text": source.get("text"),
                    "href": source.get("href"),
                }
                for source in result.get("sources") or []
            ],
        }
        self._send(200, json.dumps(body), "application/json")


def main():
    """Open the tracker. Ask uses the same generator as document-rag ask."""
    os.chdir(ROOT)
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Change Tracker at http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

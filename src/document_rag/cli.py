"""Command-line interface for PDF question answering and summarization."""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from document_rag.answer import ask
from document_rag.config import CHAT_MODEL, HF_CHAT_MODEL, RagError, configured_provider
from document_rag.diff import build_changes
from document_rag.index import Index, resolve_files
from document_rag.ingest import ingest
from document_rag.models import ApiModels
from document_rag.retrieve import Retriever
from document_rag.summarize import summarize
from document_rag.versions import scope_for_question


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser():
    root = argparse.ArgumentParser(
        description="Ask questions, compare versions, and summarize PDFs."
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("ingest", "Extract PDFs and build or replace the local index"),
        ("diff", "Compare consecutive versions and index section change notes"),
        ("search", "Show passages found by hybrid retrieval"),
        ("ask", "Answer a question with page citations"),
        ("summarize", "Summarize one indexed PDF"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--index-dir", type=Path, default=Path("index"))
        command.add_argument(
            "--json", action="store_true", help="Print machine-readable output"
        )
        if name == "ingest":
            command.add_argument("--data-dir", type=Path, required=True)
            command.add_argument("--batch-size", type=positive, default=16)
        if name in ("search", "ask"):
            command.add_argument("question")
            command.add_argument(
                "--file",
                action="append",
                default=[],
                help="Filename or stem; repeat to limit retrieval to those PDFs",
            )
        if name == "search":
            command.add_argument("--top-k", type=positive, default=6)
        if name == "diff":
            command.add_argument("--from-version", help="Older version, such as 1.3.0")
            command.add_argument("--to-version", help="Newer version, such as 1.4.0")
        if name in ("ask", "summarize", "diff"):
            command.add_argument(
                "--provider",
                choices=["anthropic", "huggingface"],
                default=configured_provider(),
                help="Text generation: anthropic needs ANTHROPIC_API_KEY; huggingface runs a local model",
            )
            command.add_argument(
                "--model",
                default=None,
                help="Anthropic model name, or a Hugging Face model id when --provider huggingface",
            )
        if name == "summarize":
            command.add_argument(
                "--file", required=True, help="Filename or stem of one indexed PDF"
            )
    return root


def generation_models(args):
    if args.model:
        chat_model = args.model
    elif args.provider == "huggingface":
        chat_model = HF_CHAT_MODEL
    else:
        chat_model = CHAT_MODEL
    return ApiModels(chat_model, provider=args.provider)


def render_sources(sources):
    for source in sources:
        version = f", version {source['version']}" if source.get("version") else ""
        section = f", section {source['section_id']}" if source.get("section_id") else ""
        print(
            f"\n[{source['source_id']}] {source['title']}{version}{section}\n"
            f"  {source['filename']}, page {source['page']}"
        )


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    try:
        if args.command == "ingest":
            result = ingest(
                args.data_dir,
                args.index_dir,
                ApiModels(),
                batch_size=args.batch_size,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            print(
                json.dumps(result, indent=2)
                if args.json
                else f"Indexed {result['documents']} PDFs and {result['chunks']} passages in "
                f"{result['ingestion_seconds']:.1f}s."
            )
            return 0
        with Index(args.index_dir) as index:
            if args.command == "diff":
                result = build_changes(
                    index,
                    generation_models(args),
                    from_version=args.from_version,
                    to_version=args.to_version,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
                if not args.json:
                    print(
                        f"Indexed {result['changes']} section change notes "
                        f"across {len(result['pairs'])} version pair(s)."
                    )
                    for note in result["notes"]:
                        print(
                            f"- {note['from_version']} -> {note['to_version']} "
                            f"{note['section_id']} {note['title']}: {note['change']}"
                        )
            elif args.command == "search":
                file_ids, where = scope_for_question(
                    args.question, resolve_files(index.db, args.file)
                )
                sources = Retriever(index, ApiModels()).search(
                    args.question, file_ids=file_ids, where=where, top_k=args.top_k
                )
                result = {"question": args.question, "sources": sources}
                if not args.json:
                    print(f"{len(sources)} matching passages:")
                    for source in sources:
                        render_sources([source])
                        print(f"  Rank: {source['rank']}\n{source['text']}")
            elif args.command == "ask":
                result = ask(
                    index,
                    generation_models(args),
                    args.question,
                    file_ids=resolve_files(index.db, args.file),
                )
                if not args.json:
                    print(result["answer"])
                    render_sources(result["sources"])
            else:
                result = summarize(index, generation_models(args), args.file)
                if not args.json:
                    print(result["answer"])
                    print(
                        f"\n({result['file']}, {result['pages']} pages, {result['mode']})"
                    )
            result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (RagError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Completed ingestion batches are cached; rerun to resume.",
            file=sys.stderr,
        )
        return 130

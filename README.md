# Medicaid Disability Manual

A local reader for three editions of the Medicaid Disability Manual. The Change Tracker page shows what changed between editions, section by section. A question command answers from the manual text and cites the passage it used.

This is a reading aid for the PDFs in this repository. It is not a determination of eligibility, and it does not replace the manuals themselves.

## Editions

The PDFs in `data/` are one manual, named so the version and date can be read from the filename:

| File | Edition |
| --- | --- |
| `Medicaid Disability Manual-V.2024.12.0-20241201.pdf` | December 2024 |
| `Medicaid Disability Manual-V.2025.2.0-20250201.pdf` | February 2025 |
| `Medicaid Disability Manual-V.2026.1.0-20260101.pdf` | January 2026 |

The January 2026 file is the edition whose first page says January 2026. The date in that filename is the file stamp `20260101`.

The same filename pattern works for other manuals: `Name-V.1.4.0-20260403.pdf`. The stem before `-V.` is the document family, the middle number is the version, and the trailing digits are the date (`YYYYMMDD` or `YYYYMM`).

## What you can do

**Change Tracker.** `site/index.html` loads the extracted text in `site/data/` and marks additions, deletions, and edits between two editions. The outline follows the manual’s numbered sections. Pick a pair of versions, or use the shortcuts for December 2024 to February 2025 and February 2025 to January 2026.

**Ask.** The Ask panel on that page searches the open manual’s sections and returns a short answer with links back to the sections it used. Questions about what changed also search the change notes in `judgments.json` when that file contains them. It is empty in this repository, so the page shows the textual diff and labels it as not reviewed for substance.

**Command line.** `document-rag` builds a separate local index from the PDFs. Use it to search passages, ask a question with page citations, summarize one PDF, or write a note for each section that changed between consecutive versions.

The page and the command line do not share an index. The page reads `site/data/`. The commands below read `index/`, which is created on your machine and is not committed.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/), for the local embedding model
- An Anthropic API key, unless you generate text with a local Hugging Face model

Embeddings always run locally (`embeddinggemma:300m` through Ollama). Answers use Claude Sonnet 4.5 by default.

## Setup

```sh
uv sync
ollama pull embeddinggemma:300m
```

Create a `.env` file in the project root. It is gitignored.

```sh
ANTHROPIC_API_KEY=your-key
```

To generate text on this machine instead, install the local extra and set the provider. The default local model is `Qwen/Qwen2.5-3B-Instruct`. Override it with `HF_CHAT_MODEL`.

```sh
uv sync --extra local
```

```sh
CHAT_PROVIDER=huggingface
```

`CHAT_PROVIDER` may be `anthropic` or `huggingface`. Each command also accepts `--provider` and `--model`.

## Run the Change Tracker

The extracted manual is already in `site/data/`. Start the local server, which serves that page and the Ask API:

```sh
uv run document-rag-site
```

Open http://127.0.0.1:8765. Set `PORT` to use another port. The server listens on `127.0.0.1` only.

Ask needs Ollama running for semantic ranking. If Ollama is down, Ask still searches with keyword ranking. Anthropic answers need `ANTHROPIC_API_KEY`. Choose **Local Hugging Face** in the Ask panel when `CHAT_PROVIDER` is not how you want that one question answered.

## Command-line index

From the project root, with Ollama running:

```sh
uv run document-rag ingest --data-dir data
uv run document-rag search "disability determination"
uv run document-rag ask "What changed about disability?"
uv run document-rag summarize --file "Medicaid Disability Manual-V.2026.1.0-20260101"
```

`ingest` extracts each PDF, splits the pages into short passages, and stores them in a local Chroma database under `index/`. Rerunning it resumes completed batches. Questions about the manual search the newest edition unless the question names a version (such as `2025.2.0`) or you pass `--file` one or more times.

`ask` retrieves passages with embeddings and BM25, then asks the model to answer only from those passages. The reply cites the file, version, section, and page. If the passages do not support an answer, the command says so.

Compare consecutive editions and index a short note for each section whose text changed:

```sh
uv run document-rag diff
uv run document-rag diff --from-version 2025.2.0 --to-version 2026.1.0
```

`diff` splits each PDF on numbered headings, ignores unchanged wording, page-number shifts, and DocuSign envelope ids, and asks the model what requirement, date, population, amount, or process changed. Notes that the model calls non-substantive are left out of the index. A question that asks what changed, such as “What changed between 2025.2.0 and 2026.1.0?”, searches those notes.

`--json` on any command prints the same result as JSON. `document-rag --help` lists the rest of the flags.

## Layout

```text
data/                  the three manual PDFs
site/index.html        Change Tracker page
site/data/             extracted sections, version config, and the document list
src/document_rag/      ingest, diff, retrieval, answers, and the local server
index/                 local Chroma index (created by ingest and diff; gitignored)
tests/                 pipeline tests that do not call the model APIs
```

`site/data/library.json` lists the manuals the page can open. Each manual is a folder with:

- `history.json` — versions, section headings, and the text of each section in each edition
- `config.json` — labels, edition dates, and which pair opens first
- `judgments.json` — optional reviewed notes on what a change means (`null` until those notes exist)

## Tests

```sh
uv run pytest
```

The tests cover version parsing, section comparison, chunk size, and the shape of an answer. They do not call Anthropic or Ollama.

## License and source text

The manuals are California Medicaid policy documents. Keep them with their original wording when you share this project, and treat generated summaries as a guide back to the cited page.

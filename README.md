# CiteSage

A citation-grounded research assistant. CiteSage downloads arXiv papers,
indexes them, and answers questions **strictly from those papers** — every
answer carries verifiable citations (paper, page, section) that are resolved
against the retrieved evidence before anything is shown. If the indexed
papers don't support an answer, it refuses instead of guessing.

100% free stack: free arXiv API, free-tier LLMs (Groq / Gemini), a local
embedding model, and local indexes. No paid services anywhere.

## How it works

```
fetch ──► extract ──► chunk ──► index ──► search / ask
arXiv     PDF text    token-    dense (Chroma)     hybrid retrieval
PDFs      + pages     aware     + sparse (BM25)    (dense + BM25, RRF)
                                chunks             + grounded answer
```

- **Hybrid retrieval** — dense vectors (all-MiniLM-L6-v2 via
  sentence-transformers, stored in Chroma) fused with a BM25 keyword index
  using reciprocal rank fusion.
- **Grounded generation** — the LLM (Groq first, Gemini as fallback) may
  only use the retrieved chunks, and every citation marker in its answer is
  validated and resolved to a real chunk. Unresolvable citations fail the
  answer rather than slipping through.

## Requirements

- Python 3.10+
- For the `ask` command only: a free API key from
  [Groq](https://console.groq.com) and/or
  [Google AI Studio](https://aistudio.google.com) (Gemini). Everything else
  runs locally with no keys.
- First `index` run downloads the MiniLM embedding model (~90 MB).

## Setup

```bash
git clone https://github.com/AtharvaAher26/citesage
cd citesage
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

(or simply `make setup`, which does exactly this)

Then configure your keys:

```bash
cp .env.example .env
# edit .env and set at least one of:
#   CITESAGE_GROQ_API_KEY=...
#   CITESAGE_GEMINI_API_KEY=...
```

## Usage

```bash
# 1. Fetch papers on a topic from arXiv (idempotent — safe to re-run)
citesage fetch "retrieval augmented generation" --max-papers 10

# 2. Extract text and build the search index
citesage extract
citesage chunk
citesage index

# …or steps 1–2's fetch/extract/chunk in one go:
citesage pipeline "retrieval augmented generation" --max-papers 5

# 3. Search the index (no LLM involved)
citesage search "how does RRF fusion weight the two rankings?" --top-k 5

# 4. Ask a question — answered only from the indexed papers, with sources
citesage ask "What are the main failure modes of dense retrieval?"
```

Example `search` output:

```
3 result(s) for: how does RRF fusion weight the two rankings?

[1] score=0.0321  dense#2  sparse#1
    Smith et al., 2024 — "Hybrid Search in Practice", p. 4, §3.2
    Reciprocal rank fusion combines rankings by summing 1/(k + rank)
    across systems, which removes the need to normalise scores...
```

Example `ask` output:

```
ANSWER:
Dense retrieval fails mainly on out-of-vocabulary terms and exact-match
queries, where the embedding space has no reason to place the query near
the relevant passage [1]. ...

SOURCES:
  [1] Smith et al., 2024 — "Hybrid Search in Practice", p. 4, §3.2

(answered by groq)
```

If the evidence isn't there, `ask` says so:

```
NO SUPPORTING EVIDENCE in the indexed papers for this question.
```

## Web UI

The same pipeline is available as a local web app — a library sidebar with
fetch-and-index, plus Ask/Search panels — served by a thin FastAPI layer
(`src/citesage/web/`) over the exact code paths the CLI uses.

```bash
pip install -e ".[web]"   # adds FastAPI + uvicorn (optional extra)
citesage serve            # -> http://127.0.0.1:8000  (--host / --port to change)
```

Then open http://127.0.0.1:8000 in a browser.

- **Library sidebar** — every fetched paper (title, authors, year, arXiv
  link) and a fetch form: enter a topic, hit *Fetch & index*, and watch the
  fetch → extract → chunk → index job progress live.
- **Ask** — answers rendered with clickable `[n]` citation chips that jump
  to the matching source card. Refusals ("no supporting evidence") are shown
  as an honest notice, exactly like the CLI.
- **Search** — raw hybrid-retrieval results with fused score, dense/sparse
  ranks, citation and snippet. Works with **no LLM key configured**; Ask
  needs a Groq/Gemini key in `.env` and says so in a banner when it's missing.
- The header pill always shows index state (papers / chunks / ready), and
  the JSON API is documented at http://127.0.0.1:8000/api/docs.

_Screenshot placeholder — run `citesage serve`, fetch a topic, and drop a
screenshot of the Ask view here._

## Configuration

All settings live in one place (`src/citesage/config.py`) and can be
overridden with `CITESAGE_*` environment variables or a `.env` file:

| Variable | Default | Purpose |
|---|---|---|
| `CITESAGE_DATA_DIR` | `data` | Where papers, chunks and indexes live |
| `CITESAGE_CHUNK_TOKENS` | `220` | Chunk size (tuned for MiniLM's 256-token window) |
| `CITESAGE_CHUNK_OVERLAP` | `30` | Overlap between consecutive chunks |
| `CITESAGE_LOG_LEVEL` | `INFO` | Logging verbosity |
| `CITESAGE_GROQ_API_KEY` | — | Groq key (primary LLM) |
| `CITESAGE_GEMINI_API_KEY` | — | Gemini key (fallback LLM) |
| `CITESAGE_GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model |
| `CITESAGE_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model |

Downloaded papers, chunks and indexes are written under `data/` (gitignored).

## Tests

```bash
pytest -q        # or: make test
```

The suite runs fully offline — LLM clients are tested against mock HTTP
transports, and the rest against small synthetic data in temp dirs. For an
offline end-to-end run of the pipeline itself, `python make_fixtures.py`
creates synthetic papers you can extract, chunk and index without arXiv.

## Project layout

```
src/citesage/
  cli.py          Typer CLI: fetch / extract / chunk / pipeline / index / search / ask
  config.py       All settings (CITESAGE_* env vars)
  fetch/          arXiv search + PDF download, paper store
  processing/     PDF text extraction, token-aware chunking
  retrieval/      Embeddings, Chroma vector store, BM25 index, hybrid (RRF) retriever
  generation/     LLM clients (Groq + Gemini fallback), grounded answerer, citation resolver
  web/            Optional FastAPI app + vanilla-JS frontend (the `serve` command)
tests/            Offline pytest suite
```

## License

MIT — see [LICENSE](LICENSE).

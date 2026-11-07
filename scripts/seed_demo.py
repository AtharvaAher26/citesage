"""Seed a small, ready-to-query demo corpus: three classic retrieval papers.

Fetches the original RAG paper, DPR, and ColBERT from arXiv and runs the
full CiteSage pipeline (fetch -> extract -> chunk -> index) into a data
directory, using the same pipeline functions the CLI and web UI use -
nothing here shells out to the `citesage` command.

    python scripts/seed_demo.py                 # into ./data (or $CITESAGE_DATA_DIR)
    python scripts/seed_demo.py --data-dir DIR  # into DIR instead
    python scripts/seed_demo.py --only 2004.04906   # just one paper

Idempotent: a second run over a complete corpus prints that there is
nothing to do and exits. Partial corpora are completed, not restarted.
Every step prints exactly what it did.

Note on IDs: CiteSage stores arXiv IDs with their version suffix
("2004.04906v2"), while humans type the bare ID ("2004.04906"). This
script takes bare IDs and resolves them to whatever the store holds.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

from citesage.config import Settings
from citesage.fetch.arxiv_client import ArxivClient, ArxivFetchError
from citesage.fetch.storage import PaperStore
from citesage.processing.chunker import chunk_paper
from citesage.processing.extractor import ExtractionError, extract_pdf
from citesage.retrieval.indexer import build_indexes

# The demo corpus: three stable, well-known papers that between them cover
# the questions a first-time visitor is most likely to ask CiteSage.
DEMO_PAPERS = [
    "2005.11401",  # Retrieval-Augmented Generation (Lewis et al., 2020)
    "2004.04906",  # Dense Passage Retrieval (Karpukhin et al., 2020)
    "2004.12832",  # ColBERT (Khattab & Zaharia, 2020)
]

_VERSION_SUFFIX = re.compile(r"v\d+$")


def _bare(arxiv_id: str) -> str:
    """'2004.04906v2' -> '2004.04906' (the form users type)."""
    return _VERSION_SUFFIX.sub("", arxiv_id)


def _store_key(store: PaperStore, bare_id: str) -> str | None:
    """The store's (versioned) key for a bare ID, or None if absent."""
    for pid in store.papers:
        if _bare(pid) == bare_id:
            return pid
    return None


def _interim_path(interim_dir: Path, store_key: str) -> Path:
    return interim_dir / f"{store_key.replace('/', '_')}.json"


def _already_seeded(store: PaperStore, wanted: list[str], interim_dir: Path,
                    chunks_path: Path, index_dir: Path) -> int | None:
    """Return the indexed chunk count if the corpus is complete, else None."""
    keys = [_store_key(store, pid) for pid in wanted]
    if any(k is None for k in keys):
        return None
    if not all(_interim_path(interim_dir, k).exists() for k in keys if k):
        return None
    if not chunks_path.exists() or not (index_dir / "bm25.json").exists():
        return None
    return sum(1 for line in chunks_path.open(encoding="utf-8") if line.strip())


def seed(data_dir: Path, wanted: list[str]) -> int:
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    index_dir = data_dir / "index"
    chunks_path = processed_dir / "chunks.jsonl"
    for d in (raw_dir, interim_dir, processed_dir, index_dir):
        d.mkdir(parents=True, exist_ok=True)

    store = PaperStore(raw_dir / "papers.json")

    done_chunks = _already_seeded(store, wanted, interim_dir, chunks_path, index_dir)
    if done_chunks is not None:
        print(f"Demo corpus already seeded in {data_dir}: "
              f"{len(wanted)} papers, {done_chunks} chunks indexed. Nothing to do.")
        return 0

    failures: list[str] = []

    # ---- fetch (metadata by ID, then PDFs) -------------------------------
    missing = [pid for pid in wanted if _store_key(store, pid) is None]
    if missing:
        print(f"Fetching metadata for {len(missing)} paper(s) from arXiv...")
        try:
            papers = ArxivClient().fetch_by_ids(missing)
        except ArxivFetchError as exc:
            print(f"error: {exc}")
            return 1
        found = {_bare(p.arxiv_id): p for p in papers}
        for pid in missing:
            if pid not in found:
                print(f"  {pid}: NOT FOUND on arXiv")
                failures.append(pid)
        to_download = [found[pid] for pid in missing if pid in found]
    else:
        to_download = []
        print("All demo papers already in the store - no metadata fetch needed.")

    client = ArxivClient()
    for paper in to_download:
        pdf_path = client.download_pdf(paper, raw_dir)
        paper.pdf_path = str(pdf_path)
        store.upsert(paper)
        print(f"  downloaded {paper.arxiv_id} - {paper.title}")
        time.sleep(1.0)  # politeness between downloads, as in the CLI
    if to_download:
        store.save()

    # Papers we already had still need their PDF path to resolve locally
    # (a store written by another data dir would point somewhere stale).
    for bare_id in wanted:
        key = _store_key(store, bare_id)
        if key is None:
            continue
        paper = store.papers[key]
        candidate = raw_dir / f"{key.replace('/', '_')}.pdf"
        if (not paper.pdf_path or not Path(paper.pdf_path).exists()) and candidate.exists():
            paper.pdf_path = str(candidate)

    # ---- extract ----------------------------------------------------------
    for bare_id in wanted:
        key = _store_key(store, bare_id)
        if key is None:
            continue  # already reported under fetch failures
        paper = store.papers[key]
        if _interim_path(interim_dir, key).exists():
            print(f"  extract {key}: already extracted, skipped")
            continue
        if not paper.pdf_path or not Path(paper.pdf_path).exists():
            print(f"  extract {key}: PDF missing locally, skipped")
            failures.append(bare_id)
            continue
        try:
            extract_pdf(paper, interim_dir)
            print(f"  extracted {key}")
        except ExtractionError as exc:
            print(f"  extract {key}: REJECTED - {exc}")
            failures.append(bare_id)

    # ---- chunk (mirrors `citesage chunk`: every interim file) -------------
    files = sorted(interim_dir.glob("*.json"))
    if not files:
        print("error: nothing to chunk - extraction produced no papers.")
        return 1
    total = 0
    with chunks_path.open("w", encoding="utf-8") as f:
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            chunks = chunk_paper(payload["arxiv_id"], payload["title"], payload["pages"])
            for c in chunks:
                f.write(c.model_dump_json() + "\n")
            total += len(chunks)
            print(f"  chunked {payload['arxiv_id']}: {len(chunks)} chunks")
    print(f"Wrote {total} chunks -> {chunks_path}")

    # ---- index --------------------------------------------------------------
    print("Building indexes (first run downloads the MiniLM model, ~90 MB)...")
    stats = build_indexes(chunks_path=chunks_path, index_dir=index_dir)
    print(f"Indexed {stats['chunks']} chunks -> {index_dir} "
          f"(dense: {stats['dense_count']} vectors, sparse: {stats['sparse_count']} docs)")

    if failures:
        print(f"Finished with problems: {len(failures)} paper(s) incomplete: "
              f"{', '.join(failures)}")
        return 1

    print(f"Demo corpus ready: {len(wanted)} papers, {total} chunks in {data_dir}.")
    if data_dir == Path(Settings().data_dir):
        print("Now run:  citesage serve   then open http://127.0.0.1:8000")
    else:
        print(f"Now run:  CITESAGE_DATA_DIR={data_dir} citesage serve   "
              "then open http://127.0.0.1:8000")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Where the demo corpus lives (default: CITESAGE_DATA_DIR or ./data)",
    )
    parser.add_argument(
        "--only", action="append", default=None, metavar="ARXIV_ID",
        help="Seed only this paper (repeatable). Default: all three classics.",
    )
    args = parser.parse_args()

    # The script prints its own step-by-step summary; keep library logging
    # to warnings so that output stays exact and readable.
    logging.basicConfig(level=logging.WARNING)

    data_dir = args.data_dir if args.data_dir is not None else Path(Settings().data_dir)
    wanted = [_bare(pid) for pid in (args.only if args.only else DEMO_PAPERS)]
    return seed(data_dir, wanted)


if __name__ == "__main__":
    sys.exit(main())

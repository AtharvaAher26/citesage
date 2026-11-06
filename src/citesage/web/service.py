"""Service layer for the web UI: wraps the CLI pipeline behind plain methods.

The route handlers in app.py stay thin - everything interesting happens here.
The wiring intentionally mirrors cli.py (same PaperStore location, same
HybridRetriever construction, same answerer.ask semantics) so the web UI and
the CLI can never drift apart in behaviour.

Heavy pieces (embedder, Chroma, the LLM clients) are built lazily on first
use, exactly like the CLI's local imports, so starting the server is cheap
and search/ask degrade gracefully when pieces are missing:

- no index yet        -> index_ready is False; search returns no results,
                         ask returns a structured "no_index" error
- no LLM key in .env  -> ask returns {"error": "no_llm_key"} (HTTP 200);
                         search keeps working, it never needed a key
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from citesage.config import Settings
from citesage.fetch.storage import PaperStore

logger = logging.getLogger(__name__)


class JobBusyError(Exception):
    """Raised when a fetch/index job is already running."""


class CiteSageService:
    """One service instance per server process (the app is single-user/local)."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        settings: Settings | None = None,
    ) -> None:
        # A fresh Settings() (not the module-level singleton) mirrors `citesage
        # ask`, and lets tests inject keys/data dirs without touching env vars.
        self.settings = settings or Settings()
        self.data_dir = Path(data_dir) if data_dir is not None else Path(self.settings.data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        self.index_dir = self.data_dir / "index"
        self.chunks_path = self.processed_dir / "chunks.jsonl"
        self.store_path = self.raw_dir / "papers.json"

        self._retriever: Any | None = None  # lazy: built on first search/ask
        self._jobs: dict[str, dict[str, Any]] = {}
        self._jobs_lock = threading.Lock()
        self._active_job_id: str | None = None

    # ------------------------------------------------------------------
    # status / library
    # ------------------------------------------------------------------

    @property
    def index_ready(self) -> bool:
        """Same check the CLI uses: the BM25 file is the index's marker."""
        return (self.index_dir / "bm25.json").exists()

    @property
    def llm_configured(self) -> bool:
        return bool(self.settings.groq_api_key or self.settings.gemini_api_key)

    def _paper_count(self) -> int:
        if not self.store_path.exists():
            return 0
        try:
            return len(PaperStore(self.store_path).papers)
        except Exception:  # a corrupt manifest must not break the status pill
            logger.exception("could not read paper store at %s", self.store_path)
            return 0

    def _chunk_count(self) -> int:
        if self.index_ready:
            try:
                from citesage.retrieval.keyword_index import BM25Index

                return BM25Index.load(self.index_dir / "bm25.json").size
            except Exception:
                logger.exception("could not read BM25 index for chunk count")
                return 0
        if self.chunks_path.exists():  # chunked but not indexed yet
            return sum(1 for line in self.chunks_path.open(encoding="utf-8") if line.strip())
        return 0

    def status(self) -> dict[str, Any]:
        return {
            "papers": self._paper_count(),
            "chunks": self._chunk_count(),
            "index_ready": self.index_ready,
            "llm_configured": self.llm_configured,
        }

    def papers(self) -> list[dict[str, Any]]:
        """Fetched papers from the PaperStore manifest, newest first."""
        if not self.store_path.exists():
            return []
        store = PaperStore(self.store_path)
        out: list[dict[str, Any]] = []
        for paper in store.papers.values():
            out.append(
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.published.year if paper.published else None,
                    "url": f"https://arxiv.org/abs/{paper.arxiv_id}",
                }
            )
        out.sort(key=lambda p: (p["year"] is not None, p["year"] or 0), reverse=True)
        return out

    # ------------------------------------------------------------------
    # retrieval (lazy, mirrors cli.search / cli.ask construction)
    # ------------------------------------------------------------------

    def _get_retriever(self) -> Any:
        if self._retriever is None:
            # Local imports on purpose: chromadb / sentence-transformers are
            # heavy, and the server must start (and serve the library) even
            # if they are slow to load.
            from citesage.retrieval.embedder import Embedder
            from citesage.retrieval.hybrid import HybridRetriever
            from citesage.retrieval.keyword_index import BM25Index
            from citesage.retrieval.vector_store import VectorStore

            vector_store = VectorStore(self.index_dir)
            bm25 = BM25Index.load(self.index_dir / "bm25.json")
            self._retriever = HybridRetriever(
                vector_store=vector_store, bm25_index=bm25, embedder=Embedder()
            )
        return self._retriever

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict[str, Any]]:
        """Ranked chunks shaped like the CLI's search output."""
        if not self.index_ready:
            return []
        results = self._get_retriever().search(query, top_k=top_k, mode=mode)
        out: list[dict[str, Any]] = []
        for hit in results:
            snippet = " ".join(hit.text.split())[:300]
            if len(hit.text) > 300:
                snippet += "..."
            out.append(
                {
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                    "dense_rank": hit.dense_rank,
                    "sparse_rank": hit.sparse_rank,
                    "citation": hit.citation,
                    "title": hit.title,
                    "paper_id": hit.paper_id,
                    "section": hit.section,
                    "page_start": hit.page_start,
                    "snippet": snippet,
                }
            )
        return out

    # ------------------------------------------------------------------
    # grounded answering
    # ------------------------------------------------------------------

    def ask(self, question: str, top_k: int = 5, mode: str = "hybrid") -> dict[str, Any]:
        """Answer strictly from the index. Never raises for expected states:
        missing key / missing index / no evidence / refusal all come back as
        structured fields so the UI can render them honestly."""
        base: dict[str, Any] = {
            "answer_text": None,
            "sources": [],
            "refused": False,
            "answered_by": None,
            "warning": None,
            "error": None,
            "index_ready": self.index_ready,
        }
        if not self.index_ready:
            base["error"] = "no_index"
            return base
        if not self.llm_configured:
            # Search works without keys; only generation needs one.
            base["error"] = "no_llm_key"
            return base

        from citesage.generation.answerer import ask as ask_question
        from citesage.generation.llm import LLMError, build_clients

        clients = build_clients(
            groq_api_key=self.settings.groq_api_key,
            gemini_api_key=self.settings.gemini_api_key,
            groq_model=self.settings.groq_model,
            gemini_model=self.settings.gemini_model,
            provider="auto",
            timeout_seconds=self.settings.llm_timeout_seconds,
            max_output_tokens=self.settings.llm_max_output_tokens,
        )
        evidence = self._get_retriever().search(question, top_k=top_k, mode=mode)
        if not evidence:
            # Same meaning as the CLI's early return: nothing to ground on.
            base["refused"] = True
            base["answer_text"] = (
                "No evidence retrieved - try different wording or index more papers."
            )
            return base

        try:
            answer = ask_question(question, evidence, clients)
        except LLMError as exc:
            base["error"] = "llm_error"
            base["warning"] = str(exc)
            return base

        base["answer_text"] = answer.text
        base["refused"] = answer.refused
        base["answered_by"] = answer.provider
        base["warning"] = answer.warning
        base["sources"] = [
            {
                "n": citation.number,
                "citation": citation.reference,
                "paper": citation.paper_title,
                "arxiv_id": citation.arxiv_id,
                "page": citation.page_start,
                "section": citation.section,
            }
            for citation in answer.citations
        ]
        return base

    # ------------------------------------------------------------------
    # fetch -> extract -> chunk -> index background job
    # ------------------------------------------------------------------

    def start_fetch(self, topic: str, max_papers: int = 10) -> str:
        """Start the pipeline in a background thread. One job at a time."""
        with self._jobs_lock:
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active and active["status"] == "running":
                    raise JobBusyError("a fetch job is already running")
            job_id = uuid.uuid4().hex[:12]
            self._jobs[job_id] = {
                "id": job_id,
                "status": "running",
                "step": "queued",
                "topic": topic,
                "max_papers": max_papers,
                "error": None,
                "detail": None,
            }
            self._active_job_id = job_id
        thread = threading.Thread(
            target=self._run_pipeline, args=(job_id, topic, max_papers), daemon=True
        )
        thread.start()
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def _set_step(self, job_id: str, step: str, detail: str | None = None) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job["step"] = step
                if detail is not None:
                    job["detail"] = detail

    def _run_pipeline(self, job_id: str, topic: str, max_papers: int) -> None:
        """The four CLI steps, in order, updating the job as we go."""
        try:
            self._step_fetch(job_id, topic, max_papers)
            self._step_extract(job_id)
            self._step_chunk(job_id)
            self._step_index(job_id)
            # The index on disk changed - drop the cached retriever so the
            # next search/ask rebuilds it against the new index.
            self._retriever = None
            with self._jobs_lock:
                job = self._jobs[job_id]
                job["status"] = "done"
                job["step"] = "done"
                job["detail"] = self.status()
        except Exception as exc:  # surfaced to the UI via the job, not a crash
            logger.exception("fetch job %s failed", job_id)
            with self._jobs_lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self._jobs_lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _step_fetch(self, job_id: str, topic: str, max_papers: int) -> None:
        # Mirrors cli.fetch: search arXiv, download only papers we lack.
        from citesage.fetch.arxiv_client import ArxivClient

        self._set_step(job_id, "fetch", f"searching arXiv for '{topic}'")
        store = PaperStore(self.store_path)
        client = ArxivClient()
        papers = client.search(topic, max_papers)
        new = 0
        for paper in papers:
            if store.has(paper.arxiv_id):
                continue
            try:
                pdf_path = client.download_pdf(paper, self.raw_dir)
            except Exception as exc:  # one bad paper must not kill the batch
                logger.warning("download failed %s: %s", paper.arxiv_id, exc)
                continue
            paper.pdf_path = str(pdf_path)
            store.upsert(paper)
            new += 1
            time.sleep(1.0)  # politeness between downloads, as in the CLI
        store.save()
        self._set_step(job_id, "fetch", f"fetched {new} new papers (store total: {len(store.papers)})")

    def _step_extract(self, job_id: str) -> None:
        # Mirrors cli.extract: every stored PDF -> data/interim/*.json
        from citesage.processing.extractor import ExtractionError, extract_pdf

        self._set_step(job_id, "extract")
        store = PaperStore(self.store_path)
        ok = skipped = 0
        for pid, paper in store.papers.items():
            if not paper.pdf_path or not Path(paper.pdf_path).exists():
                logger.warning("missing pdf for %s", pid)
                continue
            try:
                extract_pdf(paper, self.interim_dir)
                ok += 1
            except ExtractionError as exc:
                skipped += 1
                logger.warning("rejected %s: %s", pid, exc)
        self._set_step(job_id, "extract", f"extracted {ok}, rejected {skipped}")

    def _step_chunk(self, job_id: str) -> None:
        # Mirrors cli.chunk: interim JSONs -> one chunks.jsonl
        from citesage.processing.chunker import chunk_paper

        self._set_step(job_id, "chunk")
        files = sorted(self.interim_dir.glob("*.json"))
        if not files:
            raise RuntimeError("Nothing to chunk - extraction produced no papers.")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        with self.chunks_path.open("w", encoding="utf-8") as f:
            for path in files:
                payload = json.loads(path.read_text(encoding="utf-8"))
                chunks = chunk_paper(payload["arxiv_id"], payload["title"], payload["pages"])
                for c in chunks:
                    f.write(c.model_dump_json() + "\n")
                total += len(chunks)
        self._set_step(job_id, "chunk", f"wrote {total} chunks")

    def _step_index(self, job_id: str) -> None:
        # Mirrors cli.index: dense (Chroma) + sparse (BM25) from chunks.jsonl
        from citesage.retrieval.indexer import build_indexes

        self._set_step(job_id, "index", "building indexes (first run downloads the embedding model)")
        stats = build_indexes(chunks_path=self.chunks_path, index_dir=self.index_dir)
        self._set_step(job_id, "index", f"indexed {stats['chunks']} chunks")

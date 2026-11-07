from __future__ import annotations

import logging
from pathlib import Path

import arxiv
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from citesage.config import settings
from citesage.models import Paper, normalize_arxiv_id

log = logging.getLogger(__name__)

PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
USER_AGENT = "citesage/0.1 (research assistant; educational use)"


def _to_papers(results: list[arxiv.Result]) -> list[Paper]:
    """Map arxiv.Result objects to our Paper model. One place, so topic
    search and id lookup can never drift apart in what they store."""
    return [
        Paper(
            arxiv_id=normalize_arxiv_id(r.entry_id),
            title=" ".join(r.title.split()),
            authors=[a.name for a in r.authors],
            abstract=" ".join(r.summary.split()),
            published=r.published,
            categories=list(r.categories),
        )
        for r in results
    ]


class ArxivFetchError(Exception):
    """arXiv search failed after retries."""


class ArxivRateLimitError(ArxivFetchError):
    """arXiv is throttling this IP (HTTP 429). Back off and retry later."""


class ArxivClient:
    """Well-behaved arXiv client: rate-limit compliant, retried, atomic downloads."""

    def __init__(self, delay: float | None = None):
        self._client = arxiv.Client(
            page_size=20,
            delay_seconds=delay or settings.arxiv_request_delay,
            num_retries=5,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=10, min=10, max=120),
        retry=retry_if_exception_type((ArxivRateLimitError,)),
    )
    def search(self, query: str, max_results: int) -> list[Paper]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            results = list(self._client.results(search))
        except Exception as exc:
            message = str(exc)
            if "429" in message:
                # arXiv throttled us — outer retry waits 10s, 20s, 40s, then gives up
                log.warning("arXiv rate limit hit; backing off: %s", message)
                raise ArxivRateLimitError(message) from exc
            raise ArxivFetchError(f"arXiv search failed for {query!r}: {exc}") from exc

        papers = _to_papers(results)
        log.info("arxiv search ok: query=%r found=%d", query, len(papers))
        return papers

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=10, min=10, max=120),
        retry=retry_if_exception_type((ArxivRateLimitError,)),
    )
    def fetch_by_ids(self, arxiv_ids: list[str]) -> list[Paper]:
        """Look up specific papers by arXiv ID (metadata only, no PDFs).

        Exists for scripts/seed_demo.py, which needs three exact papers
        rather than a topic search. PDF downloads stay in download_pdf,
        exactly as with search results.
        """
        search = arxiv.Search(id_list=list(arxiv_ids))
        try:
            results = list(self._client.results(search))
        except Exception as exc:
            message = str(exc)
            if "429" in message:
                # Same throttling contract as search: back off, then give up.
                log.warning("arXiv rate limit hit; backing off: %s", message)
                raise ArxivRateLimitError(message) from exc
            raise ArxivFetchError(f"arXiv id lookup failed for {arxiv_ids}: {exc}") from exc

        papers = _to_papers(results)
        log.info("arxiv id lookup ok: requested=%d found=%d", len(arxiv_ids), len(papers))
        return papers

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def download_pdf(self, paper: Paper, dest_dir: Path) -> Path:
        """Stream PDF to disk atomically (tmp file + rename). Skips if already present."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            log.info("download skipped (exists): %s", paper.arxiv_id)
            return dest

        url = PDF_URL.format(arxiv_id=paper.arxiv_id)
        tmp = dest.with_suffix(".tmp")
        try:
            with requests.get(url, timeout=30, stream=True,
                              headers={"User-Agent": USER_AGENT}) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for block in resp.iter_content(chunk_size=1 << 16):
                        f.write(block)
            tmp.replace(dest)  # atomic — never a half-written PDF
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        log.info("downloaded %s (%.1f KB)", paper.arxiv_id, dest.stat().st_size / 1024)
        return dest
"""Web UI tests: FastAPI TestClient over a CiteSageService on a temp data dir.

Everything heavy is faked, mirroring the rest of the suite: a stub retriever
stands in for Chroma/BM25/embedder, a FakeClient for the LLM providers, and
the fetch pipeline runner is replaced with a controllable stub. No network,
no embedding-model downloads, no real LLM calls.
"""

import threading
import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient

from citesage.config import Settings
from citesage.fetch.storage import PaperStore
from citesage.models import Paper
from citesage.retrieval.hybrid import RetrievedChunk
from citesage.retrieval.keyword_index import BM25Document, BM25Index
from citesage.web.app import create_app
from citesage.web.service import CiteSageService


# ----------------------------------------------------------------------
# fakes (same shapes as tests/test_answerer.py and tests/test_hybrid.py)
# ----------------------------------------------------------------------


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.name = "fake/model"

    def generate(self, system, user):
        return self.text


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=5, mode="hybrid"):
        return self._hits[:top_k]


def make_chunk(**overrides):
    fields = dict(
        chunk_id="2401.00001v1::chunk_000::abc123",
        text="Reciprocal rank fusion combines dense and sparse rankings.",
        score=0.032,
        paper_id="2401.00001v1",
        title="Fixture Paper",
        section="Methods",
        page_start=2,
        page_end=2,
        dense_rank=1,
        sparse_rank=3,
    )
    fields.update(overrides)
    return RetrievedChunk(**fields)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


def write_store(data_dir, count=2):
    store = PaperStore(data_dir / "raw" / "papers.json")
    for i in range(count):
        store.upsert(
            Paper(
                arxiv_id=f"2401.0000{i + 1}v1",
                title=f"Paper {i + 1}",
                authors=["Ada Lovelace", "Alan Turing"],
                abstract="x",
                published=datetime(2024 - i, 5, 1, tzinfo=timezone.utc),
                categories=["cs.IR"],
            )
        )
    store.save()


def write_index(data_dir, docs=3):
    index = BM25Index()
    index.build(
        [
            BM25Document(
                chunk_id=f"c{n}",
                text=f"chunk text {n}",
                metadata={"paper_id": "2401.00001v1", "title": "Paper 1"},
            )
            for n in range(docs)
        ]
    )
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index.save(index_dir / "bm25.json")


def make_service(tmp_path, with_key=True, **kwargs):
    settings = Settings(
        groq_api_key="test-key" if with_key else "",
        gemini_api_key="",
        data_dir=tmp_path,
    )
    return CiteSageService(data_dir=tmp_path, settings=settings, **kwargs)


def make_client(service):
    return TestClient(create_app(service))


# ----------------------------------------------------------------------
# status + papers
# ----------------------------------------------------------------------


def test_status_shape_on_empty_install(tmp_path):
    client = make_client(make_service(tmp_path, with_key=False))
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"papers", "chunks", "index_ready", "llm_configured"}
    assert body == {
        "papers": 0,
        "chunks": 0,
        "index_ready": False,
        "llm_configured": False,
    }


def test_status_counts_store_and_index(tmp_path):
    write_store(tmp_path, count=2)
    write_index(tmp_path, docs=3)
    client = make_client(make_service(tmp_path))
    body = client.get("/api/status").json()
    assert body["papers"] == 2
    assert body["chunks"] == 3
    assert body["index_ready"] is True
    assert body["llm_configured"] is True


def test_papers_endpoint_lists_store_newest_first(tmp_path):
    write_store(tmp_path, count=2)
    client = make_client(make_service(tmp_path))
    papers = client.get("/api/papers").json()
    assert len(papers) == 2
    first = papers[0]
    assert first["title"] == "Paper 1"
    assert first["year"] == 2024
    assert first["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert first["arxiv_id"] == "2401.00001v1"
    assert first["url"] == "https://arxiv.org/abs/2401.00001v1"


def test_papers_endpoint_empty_store(tmp_path):
    client = make_client(make_service(tmp_path))
    assert client.get("/api/papers").json() == []


# ----------------------------------------------------------------------
# search
# ----------------------------------------------------------------------


def test_search_without_index_reports_state(tmp_path):
    client = make_client(make_service(tmp_path))
    resp = client.post("/api/search", json={"query": "fusion"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["index_ready"] is False
    assert body["results"] == []


def test_search_result_shape_mirrors_cli(tmp_path):
    write_index(tmp_path)
    service = make_service(tmp_path)
    service._retriever = FakeRetriever([make_chunk()])
    client = make_client(service)
    body = client.post("/api/search", json={"query": "fusion", "top_k": 5}).json()
    assert body["index_ready"] is True
    (hit,) = body["results"]
    assert hit["score"] == pytest.approx(0.032)
    assert hit["dense_rank"] == 1
    assert hit["sparse_rank"] == 3
    assert hit["citation"] == "Fixture Paper (arXiv:2401.00001v1), p. 2 [Methods]"
    assert hit["snippet"].startswith("Reciprocal rank fusion")


# ----------------------------------------------------------------------
# ask
# ----------------------------------------------------------------------


def patch_clients(monkeypatch, text):
    monkeypatch.setattr(
        "citesage.generation.llm.build_clients",
        lambda **kwargs: [FakeClient(text)],
    )


def test_ask_success_with_citations(tmp_path, monkeypatch):
    write_index(tmp_path)
    service = make_service(tmp_path)
    service._retriever = FakeRetriever([make_chunk()])
    patch_clients(monkeypatch, "Fusion merges both rankings [1].")
    client = make_client(service)
    body = client.post("/api/ask", json={"question": "how does fusion work?"}).json()
    assert body["error"] is None
    assert body["refused"] is False
    assert body["answer_text"] == "Fusion merges both rankings [1]."
    assert body["answered_by"] == "fake/model"
    (source,) = body["sources"]
    assert source["n"] == 1
    assert source["paper"] == "Fixture Paper"
    assert source["page"] == 2
    assert source["section"] == "Methods"
    assert source["citation"].startswith("[1] Fixture Paper (arXiv:2401.00001v1)")


def test_ask_refusal_when_evidence_insufficient(tmp_path, monkeypatch):
    write_index(tmp_path)
    service = make_service(tmp_path)
    service._retriever = FakeRetriever([make_chunk()])
    patch_clients(monkeypatch, "NOT_ENOUGH_EVIDENCE\nNothing here covers quantum fields.")
    client = make_client(service)
    body = client.post("/api/ask", json={"question": "quantum fields?"}).json()
    assert body["refused"] is True
    assert body["sources"] == []
    assert "quantum" in body["answer_text"]


def test_ask_without_llm_key_but_search_still_works(tmp_path):
    write_index(tmp_path)
    service = make_service(tmp_path, with_key=False)
    service._retriever = FakeRetriever([make_chunk()])
    client = make_client(service)

    body = client.post("/api/ask", json={"question": "anything?"}).json()
    assert body["error"] == "no_llm_key"
    assert body["answer_text"] is None
    assert body["refused"] is False

    search = client.post("/api/search", json={"query": "fusion"}).json()
    assert search["index_ready"] is True
    assert len(search["results"]) == 1


def test_ask_without_index_reports_no_index(tmp_path):
    client = make_client(make_service(tmp_path))  # key set, but no index files
    body = client.post("/api/ask", json={"question": "anything?"}).json()
    assert body["error"] == "no_index"
    assert body["index_ready"] is False


# ----------------------------------------------------------------------
# fetch jobs
# ----------------------------------------------------------------------


def test_fetch_job_busy_conflict(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def blocking_pipeline(self, job_id, topic, max_papers):
        started.set()
        release.wait(timeout=15)
        with self._jobs_lock:
            self._jobs[job_id]["status"] = "done"
            self._jobs[job_id]["step"] = "done"
            if self._active_job_id == job_id:
                self._active_job_id = None

    monkeypatch.setattr(CiteSageService, "_run_pipeline", blocking_pipeline)
    client = make_client(make_service(tmp_path))

    resp = client.post("/api/fetch", json={"topic": "fusion", "max_papers": 3})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert started.wait(timeout=5)

    conflict = client.post("/api/fetch", json={"topic": "other", "max_papers": 1})
    assert conflict.status_code == 409

    running = client.get(f"/api/jobs/{job_id}").json()
    assert running["status"] == "running"
    assert running["topic"] == "fusion"

    release.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "done":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("job never reached done")


def test_unknown_job_is_404(tmp_path):
    client = make_client(make_service(tmp_path))
    assert client.get("/api/jobs/does-not-exist").status_code == 404


# ----------------------------------------------------------------------
# frontend
# ----------------------------------------------------------------------


def test_root_serves_frontend(tmp_path):
    client = make_client(make_service(tmp_path))
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "CiteSage" in resp.text
    assert "app.js" in resp.text

"""FastAPI app for the CiteSage web UI.

Routes are deliberately thin: every handler delegates to CiteSageService
(service.py), which owns the pipeline. This module only translates HTTP
<-> service calls and shapes error responses.

Run it with:  citesage serve        (or: uvicorn citesage.web.app:app)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from citesage.web.service import CiteSageService, JobBusyError

STATIC_DIR = Path(__file__).parent / "static"


class FetchRequest(BaseModel):
    topic: str = Field(min_length=1)
    max_papers: int = Field(default=10, ge=1, le=50)  # same bounds as the CLI


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)


def create_app(service: CiteSageService | None = None) -> FastAPI:
    """App factory - tests inject a service pointed at a temp data dir."""
    svc = service or CiteSageService()
    app = FastAPI(title="CiteSage", docs_url="/api/docs")
    app.state.service = svc

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return svc.status()

    @app.get("/api/papers")
    def papers() -> list[dict[str, Any]]:
        return svc.papers()

    @app.post("/api/fetch", status_code=202)
    def fetch(req: FetchRequest) -> dict[str, Any]:
        try:
            job_id = svc.start_fetch(req.topic, max_papers=req.max_papers)
        except JobBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        found = svc.get_job(job_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        return found

    @app.post("/api/search")
    def search(req: SearchRequest) -> dict[str, Any]:
        # No index is a normal state for a fresh install, not an error:
        # report it in-band so the UI can say "fetch some papers first".
        if not svc.index_ready:
            return {"index_ready": False, "query": req.query, "results": []}
        return {
            "index_ready": True,
            "query": req.query,
            "results": svc.search(req.query, top_k=req.top_k),
        }

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        # Always 200 for expected states (no index, no key, refusal, LLM
        # failure): the payload's error/refused fields carry the story.
        return svc.ask(req.question, top_k=req.top_k)

    # Registered last so /api/* wins over the static catch-all.
    # html=True makes GET / serve static/index.html.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()

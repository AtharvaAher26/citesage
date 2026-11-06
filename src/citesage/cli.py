from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import typer
from rich.console import Console

from citesage.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, settings
from citesage.fetch.arxiv_client import ArxivClient, ArxivFetchError, ArxivRateLimitError
from citesage.fetch.storage import PaperStore
from citesage.logging_conf import setup_logging
from citesage.processing.chunker import chunk_paper
from citesage.processing.extractor import ExtractionError, extract_pdf
from citesage.generation.llm import LLMError, build_clients, generate_with_fallback

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="CiteSage — citation-grounded research assistant",
)
console = Console()
log = logging.getLogger(__name__)


@app.command()
def fetch(topic: str, max_papers: int = typer.Option(10, min=1, max=50)):
    """Search arXiv and download PDFs into data/raw (idempotent)."""
    setup_logging(settings.log_level)
    store = PaperStore(RAW_DIR / "papers.json")
    client = ArxivClient()

    try:
        papers = client.search(topic, max_papers)
    except ArxivRateLimitError:
        console.print("[red]arXiv is rate-limiting this IP (HTTP 429).[/]")
        console.print("This is temporary — wait 60–90 seconds, then run the same command again.")
        raise typer.Exit(1)
    except ArxivFetchError as exc:
        console.print(f"[red]Search failed:[/] {exc}")
        raise typer.Exit(1)

    new = 0
    for i, paper in enumerate(papers, start=1):
        if store.has(paper.arxiv_id):
            console.print(f"[dim]{i}. skip {paper.arxiv_id} (already fetched)[/]")
            continue
        console.print(f"{i}. downloading [cyan]{paper.arxiv_id}[/] — {paper.title[:60]}…")
        try:
            pdf_path = client.download_pdf(paper, RAW_DIR)
        except Exception as exc:  # keep going: one bad paper must not kill the batch
            log.warning("download failed %s: %s", paper.arxiv_id, exc)
            console.print(f"   [yellow]failed: {exc}[/]")
            continue
        paper.pdf_path = str(pdf_path)
        store.upsert(paper)
        new += 1
        time.sleep(1.0)  # politeness between downloads

    store.save()
    console.print(f"[green]✔ fetched {new} new papers[/] (store total: {len(store.papers)})")


@app.command()
def extract():
    """Extract text from all downloaded PDFs -> data/interim/*.json."""
    setup_logging(settings.log_level)
    store = PaperStore(RAW_DIR / "papers.json")
    ok = skipped = 0
    for pid, paper in store.papers.items():
        if not paper.pdf_path or not Path(paper.pdf_path).exists():
            log.warning("missing pdf for %s", pid)
            continue
        try:
            extract_pdf(paper, INTERIM_DIR)
            ok += 1
        except ExtractionError as exc:
            skipped += 1
            console.print(f"[yellow]rejected {pid}: {exc}[/]")
    console.print(f"[green]✔ extracted {ok}[/], rejected {skipped}")


@app.command()
def chunk():
    """Chunk all extracted papers -> data/processed/chunks.jsonl."""
    setup_logging(settings.log_level)
    files = sorted(INTERIM_DIR.glob("*.json"))
    if not files:
        console.print("[red]Nothing to chunk — run `extract` first.[/]")
        raise typer.Exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "chunks.jsonl"
    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            chunks = chunk_paper(payload["arxiv_id"], payload["title"], payload["pages"])
            for c in chunks:
                f.write(c.model_dump_json() + "\n")
            total += len(chunks)
            console.print(f"  {payload['arxiv_id']}: {len(chunks)} chunks")
    console.print(f"[green]✔ wrote {total} chunks -> {out_path}[/]")


@app.command()
def pipeline(topic: str, max_papers: int = typer.Option(5)):
    """Full Phase-1A pipeline: fetch -> extract -> chunk."""
    fetch(topic=topic, max_papers=max_papers)
    extract()
    chunk()


@app.command()
def index(
    chunks: Path = typer.Option(Path("data/processed/chunks.jsonl"), "--chunks", help="chunks.jsonl produced by 'citesage chunk'"),
    index_dir: Path = typer.Option(Path("data/index"), "--index-dir", help="Where the dense + sparse indexes live"),
) -> None:
    """Build retrieval indexes: dense (Chroma) + sparse (BM25)."""
    if not chunks.exists():
        typer.echo(f"chunks file not found: {chunks}")
        typer.echo("Run 'citesage chunk' first (or 'citesage pipeline').")
        raise typer.Exit(code=1)
    typer.echo("Building indexes (first run downloads the MiniLM model, ~90 MB)...")
    from citesage.retrieval.indexer import build_indexes  # local import: heavy deps only here

    stats = build_indexes(chunks_path=chunks, index_dir=index_dir)
    typer.echo(
        f"Indexed {stats['chunks']} chunks -> {index_dir}\n"
        f"  dense  (Chroma): {stats['dense_count']} vectors, dim {stats['dim']}\n"
        f"  sparse (BM25):   {stats['sparse_count']} docs"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language question"),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to return"),
    mode: str = typer.Option("hybrid", "--mode", help="hybrid | dense | sparse"),
    index_dir: Path = typer.Option(Path("data/index"), "--index-dir"),
) -> None:
    """Search the index and print ranked chunks with citation info."""
    if mode not in {"hybrid", "dense", "sparse"}:
        typer.echo(f"Unknown mode '{mode}' (use hybrid | dense | sparse)")
        raise typer.Exit(code=2)
    if not (index_dir / "bm25.json").exists():
        typer.echo(f"No index found in {index_dir}. Run:  citesage index")
        raise typer.Exit(code=1)
    from citesage.retrieval.embedder import Embedder  # local imports: heavy deps only here
    from citesage.retrieval.hybrid import HybridRetriever
    from citesage.retrieval.keyword_index import BM25Index
    from citesage.retrieval.vector_store import VectorStore

    vector_store = VectorStore(index_dir)
    bm25 = BM25Index.load(index_dir / "bm25.json")
    retriever = HybridRetriever(vector_store=vector_store, bm25_index=bm25, embedder=Embedder())
    results = retriever.search(query, top_k=top_k, mode=mode)
    if not results:
        typer.echo("No results.")
        return
    typer.echo(f"\n{len(results)} result(s) for: {query}")
    for position, hit in enumerate(results, start=1):
        typer.echo(
            f"\n[{position}] score={hit.score:.4f}  dense#{hit.dense_rank or '-'}  sparse#{hit.sparse_rank or '-'}"
        )
        typer.echo(f"    {hit.citation}")
        snippet = " ".join(hit.text.split())[:300]
        typer.echo(f"    {snippet}{'...' if len(hit.text) > 300 else ''}")

@app.command()
def ask(
    question: str = typer.Argument(..., help="Your research question"),
    top_k: int = typer.Option(5, "--top-k", help="Evidence chunks to retrieve"),
    mode: str = typer.Option("hybrid", "--mode", help="hybrid | dense | sparse"),
    provider: str = typer.Option("auto", "--provider", help="auto | groq | gemini"),
    index_dir: Path = typer.Option(Path("data/index"), "--index-dir"),
) -> None:
    """Answer a question strictly from the indexed papers, with resolved citations."""
    if mode not in {"hybrid", "dense", "sparse"}:
        typer.echo(f"Unknown mode '{mode}' (use hybrid | dense | sparse)")
        raise typer.Exit(code=2)
    if not (index_dir / "bm25.json").exists():
        typer.echo(f"No index found in {index_dir}. Run:  citesage index")
        raise typer.Exit(code=1)

    from citesage.config import Settings  # local imports: heavy deps only when used
    from citesage.generation.answerer import ask as ask_question
    from citesage.generation.llm import AllProvidersFailedError, LLMConfigError, build_clients
    from citesage.retrieval.embedder import Embedder
    from citesage.retrieval.hybrid import HybridRetriever
    from citesage.retrieval.keyword_index import BM25Index
    from citesage.retrieval.vector_store import VectorStore

    settings = Settings()
    try:
        clients = build_clients(
            groq_api_key=settings.groq_api_key,
            gemini_api_key=settings.gemini_api_key,
            groq_model=settings.groq_model,
            gemini_model=settings.gemini_model,
            provider=provider,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
        )
    except LLMConfigError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    vector_store = VectorStore(index_dir)
    bm25 = BM25Index.load(index_dir / "bm25.json")
    retriever = HybridRetriever(vector_store=vector_store, bm25_index=bm25, embedder=Embedder())
    evidence = retriever.search(question, top_k=top_k, mode=mode)
    if not evidence:
        typer.echo("No evidence retrieved - try different wording or index more papers.")
        return

    try:
        answer = ask_question(question, evidence, clients)
    except AllProvidersFailedError as exc:
        typer.echo(f"LLM providers unavailable: {exc}")
        typer.echo("Check the keys in .env, your quota, or your connection, then retry.")
        raise typer.Exit(code=1)
    except LLMError as exc:
        typer.echo(f"Generation failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo("")
    if answer.refused:
        typer.echo("NO SUPPORTING EVIDENCE in the indexed papers for this question.")
        typer.echo(f"Detail: {answer.text}")
        typer.echo(f"(provider: {answer.provider})")
        return

    typer.echo("ANSWER:")
    typer.echo(answer.text)
    typer.echo("")
    typer.echo("SOURCES:")
    for citation in answer.citations:
        typer.echo(f"  {citation.reference}")
    if answer.warning:
        typer.echo(f"  warning: {answer.warning}")
    typer.echo(f"\n(answered by {answer.provider})")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind"),
    port: int = typer.Option(8000, "--port", help="Port to listen on"),
) -> None:
    """Serve the web UI locally (requires the 'web' extra)."""
    try:
        # Local imports: FastAPI/uvicorn are optional deps - the CLI must
        # keep working for users who never installed the web extra.
        import uvicorn

        from citesage.web.app import create_app
    except ImportError:
        typer.echo("The web UI is not installed. Add it with:")
        typer.echo('  pip install -e ".[web]"')
        raise typer.Exit(code=1)
    typer.echo(f"CiteSage web UI -> http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()


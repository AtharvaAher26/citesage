# CiteSage web UI in a container.
#
#   docker build -t citesage .
#   docker run -p 8000:8000 -e DEMO=1 citesage
#
# DEMO=1 makes the entrypoint seed the 3-paper demo corpus on first start
# (see docker-entrypoint.sh). Ask additionally needs a free LLM key:
#   docker run -p 8000:8000 -e DEMO=1 -e CITESAGE_GROQ_API_KEY=... citesage
# Search works without any key. Expect an image of roughly 2-2.5 GB -
# PyTorch dominates; the CPU-only wheel below keeps it from being ~8 GB.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CITESAGE_DATA_DIR=/app/data \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only PyTorch first: the default PyPI wheel bundles CUDA libraries.
# Once this satisfies sentence-transformers' torch requirement, the much
# larger CUDA build is never downloaded.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Install the package (with the web extra) from source.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install ".[web]"

# Pre-download the MiniLM embedding model at build time, so the first
# index/search at runtime is not stalled on a ~90 MB download.
RUN python -c "from citesage.retrieval.embedder import Embedder; Embedder().encode(['warmup'])"

COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh && mkdir -p /app/data

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["citesage", "serve", "--host", "0.0.0.0", "--port", "8000"]

#!/bin/sh
# Container entrypoint: optionally seed the demo corpus, then serve.
#
# With DEMO=1 and no index present yet, the 3-paper demo corpus (RAG, DPR,
# ColBERT) is built first, so the UI at :8000 can answer and search the
# moment the container is up. The check is the index file itself, so
# restarts - and containers with a volume mounted over the data dir -
# never re-seed.
set -e

DATA_DIR="${CITESAGE_DATA_DIR:-/app/data}"

if [ "$DEMO" = "1" ] && [ ! -f "$DATA_DIR/index/bm25.json" ]; then
    echo "[entrypoint] DEMO=1 and no index in $DATA_DIR - seeding demo corpus..."
    # arXiv throttles busy IPs (HTTP 429); a failed seed must not kill the
    # container - serve anyway and let the UI's fetch form build a corpus.
    python scripts/seed_demo.py --data-dir "$DATA_DIR" \
        || echo "[entrypoint] WARNING: demo seeding did not complete (arXiv may be rate-limiting). Starting the UI anyway."
fi

exec "$@"

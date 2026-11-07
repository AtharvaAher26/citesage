.PHONY: setup test lint pipeline demo
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check src tests
pipeline:
	citesage pipeline "$(TOPIC)" --max-papers 5
demo:
	python scripts/seed_demo.py

.PHONY: help sync sync-dev check lint fmt test serve build-index

help:
	@echo "Targets:"
	@echo "  make sync         uv sync runtime deps"
	@echo "  make check        ruff check + format --check"
	@echo "  make test         pytest"
	@echo "  make serve        run the search HTTP service on :8000"
	@echo "  make build-index  build a retrieval index (CORPUS=... OUT=...)"

sync:
	uv sync

sync-dev:
	uv sync --extra dev

check: sync-dev
	uv run ruff check .
	uv run ruff format --check .

lint: sync-dev
	uv run ruff check --fix .

fmt: sync-dev
	uv run ruff format .

test: sync-dev
	uv run pytest

serve: sync
	uv run search-server

build-index: sync
	uv run python -m search_server.index_builder --corpus $(CORPUS) --out $(OUT)

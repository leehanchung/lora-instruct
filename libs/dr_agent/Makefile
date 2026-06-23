.PHONY: help sync sync-dev check lint fmt test

help:
	@echo "Targets:"
	@echo "  make sync       uv sync runtime deps"
	@echo "  make sync-dev   uv sync runtime + dev deps"
	@echo "  make check      ruff check + format --check"
	@echo "  make lint       ruff check --fix"
	@echo "  make fmt        ruff format"
	@echo "  make test       pytest"

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

.PHONY: help sync sync-dev check lint fmt test gen-web

help:
	@echo "Targets:"
	@echo "  make sync      uv sync runtime deps"
	@echo "  make check     ruff check + format --check"
	@echo "  make test      pytest"
	@echo "  make gen-web   run the web data-gen pipeline"

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

gen-web: sync
	uv run python -m datagen.domains.web --seeds src/datagen/domains/web/seeds.txt --out outputs/web

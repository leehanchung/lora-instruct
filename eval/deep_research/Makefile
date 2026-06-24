.PHONY: help sync sync-dev check lint fmt test eval

BENCHMARK ?= simpleqa
CONFIG    ?= run.yaml

help:
	@echo "Targets:"
	@echo "  make sync       uv sync runtime deps"
	@echo "  make check      ruff check + format --check"
	@echo "  make test       pytest"
	@echo "  make eval       run a benchmark (BENCHMARK=simpleqa CONFIG=run.yaml)"

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

eval: sync
	uv run python -m eval_harness.harness.runner --config $(CONFIG) --benchmark $(BENCHMARK)

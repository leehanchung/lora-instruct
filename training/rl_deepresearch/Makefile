.PHONY: help sync sync-dev check lint fmt test image run

SLIME_IMAGE ?= slimerl/slime:latest

help:
	@echo "Targets:"
	@echo "  make sync       uv sync runtime deps (light; engine is the container)"
	@echo "  make check      ruff check + format --check"
	@echo "  make lint       ruff check --fix"
	@echo "  make test       pytest"
	@echo "  make image      docker pull the slime engine image"
	@echo "  make run        launch a run inside the slime container (see engine/README.md)"

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

image:
	docker pull $(SLIME_IMAGE)

run:
	docker run --rm --gpus all --ipc=host --shm-size=16g \
	  --ulimit memlock=-1 --ulimit stack=67108864 \
	  -v "$(CURDIR)":/workspace -w /workspace \
	  $(SLIME_IMAGE) bash launch/run.sh

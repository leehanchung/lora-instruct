.PHONY: help sync sync-dev lint fmt check modal-deploy

MODAL_IMAGE_BUILDER_VERSION ?= 2025.06

help:
	@echo "Targets (Modal deploy):"
	@echo "  make sync          uv sync runtime deps"
	@echo "  make sync-dev      uv sync runtime + dev deps"
	@echo "  make check         ruff check + format --check"
	@echo "  make lint          ruff check --fix"
	@echo "  make fmt           ruff format"
	@echo "  make modal-deploy  deploy the Modal sandbox app"
	@echo ""
	@echo "Note: modal-deploy is primarily run from CI/CD. This target"
	@echo "exists for local debugging when you need to push quickly."

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

modal-deploy: sync
	MODAL_IMAGE_BUILDER_VERSION=$(MODAL_IMAGE_BUILDER_VERSION) \
	  uv run modal deploy src/delulu_sandbox_modal/app.py

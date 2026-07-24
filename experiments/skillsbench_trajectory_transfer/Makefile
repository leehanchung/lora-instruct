RUN ?= exp_01ky0ztpc9eqn9g67e30pjt5k9
RUN_DIR := runs/$(RUN)

.PHONY: help sync sync-dev check lint fmt test cli-smoke reproduce-analysis report verify-artifacts

help:
	@echo "Targets:"
	@echo "  make sync                 install runtime dependencies"
	@echo "  make sync-dev             install runtime and development dependencies"
	@echo "  make check                lint, format-check, tests, and CLI smoke"
	@echo "  make reproduce-analysis   rebuild and verify the recorded paired analysis"
	@echo "  make report               rebuild runs/$(RUN)/report/rebuilt.html"
	@echo "  make verify-artifacts     verify recorded artifact metadata snapshot"

sync:
	uv sync

sync-dev:
	uv sync --extra dev

check: sync-dev
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest
	$(MAKE) cli-smoke

lint: sync-dev
	uv run ruff check --fix .

fmt: sync-dev
	uv run ruff format .

test: sync-dev
	uv run pytest

cli-smoke:
	@for command in skillsbench-corpus skillsbench-overlap skillsbench-train skillsbench-evaluate skillsbench-analyze skillsbench-report; do \
		uv run $$command --help >/dev/null || exit 1; \
	done

reproduce-analysis:
	uv run skillsbench-analyze "$(RUN_DIR)/data/normalized_paired_rows.jsonl" \
		--json "$(RUN_DIR)/metrics/reproduced_paired_analysis.json" \
		--csv "$(RUN_DIR)/metrics/reproduced_paired_analysis.csv" \
		--bootstrap-replicates 10000 --seed 17 \
		--compare "$(RUN_DIR)/metrics/full_paired_analysis.json" \
		--headline "$(RUN_DIR)/metrics/final_metrics.json" \
		--reproduction-output "$(RUN_DIR)/reproduction.json" --tolerance 1e-10

report:
	uv run skillsbench-report --run-dir "$(RUN_DIR)" --output "$(RUN_DIR)/report/rebuilt.html"

verify-artifacts:
	uv run skillsbench-evaluate artifacts \
		--manifest "$(RUN_DIR)/artifacts/manifest.json" \
		--snapshot "$(RUN_DIR)/artifacts/verified_snapshot.json"

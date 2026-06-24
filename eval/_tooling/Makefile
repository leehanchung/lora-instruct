# eval driver — BenchFlow is the sole runner; --agent and --model are the knobs.
#
# A benchmark = eval/benchmarks/<name>/tasks/ (a flat dir of BenchFlow task dirs).
# Run any benchmark with any agent/model:
#
#   make eval BENCHMARK=simpleqa AGENT=pi     MODEL=vllm/qwen35-2b-base
#   make eval BENCHMARK=hotpotqa AGENT=claude MODEL=claude-haiku-4-5-20251001
#   make eval BENCHMARK=simpleqa AGENT=codex  MODEL=gpt-5.4
#
# Run all `make` commands from eval/_tooling/.

# NOTE: never put an inline `# comment` after a `VAR := value` — Make folds the
# trailing spaces into the value (silently breaks URLs/paths). Comments go above.
SHELL := /bin/bash

# eval/ root, relative to eval/_tooling/
EVAL := ..

# ---- run knobs (override on the CLI) ----
BENCHMARK ?= simpleqa
AGENT ?= pi
MODEL ?= vllm/qwen35-2b-base
SANDBOX ?= docker
CONCURRENCY ?= 5
# N: tasks to generate (gen target)
N ?= 50
# ENGINE_PORT: local engine — SGLang 30000, vLLM 8000
ENGINE_PORT ?= 30000
ENGINE_URL ?= http://localhost:$(ENGINE_PORT)/v1

# Self-hosted models (BenchFlow's `vllm/` provider = any local OpenAI endpoint)
# need the engine URL injected + a placeholder key, and a reachable engine.
# First-party agents/models (claude/codex/…) bring their own auth, so skip both.
ifneq ($(findstring vllm/,$(MODEL)),)
  ENGINE_ENV := --agent-env BENCHFLOW_PROVIDER_BASE_URL=$(ENGINE_URL) --agent-env OPENAI_API_KEY=dummy-key
  ENGINE_DEP := check-engine
else
  ENGINE_ENV :=
  ENGINE_DEP :=
endif

# Docker-Desktop-on-WSL: force host.docker.internal for the proxy (see docs/).
export BENCHFLOW_DOCKER_HOST_ADDRESS ?= host.docker.internal

.PHONY: eval eval-direct serve serve-vllm gen gen-all smoke check-engine clean help

help:
	@echo "Run from eval/_tooling/:"
	@echo "  make serve                       start the engine (SGLang :30000)"
	@echo "  make gen BENCHMARK=simpleqa N=50 generate task dirs -> benchmarks/<B>/tasks/"
	@echo "  make eval BENCHMARK=.. AGENT=.. MODEL=..   run a benchmark via BenchFlow"
	@echo "  make smoke                       sanity-check agent->engine (no Docker)"
	@echo "Knobs: BENCHMARK AGENT MODEL SANDBOX CONCURRENCY  (see top of Makefile)"

# THE eval command: BenchFlow runs benchmarks/<BENCHMARK>/tasks with the chosen agent+model.
eval: $(ENGINE_DEP)
	bench eval run \
	  --tasks-dir $(EVAL)/benchmarks/$(BENCHMARK)/tasks \
	  --agent $(AGENT) --model $(MODEL) \
	  --sandbox $(SANDBOX) --concurrency $(CONCURRENCY) \
	  --jobs-dir $(EVAL)/_runs/$(AGENT)/$(BENCHMARK) \
	  $(ENGINE_ENV)

serve:
	bash scripts/serve_sglang.sh

serve-vllm:
	bash scripts/serve_qwen.sh

gen:
	uv run --with datasets python scripts/gen_tasks.py --benchmark $(BENCHMARK) --n $(N)

gen-all:
	$(MAKE) gen BENCHMARK=simpleqa N=$(N)
	$(MAKE) gen BENCHMARK=hotpotqa N=$(N)

smoke: check-engine
	bash scripts/smoke_pi.sh

check-engine:
	@curl -sf -m 10 http://127.0.0.1:$(ENGINE_PORT)/v1/models >/dev/null \
	  && echo "engine up on :$(ENGINE_PORT)" \
	  || (echo "engine NOT reachable on :$(ENGINE_PORT) — run 'make serve' first" && exit 1)

# Docker-free DEBUG fallback (NOT BenchFlow): pi -> engine direct, same tasks+scorer.
# For sanity-checking the model while the BenchFlow proxy bug is open
# (docs/benchflow-engine-500.md). Not the official path.
eval-direct: check-engine
	python3 scripts/run_local.py --benchmark $(BENCHMARK) --n $(N)

clean:
	rm -rf $(EVAL)/benchmarks/*/tasks $(EVAL)/_runs

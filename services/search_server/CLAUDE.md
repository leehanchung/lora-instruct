# CLAUDE.md — search-server

The tool/search HTTP service. Keep it behind a stable contract so the agent and
trainer never depend on the backend implementation.

## Commands

- `make serve` — run on :8000
- `make build-index CORPUS=... OUT=...` — build a retrieval index
- `make check` / `make test`

## Conventions

- **Heavy retrieval deps live ONLY here** (bm25/faiss/sentence-transformers).
  Never add them to `libs/dr_agent`.
- **Don't change the contract lightly** — `/search`, `/retrieve`, `/visit` are
  consumed by `dr_agent.tools`; a breaking change ripples to training and eval.
- **Secrets via env vars** (`SERPER_API_KEY`, `JINA_API_KEY`); never commit keys.
- Prefer a **self-hosted index** for RL reproducibility; reserve the live-web
  backend for eval/serving where freshness matters.

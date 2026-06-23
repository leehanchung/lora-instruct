# CLAUDE.md — dr-agent (shared library)

This is the shared deep-research agent library. It is imported by `training/`,
`eval/`, and `apps/`. **Changes here ripple to all three** — treat it as a
public API and be conservative.

## Commands

- `make check` — ruff check + format check
- `make test` — pytest
- `make sync-dev` — install runtime + dev deps (uv)

## Conventions

- **No heavy ML deps here.** This lib must stay light enough to import from a
  serving app. Trainer/SGLang/Megatron deps live in `training/`, faiss/bm25 in
  `services/search_server`. The agent reaches tools over HTTP, not via imports.
- **Tools are defined once.** If you need a search/visit/scholar tool anywhere in
  the monorepo, add or reuse it here — never re-implement it in a recipe or eval.
- **Rewards are a flat registry** keyed by a `data_source` field on each data row
  (see `rewards/__init__.py`). Add a task = add one scorer file + register it.
- **Prompts are templates** under `prompts/`, loaded via jinja — not inlined.

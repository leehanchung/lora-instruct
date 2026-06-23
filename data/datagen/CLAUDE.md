# CLAUDE.md — datagen

Synthetic task generation. Output conforms to the `dr_agent` scoring schema so it
serves as both training and eval data.

## Commands

- `make gen-web` — run the web pipeline
- `make check` / `make test`

## Conventions

- **core/ is domain-agnostic; domains/<name>/ subclass it** with the identical
  stage-named layout. Don't put domain logic in `core/`.
- **Stages write to separate `raw/verified/final` dirs**, never mutate in place —
  keeps partial reruns clean and provenance auditable.
- **Prompts live in each domain's `prompts.py`**, versioned with the code.
- **Heavy per-domain deps are extras** in `pyproject.toml` (`[sec]`, `[indexing]`),
  not base deps.
- `outputs/` is gitignored; never commit generated datasets.

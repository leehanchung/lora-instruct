# CLAUDE.md — delulu_sandbox_modal

The **Modal sandbox** half of delulu. A Modal function that runs Claude Code
ephemerally, one task at a time, in an isolated sandbox — invoked by the
Discord bot (see `apps/delulu_discord`). Entry point:
`src/delulu_sandbox_modal/app.py`.

Work from this directory (or `make -C apps/delulu_sandbox_modal <target>`).
This project owns its `pyproject.toml`, `uv.lock`, `.venv`, and `Makefile` —
the toolchain is **uv**, not repo-root tooling.

## Commands

- `make check` — ruff check + ruff format --check (run before pushing)
- `make test` — pytest
- `make lint` / `make fmt` — ruff check --fix / ruff format
- `make sync-dev` — `uv sync --extra dev`
- `make modal-deploy` — `modal deploy src/delulu_sandbox_modal/app.py`

## Gotchas

- **`modal-deploy` is primarily CI-driven.** CI deploys from a GitHub runner
  using the `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` repo secrets with
  `MODAL_IMAGE_BUILDER_VERSION` pinned. The local target exists for quick
  debugging pushes.
- Repo provisioning is serialized with `max_containers=1` — filesystem locks
  don't work on Modal Volumes, so don't reintroduce flock-style coordination.
- Plans for this app live in `apps/delulu_sandbox_modal/prd/`.

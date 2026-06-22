# CLAUDE.md — delulu_discord

The always-on **Discord bot** half of delulu. A discord.py gateway client
routes @mention/thread messages to Claude Code runs dispatched into
ephemeral Modal sandboxes (see `apps/delulu_sandbox_modal`), then streams
results back. Runs as a Docker container on a VPS droplet.

Work from this directory (or `make -C apps/delulu_discord <target>`). This
project owns its `pyproject.toml`, `uv.lock`, `.venv`, and `Makefile` — the
toolchain is **uv**, not repo-root tooling.

## Commands

- `make check` — ruff check + ruff format --check (run before pushing)
- `make test` — pytest
- `make lint` / `make fmt` — ruff check --fix / ruff format
- `make sync-dev` — `uv sync --extra dev`
- `make deploy` — build the image + restart the container (**droplet only**)
- `make logs` — `journalctl -f CONTAINER_NAME=disco`

## Gotchas

- **Deploy runs on the droplet**, not locally. CI SSHes in, `git pull`, then
  `make -C apps/delulu_discord deploy` (docker build + restart). Don't run
  `make deploy` on your machine — it expects the droplet's env-file and
  Modal config at `/root/...`.
- **Persistent state is a SQLite session DB** at `/data/sessions.db`, backed
  by the named Docker volume `disco-data` so `/setrepo` bindings and thread
  sessions survive `docker rm` across deploys.
- Container logs go to **journald** (survive `docker rm`); plain `docker logs`
  loses history on rebuild.
- Plans for this app live in `apps/delulu_discord/prd/`.

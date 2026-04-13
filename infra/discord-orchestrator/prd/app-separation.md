# App separation refactor

Plan to split the current single-package layout into two independent apps
under an `apps/` parent, each with its own deploy target and its own
Makefile scoped to that target. The Discord bot (`delulu_discord`)
deploys to a VPS; the Modal sandbox function (`delulu_sandbox_modal`)
deploys to Modal — primarily via CI/CD, but with a local-dev Makefile
target for debugging. Nothing in this document is in the code yet.

## Context / problem

Today's layout:

```
infra/discord-orchestrator/
├── pyproject.toml          # one project, one lockfile, one dep set
├── uv.lock
├── Dockerfile              # one Dockerfile (for the bot)
├── Makefile                # targets for both bot AND modal deploys
└── src/
    ├── bot/                # Discord bot process
    │   ├── main.py
    │   ├── handlers.py
    │   └── session_manager.py
    ├── modal_dispatch/     # Modal App + sandbox function + client dispatcher
    │   ├── app.py
    │   └── sandbox.py      # this is actually a *bot-side* client wrapper
    └── config/
        └── settings.py     # bot-side pydantic-settings
```

Three problems:

1. **Two apps, one package.** `src/bot/` runs on a VPS in a Docker
   container. `src/modal_dispatch/app.py` runs inside ephemeral Modal
   sandboxes. They share **zero runtime code**. The only thing crossing
   the boundary is a contract: a function name (`run_claude_code`) and
   its kwargs.

2. **Confusing naming.** `src/modal_dispatch/sandbox.py` is the
   *client-side* dispatcher the bot uses to call into Modal. It never
   runs inside a sandbox. The name is a lie inherited from an earlier
   iteration.

3. **Coupled tooling and Makefile.** One `pyproject.toml` forces both
   apps to share deps, Python version, ruff config, and uv.lock. One
   `Makefile` has VPS-deploy targets *and* Modal-deploy targets mingled
   together, so editing either requires understanding both.

## Goals / non-goals

**Goals**
- Two fully independent deployable units under `apps/`, each with its own
  `pyproject.toml`, `uv.lock`, Dockerfile/Modalfile, and Makefile.
- Each sub-Makefile scoped to **one** deployment target:
  - `apps/delulu_discord/Makefile` — VPS/Docker only, **no** `modal-deploy`.
  - `apps/delulu_sandbox_modal/Makefile` — Modal only, **no** Docker.
- Modal deploys happen primarily via CI/CD (with Modal tokens as GitHub
  secrets). The local `modal-deploy` target exists so developers can run
  it from their laptop when debugging without needing CI.
- Zero Python imports crossing the boundary between the two apps.
- A top-level `apps/`-aware `Makefile` at
  `infra/discord-orchestrator/Makefile` that dispatches to each
  sub-project.
- Tests for one app don't pull in the other's dependencies.
- Clear naming: no more `modal_dispatch/sandbox.py` that's really a client.

**Non-goals**
- Not making this a uv workspace (considered below, rejected).
- Not adding a shared library yet — neither app has runtime code to
  share today. Defer until the streaming PRD lands a typed event model
  that genuinely needs to exist on both sides.
- Not renaming at the Discord / Modal level (same bot, same App name).

## Proposed layout

```
infra/discord-orchestrator/
├── apps/
│   ├── delulu_discord/               # Discord bot — runs on VPS
│   │   ├── pyproject.toml            # discord.py, modal (client), pydantic-settings, structlog
│   │   ├── uv.lock
│   │   ├── .python-version
│   │   ├── Dockerfile                # python:3.14-slim, installs ONLY bot deps
│   │   ├── Makefile                  # VPS-only: sync, check, image, restart, deploy, logs, stop, shell
│   │   ├── src/
│   │   │   └── delulu_discord/
│   │   │       ├── __init__.py
│   │   │       ├── main.py           # Discord client + on_message + @mention gating
│   │   │       ├── handlers.py       # MessageHandler
│   │   │       ├── session_manager.py
│   │   │       ├── dispatcher.py     # (was src/modal_dispatch/sandbox.py)
│   │   │       └── settings.py       # (was src/config/settings.py)
│   │   └── tests/
│   │
│   └── delulu_sandbox_modal/         # Modal sandbox function — runs in Modal
│       ├── pyproject.toml            # modal, structlog
│       ├── uv.lock
│       ├── .python-version
│       ├── Makefile                  # Modal-only: sync, check, modal-deploy
│       ├── src/
│       │   └── delulu_sandbox_modal/
│       │       ├── __init__.py
│       │       └── app.py            # Modal App, image, volume, secret, run_claude_code
│       └── tests/
│
├── Makefile                          # top-level dispatcher — aggregates apps/
├── README.md                         # documents the whole system
├── ARCHITECTURE.md
└── prd/
    ├── streaming.md
    ├── repo-provisioning.md
    └── app-separation.md             # this file
```

No root-level `pyproject.toml` or `uv.lock`. The root of
`infra/discord-orchestrator/` is just the `apps/` parent, docs, PRDs,
and a thin dispatching `Makefile`.

### File-by-file migration

| Current | New |
|---|---|
| `src/bot/main.py` | `apps/delulu_discord/src/delulu_discord/main.py` |
| `src/bot/handlers.py` | `apps/delulu_discord/src/delulu_discord/handlers.py` |
| `src/bot/session_manager.py` | `apps/delulu_discord/src/delulu_discord/session_manager.py` |
| `src/modal_dispatch/sandbox.py` | `apps/delulu_discord/src/delulu_discord/dispatcher.py` ⬅ **moves AND renames** |
| `src/config/settings.py` | `apps/delulu_discord/src/delulu_discord/settings.py` |
| `src/modal_dispatch/app.py` | `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/app.py` |
| `src/__init__.py`, `src/config/__init__.py` | deleted (inlined into package init) |
| `Dockerfile` | `apps/delulu_discord/Dockerfile` |
| top-level `Makefile` (bot + modal mingled) | **split three ways**: `apps/delulu_discord/Makefile` (VPS), `apps/delulu_sandbox_modal/Makefile` (Modal), and a top-level dispatcher |
| `pyproject.toml` (one) | `apps/delulu_discord/pyproject.toml`, `apps/delulu_sandbox_modal/pyproject.toml` |
| `uv.lock` | split into `apps/delulu_discord/uv.lock` and `apps/delulu_sandbox_modal/uv.lock` |
| `.python-version` | duplicated into each sub-project |

### Import path changes

Every import like `from src.bot.handlers import ...` becomes
`from delulu_discord.handlers import ...`. Every
`from src.modal_dispatch.app import ...` becomes
`from delulu_sandbox_modal.app import ...`.

Concretely:
- `apps/delulu_discord/src/delulu_discord/main.py`:
  - `from delulu_discord.handlers import MessageHandler`
  - `from delulu_discord.session_manager import SessionManager`
  - `from delulu_discord.settings import Settings`
  - `from delulu_discord.dispatcher import SandboxDispatcher`
- `apps/delulu_discord/src/delulu_discord/dispatcher.py`:
  - Imports only `modal` (the client). Does **not** import anything from
    `delulu_sandbox_modal` — it looks up the function by name via
    `modal.Function.from_name(settings.modal_app_name, "run_claude_code")`.
- `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/app.py`:
  - Imports only `modal` and `structlog`. The inline imports of `os`,
    `subprocess`, `shutil` inside `run_claude_code` stay inside the
    function body — sandbox-side code.

The `sandbox.py → dispatcher.py` rename is cosmetic but important. The
file has always been a client-side wrapper around
`Function.from_name(...).remote()`. Calling it `sandbox.py` and putting
it under `modal_dispatch/` implied it ran inside the sandbox. It never
did.

## Why not a uv workspace?

Considered and rejected:

- **Uv workspaces share a single lockfile across members.** That's nice
  for version consistency but defeats independence — a dep change in
  one member invalidates the other's resolution.
- **Workspace members share a Python pin.** Today they're both on 3.14,
  but if Modal's image builder ever lags again (as it did for 3.13/3.14),
  we'd want to pin the bot newer than the sandbox — awkward in a workspace.
- **Dockerfile for a workspace member is more complex** — it needs to
  copy the root `pyproject.toml` + member's `pyproject.toml`. Possible
  but fiddly, and harder to explain.
- **No shared runtime code today.** The streaming PRD may eventually
  justify a third workspace member with typed events, but that decision
  is easier to make once that refactor lands.

Two fully independent projects. Revisit if we ever have real shared
runtime code.

## Makefiles — scoped to their deploy target

### `apps/delulu_discord/Makefile` — VPS/Docker only

Knows about `docker build`, `docker run`, the bot container, the env
file, and the Modal credentials mount. **Does not** know about
`modal deploy`.

```make
.PHONY: help sync sync-dev lint fmt check image restart deploy logs stop shell

BOT_NAME   ?= disco
IMAGE      ?= delulu-discord
ENV_FILE   ?= /root/disco.env
MODAL_TOML ?= /root/.modal.toml

help:
	@echo "Targets (VPS deploy):"
	@echo "  make sync          uv sync runtime deps"
	@echo "  make sync-dev      uv sync runtime + dev deps"
	@echo "  make check         ruff check + format --check"
	@echo "  make lint          ruff check --fix"
	@echo "  make fmt           ruff format"
	@echo "  make image         docker build the bot image"
	@echo "  make restart       stop+remove+run the $(BOT_NAME) container"
	@echo "  make deploy        image + restart"
	@echo "  make logs          tail bot container logs"
	@echo "  make stop          stop and remove the bot container"
	@echo "  make shell         exec a shell inside the running container"

sync:       ; uv sync
sync-dev:   ; uv sync --extra dev
check: sync-dev
	uv run ruff check .
	uv run ruff format --check .
lint: sync-dev  ; uv run ruff check --fix .
fmt: sync-dev   ; uv run ruff format .

image:
	docker build -t $(IMAGE) .

restart:
	-docker stop $(BOT_NAME)
	-docker rm $(BOT_NAME)
	docker run -d \
	  --name $(BOT_NAME) \
	  --restart=unless-stopped \
	  --env-file $(ENV_FILE) \
	  -v $(MODAL_TOML):/root/.modal.toml:ro \
	  $(IMAGE)

deploy: image restart

logs:  ; docker logs -f $(BOT_NAME)
stop:
	-docker stop $(BOT_NAME)
	-docker rm $(BOT_NAME)
shell: ; docker exec -it $(BOT_NAME) /bin/sh
```

### `apps/delulu_sandbox_modal/Makefile` — Modal only

Knows about `modal deploy`. **Does not** know about Docker.

```make
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

sync:       ; uv sync
sync-dev:   ; uv sync --extra dev
check: sync-dev
	uv run ruff check .
	uv run ruff format --check .
lint: sync-dev  ; uv run ruff check --fix .
fmt: sync-dev   ; uv run ruff format .

modal-deploy: sync
	MODAL_IMAGE_BUILDER_VERSION=$(MODAL_IMAGE_BUILDER_VERSION) \
	  uv run modal deploy src/delulu_sandbox_modal/app.py
```

### Top-level `infra/discord-orchestrator/Makefile`

Thin dispatcher. Does not know any deploy details itself.

```make
.PHONY: help check deploy-bot deploy-modal deploy-all

APPS := apps/delulu_discord apps/delulu_sandbox_modal

help:
	@echo "Top-level targets:"
	@echo "  make check         run ruff on all apps/"
	@echo "  make deploy-bot    deploy the Discord bot to the VPS"
	@echo "  make deploy-modal  deploy the Modal sandbox app"
	@echo "  make deploy-all    modal-deploy then bot-deploy"
	@echo ""
	@echo "Sub-project targets available via:"
	@echo "  make -C apps/delulu_discord <target>"
	@echo "  make -C apps/delulu_sandbox_modal <target>"

check:
	$(MAKE) -C apps/delulu_discord check
	$(MAKE) -C apps/delulu_sandbox_modal check

deploy-bot:
	$(MAKE) -C apps/delulu_discord deploy

deploy-modal:
	$(MAKE) -C apps/delulu_sandbox_modal modal-deploy

deploy-all: deploy-modal deploy-bot
```

Top-level has no direct knowledge of Docker or Modal — each sub-Makefile
owns its own domain.

## Dockerfile — bot only

`apps/delulu_discord/Dockerfile`:

```dockerfile
FROM python:3.14-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock .python-version ./
RUN uv pip install --system .
COPY src/ src/
CMD ["python", "-m", "delulu_discord.main"]
```

Build context becomes `apps/delulu_discord/`, which physically cannot
pull in Modal sandbox code even by accident.

## pyproject.toml splits

### `apps/delulu_discord/pyproject.toml`

```toml
[project]
name = "delulu-discord"
version = "0.1.0"
description = "Discord bot that dispatches Claude Code tasks to Modal sandboxes"
requires-python = ">=3.14"
dependencies = [
    "discord.py>=2.4,<3",
    "modal>=1.0",                 # client only — used for Function.from_name + .remote
    "pydantic>=2.0,<3",
    "pydantic-settings>=2.0,<3",
    "structlog>=24.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.15",
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[project.scripts]
delulu-discord = "delulu_discord.main:main"

[tool.ruff]
target-version = "py314"
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

### `apps/delulu_sandbox_modal/pyproject.toml`

```toml
[project]
name = "delulu-sandbox-modal"
version = "0.1.0"
description = "Modal sandbox function that runs Claude Code for the delulu Discord bot"
requires-python = ">=3.14"
dependencies = [
    "modal>=1.0",
    "structlog>=24.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.15",
    "pytest>=8.0",
]

[tool.ruff]
target-version = "py314"
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

Note how tiny `delulu_sandbox_modal`'s deps are: just `modal` and
`structlog`. No Discord, no pydantic. This alone is a good reason to
separate — the Modal image definition no longer pulls in unrelated
packages during dev-side resolution.

## Pre-commit impact

The existing hook entry in the repo-root `.pre-commit-config.yaml` is:

```yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.15.10
  hooks:
    - id: ruff
      files: ^infra/discord-orchestrator/.*\.py$
    - id: ruff-format
      files: ^infra/discord-orchestrator/.*\.py$
```

The `files:` pattern already matches `infra/discord-orchestrator/apps/**`.
Ruff walks up from each changed file to find the nearest `pyproject.toml`,
so each sub-project's config wins for its own files. **Nothing to change
in the pre-commit config.**

## CI/CD impact

Once separated, the CI/CD refactor becomes trivially obvious:

- **`deploy-modal`** job: checkout repo, `cd apps/delulu_sandbox_modal`,
  `make sync-dev modal-deploy` with `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`
  as env vars. Runs on GitHub Actions runner, never touches the VPS.
- **`deploy-bot`** job: SSH into the VPS, `cd
  /root/SMILE-factory/infra/discord-orchestrator/apps/delulu_discord`,
  `make deploy`. VPS only needs Docker.

Path filters become clean and unambiguous:

```yaml
filters: |
  modal:
    - 'infra/discord-orchestrator/apps/delulu_sandbox_modal/**'
  bot:
    - 'infra/discord-orchestrator/apps/delulu_discord/**'
```

No more shared `pyproject.toml` / `uv.lock` triggering both jobs. A
change to the bot only triggers bot deploy. A change to the sandbox
only triggers Modal deploy.

## Migration steps (commit-by-commit)

Ship in three commits so each is independently revertable and
independently verifiable.

### Commit 1 — move files, no logic changes

- Create `apps/delulu_discord/` and `apps/delulu_sandbox_modal/` with
  the new layout.
- Move files into place and rename `sandbox.py → dispatcher.py`.
- Rewrite imports (`src.bot.*` → `delulu_discord.*`,
  `src.modal_dispatch.*` → `delulu_sandbox_modal.*`).
- Split `pyproject.toml` into two sub-projects; delete the old one.
- Create `apps/delulu_discord/Makefile`,
  `apps/delulu_sandbox_modal/Makefile`, and the top-level dispatching
  `Makefile`. Delete the old mingled one.
- Move `Dockerfile` to `apps/delulu_discord/Dockerfile`.
- Regenerate each sub-project's `uv.lock` with
  `uv lock` inside its directory.
- Delete the old `src/` tree.

**Verification**: from `infra/discord-orchestrator/`, `make check`
passes (both sub-projects lint cleanly). Inside each sub-project,
`uv run python -c "from delulu_discord.main import create_bot"` (or the
modal equivalent) succeeds.

### Commit 2 — update droplet wiring

- Update `README.md` layout and manual-deploy sections to reference the
  new paths (`make -C apps/delulu_discord deploy`, etc).
- Update `.github/workflows/discord-orchestrator-deploy.yml` to use the
  new sub-project targets and path filters. Leave the one-job-SSH shape
  for now — the two-job CI/CD refactor is a separate PR on top of this.
- Rebuild and redeploy manually on the droplet: `git pull`,
  `cd apps/delulu_discord`, `make deploy`.

**Verification**: send a test `@delulu` mention, expect the bot to
respond. Check `docker logs disco` for a clean startup.

### Commit 3 — cleanup

- Remove any lingering references to `src.bot.*` / `src.modal_dispatch.*`
  or the old layout in README, ARCHITECTURE, and the other PRDs (update
  streaming PRD and repo-provisioning PRD to reference the new paths and
  package names).
- Verify `make check` from `infra/discord-orchestrator/` passes.

## What won't break during migration

- **Active sessions.** In-memory `SessionManager` resets on bot restart
  regardless.
- **Modal volume paths.** Untouched — `/vol/workspaces/<thread_id>`,
  `/vol/claude-home/`.
- **Modal App name.** Stays `discord-orchestrator`. `Function.from_name`
  still looks up the same name. (The *package* is `delulu_sandbox_modal`
  but the deployed Modal App identifier is unchanged.)
- **Discord bot user ID**. Unchanged by renames in the code.
- **CI/CD secrets.** Existing `DROPLET_HOST` / `DROPLET_SSH_KEY` work.
  New `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` are a separate CI/CD
  refactor, not part of this one.

## Open questions

1. **Root `pyproject.toml` — keep or delete?** Recommend **delete**. The
   apps each have their own, and pre-commit/ruff walks up to find the
   nearest. One less file, one less place to look.
2. **Settings fields `modal_app_name` / `modal_volume_name`** — they
   describe things the `delulu_sandbox_modal` app creates, but they're
   read by the bot. Keep them in `delulu_discord.settings`? Or define
   them as constants in `delulu_sandbox_modal` and duplicate/import?
   **Leaning: keep in bot settings**, accept the duplication risk as
   low-cost. They change ~once a year and the "source of truth" is
   really the decorator in `delulu_sandbox_modal/app.py`.
3. **Tests scaffolding — add in this refactor or defer?** **Defer.** No
   tests exist today; adding empty `tests/` directories would just be
   noise. Add when we have tests to write.

## Out of scope — defer

- Separating the CI/CD workflow into two jobs with Modal tokens (its
  own refactor, built on top of this one).
- Shared type package for streaming events (only needed once streaming
  PRD lands).
- Unit tests.

## Effort estimate

- Commit 1 (move files): 1–2 hours (mechanical but careful — lots of
  import paths, two new Makefiles, two new pyproject.tomls, two
  regenerated lockfiles).
- Commit 2 (droplet wiring + workflow paths): 45 min.
- Commit 3 (docs cleanup): 30 min.
- Manual end-to-end testing: 30 min.

Call it half a day.

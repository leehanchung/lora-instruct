# App separation refactor

Plan to split the current single-package layout into two independent apps
that share nothing at runtime: the Discord bot (VPS-deployed long-running
process) and the Modal sandbox function (serverless, deployed separately).
Nothing in this document is in the code yet.

## Context / problem

Today's layout:

```
infra/discord-orchestrator/
├── pyproject.toml          # one project, one lockfile, one dep set
├── uv.lock
├── Dockerfile              # one Dockerfile (for the bot)
├── Makefile                # targets for both bot and modal deploys
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

1. **Two apps, one package.** `src/bot/` runs on a VPS in a Docker container.
   `src/modal_dispatch/app.py` runs inside ephemeral Modal sandboxes. They
   share **zero runtime code**. The only thing crossing the boundary is a
   contract: a function name (`run_claude_code`) and its kwargs. They are
   as independent as two separate microservices.

2. **Confusing naming.** `src/modal_dispatch/sandbox.py` is actually the
   *client-side* dispatcher that the bot uses to call into Modal. It has
   nothing to do with the sandbox itself — the name is a lie inherited from
   an earlier iteration where the file was half-sandbox, half-dispatcher.

3. **Coupled tooling.** One `pyproject.toml` forces both apps to share
   dependencies, Python version, ruff config, and uv.lock. A dependency
   needed by only one side (e.g. `discord.py`) gets pulled into the other's
   dep resolution. One file change invalidates both lockfiles.

## Goals / non-goals

**Goals**
- Two fully independent deployable units, each with its own
  `pyproject.toml`, `uv.lock`, Dockerfile/Modalfile, and Makefile.
- Zero Python imports crossing the boundary between the two apps.
- A top-level `Makefile` that dispatches to each sub-project so the "run
  everything" workflow from the project root keeps working.
- Tests for one app don't pull in the other's dependencies.
- Clear naming: no more `modal_dispatch/sandbox.py` that's really a client.

**Non-goals**
- Not making this a uv workspace (considered below, rejected).
- Not adding a shared library yet — neither app has runtime code to
  share today. Defer until the streaming PRD is implemented and we need
  a typed event model on both sides.
- Not renaming at the Discord / Modal level — same bot, same App name.

## Proposed layout

```
infra/discord-orchestrator/
├── bot/                          # Discord bot — runs on VPS
│   ├── pyproject.toml            # discord.py, modal (client), pydantic-settings, structlog
│   ├── uv.lock
│   ├── .python-version
│   ├── Dockerfile                # python:3.14-slim, installs ONLY bot deps
│   ├── Makefile                  # bot-specific targets: deploy, logs, restart, shell
│   ├── src/
│   │   └── disco_bot/
│   │       ├── __init__.py
│   │       ├── main.py           # entrypoint — Discord client setup + on_message
│   │       ├── handlers.py       # MessageHandler — thread/prompt routing
│   │       ├── session_manager.py
│   │       ├── dispatcher.py     # (was src/modal_dispatch/sandbox.py)
│   │       └── settings.py       # (was src/config/settings.py)
│   └── tests/
│
├── modal_app/                    # Modal sandbox function — runs in Modal
│   ├── pyproject.toml            # modal, structlog
│   ├── uv.lock
│   ├── .python-version
│   ├── Makefile                  # modal-specific: modal-deploy, lint, check
│   ├── src/
│   │   └── disco_modal/
│   │       ├── __init__.py
│   │       └── app.py            # Modal App, image, volume, secret, run_claude_code
│   └── tests/
│
├── Makefile                      # top-level — dispatches to bot/ and modal_app/
├── pyproject.toml                # (minimal) — just enough for ruff/pre-commit at the root
├── README.md                     # unchanged (documents the whole system)
├── ARCHITECTURE.md               # unchanged
└── prd/
    ├── streaming.md
    ├── repo-provisioning.md
    └── app-separation.md         # this file
```

### File-by-file migration

| Current | New |
|---|---|
| `src/bot/main.py` | `bot/src/disco_bot/main.py` |
| `src/bot/handlers.py` | `bot/src/disco_bot/handlers.py` |
| `src/bot/session_manager.py` | `bot/src/disco_bot/session_manager.py` |
| `src/modal_dispatch/sandbox.py` | `bot/src/disco_bot/dispatcher.py` ⬅ **moves AND renames** |
| `src/config/settings.py` | `bot/src/disco_bot/settings.py` |
| `src/modal_dispatch/app.py` | `modal_app/src/disco_modal/app.py` |
| `src/config/__init__.py` | deleted (inlined) |
| `src/__init__.py` | deleted |
| `Dockerfile` | `bot/Dockerfile` |
| top-level `Makefile` (bot + modal targets) | split: `bot/Makefile`, `modal_app/Makefile`, top-level dispatcher |
| `pyproject.toml` (one) | `bot/pyproject.toml`, `modal_app/pyproject.toml`, plus a minimal root `pyproject.toml` |
| `uv.lock` | split into `bot/uv.lock` and `modal_app/uv.lock` |
| `.python-version` | duplicated into `bot/.python-version` and `modal_app/.python-version` |

### Import path changes

Every import like `from src.bot.handlers import ...` becomes
`from disco_bot.handlers import ...`. Every
`from src.modal_dispatch.app import ...` becomes
`from disco_modal.app import ...`.

Concretely:
- `bot/src/disco_bot/main.py`:
  - `from disco_bot.handlers import MessageHandler`
  - `from disco_bot.session_manager import SessionManager`
  - `from disco_bot.settings import Settings`
  - `from disco_bot.dispatcher import SandboxDispatcher`
- `bot/src/disco_bot/dispatcher.py`:
  - Only imports `modal` (the client). Does NOT import anything from
    `disco_modal` — it looks up the function by name via
    `modal.Function.from_name(...)`.
- `modal_app/src/disco_modal/app.py`:
  - Imports only `modal` and `structlog` (the inline imports of `os`,
    `subprocess`, `shutil` inside `run_claude_code` stay inside the function
    body — sandbox code).

The `sandbox.py → dispatcher.py` rename is cosmetic but important. The file
has always been a client-side wrapper around `Function.from_name(...).remote()`.
Calling it `sandbox.py` and putting it under `modal_dispatch/` implied it ran
inside the sandbox. It never did.

## Why not a uv workspace?

Considered and rejected:

- **Uv workspaces share a single lockfile across members.** That's nice for
  version consistency but defeats the goal of independence — a dep change in
  one member invalidates the other's resolution.
- **Workspace members share a Python pin.** Today they're both on 3.14, but
  if Modal's image builder ever lags behind again (as it did for 3.13/3.14),
  we'd want to pin the bot newer than the sandbox — a workspace makes that
  awkward.
- **Dockerfile for a workspace member is more complex.** Need to copy the
  root `pyproject.toml`, the member's `pyproject.toml`, and only the
  member's `src/`. Possible but fiddly.
- **No shared runtime code to justify the overhead.** The streaming PRD
  will eventually add a shared event type, but that can live as a tiny
  third workspace member later — or as duplicated type dicts, given how
  small the contract is.

Go with two fully independent projects. Revisit if we ever have real shared
runtime code.

## Top-level Makefile

```make
.PHONY: help bot-check bot-deploy modal-check modal-deploy check deploy

help:
	@echo "Top-level targets:"
	@echo "  make bot-check      run ruff + tests in bot/"
	@echo "  make bot-deploy     build & restart the bot container on this host"
	@echo "  make modal-check    run ruff + tests in modal_app/"
	@echo "  make modal-deploy   deploy the Modal sandbox app"
	@echo "  make check          bot-check + modal-check"
	@echo "  make deploy         modal-deploy + bot-deploy"

bot-check:
	$(MAKE) -C bot check

bot-deploy:
	$(MAKE) -C bot deploy

modal-check:
	$(MAKE) -C modal_app check

modal-deploy:
	$(MAKE) -C modal_app modal-deploy

check: bot-check modal-check

deploy: modal-deploy bot-deploy
```

Each sub-project keeps its current set of targets (`lint`, `fmt`, `check`,
`deploy`, `logs`, `shell`, etc.) in its own `Makefile`, specialized to the
commands that make sense for that app.

## Dockerfile — the bot only

The current `Dockerfile` becomes `bot/Dockerfile`, and only knows about
`bot/pyproject.toml` and `bot/src/disco_bot/`:

```dockerfile
FROM python:3.14-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock .python-version ./
RUN uv pip install --system .
COPY src/ src/
CMD ["python", "-m", "disco_bot.main"]
```

The build context becomes `bot/` (not `infra/discord-orchestrator/`), which
means the bot image can't accidentally pull in modal_app files.

The top-level Docker-build flow from the droplet becomes:

```bash
cd /root/lora-instruct/infra/discord-orchestrator/bot
docker build -t discord-orchestrator .
docker run -d --name disco ... discord-orchestrator
```

(Or wrapped by `make bot-deploy` from the `infra/discord-orchestrator/` root.)

## pyproject.toml splits

### `bot/pyproject.toml`

```toml
[project]
name = "disco-bot"
version = "0.1.0"
description = "Discord bot that dispatches Claude Code tasks to Modal sandboxes"
requires-python = ">=3.14"
dependencies = [
    "discord.py>=2.4,<3",
    "modal>=1.0",                 # client only — used to call Function.from_name
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
disco-bot = "disco_bot.main:main"

[tool.ruff]
target-version = "py314"
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

### `modal_app/pyproject.toml`

```toml
[project]
name = "disco-modal"
version = "0.1.0"
description = "Modal sandbox function that runs Claude Code for the Discord bot"
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

Note: `modal_app` has far fewer deps. No `discord.py`, no `pydantic-settings`.
This alone is a good reason to separate — the Modal image builder won't see
discord.py in the dep graph (even though Modal's `.pip_install(...)` is what
actually installs into the sandbox image, tooling around resolution is
cleaner when deps match reality).

### Root `pyproject.toml` (minimal)

Keep a tiny one at `infra/discord-orchestrator/pyproject.toml` **only** for:
- Scoped ruff config that pre-commit can still find via the existing hook
- A place to hang top-level metadata (`name = "discord-orchestrator"`,
  description, etc.)

No deps, no lockfile at this level. Or alternatively delete it entirely and
let pre-commit's ruff hook use each sub-project's `pyproject.toml` directly
— ruff walks up from the changed file to find the nearest config.

**Recommend: delete the root `pyproject.toml`.** Keep the layout obvious.

## Pre-commit impact

The existing hook entry in the repo root is:

```yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.15.10
  hooks:
    - id: ruff
      files: ^infra/discord-orchestrator/.*\.py$
    - id: ruff-format
      files: ^infra/discord-orchestrator/.*\.py$
```

The `files: ^infra/discord-orchestrator/.*\.py$` pattern already matches
`infra/discord-orchestrator/bot/**` and `infra/discord-orchestrator/modal_app/**`.
Ruff will find each sub-project's `pyproject.toml` via its tree walk. Nothing
to change at the repo root.

## CI/CD impact (the whole reason we're doing this)

Once separated, the two CI/CD jobs map naturally:

- **`deploy-modal`** job runs `make -C modal_app modal-deploy` with Modal
  tokens as env vars. No SSH, no droplet involvement.
- **`deploy-bot`** job SSHes into the droplet and runs
  `make -C bot deploy`. Droplet only needs Docker; doesn't need `uv` or
  the `modal` CLI installed.

Path filters become cleaner too:

```yaml
filters: |
  modal:
    - 'infra/discord-orchestrator/modal_app/**'
  bot:
    - 'infra/discord-orchestrator/bot/**'
```

No more ambiguity about which job fires on which file change.

## Migration steps (commit-by-commit)

Ship in three commits so each is independently revertable and
independently verifiable.

### Commit 1 — move files, no logic changes

- Create `bot/` and `modal_app/` directories with the new layout.
- Move files into place and rename `sandbox.py` → `dispatcher.py`.
- Rewrite import paths (`src.bot.*` → `disco_bot.*`,
  `src.modal_dispatch.*` → `disco_modal.*`).
- Split `pyproject.toml` into `bot/pyproject.toml` and
  `modal_app/pyproject.toml`. Delete the root one (or keep minimal).
- Create `bot/Makefile`, `modal_app/Makefile`, and a top-level
  dispatching `Makefile`.
- Move `Dockerfile` to `bot/Dockerfile`.
- Regenerate `bot/uv.lock` and `modal_app/uv.lock`.
- Delete the old `src/` tree and old top-level `pyproject.toml`.

**Verification**: `make check` at `infra/discord-orchestrator/` should pass
(both sub-projects lint cleanly). `uv run python -c "from disco_bot.main
import create_bot"` should succeed inside `bot/`. Same for
`from disco_modal.app import app` inside `modal_app/`.

### Commit 2 — update droplet wiring

- Update `README.md` layout section and manual-deploy instructions.
- Update `.github/workflows/discord-orchestrator-deploy.yml` to use the new
  sub-make targets (`make -C bot deploy`, etc). Path filters stay the same
  shape, just the paths change.
- Update `docker build` / `docker run` paths in docs.
- On the droplet: `git pull` then `make bot-deploy` should rebuild and
  restart cleanly.

**Verification**: run the workflow via `workflow_dispatch`, expect a
successful deploy. Send a test @mention, expect the bot to respond.

### Commit 3 — cleanup

- Remove any lingering references to `src.bot` / `src.modal_dispatch` / the
  old layout in README, ARCHITECTURE, PRDs (update streaming and
  repo-provisioning PRDs to reference the new paths).
- Verify `make check` at the root passes.

## What will break during migration

- **Active sessions.** The in-memory `SessionManager` resets every bot
  restart regardless, so the first deploy after the refactor behaves
  like any other restart. No special handling needed.
- **Modal volume paths.** Untouched — still `/vol/workspaces/<thread_id>`
  and `/vol/claude-home/`. The sandbox function code moves to a new import
  path but the runtime behavior and volume layout are identical.
- **Modal App name.** Stays `discord-orchestrator`. `Function.from_name`
  in the dispatcher still looks up the same name.
- **Discord bot user ID**. Unchanged.
- **CI/CD secrets.** Existing secrets still work. New secrets
  (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) are a separate CI/CD refactor,
  not part of this one.

## Open questions

1. **Root `pyproject.toml`: delete or keep minimal?** Leaning delete — one
   less file, one less question about what lives there. Pre-commit works
   either way because ruff walks up for config.
2. **`modal_app/` naming.** "`modal_app`" vs "`modal_fn`" vs "`sandbox`" vs
   "`modal_runtime`". `modal_app` is fine but not perfect — the directory
   holds one function, not a full app. Open to bikeshedding.
3. **Settings class membership.** `Settings` lives in the bot, but some
   fields (`modal_app_name`, `modal_volume_name`) are really Modal
   contract values. Keep them in bot settings and accept the cross-reference,
   or define them as constants in `disco_modal` and import them (adds a
   dep)? Leaning: keep in bot, accept the duplication/drift risk as
   low-cost.
4. **Tests directory.** Currently empty. Add `tests/` scaffolding in this
   refactor or defer until we have tests to write?
5. **`disco_bot` / `disco_modal` package names.** These are what you'd
   `import` in code. "`disco`" comes from the container name (`disco`). Any
   preference? Alternatives: `discord_orchestrator_bot` (verbose),
   `orchestrator_bot` (decent), `bot` (collides with lots of things).

## Out of scope — defer

- Separating the CI/CD workflow into two jobs (its own refactor, built on
  top of this one — see the earlier discussion).
- Shared type package (only needed once streaming PRD lands).
- Unit tests (separate effort).
- Moving PRDs out of `infra/discord-orchestrator/prd/` (they're fine here).

## Effort estimate

- Commit 1 (move files): 1–2 hours (mechanical but careful — lots of import
  paths to update, two new Makefiles, two new pyproject.tomls, regenerate
  both lockfiles).
- Commit 2 (droplet wiring): 30 min.
- Commit 3 (docs cleanup): 30 min.
- Manual testing end-to-end: 30 min.

Call it half a day.

# SMILE-factory

This repository hosts two projects:

- **[Delulu Discord Orchestrator](#discord-orchestrator)** *(active)* — a Discord bot that dispatches Claude Code tasks to ephemeral Modal sandboxes. Lives under `apps/delulu_discord` and `apps/delulu_sandbox_modal`.
- **[LoRA-Instruct](#lora-instruct-archived)** *(archived)* — earlier fine-tuning work with HuggingFace PEFT + LoRA, kept in-tree pending a later migration. Lives at the repo root (`finetune.py`, `dataset/`, `inference/`, etc.).

---

# Discord Orchestrator

Discord bot that dispatches Claude Code tasks to ephemeral Modal Sandboxes.

## How it works

1. User `@mentions` the bot in any channel it can see
2. Bot creates a Discord thread off that message and a new Claude Code session
3. Bot dispatches the task to a Modal Sandbox (ephemeral container with Claude Code installed)
4. Sandbox runs the task against a per-thread workspace on the Modal Volume, returns output, dies
5. Bot posts the result in the thread
6. Replies inside the thread auto-continue the same session via `claude --continue` — no need to re-mention the bot

The bot process itself is tiny (~50MB RAM) and just dispatches jobs — all
heavy lifting (and all Claude Code execution) happens inside Modal sandboxes.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

### Architecture

```mermaid
flowchart LR
    U([User])

    subgraph Discord
      Ch["any channel<br/>(@mention gated)"]
      Th["thread"]
    end

    subgraph Droplet["DO Droplet"]
      Bot["disco<br/>(bot container)"]
      SM["session_manager<br/>thread_id → session"]
      Bot --- SM
    end

    subgraph Modal["Modal"]
      App["discord-orchestrator App"]
      Fn["run_claude_code<br/>(ephemeral sandbox)"]
      Vol[("claude-workspaces<br/>Volume")]
      Sec["claude-oauth<br/>Secret"]
      App --- Fn
    end

    subgraph Anthropic["Anthropic"]
      API["Claude API<br/>(Pro/Max OAuth)"]
    end

    U -->|@mention| Ch
    Ch -->|gateway event| Bot
    Bot -->|create| Th
    Bot -->|Function.from_name.remote| Fn
    Sec -.->|seeds /vol/claude-home/.claude| Vol
    Vol <-.->|HOME=/vol/claude-home<br/>cwd=/vol/workspaces/<thread_id>| Fn
    Fn -->|claude -p --continue| API
    Fn -->|stdout| Bot
    Bot -->|reply| Th
    Th --> U
```

**Runtime request flow** — an `@mention` in any channel triggers the bot
to open a thread, look up or create a session, and call the deployed Modal
function via `Function.from_name(...).remote(...)`. Modal spins up an
ephemeral container from a pre-baked image (Node + Claude Code), runs
`claude --print` with `HOME=/vol/claude-home` and `cwd=/vol/workspaces/<thread_id>`,
captures stdout, and exits. Because both Claude Code's state directory
(`~/.claude/`) and the workspace are on the persistent Modal Volume,
follow-up replies in the same thread can resume via `claude --continue`
— which keys off cwd — even across ephemeral sandbox invocations and bot
restarts.

**Key invariants worth knowing:**

- **The bot never runs Claude Code itself.** It only holds a Discord
  WebSocket connection and a hydrated Modal function reference. Any VPS
  with Docker can host it.
- **Sandboxes are ephemeral but the volume is not.** Workspaces
  (`/vol/workspaces/<thread_id>`) and Claude Code's entire home
  (`/vol/claude-home/.claude/` — credentials, settings, session history)
  survive across invocations. That's what makes session resume,
  refresh-token rotation, and `--continue` all work.
- **Auth is one-way from laptop → Modal Secret → Volume.** The secret only
  seeds the volume on first run; after that, the volume's copy is the
  source of truth. The droplet never stores Claude credentials at rest.
- **`workspace_path` is deterministic per `thread_id`.** The in-memory
  session map is a convenience; even if it's wiped by a bot restart, the
  same Discord thread always maps to the same volume directory.

## Prerequisites

- Python 3.14 and [uv](https://docs.astral.sh/uv/)
- A Discord account
- A [Modal](https://modal.com/) account (`uv tool install modal` or `pip install modal`)
- A Claude Pro or Max subscription (Claude Code authenticates via OAuth, no API key needed)
- *(For deployment)* a server to run the bot — anything that can run Docker. A
  $4–6/mo Digital Ocean droplet is plenty.

---

## Setup

The setup has three parts:

1. **Discord** — create a bot application and invite it to your server.
2. **Modal** — authenticate, upload Claude OAuth credentials as a secret, deploy the sandbox app.
3. **Bot host** — run the bot process (locally or on a server).

### 1. Create the Discord bot

1. Go to https://discord.com/developers/applications → **New Application**.
2. Left sidebar → **Bot**. Scroll to **Privileged Gateway Intents** and enable
   **Message Content Intent**. Save changes.
3. At the top of the **Bot** page, click **Reset Token** → **Copy**. This is
   your `DISCORD_BOT_TOKEN` — save it somewhere, you only see it once.
4. Left sidebar → **OAuth2 → URL Generator**:
   - Scopes: `bot`
   - Bot permissions: `View Channels`, `Send Messages`, `Create Public Threads`,
     `Send Messages in Threads`, `Read Message History`
5. Open the generated URL, pick your server, **Authorize**.

### 2. Set up Modal

```bash
cd apps/delulu_sandbox_modal
uv sync

# Authenticate Modal (opens a browser)
uv run modal setup
```

Now generate dedicated Claude Code OAuth credentials for the bot. Using a
throwaway `HOME` keeps this session isolated from your daily `claude login`,
so nothing you do on your laptop later can invalidate the bot's auth:

```bash
HOME=/tmp/claude-bot-home claude login
# follow the browser flow

uv run modal secret create claude-oauth \
  CLAUDE_CREDENTIALS_JSON="$(cat /tmp/claude-bot-home/.claude/.credentials.json)"

rm -rf /tmp/claude-bot-home
```

The sandbox points `HOME` at `/vol/claude-home` so Claude Code's entire
state directory (credentials, settings, and session history under
`~/.claude/projects/`) lives on the Modal Volume. Rotated refresh tokens
and `--continue` history both persist across ephemeral sandbox
invocations. The secret only seeds the volume on first run — after that
the volume copy is the source of truth. If auth ever breaks, delete
`claude-home/` from the volume and re-create the secret.

Deploy the sandbox app (from the repo root):

```bash
make deploy-modal
# — or, scoped to the sub-project —
make -C apps/delulu_sandbox_modal modal-deploy
```

### 3. Run the bot

Configure the bot's environment:

```bash
cp apps/delulu_discord/.env.example apps/delulu_discord/.env
# Edit the copy and paste DISCORD_BOT_TOKEN
```

#### Option A — locally

```bash
cd apps/delulu_discord
uv run delulu-discord
```

#### Option B — Docker (any host)

```bash
make -C apps/delulu_discord deploy   # = image + restart
make -C apps/delulu_discord logs     # tail the container
```

The bot Makefile expects `/root/disco.env` for the bot env file and
`/root/.modal.toml` for the Modal credentials. Override via env vars if
your paths differ:

```bash
make -C apps/delulu_discord deploy ENV_FILE=$PWD/apps/delulu_discord/.env MODAL_TOML=$HOME/.modal.toml
```

The Modal credential mount is what lets the bot dispatch sandboxes —
without it every dispatch fails with an auth error.

---

## Deploying to a Digital Ocean droplet

The bot only needs Python + Docker — no Node, no Claude Code CLI, nothing else.
A $4–6/mo basic droplet (Ubuntu 24.04, 512MB+) is plenty.

```bash
ssh root@<droplet-ip>
apt update && apt install -y docker.io git
systemctl enable --now docker

git clone https://github.com/<you>/SMILE-factory.git
cd SMILE-factory
cp apps/delulu_discord/.env.example apps/delulu_discord/.env
nano apps/delulu_discord/.env   # paste DISCORD_BOT_TOKEN
```

Then from your **laptop**, copy your Modal credentials up:

```bash
scp ~/.modal.toml root@<droplet-ip>:/root/.modal.toml
```

Create the bot env file at the path the Makefile expects:

```bash
cat > /root/disco.env <<'EOF'
DISCORD_BOT_TOKEN=your-discord-bot-token-here
EOF
chmod 600 /root/disco.env
```

Then build and run the container via the bot sub-project's Makefile:

```bash
cd /root/SMILE-factory
make deploy-bot                        # = make -C apps/delulu_discord deploy
make -C apps/delulu_discord logs       # tail to confirm it connected to Discord
```

The container runs with `--restart=unless-stopped`, so it survives
crashes and reboots.

The Claude OAuth credentials are **never** stored on the droplet — they
live only as a Modal Secret and inside the Modal Volume. The droplet only
holds the Discord bot token and Modal client credentials.

---

## Testing it

The bot only responds to explicit `@mentions`, so invite it to any
channel you like — `#general` works. Send a message mentioning the bot:

```
@corchestra tell me a joke
```

Within ~10 seconds the bot should open a thread off your message (named
after the prompt, e.g. `"tell me a joke"`) and post Claude Code's
response inside it. Reply in the thread without mentioning the bot —
it'll auto-continue the same Claude Code session, so follow-ups can
reference earlier context.

If nothing happens:

```bash
make -C apps/delulu_discord logs
```

Common failure modes:
- **No reaction to your mention** — Message Content Intent not enabled,
  or the bot isn't allowed to see / post in that channel.
- **Modal auth error** — `/root/.modal.toml` not mounted into the container.
- **Claude auth error** — `claude-oauth` secret missing, or `/vol/claude-home/.claude/.credentials.json` is stale (delete it on the volume and re-seed).
- **`--continue` starts a new session** — usually means `HOME` isn't pointed at the volume inside the sandbox; check `sandbox.exec` logs in Modal.

---

## Local development

### Lint and format

Ruff (lint + formatter) is the only style tooling. Each sub-project
owns its own `[tool.ruff]` config under `apps/delulu_discord/pyproject.toml`
and `apps/delulu_sandbox_modal/pyproject.toml`. Ruff is installed as a
dev extra in each, so run `make sync-dev` inside the sub-project once
(or whenever you `uv sync` fresh).

```bash
# top-level, runs ruff across both apps
make check

# or scoped to one sub-project
make -C apps/delulu_discord check
make -C apps/delulu_discord lint      # ruff check --fix
make -C apps/delulu_discord fmt       # ruff format (writes)
```

### Pre-commit hook

The repo-root [`.pre-commit-config.yaml`](.pre-commit-config.yaml)
includes a ruff + ruff-format hook scoped to files under `apps/`. To
enable it on your laptop:

```bash
uv tool install pre-commit   # one-time
pre-commit install           # installs the git hook
```

After this, every `git commit` runs trailing-whitespace, EOF, TOML/YAML,
ruff lint (with auto-fix), and ruff-format on the staged files.
If ruff rewrites anything you'll need to re-stage and commit again.

To run manually across all changed files without committing:

```bash
pre-commit run --all-files
```

---

## Updating manually

```bash
cd ~/SMILE-factory && git pull
make deploy-bot              # rebuild image + restart bot container
```

If you changed anything under `apps/delulu_sandbox_modal/`, also
redeploy the Modal sandbox:

```bash
make deploy-modal            # or: make deploy-all to do both in order
```

`make help` (at the top level or inside either sub-project) lists
available targets.

---

## CI/CD (GitHub Actions)

The workflow at
[`.github/workflows/discord-orchestrator-deploy.yml`](.github/workflows/discord-orchestrator-deploy.yml)
deploys on every push to `main` that touches `apps/**` or the top-level
`Makefile`.

It's a single job that SSHes into the droplet and runs the same
`make` targets you'd run manually: `git pull`, then
`make -C apps/delulu_discord deploy` (rebuild image + restart
container), plus `make -C apps/delulu_sandbox_modal sync modal-deploy`
conditionally when files under `apps/delulu_sandbox_modal/` actually
changed. No container registry, no image pushes — the droplet is both
the build and run target.

### One-time droplet prep

The workflow assumes the droplet is already set up as described in
[Deploying to a Digital Ocean droplet](#deploying-to-a-digital-ocean-droplet)
above — i.e.:

- Docker installed
- `/root/SMILE-factory` checked out (the workflow runs `git pull` inside it)
- `/root/disco.env` containing `DISCORD_BOT_TOKEN=...`
- `/root/.modal.toml` as a real file (from `uv run modal token new` on the droplet)
- `uv` installed and on `$PATH` for the `root` user (needed for the
  conditional `uv run modal deploy` step)

If you can run the manual deploy from the [Updating manually](#updating-manually)
section successfully, the workflow will work too — it runs the same commands.

### One-time GitHub setup

Generate a dedicated SSH deploy key (don't reuse your personal key):

```bash
# On your laptop
ssh-keygen -t ed25519 -f ~/.ssh/disco_deploy -N "" -C "github-actions-disco"

# Install the public half on the droplet
ssh-copy-id -i ~/.ssh/disco_deploy.pub root@<droplet-ip>

# Test the key works
ssh -i ~/.ssh/disco_deploy root@<droplet-ip> 'echo ok'

# Print the private half — you'll paste this into GitHub
cat ~/.ssh/disco_deploy
```

Add these as **Repository secrets** under **Settings → Secrets and variables
→ Actions** on GitHub:

| Secret | Value |
|---|---|
| `DROPLET_HOST` | The droplet's public IPv4 address |
| `DROPLET_SSH_KEY` | Full contents of `~/.ssh/disco_deploy` (including the `-----BEGIN`/`-----END` lines) |

That's it — no Modal tokens or registry secrets needed, because everything
happens on the droplet which already has its own Modal auth.

### Testing the workflow

The workflow has a `workflow_dispatch` trigger, so you can run it manually
without pushing a commit:

1. GitHub → **Actions** tab → **discord-orchestrator deploy** →
   **Run workflow** → pick `main` → **Run workflow**.
2. Watch the logs in real time. You should see the SSH step print the
   old/new SHAs, the Modal redeploy (or skip) message, the Docker build,
   and `Deploy complete`.
3. On the droplet, `docker ps` should show the `disco` container running,
   and `docker logs disco` should show a clean startup.

If it fails, the SSH step prints everything to the workflow log — no need
to SSH in to debug. Common failure modes:

- **"Permission denied (publickey)"** — `DROPLET_SSH_KEY` is wrong, or the
  public key isn't in `/root/.ssh/authorized_keys` on the droplet.
- **"uv: command not found"** — `uv` isn't on the `root` user's `$PATH` in
  non-interactive SSH sessions. Fix: `echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.bashrc` on the droplet (the workflow already re-exports this inside the script, but this ensures it also works for manual SSH).
- **"error: unable to connect to Docker"** — `systemctl status docker` on
  the droplet, `systemctl enable --now docker` if it's not running.

### Rollback

Rollback is a manual `git` operation on the droplet:

```bash
ssh root@<droplet-ip>
cd /root/SMILE-factory
git log --oneline -10             # find the commit you want to roll back to
git reset --hard <good-sha>

make deploy-bot                   # or `make deploy-all` if sandbox code rolled back too
```

Then force-revert on the remote once you've confirmed the rollback works,
so the next push doesn't redeploy the broken version.

---

## Project structure

```
SMILE-factory/
├── apps/
│   ├── delulu_discord/                     # VPS-hosted Discord bot
│   │   ├── pyproject.toml                  # discord.py, modal (client), pydantic, structlog
│   │   ├── uv.lock
│   │   ├── .python-version
│   │   ├── .env.example                    # bot environment template
│   │   ├── Dockerfile                      # bot image only — no sandbox deps
│   │   ├── Makefile                        # VPS targets: sync / check / image / deploy / logs
│   │   └── src/delulu_discord/
│   │       ├── main.py                     # Bot entrypoint + @mention gating
│   │       ├── handlers.py                 # Message routing + attachment download + result posting
│   │       ├── session_manager.py          # Thread ↔ session mapping with TTL
│   │       ├── dispatcher.py               # SandboxDispatcher (client-side Function.from_name wrapper)
│   │       └── settings.py                 # Pydantic Settings from env vars
│   │
│   └── delulu_sandbox_modal/               # Modal-deployed sandbox function
│       ├── pyproject.toml                  # modal, structlog — nothing else
│       ├── uv.lock
│       ├── .python-version
│       ├── Makefile                        # Modal targets: sync / check / modal-deploy
│       └── src/delulu_sandbox_modal/
│           └── app.py                      # Modal App, image, volume, secret, run_claude_code
│
├── Makefile                                # top-level dispatcher (check / deploy-bot / deploy-modal / deploy-all)
├── ARCHITECTURE.md                         # Full design doc
├── README.md                               # this file
├── prd/                                    # planning docs
├── infra/managed-agents/                   # sibling infra work (unrelated to delulu)
└── (LoRA-Instruct files — archived, see below)
```

---

# LoRA-Instruct (archived)

> Retained in-tree pending a later migration. The delulu Discord
> orchestrator above is the active project in this repo.

This repository contains code for fine-tuning permissive open source LLMs using [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685).

Code is tested using [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset.

- Estimated training time for fine-tuning RedPajama-INCITE-Base-7B-v0.1 with a single RTX 3090 and Stanford Alpaca is ~12 hours.
- Estimated training time for fine-tuning RedPajama-INCITE-Base-7B-v0.1 with RTX 3090 and RTX Titan and Stanford Alpaca is ~6.5 hours.
- Currently only supports LoRA Instruct fine-tuning [RedPajama-INCITE-Base-7B-v0.1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1).


Inspired by [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)

## Trained Models
| Model | Runs | Training Time  | Link |
|:-------|:----:|:----:|:-----:|
| LLaMA 3B | :white_large_square: |  |  |
| LLaMA 7B | :white_large_square: |  |  |
| RedPajama 3B | :white_check_mark: | 1:44:14 | |
| RedPajama 7B | :white_check_mark: | 3:09:58 | |
| MPT 3B | :white_large_square: |  |  |
| MPT 7B | :white_large_square: |  |  |
| Falcon 7B | :white_check_mark: |  |  |

#### Training Hardware Spec
```
Ubuntu 20.04.1 LTS (WSL2)

Driver Version: 531.41
CUDA Version: 12.1
cuDNN version: 8.5.0
```

### Local Setup

Install dependencies
```bash
poetry install
```

To fine-tune using NVidia 2000 series GPU or earlier, please comment out this line in `finetune.py`
```python
model = prepare_model_for_int8_training(model)
```

### Training (`finetune.py`)

This file contains a straightforward application of PEFT / LoRA to decoder only model,
as well as some code related to prompt construction and tokenization.

Example usage:
```bash
python finetune.py \
    --base_model 'togethercomputer/RedPajama-INCITE-Base-7B-v0.1' \
    --output_dir './lora-redpajama'
```

#### Distributed Training with 🤗 Accelerate

We uses HuggingFace's `accelerate` library for distributed training. The following is an example for distributed training with two GPUs.

* NOTE: please set the following environment variables
```bash
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1
```

```bash
torchrun \
    --nproc_per_node=2 \
    --master_port=1234 \
    finetune.py \
    --base_model 'togethercomputer/RedPajama-INCITE-Base-7B-v0.1' \
    --output_dir './lora-redpajama'
```

## References
- [LoRA: Low-Rank Adapation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods](https://github.com/huggingface/peft)
- [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca)
- [EMNLP 2022 Tutorial: Modular and Parameter-Efficient Fine-Tuning for NLP Models](https://www.youtube.com/watch?v=KoOlcX3XLd4)

# Discord Orchestrator

Discord bot that dispatches Claude Code tasks to ephemeral Modal Sandboxes.

## How it works

1. User sends a message in a `#claude-*` channel
2. Bot creates a Discord thread and a new Claude Code session
3. Bot dispatches the task to a Modal Sandbox (ephemeral container with Claude Code installed)
4. Sandbox runs the task, returns output, dies
5. Bot posts the result in the thread
6. Replies in the thread resume the same session (within 1hr TTL)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Discord bot token (with Message Content intent enabled)
- A Modal account with `modal` CLI authenticated
- An Anthropic API key

### Install

```bash
cd infra/discord-orchestrator
uv sync
```

### Configure

```bash
cp .env.example .env
# Edit .env with your tokens
```

Create the Modal secret:
```bash
modal secret create anthropic-key ANTHROPIC_API_KEY=sk-ant-...
```

### Run

```bash
# Start the bot locally
uv run bot

# Or with Docker
docker build -t discord-orchestrator .
docker run --env-file .env discord-orchestrator
```

### Deploy the Modal app

```bash
modal deploy src/modal_dispatch/app.py
```

## Project Structure

```
discord-orchestrator/
├── src/
│   ├── bot/
│   │   ├── main.py              # Bot entrypoint
│   │   ├── handlers.py          # Message routing + result posting
│   │   └── session_manager.py   # Thread ↔ session mapping with TTL
│   ├── modal_dispatch/
│   │   ├── app.py               # Modal app, image, volume, secrets
│   │   ├── sandbox.py           # Claude Code execution + dispatcher
│   │   └── workspace.py         # Workspace provisioning (git clone)
│   └── config/
│       └── settings.py          # Pydantic Settings from env vars
├── ARCHITECTURE.md              # Full design doc
├── Dockerfile                   # For hosting the bot process
├── pyproject.toml               # uv project config
└── .env.example                 # Environment variable template
```

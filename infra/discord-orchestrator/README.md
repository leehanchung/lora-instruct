# Discord Orchestrator

Discord bot that dispatches Claude Code tasks to ephemeral Modal Sandboxes.

## How it works

1. User sends a message in a `#claude-*` channel
2. Bot creates a Discord thread and a new Claude Code session
3. Bot dispatches the task to a Modal Sandbox (ephemeral container with Claude Code installed)
4. Sandbox runs the task, returns output, dies
5. Bot posts the result in the thread
6. Replies in the thread resume the same session (within 1hr TTL)

The bot process itself is tiny (~50MB RAM) and just dispatches jobs — all
heavy lifting (and all Claude Code execution) happens inside Modal sandboxes.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
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
cd infra/discord-orchestrator
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

The sandbox persists `.credentials.json` to a Modal Volume so rotated refresh
tokens survive across invocations. The secret only seeds the volume on first
run — after that the volume copy is the source of truth. If auth ever breaks,
delete `claude-auth/` from the volume and re-create the secret.

Deploy the sandbox app:

```bash
uv run modal deploy src/modal_dispatch/app.py
```

### 3. Run the bot

Configure the bot's environment:

```bash
cp .env.example .env
# Edit .env and paste DISCORD_BOT_TOKEN
```

#### Option A — locally

```bash
uv run bot
```

#### Option B — Docker (any host)

```bash
docker build -t discord-orchestrator .
docker run -d --name disco --restart=unless-stopped \
  --env-file .env \
  -v ~/.modal.toml:/root/.modal.toml:ro \
  discord-orchestrator

docker logs -f disco
```

The `~/.modal.toml` mount gives the bot the Modal credentials it needs to
dispatch sandboxes. Without it, every dispatch will fail with an auth error.

---

## Deploying to a Digital Ocean droplet

The bot only needs Python + Docker — no Node, no Claude Code CLI, nothing else.
A $4–6/mo basic droplet (Ubuntu 24.04, 512MB+) is plenty.

```bash
ssh root@<droplet-ip>
apt update && apt install -y docker.io git
systemctl enable --now docker

git clone https://github.com/<you>/lora-instruct.git
cd lora-instruct/infra/discord-orchestrator
cp .env.example .env
nano .env   # paste DISCORD_BOT_TOKEN
```

Then from your **laptop**, copy your Modal credentials up:

```bash
scp ~/.modal.toml root@<droplet-ip>:/root/.modal.toml
```

Back on the droplet, build and run:

```bash
docker build -t discord-orchestrator .
docker run -d --name disco --restart=unless-stopped \
  --env-file .env \
  -v /root/.modal.toml:/root/.modal.toml:ro \
  discord-orchestrator
docker logs -f disco
```

`--restart=unless-stopped` keeps the bot alive across crashes and reboots.

The Claude OAuth credentials are **never** stored on the droplet — they live
only as a Modal Secret and inside the Modal Volume.

---

## Testing it

In your Discord server, create a channel literally named `claude-test` (or
anything starting with `claude-`). Send a message. Within ~10 seconds the bot
should open a thread and post Claude Code's response.

If nothing happens:

```bash
docker logs disco
```

Common failure modes:
- **No Discord events** — Message Content Intent not enabled, or bot not in the channel.
- **Modal auth error** — `~/.modal.toml` not mounted into the container.
- **Claude auth error** — `claude-oauth` secret missing, or refresh token invalidated.

---

## Updating

```bash
cd ~/lora-instruct && git pull
cd infra/discord-orchestrator
docker build -t discord-orchestrator . && docker restart disco
```

If you changed anything under `src/modal_dispatch/`, also redeploy the sandbox:

```bash
uv run modal deploy src/modal_dispatch/app.py
```

---

## Project structure

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

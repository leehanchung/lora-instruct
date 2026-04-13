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

## Updating manually

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

## CI/CD (GitHub Actions)

The workflow at
[`.github/workflows/discord-orchestrator-deploy.yml`](../../.github/workflows/discord-orchestrator-deploy.yml)
deploys on every push to `main` that touches `infra/discord-orchestrator/**`.

It's a single job that SSHes into the droplet and runs the same commands
you'd run manually: `git pull`, rebuild the Docker image, restart the
container, and (only if files under `src/modal_dispatch/` actually changed)
redeploy the Modal sandbox app. No container registry, no image pushes —
the droplet is both the build and run target.

### One-time droplet prep

The workflow assumes the droplet is already set up as described in
[Deploying to a Digital Ocean droplet](#deploying-to-a-digital-ocean-droplet)
above — i.e.:

- Docker installed
- `/root/lora-instruct` checked out (the workflow runs `git pull` inside it)
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
cd /root/lora-instruct
git log --oneline -10             # find the commit you want to roll back to
git reset --hard <good-sha>

cd infra/discord-orchestrator
docker build -t discord-orchestrator .
docker stop disco && docker rm disco
docker run -d --name disco --restart=unless-stopped \
  --env-file /root/disco.env \
  -v /root/.modal.toml:/root/.modal.toml:ro \
  discord-orchestrator
```

Then force-revert on the remote once you've confirmed the rollback works,
so the next push doesn't redeploy the broken version.

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

# Discord Orchestrator — Architecture

## Overview

A lightweight system that lets users trigger Claude Code tasks from Discord. Modal provides ephemeral compute, Discord provides the conversational interface, and a persistent Modal Volume holds workspaces and session history.

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   Discord    │         │   Bot Process    │         │  Modal Sandbox   │
│   (User)     │◄───────►│   (always-on)    │────────►│  (ephemeral)     │
│              │  d.py    │                  │  modal   │                  │
│  threads =   │  gateway │  - route msgs    │  spawn   │  - claude code   │
│  sessions    │         │  - map threads   │         │  - task exec     │
└──────────────┘         │  - post results  │         │  - stdout/stderr │
                         └──────────────────┘         └────────┬─────────┘
                                                               │
                                                      ┌────────▼─────────┐
                                                      │  Modal Volume    │
                                                      │  (persistent)    │
                                                      │                  │
                                                      │  /workspaces/    │
                                                      │    <session-id>/ │
                                                      │  /.claude/       │
                                                      │    sessions/     │
                                                      └──────────────────┘
```

## Components

### 1. Discord Bot (`src/bot/`)

**Always-on process** hosted on a cheap VM (Fly.io, Railway, or local machine). Responsibilities:

- Listen for messages via discord.py gateway connection
- **Thread-session mapping**: new channel message → create Discord thread + allocate session ID. Reply in thread → reuse session ID.
- Dispatch work to Modal via `modal_dispatch`
- Post results back into the originating thread
- Truncate long outputs to fit Discord's 2000-char limit (with file upload fallback)

Key files:
- `main.py` — bot entrypoint, client setup, event registration
- `handlers.py` — message/command event handlers
- `session_manager.py` — thread↔session mapping, TTL expiry

### 2. Modal Sandbox Dispatcher (`src/modal_dispatch/`)

**Ephemeral containers** spun up per task. Each sandbox:

- Mounts the persistent volume at `/vol`
- Has Claude Code installed (via image build)
- Receives: session_id, workspace_path, prompt, whether to resume
- Runs `claude` CLI with appropriate flags
- Returns stdout/stderr as the result
- Dies after completion (~4GB RAM, configurable)

Key files:
- `app.py` — Modal app definition, image build spec, volume config
- `sandbox.py` — the function that runs inside each sandbox
- `workspace.py` — workspace provisioning (git clone, directory setup)

### 3. Config (`src/config/`)

- `settings.py` — Pydantic Settings model, loads from env vars / .env

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Workspace content** | Git clone onto volume | Most common use case; user provides repo URL, we clone into `/vol/workspaces/<session>/` |
| **Streaming** | Post final result | Simpler to implement; Discord message edit rate limits make streaming janky. Future enhancement. |
| **Bot hosting** | Separate always-on process | Bot is lightweight (~50MB RAM), doesn't need Modal. Fly.io or Railway for $5/mo. |
| **Auth** | Modal Secret | Anthropic API key stored as a Modal Secret, injected into sandbox env. Discord token lives with the bot process. |
| **Session TTL** | 1 hour default | Replies in a thread within 1hr resume the Claude Code session. After that, new session, same workspace. |

## Session Lifecycle

```
1. User sends message in #claude channel
   → Bot creates Discord thread (titled from first ~50 chars)
   → Bot generates session_id (uuid4)
   → Bot stores {thread_id: session_id, workspace_path, created_at}

2. Bot calls modal_dispatch.run_task(session_id, prompt, resume=False)
   → Modal spins up sandbox
   → Sandbox runs: claude --print -p "<prompt>" --output-format text
   → Sandbox returns output string

3. Bot posts output to the thread
   → If > 1900 chars, uploads as file attachment

4. User replies in the thread
   → Bot looks up session_id from thread_id
   → If session age < TTL: resume=True (--resume flag)
   → If session age >= TTL: resume=False (fresh session, same workspace)
   → Dispatch again
```

## Volume Layout

```
/vol/
├── workspaces/
│   ├── <session-id-1>/
│   │   ├── .git/
│   │   └── <repo contents>
│   └── <session-id-2>/
│       └── ...
└── .claude/
    └── sessions/
        └── <session history files>
```

## Future Enhancements

- **Streaming output** via Discord message edits (with rate limit backoff)
- **Slash commands** for workspace management (`/clone`, `/workspace list`, `/workspace delete`)
- **File upload** — attach files in Discord, they get placed in the workspace
- **Multi-repo** — switch between repos within a session
- **Cost tracking** — log Modal compute seconds per user/session
- **GitHub webhook integration** — trigger Claude Code on PR events

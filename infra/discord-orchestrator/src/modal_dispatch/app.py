"""Modal app definition — image, volume, secrets, and the sandbox function.

This is the sole deploy target: `modal deploy src/modal_dispatch/app.py`.

The sandbox function `run_claude_code` is defined here (not in sandbox.py)
so Modal's deploy importer and Python's package importer agree on a single
`app` instance. Previously the function lived in sandbox.py and app.py did
`from src.modal_dispatch import sandbox` at the bottom — which worked when
imported as a package but broke under `modal deploy`, because Modal imports
this file as a standalone script, which then imports `src.modal_dispatch.app`
a *second* time and registers the decorator on a different `app` object.
"""

from __future__ import annotations

import modal
import structlog

logger = structlog.get_logger()

# ── Modal App ────────────────────────────────────────────────
app = modal.App("discord-orchestrator")

# ── Persistent Volume ────────────────────────────────────────
# Shared across all sandboxes. Holds workspaces and Claude Code session history.
volume = modal.Volume.from_name("claude-workspaces", create_if_missing=True)

# ── Container Image ──────────────────────────────────────────
# Pre-bake Claude Code into the image so sandbox startup is fast.
sandbox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .run_commands(
        # Install Node.js (required by Claude Code)
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        # Install Claude Code globally
        "npm install -g @anthropic-ai/claude-code",
    )
)

# ── Secrets ──────────────────────────────────────────────────
# Claude Code OAuth credentials (Pro/Max subscription). Seeded into the volume on
# first run, then refreshed-in-place on the volume so rotated refresh tokens
# survive across sandbox invocations.
#
# Create with:
#   modal secret create claude-oauth \
#     CLAUDE_CREDENTIALS_JSON="$(cat ~/.claude/.credentials.json)"
claude_oauth_secret = modal.Secret.from_name("claude-oauth")


# ── Modal Function (runs inside the sandbox) ─────────────────
@app.function(
    image=sandbox_image,
    volumes={"/vol": volume},
    secrets=[claude_oauth_secret],
    memory=4096,
    timeout=300,
)
def run_claude_code(
    session_id: str,
    workspace_path: str,
    prompt: str,
    *,
    resume: bool = False,
) -> str:
    """Execute a Claude Code task inside an ephemeral Modal sandbox."""
    import os
    import shutil
    import subprocess

    os.makedirs(workspace_path, exist_ok=True)

    # ── Claude Code OAuth credentials ────────────────────────
    # Persisted on the volume so refreshed/rotated refresh tokens survive
    # across sandbox invocations. Seeded from the secret on first run.
    persisted_creds = "/vol/claude-auth/.credentials.json"
    home_claude_dir = os.path.expanduser("~/.claude")
    home_creds = os.path.join(home_claude_dir, ".credentials.json")
    os.makedirs(home_claude_dir, exist_ok=True)
    os.makedirs(os.path.dirname(persisted_creds), exist_ok=True)

    if not os.path.exists(persisted_creds):
        seed = os.environ.get("CLAUDE_CREDENTIALS_JSON")
        if not seed:
            raise RuntimeError(
                "No persisted Claude credentials and CLAUDE_CREDENTIALS_JSON "
                "secret is empty — re-create the claude-oauth Modal secret."
            )
        with open(persisted_creds, "w") as f:
            f.write(seed)
        os.chmod(persisted_creds, 0o600)
        volume.commit()

    shutil.copyfile(persisted_creds, home_creds)
    os.chmod(home_creds, 0o600)

    cmd = [
        "claude",
        "--print",
        "-p", prompt,
        "--output-format", "text",
    ]
    if resume:
        cmd.append("--resume")

    # ANTHROPIC_API_KEY is stripped — Claude Code prefers it over OAuth if set,
    # which would bypass the Pro/Max subscription auth.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["CLAUDE_PROJECT_DIR"] = workspace_path

    logger.info(
        "sandbox.exec",
        session_id=session_id,
        resume=resume,
        workspace=workspace_path,
        prompt_preview=prompt[:80],
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=workspace_path,
        env=env,
        timeout=280,
    )

    if result.returncode != 0:
        logger.warning(
            "sandbox.nonzero_exit",
            session_id=session_id,
            returncode=result.returncode,
            stderr=result.stderr[:500],
        )

    # Persist any refreshed OAuth credentials back to the volume. The refresh
    # token rotates on every refresh, so the next sandbox MUST see the new file.
    if os.path.exists(home_creds):
        shutil.copyfile(home_creds, persisted_creds)
        os.chmod(persisted_creds, 0o600)

    volume.commit()

    output = result.stdout.strip()
    if result.returncode != 0 and result.stderr.strip():
        output += f"\n\n[stderr]\n{result.stderr.strip()}"

    return output

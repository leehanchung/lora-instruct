"""Modal app definition — image, volume, secrets, and the sandbox function.

This is the sole deploy target: `modal deploy src/delulu_sandbox_modal/app.py`.

The sandbox function `run_claude_code` is defined here (not in a separate
module) so Modal's deploy importer and Python's package importer agree on a
single `app` instance. Previously the function lived in a sibling file and
this module re-imported it at the bottom — which worked when imported as a
package but broke under `modal deploy`, because Modal imports this file as a
standalone script, which then imports the package path a *second* time and
registers the decorator on a different `app` object.
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
    # Requires Modal Image Builder 2025.06 or newer — the default 2023.12
    # builder only supports Python 3.10–3.12. Set workspace-wide with:
    #   modal config set image_builder_version 2025.06
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("git", "curl")
    # app.py is imported inside the sandbox to call run_claude_code, so any
    # third-party module it imports at module level must be present here too.
    .pip_install("structlog>=24.0")
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
    attachments: list[tuple[str, bytes]] | None = None,
    message_id: int | None = None,
) -> str:
    """Execute a Claude Code task inside an ephemeral Modal sandbox."""
    import os
    import re
    import shutil
    import subprocess

    os.makedirs(workspace_path, exist_ok=True)

    # ── Materialize Discord attachments onto the workspace ───
    # Claude Code only sees what's in the prompt and on disk, so any files
    # the user attached have to be written into the workspace and then
    # referenced explicitly in the prompt. We scope each message's files
    # under _attachments/<message_id>/ so repeated uploads in the same
    # thread can't collide on filename.
    attachment_lines: list[str] = []
    if attachments:
        bucket = str(message_id) if message_id is not None else "latest"
        att_dir = os.path.join(workspace_path, "_attachments", bucket)
        os.makedirs(att_dir, exist_ok=True)
        for raw_name, data in attachments:
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name) or "file"
            path = os.path.join(att_dir, safe)
            with open(path, "wb") as f:
                f.write(data)
            rel = os.path.relpath(path, workspace_path)
            attachment_lines.append(f"- {rel} ({len(data)} bytes)")

    if attachment_lines:
        prompt = (
            f"{prompt}\n\n"
            "[The user attached the following files to this message. "
            "Read them from the workspace before responding:]\n" + "\n".join(attachment_lines)
        )

    # ── Persistent Claude Code home on the Modal Volume ──────
    # Point HOME at /vol/claude-home so Claude Code reads and writes its
    # entire state directory (credentials, settings, session history under
    # ~/.claude/projects/<hash-of-cwd>/) directly on the persistent volume.
    # Combined with a deterministic per-thread workspace_path, this is what
    # makes `claude --continue` work across ephemeral sandbox invocations.
    claude_home = "/vol/claude-home"
    claude_config = f"{claude_home}/.claude"
    creds_file = f"{claude_config}/.credentials.json"
    os.makedirs(claude_config, exist_ok=True)

    # One-time migration from the previous /vol/claude-auth/ layout.
    legacy_creds = "/vol/claude-auth/.credentials.json"
    if not os.path.exists(creds_file) and os.path.exists(legacy_creds):
        shutil.copyfile(legacy_creds, creds_file)
        os.chmod(creds_file, 0o600)

    # Seed credentials from the Modal secret on first run.
    if not os.path.exists(creds_file):
        seed = os.environ.get("CLAUDE_CREDENTIALS_JSON")
        if not seed:
            raise RuntimeError(
                "No persisted Claude credentials and CLAUDE_CREDENTIALS_JSON "
                "secret is empty — re-create the claude-oauth Modal secret."
            )
        with open(creds_file, "w") as f:
            f.write(seed)
        os.chmod(creds_file, 0o600)

    cmd = [
        "claude",
        "--print",
        "-p",
        prompt,
        "--output-format",
        "text",
    ]
    if resume:
        # --continue resumes the most recent Claude Code session in the cwd.
        # Since workspace_path is stable per thread (see session_manager) and
        # ~/.claude/projects/<cwd-hash>/ lives on the volume, this finds the
        # prior conversation. --resume wants an explicit Claude-internal UUID
        # which we don't track, so --continue is the right primitive here.
        cmd.append("--continue")

    # ANTHROPIC_API_KEY is stripped — Claude Code prefers it over OAuth if set,
    # which would bypass the Pro/Max subscription auth.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["HOME"] = claude_home
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

    # Persist everything Claude Code wrote: rotated refresh tokens, new
    # session history files, any settings updates.
    volume.commit()

    output = result.stdout.strip()
    if result.returncode != 0 and result.stderr.strip():
        output += f"\n\n[stderr]\n{result.stderr.strip()}"

    return output

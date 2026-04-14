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

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import modal
import structlog

if TYPE_CHECKING:
    # Imported for type-checking only. At runtime the generator yields plain
    # dicts — keeping this off the runtime import graph avoids any chance of
    # re-entering app.py through a sibling module during ``modal deploy``
    # (see the module docstring for the full rationale).
    from delulu_sandbox_modal.events import Event

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


# ── Stream-json helpers (module-level for testability) ───────
# These parse Claude Code's ``--output-format stream-json`` lines into the
# internal event dicts declared in ``events.py``. They live here rather
# than in a sibling module because ``app.py`` is imported as ``__main__``
# by ``modal deploy`` (see the module docstring); keeping helpers inline
# avoids the double-import footgun that broke the previous layout.


def _summarize_tool_input(tool: str, input_data: dict[str, Any]) -> str:
    """Return a short one-liner describing a tool invocation for display."""
    if tool in ("Read", "Edit", "Write"):
        path = input_data.get("file_path") or ""
        return f"`{path}`" if path else ""
    if tool == "Bash":
        cmd = input_data.get("command") or ""
        return f"`{cmd[:80]}`" if cmd else ""
    if tool in ("Grep", "Glob"):
        pattern = input_data.get("pattern") or ""
        return f"`{pattern}`" if pattern else ""
    return ""


def _summarize_tool_result(content: Any) -> str:
    """Return a short preview of a tool result.

    Claude Code's ``tool_result`` content may be a string, a list of
    content blocks, or something else depending on CC version — parse
    defensively.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped:
            return ""
        first_line = stripped.splitlines()[0]
        return (first_line[:80] + "…") if len(first_line) > 80 else first_line
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return _summarize_tool_result(block.get("text"))
        return ""
    return str(content)[:80]


def _flatten_stream_event(
    parsed: dict[str, Any],
    tool_names_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    """Map a Claude Code stream-json line to zero-or-more internal events.

    Unknown event types are skipped. ``tool_names_by_id`` is mutated so
    subsequent ``tool_result`` blocks can be labelled with their tool name.
    """
    out: list[dict[str, Any]] = []
    kind = parsed.get("type")

    if kind == "assistant":
        message = parsed.get("message") or {}
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text") or ""
                if text:
                    out.append({"type": "text", "text": text})
            elif btype == "tool_use":
                tool_id = block.get("id") or ""
                tool = block.get("name") or "?"
                if tool_id:
                    tool_names_by_id[tool_id] = tool
                summary = _summarize_tool_input(tool, block.get("input") or {})
                out.append({"type": "tool_use", "tool": tool, "summary": summary})
            elif btype == "thinking":
                thought = block.get("thinking") or ""
                if thought:
                    out.append({"type": "thinking", "text": thought})
    elif kind == "user":
        message = parsed.get("message") or {}
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id") or ""
                tool = tool_names_by_id.get(tool_id, "")
                is_error = bool(block.get("is_error"))
                summary = _summarize_tool_result(block.get("content"))
                out.append(
                    {
                        "type": "tool_result",
                        "tool": tool,
                        "ok": not is_error,
                        "summary": summary,
                    }
                )
    return out


# ── Provisioning Modal Function ──────────────────────────────
# Dedicated function for git operations (clone, fetch, worktree add).
# `max_containers=1` is load-bearing: it's how we get cross-container
# mutual exclusion on provisioning without writing any lock code. The
# flock scout at `tools/verify_flock.py` empirically confirmed that no
# filesystem lock primitive (flock, mkdir, link) provides cross-
# container atomicity on Modal Volumes, so concurrency has to live at
# Modal's orchestration layer instead. See the "Concurrency" section
# of `prd/repo-provisioning.md` for the full rationale.
#
# The image intentionally omits nodejs/claude-code — this function
# only needs git, and reusing `sandbox_image` would pull in the
# ~200MB Claude Code install on every provisioning container spin-up.
provisioner_image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("git", "ca-certificates")
    .pip_install("structlog>=24.0")
)


@app.function(
    image=provisioner_image,
    volumes={"/vol": volume},
    max_containers=1,
    timeout=300,
)
def provision_workspace(
    thread_id: int,
    repo_url: str | None,
    ref: str = "HEAD",
) -> str:
    """Ensure a workspace exists at ``/vol/workspaces/<thread_id>`` and return its path.

    Called via ``.remote()`` from ``run_claude_code``. Modal's
    `max_containers=1` serializes all concurrent invocations of this
    function at the orchestration layer, so the underlying git work
    runs without any locking primitive of its own.

    Commits the volume before returning so the caller's subsequent
    ``volume.reload()`` picks up the newly-created worktree.
    """
    # Import inside the function body so the top-level module import
    # of `app.py` doesn't pull in repo_provisioner at deploy-parse
    # time. (Not strictly necessary — repo_provisioner has no Modal
    # decorators and is safe to import at module level — but keeping
    # heavyweight imports lazy matches the existing pattern in
    # `run_claude_code` and avoids any surprise at `modal deploy`.)
    from delulu_sandbox_modal import repo_provisioner

    workspace_path, timings = repo_provisioner.provision_workspace(
        thread_id=thread_id,
        repo_url=repo_url,
        ref=ref,
    )

    logger.info(
        "provision.timing",
        thread_id=thread_id,
        repo_url=repo_url,
        ref=ref,
        workspace_path=workspace_path,
        total_ms=timings.total_ms,
        cold_clone_ms=timings.cold_clone_ms,
        fetch_ms=timings.fetch_ms,
        worktree_ms=timings.worktree_ms,
        short_circuit=timings.short_circuit,
    )

    # Commit so the caller (`run_claude_code` in its own container)
    # can `volume.reload()` and see the fresh worktree + bare cache.
    volume.commit()
    return workspace_path


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
    prompt: str,
    *,
    workspace_path: str | None = None,
    thread_id: int | None = None,
    repo_url: str | None = None,
    ref: str = "HEAD",
    resume: bool = False,
    attachments: list[tuple[str, bytes]] | None = None,
    message_id: int | None = None,
) -> Iterator[Event]:
    """Execute a Claude Code task inside an ephemeral Modal sandbox.

    This is a Modal **generator function**: callers invoke it via
    ``.remote_gen(...)`` and receive an iterator of event dicts as
    Claude Code progresses. A terminal ``DoneEvent`` carries the final
    assistant text and run stats; on nonzero subprocess exit an
    ``ErrorEvent`` is yielded instead and the function returns.

    Workspace derivation — two paths, transitional during the
    repo-provisioning rollout:

    1.  **Legacy (Phase 1):** caller passes ``workspace_path``
        directly (current bot behavior). Used as-is, no provisioning
        call. Keeps the bot side working without changes while Phase
        2 rewires `dispatcher.run_task` to pass `thread_id` instead.

    2.  **New (Phase 2+):** caller passes ``thread_id`` (and
        optionally ``repo_url`` / ``ref``). The sandbox derives the
        workspace via ``provision_workspace.remote(...)``, which
        handles the bare-cache clone, fetch, and worktree creation
        under Modal's ``max_containers=1`` serialization. The
        subsequent ``volume.reload()`` picks up the committed state
        so this container sees the fresh worktree.

    Once Phase 2 lands, the ``workspace_path`` kwarg can be removed
    in a follow-up — keeping it now minimizes the blast radius of
    the signature change.
    """
    import json
    import os
    import re
    import shutil
    import subprocess
    import threading

    # ── Workspace derivation ────────────────────────────────
    if workspace_path is None:
        if thread_id is None:
            raise ValueError(
                "run_claude_code requires either `workspace_path` (legacy) or "
                "`thread_id` (for provision_workspace.remote). Got neither."
            )
        workspace_path = provision_workspace.remote(
            thread_id=thread_id,
            repo_url=repo_url,
            ref=ref,
        )
        # The provisioning function committed the volume. Reload so
        # this container sees the newly-created worktree on its
        # mounted view of /vol — without this, the cwd we're about
        # to use doesn't exist from this container's perspective.
        volume.reload()

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
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name).strip(".") or "file"
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
        "stream-json",
        # --output-format stream-json requires --verbose in Claude Code;
        # without it, CC errors out before producing any events.
        "--verbose",
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

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace_path,
        env=env,
        bufsize=1,
    )

    # Belt-and-suspenders wall clock kill. Modal's function-level timeout
    # (300s) is the real ceiling; this matches the old subprocess.run
    # timeout=280 so we surface a clean ErrorEvent before Modal reaps us.
    killer = threading.Timer(280.0, proc.kill)
    killer.daemon = True
    killer.start()

    tool_names_by_id: dict[str, str] = {}
    accumulated_text: list[str] = []
    final_text = ""
    num_turns = 0
    duration_ms = 0

    # Drain stderr concurrently so the subprocess never blocks trying to write
    # to a full pipe while we're still reading stdout.  Without this, verbose
    # Claude Code output (~64 KB+ on stderr) can deadlock the generator until
    # the 280 s kill timer fires.
    stderr_lines: list[str] = []

    def _drain_stderr() -> None:
        if proc.stderr is not None:
            for _line in proc.stderr:
                stderr_lines.append(_line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("sandbox.stream_json.malformed", preview=line[:200])
                continue

            kind = parsed.get("type")

            if kind == "result":
                final_text = parsed.get("result") or "".join(accumulated_text)
                try:
                    num_turns = int(parsed.get("num_turns") or 0)
                except (TypeError, ValueError):
                    num_turns = 0
                try:
                    duration_ms = int(parsed.get("duration_ms") or 0)
                except (TypeError, ValueError):
                    duration_ms = 0
                continue

            if kind == "system":
                continue

            for event in _flatten_stream_event(parsed, tool_names_by_id):
                if event["type"] == "text":
                    accumulated_text.append(event["text"])
                yield event  # type: ignore[misc]

        returncode = proc.wait()
        stderr_thread.join()
        stderr_text = "".join(stderr_lines)
    finally:
        killer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        # Persist everything Claude Code wrote: rotated refresh tokens, new
        # session history files, any settings updates. Must be inside finally
        # so it runs even when the caller abandons the generator (GeneratorExit
        # is thrown at a yield), preventing loss of OAuth tokens and session
        # history on mid-stream cancellation.
        volume.commit()

    if returncode != 0:
        logger.warning(
            "sandbox.nonzero_exit",
            session_id=session_id,
            returncode=returncode,
            stderr=stderr_text[:500],
        )
        yield {
            "type": "error",
            "message": (f"Claude Code exited with code {returncode}: {stderr_text.strip()[:500]}"),
        }
        return

    yield {
        "type": "done",
        "final_text": final_text or "".join(accumulated_text),
        "num_turns": num_turns,
        "duration_ms": duration_ms,
    }

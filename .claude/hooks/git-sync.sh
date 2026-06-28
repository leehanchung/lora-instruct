#!/usr/bin/env bash
# Claude Code PreToolUse hook — keep the working branch synced with origin/main.
#
# Reads the hook JSON from stdin, pulls out the Bash command, and:
#   * `git switch -c` / `git checkout -b`  -> fast-forward local main to
#     origin/main first, so the new branch is cut from the latest main.
#   * `git push` (non-main, not a delete)  -> rebase the current branch onto
#     origin/main; if that rewrote already-pushed history, force-push (with
#     lease) here and block the original push so it can't fail non-fast-forward.
#
# Exits 0 (allow) in the normal case. Exits 2 (block) only after it has itself
# force-pushed a rebased branch, so the original push doesn't run a doomed
# non-fast-forward push.
set -u

input="$(cat)"

# Pull the command out of the hook payload (jq if present, else a crude fallback).
if command -v jq >/dev/null 2>&1; then
  cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // ""')"
else
  cmd="$(printf '%s' "$input" | sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
fi

# Cheap short-circuit: ignore commands that don't mention git at all. (The hook
# matches on the Bash tool, not a command prefix, so it also sees chained
# commands like `cd path && git push`.)
case "$cmd" in *git*) ;; *) exit 0 ;; esac

# Only act inside a git work tree.
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"

case "$cmd" in
  *"git switch -c"*|*"git checkout -b"*)
    # Refresh local main (fast-forward only) so the new branch starts from latest.
    if [ "$branch" = "main" ]; then
      if git fetch -q origin main 2>/dev/null; then
        git merge -q --ff-only origin/main 2>/dev/null || true
      fi
    else
      # ff-only ref update — refuses (no-op) rather than discarding local-only main commits.
      git fetch -q origin main:main 2>/dev/null || true
    fi
    ;;
  *"git push"*)
    # Never rebase on a remote-deletion push (`git push --delete`, `git push origin :branch`).
    case "$cmd" in *"--delete"*|*" :"*) exit 0 ;; esac
    [ "$branch" = "main" ] && exit 0
    git fetch -q origin main 2>/dev/null || exit 0
    before="$(git rev-parse HEAD 2>/dev/null)"
    if ! git rebase origin/main >/dev/null 2>&1; then
      git rebase --abort >/dev/null 2>&1 || true
      echo "git-sync: rebase of '$branch' onto origin/main hit conflicts — resolve manually, then re-push." >&2
      exit 0
    fi
    # If the rebase rewrote history AND the branch is already on the remote, a
    # plain push would be rejected. Force-push (with lease) here, then block the
    # original push so it can't fail.
    if [ "$before" != "$(git rev-parse HEAD 2>/dev/null)" ] \
       && git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
      if git push --force-with-lease >/dev/null 2>&1; then
        echo "git-sync: rebased '$branch' onto origin/main and force-pushed (with lease); original push skipped." >&2
        exit 2
      fi
      # Force-push failed: don't block — let the original push run and surface git's error.
      echo "git-sync: rebased '$branch' onto origin/main; force-push failed — re-push manually with --force-with-lease." >&2
      exit 0
    fi
    ;;
esac
exit 0

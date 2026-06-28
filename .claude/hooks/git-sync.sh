#!/usr/bin/env bash
# Claude Code PreToolUse hook — keep the working branch synced with origin/main.
#
# Reads the hook JSON from stdin, pulls out the Bash command, and:
#   * `git switch -c` / `git checkout -b`  -> fast-forward local main to
#     origin/main first, so the new branch is cut from the latest main.
#   * `git push` (when not on main)        -> rebase the current branch onto
#     origin/main so the PR isn't behind. (Re-push with --force-with-lease.)
#
# Side-effects only; always exits 0 so it never hard-blocks the tool call.
set -u

input="$(cat)"

# Pull the command out of the hook payload (jq if present, else a crude fallback).
if command -v jq >/dev/null 2>&1; then
  cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // ""')"
else
  cmd="$(printf '%s' "$input" | sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\(.*\)".*/\1/p')"
fi

# Only act inside a git work tree.
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"

case "$cmd" in
  *"git switch -c"*|*"git checkout -b"*)
    # Refresh local main so the about-to-be-created branch starts from latest.
    git fetch -q origin main 2>/dev/null || exit 0
    if [ "$branch" = "main" ]; then
      git merge -q --ff-only origin/main 2>/dev/null || true
    else
      git branch -f main origin/main 2>/dev/null || true
    fi
    ;;
  *"git push"*)
    # Rebase the feature branch onto the latest main before it's pushed.
    [ "$branch" = "main" ] && exit 0
    git fetch -q origin main 2>/dev/null || exit 0
    if ! git rebase origin/main >/dev/null 2>&1; then
      git rebase --abort >/dev/null 2>&1 || true
      echo "git-sync: rebase of '$branch' onto origin/main hit conflicts — resolve manually, then re-push with --force-with-lease." >&2
    fi
    ;;
esac
exit 0

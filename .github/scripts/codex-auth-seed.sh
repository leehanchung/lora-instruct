#!/usr/bin/env bash
#
# codex-auth-seed.sh — (re)seed the Codex CI auth from a local login.
#
# The scheduled "Codex Auth Refresh" workflow keeps auth.json alive once
# it is healthy, but it CANNOT revive a token whose refresh token has
# already been consumed/expired. Run this script for the one-time
# bootstrap, and again whenever the refresh workflow reports the token
# is dead. It pushes your freshly-minted local ~/.codex/auth.json to
# both durable stores the CI uses:
#   - the CODEX_AUTH_JSON GitHub secret  (bootstrap seed / fallback)
#   - /root/.codex-ci/auth.json on the droplet  (the live store)
#
# Prereqs on YOUR machine:
#   1. codex login        # interactive — produces ~/.codex/auth.json
#   2. gh auth login       # gh CLI authed with admin on the repo
#   3. SSH access to the droplet (same key/host as the bot deploy)
#
# IMPORTANT: use a ChatGPT account DEDICATED to CI. Signing into that
# same account elsewhere (local codex, VS Code extension, ChatGPT
# mobile) rotates the refresh token and breaks the CI copy.
#
# Usage:
#   DROPLET_HOST=1.2.3.4 ./.github/scripts/codex-auth-seed.sh
#
# Optional env:
#   AUTH_JSON          local auth.json path     (default ~/.codex/auth.json)
#   REPO               owner/repo               (default: inferred via gh)
#   DROPLET_USER       droplet ssh user         (default root)
#   DROPLET_AUTH_PATH  remote path              (default /root/.codex-ci/auth.json)
#   SKIP_SECRET=1      don't update the GitHub secret
#   SKIP_DROPLET=1     don't push to the droplet

set -euo pipefail

AUTH_JSON="${AUTH_JSON:-$HOME/.codex/auth.json}"
DROPLET_USER="${DROPLET_USER:-root}"
DROPLET_AUTH_PATH="${DROPLET_AUTH_PATH:-/root/.codex-ci/auth.json}"

# --- validate the local auth.json -----------------------------------
if [ ! -f "$AUTH_JSON" ]; then
  echo "error: $AUTH_JSON not found. Run 'codex login' first." >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required." >&2
  exit 1
fi
if ! jq -e '.tokens.refresh_token // empty' "$AUTH_JSON" >/dev/null 2>&1; then
  echo "error: $AUTH_JSON has no tokens.refresh_token." >&2
  echo "       Is this a ChatGPT account login (not an API-key auth.json)?" >&2
  exit 1
fi
LAST_REFRESH="$(jq -r '.last_refresh // "unknown"' "$AUTH_JSON")"
echo "Local auth.json looks valid (has refresh_token, last_refresh=${LAST_REFRESH})."

# --- push to the GitHub secret --------------------------------------
if [ "${SKIP_SECRET:-0}" != "1" ]; then
  if [ -z "${REPO:-}" ]; then
    REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)"
  fi
  if [ -z "${REPO:-}" ]; then
    echo "error: REPO not set and could not be inferred. Pass REPO=owner/repo." >&2
    exit 1
  fi
  gh secret set CODEX_AUTH_JSON --repo "$REPO" < "$AUTH_JSON"
  echo "✓ Set CODEX_AUTH_JSON secret on ${REPO}"
fi

# --- push to the droplet --------------------------------------------
if [ "${SKIP_DROPLET:-0}" != "1" ]; then
  if [ -z "${DROPLET_HOST:-}" ]; then
    echo "error: DROPLET_HOST not set. Pass DROPLET_HOST=<ip/host> (or SKIP_DROPLET=1)." >&2
    exit 1
  fi
  remote_dir="$(dirname "$DROPLET_AUTH_PATH")"
  # Paths are computed locally and expanded client-side on purpose.
  # shellcheck disable=SC2029
  ssh "${DROPLET_USER}@${DROPLET_HOST}" "mkdir -p '${remote_dir}' && chmod 700 '${remote_dir}'"
  scp -q "$AUTH_JSON" "${DROPLET_USER}@${DROPLET_HOST}:${DROPLET_AUTH_PATH}"
  # shellcheck disable=SC2029
  ssh "${DROPLET_USER}@${DROPLET_HOST}" "chmod 600 '${DROPLET_AUTH_PATH}'"
  echo "✓ Pushed auth.json to ${DROPLET_USER}@${DROPLET_HOST}:${DROPLET_AUTH_PATH}"
fi

echo "Done. The scheduled 'Codex Auth Refresh' workflow will keep it alive from here."

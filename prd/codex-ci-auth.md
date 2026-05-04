# Codex CLI auth in CI/CD via ChatGPT account

Plan to authenticate the Codex CLI inside this repo's GitHub Actions
workflows using a ChatGPT account session (`auth.json`) rather than an
`OPENAI_API_KEY`. The motivation is cost: usage on a ChatGPT
subscription is included in the plan, while API-key calls are metered
per-token.

This is a *plan*, not a spec of implemented behavior. Nothing in this
document has been executed yet.

## Context / problem

We want to run `codex` from CI (concrete first use case is TBD — see
**Q1** below). Two auth paths exist:

1. **API key** (`OPENAI_API_KEY`). Standard pattern, trivially seeded
   into `secrets.OPENAI_API_KEY`. Billed per-token against the API
   account.
2. **ChatGPT account session** (`auth.json` produced by `codex login`).
   Usage counts against the user's ChatGPT plan instead of API
   billing. The OpenAI docs describe this path as "use this guide
   only if you specifically need to run the workflow as your Codex
   account" — i.e. it's the non-default route, but it's the right one
   for cost reasons here.

This PRD covers path (2). API-key fallback is intentionally out of
scope.

## Goals / non-goals

**Goals**
- A reusable pattern for invoking `codex` from
  `.github/workflows/delulu-deploy.yaml` (or a sibling workflow)
  authenticated as the repo owner's ChatGPT account.
- Survive the ChatGPT session refresh cycle (~8-day token rotation,
  per OpenAI docs) without manual reseeding on every run.
- Operational hygiene: `auth.json` mode `600`, `$CODEX_HOME` mode
  `700`, never logged, never committed.
- Single source of truth for the auth file — no concurrent jobs racing
  on the same `auth.json`.

**Non-goals**
- Replacing the existing droplet/Modal deploy paths.
- Adding Codex usage to any specific feature. This PRD lands the
  *capability*; the first real use case ships in its own PR.
- Supporting both API-key and account auth in parallel. Pick one.
- Self-hosted runners. The plan stays on `ubuntu-latest`.

## Open questions (must resolve before implementation)

| Q | Question | Blocks | Default if unanswered |
|---|---|---|---|
| Q1 | What is the first concrete CI job that calls `codex`? (PR triage? Scheduled refactor? Discord-triggered job?) | All — shapes whether this is on push, schedule, or workflow_dispatch | workflow_dispatch only, as a smoke test |
| Q2 | Where do we persist the refreshed `auth.json` between ephemeral runs? | Approach — see "Persistence options" below | Option A (droplet stash) — infra already exists |
| Q3 | Do we accept the rate-limit coupling between CI and interactive ChatGPT use? | Risk acceptance | yes — flag in PRD, revisit if it bites |

## Approach

Per the OpenAI CI/CD auth guide, the lifecycle is:

1. Run `codex login` once on a trusted local machine. This writes
   `~/.codex/auth.json` (OAuth bundle).
2. Store that file as a GitHub repo secret (`CODEX_AUTH_JSON`,
   contents pasted verbatim — it's already JSON).
3. On each CI run: write the secret to `$CODEX_HOME/auth.json` with
   mode `600`, run `codex exec ...`, then **persist the refreshed
   file back** to whichever store we picked (Q2). Codex auto-refreshes
   tokens older than ~8 days and on `401` retries; if we don't
   persist the refresh, the seeded copy goes stale and one day stops
   working mid-run.
4. If refresh ever fails (manual reseed): rerun `codex login`
   locally, replace the stored copy.

### Workflow skeleton (sketch — not final)

```yaml
codex-job:
  runs-on: ubuntu-latest
  concurrency:
    group: codex-auth        # serialize: one auth.json at a time
    cancel-in-progress: false
  steps:
    - name: Restore auth.json
      env:
        CODEX_AUTH_JSON: ${{ secrets.CODEX_AUTH_JSON }}
      run: |
        export CODEX_HOME="$HOME/.codex"
        mkdir -p "$CODEX_HOME" && chmod 700 "$CODEX_HOME"
        printf '%s' "$CODEX_AUTH_JSON" > "$CODEX_HOME/auth.json"
        chmod 600 "$CODEX_HOME/auth.json"

    - name: Install codex
      run: npm i -g @openai/codex   # pin version in real impl

    - name: Run codex
      run: codex exec --json "..."

    - name: Persist refreshed auth.json
      if: always()
      run: |
        # Q2-dependent: see "Persistence options" below
        ...
```

## Persistence options (Q2)

Three candidates, ranked by cost-to-implement against this repo:

### Option A — Stash on the droplet (recommended default)

The `delulu-discord-deploy` job already SSHes into a DigitalOcean
droplet via `appleboy/ssh-action` with `secrets.DROPLET_SSH_KEY`.
That host has stable disk and is already trusted for deploy
credentials.

- **Restore:** `scp` (or inline cat over SSH) `auth.json` from
  `/root/.codex-ci/auth.json` on the droplet onto the runner.
- **Persist:** `scp` the refreshed file back after the run.
- **Bootstrap:** first-run seed comes from `secrets.CODEX_AUTH_JSON`
  if the droplet copy is missing.

Pros: no new dependency; same trust boundary as the existing CD path.
Cons: the droplet becomes the source of truth for an OAuth token; if
the droplet is rebuilt, reseed from the secret.

### Option B — External secret manager

1Password CLI, Doppler, AWS Secrets Manager, etc. Workflow reads on
entry, writes on exit.

Pros: cleanest separation; rotation/audit out of the box.
Cons: new infra and a new secret to manage. Overkill for a one-user
repo.

### Option C — Rewrite the GitHub secret via PAT

`gh secret set CODEX_AUTH_JSON --body @auth.json` from inside the run,
authed with a fine-scoped PAT (or a GitHub App) that has
`secrets: write`.

Pros: no external state; everything stays in GitHub.
Cons: requires a long-lived PAT with secret-write scope, which is a
notably higher-blast-radius credential than the OAuth bundle it's
protecting. Probably not worth it.

**Default if Q2 unanswered:** Option A.

## Operational rules (from the OpenAI docs)

- One `auth.json` per workflow stream — enforce with a
  `concurrency.group` so two Codex jobs can't race the same file.
- `auth.json` mode `600`; `$CODEX_HOME` mode `700`.
- Never `cat` `auth.json` to logs; never commit it; never include it
  in artifacts.
- On `401` failures or refresh errors, rerun `codex login` locally
  and overwrite the stored copy. Don't try to debug expired tokens
  from CI logs.

## Risks

- **Rate-limit coupling.** ChatGPT plans have usage limits. A noisy
  CI loop could degrade the owner's interactive ChatGPT experience or
  trip plan limits. Mitigation: gate the Codex job tightly (specific
  paths, `workflow_dispatch`, or scheduled with a low cadence) and
  monitor first uses before scaling up.
- **Token drift on ephemeral runners.** If persistence (Q2) silently
  fails, the stored `auth.json` ages out and the next run breaks at
  some unpredictable future date. Mitigation: the persist step is
  `if: always()` and any non-zero exit must fail the job loudly, not
  just warn.
- **Concurrency races.** Two simultaneous Codex jobs writing back
  different refreshed tokens corrupt each other. Mitigation:
  workflow-level `concurrency.group: codex-auth` with
  `cancel-in-progress: false` to serialize.
- **Secret leakage via shell expansion.** `printf '%s' "$VAR"` is
  safe; `echo $VAR` and herestring patterns can mangle JSON or leak
  via `set -x`. Mitigation: explicit `set +x` around auth handling
  and a review checklist when this lands.
- **TOS posture.** Using a ChatGPT account session for automated CI
  is what the OpenAI doc explicitly enables, so this is sanctioned —
  but it's worth re-reading the ChatGPT plan terms once before
  scaling beyond the first use case.

## Out of scope — explicit parks

- **API-key fallback path.** Not building both. If account auth
  proves operationally painful, that's a future PRD.
- **Multiple Codex identities.** One account, one `auth.json`.
- **Self-hosted runner pattern.** The OpenAI doc covers this; we
  don't need it.

## Effort estimate

- Bootstrap (run `codex login` locally, paste secret, write workflow
  job, pick Q2 option, smoke-test via `workflow_dispatch`): ~1
  evening.
- First real use case: scoped by Q1, separate PR.

## Sources / attributions

- **OpenAI Codex docs — "Authenticating Codex CLI in CI/CD
  pipelines":** <https://developers.openai.com/codex/auth/ci-cd-auth>.
  Source for: the `auth.json` / `CODEX_HOME` mechanism, the ~8-day
  refresh window, the `401`-triggered refresh-and-retry path, file
  mode requirements (`600` / `700`), the explicit guidance that this
  path is for running as your Codex account rather than the default
  API-key flow, and the "one auth.json per runner / serialized
  workflow stream" rule. Workflow snippets in this PRD are adapted
  from the ephemeral-runner pattern in that doc.
- **Repo-local context:** existing droplet SSH path
  (`appleboy/ssh-action` step in
  `.github/workflows/delulu-deploy.yaml`), which Option A reuses.

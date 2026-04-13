# Monorepo reorganization

Plan to reshape the repo root from its two-era layout (LoRA-Instruct
files scattered at root + tidy `apps/` tree for delulu) into a
category-first monorepo structure that scales to many projects.

This is a *plan*, not a spec of implemented behavior. Nothing in this
document has been executed yet.

## Context / problem

The root is two eras mashed together:

**LoRA-Instruct era** (still at repo root): `finetune.py`, `utils/`,
`templates/`, `dataset/`, `inference/`, `notebook/`, `poetry.lock`,
`pyproject.toml`, `tox.ini`.

**Delulu era** (already tidy under `apps/`): `apps/delulu_discord`,
`apps/delulu_sandbox_modal`.

**Unclassified**: `scripts/` (job scraping), `infra/managed-agents/`
(substantial — has its own pyproject, Dockerfile, deploy dir, and
harnesses; likely a third project, not a wrapper), `Research/` (empty),
`docs/`, `prd/`.

**Vestigial**: root `uv.lock` (52 bytes, just metadata header — orphan
from the dissolved `infra/discord-orchestrator/` wrapper), `tox.ini`
(py310/flake8 legacy; current toolchain is poetry + ruff for LoRA and uv
+ ruff for apps).

Three concrete pain points:

1. **The root README and `CLAUDE.md` one-liner still describe the repo
   as "LoRA-Instruct: LoRA fine-tuning".** The active work is the
   delulu Discord orchestrator and the intended future work is a
   category-first monorepo hosting many projects. The one-liner is a
   lie of omission.
2. **No home for growth.** Future LLM training projects, eval
   harnesses, shared libraries, and non-notebook research writeups
   have nowhere obvious to land. Each new project would have to invent
   its own layout.
3. **No agent guidance for multi-project work.** Claude Code (and
   humans) default to repo-root thinking, but most tasks are scoped to
   one project and should operate from that project's directory with
   its own deps, lockfile, and Makefile.

## Goals / non-goals

**Goals**
- A category-first layout at the root that has explicit homes for
  production services, training projects, shared data, shared evals,
  research writeups, shared infra, and shared libraries.
- Each project is self-contained (own deps, own lockfile, own Makefile,
  own `CLAUDE.md`) so it can be worked on from its own directory
  without repo-root tooling.
- Agent guidance baked into `CLAUDE.md` so future Claude Code sessions
  default to project-scoped work (cwd, search scope, PR boundaries).
- Every move happens as its own PR through the conventional-commits
  workflow — no big-bang reorg.
- No operational downtime for delulu during the migration.

**Non-goals**
- Rewriting any LoRA-Instruct code. Moves only.
- Introducing a uv workspace or poetry workspace. Each project stays
  independent.
- Adding cross-project shared code today. `libs/` gets a slot but
  starts empty.
- Extracting LoRA-Instruct or delulu into separate repos. The whole
  point is a monorepo.

## Target layout

```
apps/              # production services
  delulu/            # grouped by app name, not deployable
    discord/           ← apps/delulu_discord
    sandbox_modal/     ← apps/delulu_sandbox_modal
  <future-app>/

projects/          # research/training projects — each self-contained
  lora_instruct/
    training/        ← finetune.py + utils/
    data/            ← dataset/ + templates/
    inference/       ← inference/
    notebooks/       ← notebook/
    evals/             # project-specific benchmarks
    pyproject.toml     # project-owned deps
    CLAUDE.md
    README.md
  <future-project>/

data/              # cross-project datasets + scrapers only
  scrapers/          ← scripts/job_postings (if general-purpose)
  shared/              # datasets reused by 2+ projects

evals/             # cross-project eval harnesses + benchmarks
  harnesses/
  runs/                # output artifacts

research/          # long-form writeups, notes, exploratory scripts
  <topic>/             # one dir per investigation
                       # minimal notebook usage — markdown + small scripts

infra/             # shared infra (TBD: managed-agents classification)

libs/              # cross-project Python packages — empty slot, ready

docs/ prd/ .github/ CLAUDE.md README.md ARCHITECTURE.md
```

### Rule of thumb: `apps/` vs `projects/`

- **`apps/`** = runs as a production service with users hitting it.
  Deployables.
- **`projects/`** = research/training work that produces *artifacts*
  (models, datasets, reports). Not services.

## Agent working rules

Land in `CLAUDE.md` as a "Working in the monorepo" section before the
structural moves, so Claude Code sessions during the migration already
know the target shape. Key rules:

- Identify the project before touching anything.
- Work from the project root, not the repo root — `cd` in or
  `make -C <path>`.
- Default `Grep`/`Glob` to the project dir; broaden only for
  genuinely cross-cutting work.
- No imports between projects. Shared code goes in `libs/` via a
  separate PR.
- One PR per project. Cross-cutting files (`.github/`, root docs,
  `prd/`, `.pre-commit-config.yaml`) can ship as their own PR.
- Per-project `CLAUDE.md` wins over root for project-specific
  commands and conventions.

## Open questions (blockers for specific waves)

| Q | Question | Blocks | Default if unanswered |
|---|---|---|---|
| Q1 | What is `infra/managed-agents/`? App, infra, or project? | Wave 5 | stays in `infra/` |
| Q2 | Is `scripts/job_postings` LoRA-owned or general? | Wave 4 | folded into Wave 3 as LoRA data |
| Q3 | Is LoRA-Instruct active or archived? | Wave 3 | active → `projects/lora_instruct/` |
| Q4 | Does LoRA get its own `pyproject.toml`? | Wave 3 | yes (matches `apps/` pattern) |
| Q5 | Regroup `apps/delulu_*` into `apps/delulu/{discord,sandbox_modal}/`? | Wave 6 | yes (scales better; can skip) |

## Wave-by-wave plan

Each wave ships as its own PR (or small set). Waves are ordered by
dependency and risk — cheapest and safest first, operational risk last.

### Wave 0 — land what's already in flight

Not new work; gates everything downstream.

- **PR #20** (`ci: enforce conventional-commit PR titles`) merges.
- **PR #21** (`docs: update PRDs and add git workflow to CLAUDE.md`) merges.
- After #20 runs cleanly on a real PR, add `PR title / lint` to the
  required status checks on `main` via `gh api`.

### Wave 1 — cheap cleanups (no structural moves)

All safe, all reversible, no path dependencies. Two PRs:

- **PR** — `chore: remove vestigial root files`
  - Delete root `uv.lock` (orphan from dissolved wrapper).
  - Delete `tox.ini` (py310/flake8 legacy).
  - Delete empty `Research/` directory.
  - *No effect on any project.*

- **PR** — `docs: update CLAUDE.md for monorepo identity`
  - Replace the "LoRA-Instruct: LoRA fine-tuning..." one-liner with a
    monorepo description.
  - *Standalone, doesn't depend on any moves.*

### Wave 2 — declare the target layout

- **PR** — `docs: add monorepo working rules to CLAUDE.md`
  - Add the "Working in the monorepo" section to root `CLAUDE.md`.
  - Include a "*reorg in progress — not all directories exist yet*"
    note, removed in Wave 7.
  - No file moves, no stub directories.

Lands before structural moves so the agent rules are authoritative
during the migration window.

### Wave 3 — LoRA-Instruct into `projects/`

**Blocked on Q3, Q4.** Assuming active + own pyproject:

- **PR** — `refactor: move LoRA-Instruct under projects/lora_instruct`
  - `git mv finetune.py projects/lora_instruct/training/finetune.py`
  - `git mv utils/ projects/lora_instruct/training/utils/`
  - `git mv templates/ projects/lora_instruct/data/templates/`
  - `git mv dataset/ projects/lora_instruct/data/datasets/`
  - `git mv inference/ projects/lora_instruct/inference/`
  - `git mv notebook/ projects/lora_instruct/notebooks/`
  - Move root `pyproject.toml` + `poetry.lock` →
    `projects/lora_instruct/`.
  - Add `projects/lora_instruct/CLAUDE.md` (project-scoped Quick
    Reference).
  - Add `projects/lora_instruct/README.md` (split from root README's
    LoRA section).
  - Update imports: `from utils.X` → rooted in the new package layout.
    Simplest approach: make `projects/lora_instruct/` the package root
    with `src/lora_instruct/` inside, or declare `training/` as a
    package in `pyproject.toml`.
  - Update `.pre-commit-config.yaml` if it references any moved paths.
  - Delete any LoRA-related targets from the root `Makefile`.
  - **Verification:** from `projects/lora_instruct/`, `poetry install`
    + `poetry run ruff check .` pass. `python training/finetune.py
    --help` prints.

*High file-count, low runtime risk — LoRA isn't deployed from this
repo.*

### Wave 4 — data scrapers

**Blocked on Q2.**

If LoRA-owned → fold into Wave 3 and skip this wave. If general:

- **PR** — `refactor: move job scrapers under data/scrapers`
  - `git mv scripts/job_postings/ data/scrapers/job_postings/`
  - `git mv scripts/scrape_ai_trainer_jobs.py data/scrapers/`
  - `git mv scripts/job_data.py data/scrapers/`
  - If `scripts/` is now empty, delete it.

### Wave 5 — `infra/managed-agents/` classification

**Blocked on Q1.** Three branches depending on the answer:

- **If app:** `refactor: move managed-agents to apps/` — add its own
  `CLAUDE.md`, verify deploy wiring still works.
- **If infra:** `refactor: tidy infra/managed_agents structure` —
  likely just rename the hyphen to underscore for import-friendliness
  and stop; it stays put.
- **If project:** `refactor: move managed-agents to projects/` — add
  its own `CLAUDE.md`.

Investigate the directory's Dockerfile, deploy scripts, and harnesses
before committing to a branch. The Dockerfile + deploy dir imply it's a
real deployable, leaning app.

### Wave 6 — `apps/delulu/` grouping layer

**Blocked on Q5 (optional — skip if no).**

Higher-risk PR: touches workflows, Dockerfile contexts, PRDs, pre-commit
config. This is the only wave with real operational risk — deploys are
driven by path filters.

- **PR** — `refactor: group delulu apps under apps/delulu/`
  - `git mv apps/delulu_discord apps/delulu/discord`
  - `git mv apps/delulu_sandbox_modal apps/delulu/sandbox_modal`
  - Update `.github/workflows/discord-orchestrator-deploy.yml` path
    filters (`apps/delulu_*/**` → `apps/delulu/**`).
  - Update `.github/workflows/claude-review.yml` prompt — it mentions
    `apps/delulu_discord` and `apps/delulu_sandbox_modal` in two places.
  - Update the droplet deploy path in README / workflow
    (`/root/SMILE-factory/apps/delulu_discord` →
    `/root/SMILE-factory/apps/delulu/discord`).
  - Update `prd/streaming.md` and `prd/repo-provisioning.md` path
    references.
  - Update `.pre-commit-config.yaml` if it has `apps/delulu_*` globs.
  - Update the top-level dispatching `Makefile`.

- **Verification:** open a trivial PR that touches
  `apps/delulu/discord/README.md` and confirm
  `discord-orchestrator deploy` picks up the change via the new path
  filter before merging the reorg PR itself.

### Wave 7 — per-project CLAUDE.md + root cleanup

- **PR** — `docs: add per-project CLAUDE.md files`
  - Add `apps/delulu/discord/CLAUDE.md` and
    `apps/delulu/sandbox_modal/CLAUDE.md` (project-scoped commands and
    gotchas).
  - Tighten root `CLAUDE.md` Quick Reference to point at per-project
    docs instead of listing LoRA commands.
  - Remove the "*reorg in progress*" note from the Wave 2 section.
  - Update `ARCHITECTURE.md` if it still describes the old layout.

## Summary table

| Wave | PRs | Blocked on | Operational risk |
|---|---|---|---|
| 0 | #20, #21 merge | — | none |
| 1 | 2 | — | none |
| 2 | 1 | — | none |
| 3 | 1 | Q3, Q4 | none |
| 4 | 1 | Q2 | none |
| 5 | 1 | Q1 | low–medium |
| 6 | 1 | Q5 (optional) | medium (deploys) |
| 7 | 1 | waves 3–6 done | none |

**Total: ~8 PRs** once Q1–Q5 are answered. Waves 1 and 2 can start
immediately — they don't depend on any open question.

## Risks and mitigations

- **Deploy breakage in Wave 6.** Path filters in
  `discord-orchestrator-deploy.yml` drive bot redeploys. If the filter
  doesn't match the new paths, either (a) no deploy fires on bot
  changes and the droplet drifts, or (b) every unrelated change fires
  a deploy. Mitigation: test the filter on a trivial PR on a branch
  before merging the reorg PR. Verification step is listed above.
- **Import rewrites in Wave 3.** LoRA code has `from utils.X` and
  `from templates.X` imports that assume root-level packages. Moving
  them under `projects/lora_instruct/` requires either declaring a
  package root in the new `pyproject.toml` or rewriting imports.
  Mitigation: do it as a package-root move (one `pyproject.toml`
  change, one `__init__.py` if missing), not a sweeping import
  rewrite.
- **Stranded references in PRDs and docs.** `prd/streaming.md`,
  `prd/repo-provisioning.md`, `ARCHITECTURE.md`, and `README.md` all
  reference `apps/delulu_discord` paths. Each affected wave's PR must
  grep for and update these references in the same commit.
- **`.pre-commit-config.yaml` path patterns.** Must update in lockstep
  with any move that changes which directories `ruff` runs against.
  Mitigation: run `pre-commit run --all-files` on each reorg PR.
- **`poetry.lock` vs `uv.lock`.** LoRA uses poetry, apps use uv. The
  reorg doesn't unify these and shouldn't — each project's choice is
  independent. Just make sure the move keeps each project's toolchain
  intact.

## Out of scope — explicit parks

- **Second training project.** Structure is ready for it; no
  speculative scaffolding.
- **Cross-project shared eval harness.** `evals/harnesses/` starts
  empty; first real shared harness drives the first commit.
- **`libs/` population.** Empty slot now; first cross-project shared
  module drives it.
- **Monorepo-wide CI aggregation.** Each project keeps its own CI
  paths. Revisit if test runs become painful to coordinate.
- **Workspace tooling (uv or poetry workspaces).** Considered for the
  app-separation refactor and rejected; same rejection applies here
  until there's real shared runtime code.

## Effort estimate

- Wave 1: ~30 min (two trivial PRs).
- Wave 2: ~30 min (one doc PR).
- Wave 3: 2–3 hours (file moves, import rewrites, pyproject split,
  verification).
- Wave 4: ~30 min (if standalone).
- Wave 5: 1–2 hours (depends on branch taken after investigation).
- Wave 6: 2–3 hours (high touch count across workflows, Dockerfiles,
  PRDs, Makefiles; plus dry-run verification).
- Wave 7: 1 hour (per-project CLAUDE.md files + root tidy).

Call it a full day of focused work spread across several sessions, so
each PR can run through CI and get eyeballed independently.

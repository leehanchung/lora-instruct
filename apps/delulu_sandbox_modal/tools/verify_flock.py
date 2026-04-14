"""Flock semantics scout for Modal Volumes.

Verifies that ``fcntl.flock(LOCK_EX)`` provides cross-container
mutual exclusion when the lock file lives on a Modal Volume. This is
a load-bearing assumption of the ``repo_provisioner.py`` concurrency
design in ``prd/repo-provisioning.md``: if flock is a no-op on the
volume, the whole locking strategy has to switch to ``os.link``-based
atomic rename locks before any of the provisioning code is written.

Run with:

    modal run apps/delulu_sandbox_modal/tools/verify_flock.py::verify

The scout spawns ``NUM_WORKERS`` function invocations concurrently,
each of which tries to acquire an exclusive lock on the same file on
the shared volume. Each worker records its wall-clock enter and exit
times around a ``HOLD_SECONDS`` critical section, and the entrypoint
collects them and checks two signals:

1.  **Non-overlapping enter/exit windows.** If flock works, the
    earliest worker's exit should precede the next worker's enter
    for every adjacent pair (modulo a small tolerance for inter-
    container clock skew).
2.  **Total wall clock.** If flock works, all workers serialized, so
    total wall clock should be at least
    ``NUM_WORKERS * HOLD_SECONDS`` (plus container spin-up). If
    flock is a no-op and workers ran in parallel, total wall clock
    would be ~``HOLD_SECONDS`` regardless of ``NUM_WORKERS``.

Both signals must agree for PASS. If either fails, report FAIL and
the repo provisioner has to use ``os.link``-based locks instead.

Shares the production ``claude-workspaces`` volume on purpose — we
want to test flock semantics against the exact filesystem the repo
provisioner will use, not a synthetic test volume. State is written
to ``/vol/.flock-scout/`` and cleaned up at the start and end of
each run, so production data (``/vol/workspaces/*``,
``/vol/repo-cache/*``) is untouched.
"""

from __future__ import annotations

import fcntl
import os
import time

import modal

app = modal.App("flock-scout")

# Reuse the production volume so the test exercises the real backend
# the repo provisioner will run against. The scout writes to a
# dedicated hidden directory and cleans up after itself.
volume = modal.Volume.from_name("claude-workspaces", create_if_missing=True)

VOLUME_ROOT = "/vol"
SCOUT_DIR = f"{VOLUME_ROOT}/.flock-scout"
LOCK_PATH = f"{SCOUT_DIR}/test.lock"

# Three workers is enough to distinguish serialized from parallel
# behavior without wasting Modal minutes on a longer run. Increase
# locally if the result looks ambiguous.
NUM_WORKERS = 3

# Each worker holds the lock for this long. Long enough to exceed
# any reasonable inter-container clock skew (typically sub-second on
# well-synced hosts), short enough that the whole scout finishes in
# under a minute.
HOLD_SECONDS = 2.0

# Slack we allow when checking "worker B entered after worker A
# exited". Clock skew between Modal containers is small but nonzero.
TOLERANCE_SECONDS = 0.2

scout_image = (
    # Python 3.12, not 3.14. The production sandbox uses 3.14, which
    # requires Modal Image Builder 2025.06+ (set workspace-wide via
    # `modal config set image_builder_version 2025.06`). The scout
    # deliberately avoids that dependency so it runs out-of-the-box
    # on a fresh Modal config: its job is to test kernel-level flock
    # semantics on a Modal Volume, which is identical regardless of
    # which Python version the userspace process is running on.
    # Only stdlib (`fcntl`, `os`, `time`, `shutil`) is used here.
    modal.Image.debian_slim(python_version="3.12")
)


@app.function(image=scout_image, volumes={VOLUME_ROOT: volume}, timeout=120)
def clear_state() -> None:
    """Remove any leftover scout state from a previous run.

    Commits the volume so subsequent worker containers see the
    cleared state at mount time — otherwise a fresh worker might
    mount the volume with stale data from a prior invocation still
    present.
    """
    import shutil

    if os.path.isdir(SCOUT_DIR):
        shutil.rmtree(SCOUT_DIR)
    os.makedirs(SCOUT_DIR, exist_ok=True)
    volume.commit()


@app.function(image=scout_image, volumes={VOLUME_ROOT: volume}, timeout=120)
def lock_worker(worker_id: int) -> dict[str, float | int]:
    """Acquire LOCK_EX on the shared file, hold HOLD_SECONDS, release.

    Returns wall-clock timings for the critical section so the
    entrypoint can check whether workers actually serialized.
    """
    os.makedirs(SCOUT_DIR, exist_ok=True)
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        wait_start = time.time()
        fcntl.flock(fd, fcntl.LOCK_EX)
        enter_time = time.time()
        wait_seconds = enter_time - wait_start

        # Simulate work inside the critical section.
        time.sleep(HOLD_SECONDS)

        exit_time = time.time()
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

    return {
        "worker_id": worker_id,
        "wait_seconds": wait_seconds,
        "enter_time": enter_time,
        "exit_time": exit_time,
    }


@app.local_entrypoint()
def verify() -> None:
    """Run the flock scout end-to-end and print PASS/FAIL."""
    print(f"▶  Clearing scout state at {SCOUT_DIR}...")
    clear_state.remote()

    print(f"▶  Spawning {NUM_WORKERS} concurrent workers, each holding the")
    print(f"   lock for {HOLD_SECONDS}s...")
    wall_start = time.time()

    # `.map()` dispatches all workers concurrently and returns their
    # results in input order. This is the cleanest way to start a
    # race test on Modal — every worker is in flight before any of
    # them returns.
    results = list(lock_worker.map(range(NUM_WORKERS)))
    total_wall = time.time() - wall_start

    # Sort by enter_time so we can check adjacency.
    results.sort(key=lambda r: r["enter_time"])

    earliest_enter = min(r["enter_time"] for r in results)

    print()
    print(f"   Total wall clock: {total_wall:.2f}s")
    print()
    print(f"   {'worker':<10}{'wait':>10}{'enter (rel)':>16}{'exit (rel)':>16}")
    for r in results:
        enter_rel = r["enter_time"] - earliest_enter
        exit_rel = r["exit_time"] - earliest_enter
        print(
            f"   {r['worker_id']:<10}"
            f"{r['wait_seconds']:>9.2f}s"
            f"{enter_rel:>15.2f}s"
            f"{exit_rel:>15.2f}s"
        )
    print()

    # Signal 1: adjacent workers must not overlap.
    serialized = True
    overlaps: list[str] = []
    for i in range(1, len(results)):
        prev_exit = results[i - 1]["exit_time"]
        curr_enter = results[i]["enter_time"]
        if curr_enter + TOLERANCE_SECONDS < prev_exit:
            serialized = False
            overlap = prev_exit - curr_enter
            overlaps.append(
                f"worker {results[i]['worker_id']} entered "
                f"{overlap:.2f}s before worker {results[i - 1]['worker_id']} exited"
            )

    # Signal 2: total wall clock should be at least NUM_WORKERS * HOLD
    # (minus a small slack for inter-worker acquire overhead).
    expected_min_wall = NUM_WORKERS * HOLD_SECONDS - 0.5
    wall_clock_ok = total_wall >= expected_min_wall

    print("─" * 60)
    if serialized and wall_clock_ok:
        print("✅ PASS — flock provides cross-container mutual exclusion")
        print("          on the Modal Volume. Proceed with fcntl.flock in")
        print("          repo_provisioner.py.")
    elif not serialized:
        print("❌ FAIL — workers overlapped inside the critical section.")
        print("          flock is NOT honored on the Modal Volume backend.")
        for line in overlaps:
            print(f"          • {line}")
        print()
        print("          Fall back to os.link-based atomic rename locks in")
        print("          repo_provisioner.py and update the PRD concurrency")
        print("          section to reflect the fallback strategy.")
    else:
        # Serialized by enter/exit order, but total wall clock too
        # short — could be coincidental ordering of a race. Flag as
        # suspicious rather than pass so we re-run.
        print("⚠️  SUSPICIOUS — enter/exit windows look serialized but total")
        print(f"          wall clock ({total_wall:.2f}s) is below the expected")
        print(f"          minimum ({expected_min_wall:.2f}s). This can happen")
        print("          if flock is a no-op and three workers happened to")
        print("          finish in tidy order by chance. Re-run with more")
        print("          workers or longer HOLD_SECONDS to disambiguate.")

    print()
    print("▶  Cleaning up scout state...")
    clear_state.remote()

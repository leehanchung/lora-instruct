"""Constraint-based bin-packing scheduler for heterogeneous workers.

Scheduling pipeline:
  1. FILTER  — hard constraints (labels, resource floors, max sessions)
  2. SCORE   — rank eligible workers by fit
  3. SELECT  — pick from top candidates based on strategy

This is the same filter→score→select pipeline K8s uses internally,
but operating on bubblewrap processes instead of pods.

Scoring dimensions (weighted):
  - Resource fit: penalize waste (bin-pack) or reward spread
  - Label preference: soft preferences for worker labels
  - Affinity: prefer workers running sessions with matching tags
  - Anti-affinity: penalize workers running sessions with conflicting tags
  - Freshness: slight preference for workers with recent heartbeats
"""

import logging
import math
import random
from dataclasses import dataclass, field

from .base import ResourceRequest, SchedulingStrategy, WorkerInfo

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """Debugging aid: see why a worker was scored the way it was."""

    worker_id: str
    eligible: bool = True
    rejection_reason: str = ""
    resource_score: float = 0.0
    label_score: float = 0.0
    affinity_score: float = 0.0
    anti_affinity_score: float = 0.0
    total_score: float = 0.0


@dataclass
class SchedulerConfig:
    """Tuning knobs for the scheduler."""

    strategy: SchedulingStrategy = SchedulingStrategy.BIN_PACK

    # Score weights (must sum to ~1.0 for interpretability, but not enforced)
    weight_resource_fit: float = 0.4
    weight_label_preference: float = 0.2
    weight_affinity: float = 0.2
    weight_anti_affinity: float = 0.2

    # How many top candidates to randomly pick from (avoids thundering herd)
    top_k: int = 3

    # Stale worker threshold: ignore workers whose last heartbeat is older
    # than this many seconds ago (0 = disabled)
    stale_heartbeat_seconds: float = 60.0


class Scheduler:
    """Stateless scheduler: takes workers + request, returns best worker."""

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    def select(
        self,
        workers: list[WorkerInfo],
        request: ResourceRequest,
        session_tags: list[str] | None = None,
    ) -> tuple[WorkerInfo, list[ScoreBreakdown]]:
        """Pick the best worker for a session.

        Args:
            workers: All known workers (healthy and unhealthy).
            request: Resource requirements and constraints.
            session_tags: Tags for the session being scheduled.

        Returns:
            Tuple of (selected worker, score breakdowns for all workers).

        Raises:
            RuntimeError: If no eligible worker exists.
        """
        session_tags = session_tags or []
        breakdowns: list[ScoreBreakdown] = []

        # Phase 1: FILTER
        eligible: list[tuple[WorkerInfo, ScoreBreakdown]] = []
        for w in workers:
            bd = ScoreBreakdown(worker_id=w.worker_id)

            reason = self._check_hard_constraints(w, request)
            if reason:
                bd.eligible = False
                bd.rejection_reason = reason
                breakdowns.append(bd)
                continue

            eligible.append((w, bd))

        if not eligible:
            reasons = {b.rejection_reason for b in breakdowns if b.rejection_reason}
            raise RuntimeError(
                f"No eligible workers for request "
                f"(cpu={request.cpu_millicores}m, mem={request.memory_mb}Mi, "
                f"gpu={request.gpu_count}, labels={request.required_labels}). "
                f"Rejection reasons: {reasons}"
            )

        # Phase 2: SCORE
        for w, bd in eligible:
            bd.resource_score = self._score_resource_fit(w, request)
            bd.label_score = self._score_label_preference(w, request)
            bd.affinity_score = self._score_affinity(w, request, session_tags)
            bd.anti_affinity_score = self._score_anti_affinity(w, request, session_tags)

            bd.total_score = (
                self.config.weight_resource_fit * bd.resource_score
                + self.config.weight_label_preference * bd.label_score
                + self.config.weight_affinity * bd.affinity_score
                + self.config.weight_anti_affinity * bd.anti_affinity_score
            )
            breakdowns.append(bd)

        # Phase 3: SELECT
        eligible.sort(key=lambda pair: pair[1].total_score, reverse=True)
        top_k = eligible[: min(self.config.top_k, len(eligible))]

        if self.config.strategy == SchedulingStrategy.RANDOM:
            selected = random.choice(top_k)[0]
        else:
            # For bin-pack and spread, scoring already reflects the strategy,
            # so just pick randomly from top-k to break ties
            selected = random.choice(top_k)[0]

        logger.info(
            f"Scheduled on {selected.worker_id} "
            f"(score={next(b.total_score for b in breakdowns if b.worker_id == selected.worker_id):.3f}, "
            f"eligible={len(eligible)}/{len(workers)})"
        )

        return selected, breakdowns

    # ------------------------------------------------------------------
    # Phase 1: Hard constraint checks
    # ------------------------------------------------------------------

    def _check_hard_constraints(
        self, worker: WorkerInfo, request: ResourceRequest
    ) -> str:
        """Return empty string if eligible, otherwise rejection reason."""
        if not worker.healthy:
            return "unhealthy"

        cap = worker.capabilities

        if worker.active_sessions >= worker.max_sessions:
            return f"at max sessions ({worker.max_sessions})"

        if cap.available_cpu_millicores < request.cpu_millicores:
            return (
                f"insufficient CPU "
                f"({cap.available_cpu_millicores}m < {request.cpu_millicores}m)"
            )

        if cap.available_memory_mb < request.memory_mb:
            return (
                f"insufficient memory "
                f"({cap.available_memory_mb}Mi < {request.memory_mb}Mi)"
            )

        if cap.available_gpu_count < request.gpu_count:
            return (
                f"insufficient GPU "
                f"({cap.available_gpu_count} < {request.gpu_count})"
            )

        if cap.available_disk_mb < request.disk_mb:
            return (
                f"insufficient disk "
                f"({cap.available_disk_mb}Mi < {request.disk_mb}Mi)"
            )

        # Required labels must all match
        for key, value in request.required_labels.items():
            worker_value = cap.labels.get(key)
            if worker_value != value:
                return (
                    f"missing required label {key}={value} "
                    f"(has {key}={worker_value})"
                )

        return ""

    # ------------------------------------------------------------------
    # Phase 2: Scoring functions (each returns 0.0 to 1.0)
    # ------------------------------------------------------------------

    def _score_resource_fit(
        self, worker: WorkerInfo, request: ResourceRequest
    ) -> float:
        """Score based on how well resources fit.

        BIN_PACK: prefer workers where the request fills remaining capacity
                  tightly (minimize waste → maximize space on other workers).
        SPREAD:   prefer workers with the most headroom (even distribution).
        """
        cap = worker.capabilities

        # Utilization ratios AFTER placing this session
        cpu_used_after = (
            (cap.total_cpu_millicores - cap.available_cpu_millicores + request.cpu_millicores)
            / max(cap.total_cpu_millicores, 1)
        )
        mem_used_after = (
            (cap.total_memory_mb - cap.available_memory_mb + request.memory_mb)
            / max(cap.total_memory_mb, 1)
        )

        avg_utilization = (cpu_used_after + mem_used_after) / 2.0

        if self.config.strategy == SchedulingStrategy.BIN_PACK:
            # Higher utilization = better score (pack tightly)
            # But penalize if we'd exceed 90% (leave headroom for spikes)
            if avg_utilization > 0.95:
                return 0.3  # Functional but too tight
            return min(avg_utilization, 1.0)

        elif self.config.strategy == SchedulingStrategy.SPREAD:
            # Lower utilization = better score (spread out)
            return max(1.0 - avg_utilization, 0.0)

        else:
            # Random: resource fit doesn't matter
            return 0.5

    def _score_label_preference(
        self, worker: WorkerInfo, request: ResourceRequest
    ) -> float:
        """Score based on soft label preferences (0-1)."""
        if not request.preferred_labels:
            return 0.5  # Neutral when no preferences

        matches = sum(
            1
            for key, value in request.preferred_labels.items()
            if worker.capabilities.labels.get(key) == value
        )

        return matches / len(request.preferred_labels)

    def _score_affinity(
        self,
        worker: WorkerInfo,
        request: ResourceRequest,
        session_tags: list[str],
    ) -> float:
        """Score based on co-location with sessions sharing affinity tags.

        Use case: sessions using the same model benefit from shared cache
        on the same worker.
        """
        if not request.affinity_tags:
            return 0.5  # Neutral

        active_tags = set(worker.capabilities.active_session_tags)

        matches = sum(1 for tag in request.affinity_tags if tag in active_tags)
        return matches / len(request.affinity_tags)

    def _score_anti_affinity(
        self,
        worker: WorkerInfo,
        request: ResourceRequest,
        session_tags: list[str],
    ) -> float:
        """Score based on avoiding co-location with conflicting sessions.

        Use case: don't put two sessions from the same user on the same
        worker (blast radius isolation).
        """
        if not request.anti_affinity_tags:
            return 0.5  # Neutral

        active_tags = set(worker.capabilities.active_session_tags)

        conflicts = sum(
            1 for tag in request.anti_affinity_tags if tag in active_tags
        )

        if conflicts == 0:
            return 1.0  # No conflicts: perfect
        # Each conflict reduces score
        return max(0.0, 1.0 - (conflicts / len(request.anti_affinity_tags)))


__all__ = ["Scheduler", "SchedulerConfig", "ScoreBreakdown"]

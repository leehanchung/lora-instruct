"""Reward registry — flat, per-task scorers dispatched by a `data_source` field.

This is the single scoring surface used by BOTH:
  * RL training (training/rl_deepresearch/plugins/reward.py shims to score()), and
  * offline eval (eval/harness/score.py calls score()).

Add a new task type = add one scorer module + register it here. (Pattern from
Search-R1's reward_score/ and DR-Tulu's search_rewards/.)
"""

from __future__ import annotations

from collections.abc import Callable

from dr_agent.rewards import em, f1
from dr_agent.rewards.schemas import RewardResult, Row

# Maps a row's `data_source` to a scorer fn: (Row) -> RewardResult.
_REGISTRY: dict[str, Callable[[Row], RewardResult]] = {
    "exact_match": em.score,
    "f1": f1.score,
    # "rubric": rubric.score,        # TODO
    # "citation": citation.score,    # TODO
}


def register(data_source: str, fn: Callable[[Row], RewardResult]) -> None:
    """Register a scorer for a `data_source` tag."""
    _REGISTRY[data_source] = fn


def score(row: Row) -> RewardResult:
    """Score a single row by dispatching on its `data_source`."""
    try:
        scorer = _REGISTRY[row.data_source]
    except KeyError as exc:
        raise KeyError(
            f"no reward scorer registered for data_source={row.data_source!r}; "
            f"known: {sorted(_REGISTRY)}"
        ) from exc
    return scorer(row)


__all__ = ["register", "score", "Row", "RewardResult"]

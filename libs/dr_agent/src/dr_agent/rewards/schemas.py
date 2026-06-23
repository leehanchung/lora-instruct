"""Shared row + result schemas for scoring.

One schema used by data-gen output, RL rollouts, and eval — so a row produced by
data/datagen can be scored unchanged in both training and eval.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Row(BaseModel):
    """A single (task, prediction, ground-truth) record to be scored.

    `data_source` selects the scorer in the reward registry.
    """

    data_source: str
    prediction: str
    ground_truth: str | list[str]
    # Free-form extras: rubric trees, supporting items, citations, etc.
    extra: dict = Field(default_factory=dict)


class RewardResult(BaseModel):
    """The score for one row."""

    score: float
    # Optional breakdown (e.g. {"format": 0.1, "f1": 0.7}) for analysis.
    components: dict[str, float] = Field(default_factory=dict)

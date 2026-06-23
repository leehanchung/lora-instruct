"""Stage base classes + the shared Task schema.

Each pipeline stage is a class with a single `run` method. A domain subclasses
the ones it needs and overrides domain specifics (prompts, sources). Stages
accumulate fields onto Task objects so partial pipelines are debuggable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A generated deep-research task with ground truth baked in.

    Mirrors the dr_agent scoring schema so generated rows score unchanged in both
    training and eval.
    """

    id: str
    level: int = 0  # multi-hop difficulty (extend stage increments)
    question: str
    truth: str
    data_source: str = "f1"  # which dr_agent reward scorer applies
    supporting_items: list[dict] = Field(default_factory=list)
    distractors: list[dict] = Field(default_factory=list)
    passed_verification: bool | None = None


class _Stage(ABC):
    """A pipeline stage. Reads tasks, returns (mutated/new) tasks."""

    @abstractmethod
    async def run(self, tasks: list[Task]) -> list[Task]: ...


class BaseExplorer(_Stage):
    """Stage 1: generate candidate tasks from seeds."""


class BaseVerifier(_Stage):
    """Stage 2/4: validate quotes / filter unsupported tasks or distractors."""


class BaseDistractor(_Stage):
    """Stage 3: mine hard distractor passages."""


class BaseExtender(_Stage):
    """Stage 5: chain single-hop tasks into multi-hop ones (level += 1)."""

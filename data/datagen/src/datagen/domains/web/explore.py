"""Stage 1 (web): generate candidate tasks from seed topics."""

from __future__ import annotations

from datagen.core import BaseExplorer, Task
from datagen.domains.web import prompts


class WebExplorer(BaseExplorer):
    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model

    async def run(self, tasks: list[Task]) -> list[Task]:
        # TODO: for each seed, call the model with prompts.EXPLORE to draft a
        # question + truth + supporting_items grounded in real pages.
        _ = prompts.EXPLORE
        raise NotImplementedError

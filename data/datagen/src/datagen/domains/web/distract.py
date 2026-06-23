"""Stage 3 (web): mine hard distractor passages for each task."""

from __future__ import annotations

from datagen.core import BaseDistractor, Task


class WebDistractor(BaseDistractor):
    async def run(self, tasks: list[Task]) -> list[Task]:
        # TODO: retrieve near-miss passages; attach to task.distractors.
        raise NotImplementedError

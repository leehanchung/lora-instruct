"""Stage 2/4 (web): verify that supporting quotes actually entail the truth."""

from __future__ import annotations

from datagen.core import BaseVerifier, Task


class WebVerifier(BaseVerifier):
    async def run(self, tasks: list[Task]) -> list[Task]:
        # TODO: check each task's supporting_items; set passed_verification.
        raise NotImplementedError

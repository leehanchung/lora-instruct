"""Stage 5 (web): chain single-hop tasks into multi-hop tasks (level += 1)."""

from __future__ import annotations

from datagen.core import BaseExtender, Task


class WebExtender(BaseExtender):
    async def run(self, tasks: list[Task]) -> list[Task]:
        # TODO: compose two verified tasks into one multi-hop task at level+1.
        raise NotImplementedError

"""Tool abstraction. Concrete tools (search, visit, scholar, python) subclass Tool.

A Tool is a thin client: it serializes args, calls the tool HTTP service, and
returns an observation string. No tool does retrieval in-process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx


class Tool(ABC):
    """Base class for an agent tool."""

    name: str
    description: str

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url.rstrip("/")

    @abstractmethod
    async def call(self, client: httpx.AsyncClient, **kwargs) -> str:
        """Invoke the tool and return an observation string for the agent."""
        ...


class ToolSet:
    """A named collection of tools an agent may use."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

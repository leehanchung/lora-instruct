"""Typed event shapes emitted by ``run_claude_code``.

The sandbox yields these across the Modal boundary to the bot. Since
``delulu_discord`` does not import from ``delulu_sandbox_modal`` (the two
apps are independent packages), the bot currently receives them as plain
dicts — these TypedDicts exist so the sandbox-side code has a clear
schema to write against and to make future promotion to a shared
``delulu_shared`` package mechanical.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class ToolUseEvent(TypedDict):
    type: Literal["tool_use"]
    tool: str
    summary: str


class ToolResultEvent(TypedDict):
    type: Literal["tool_result"]
    tool: str
    ok: bool
    summary: str


class ThinkingEvent(TypedDict):
    type: Literal["thinking"]
    text: str


class AssistantTextEvent(TypedDict):
    type: Literal["text"]
    text: str


class DoneEvent(TypedDict):
    type: Literal["done"]
    final_text: str
    num_turns: int
    duration_ms: int


class ErrorEvent(TypedDict):
    type: Literal["error"]
    message: str


Event = ToolUseEvent | ToolResultEvent | ThinkingEvent | AssistantTextEvent | DoneEvent | ErrorEvent

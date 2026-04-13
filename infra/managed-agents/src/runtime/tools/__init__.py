"""Tool registry and base classes for agent runtime."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution."""

    output: str
    error: Optional[str] = None
    exit_code: int = 0


class Tool:
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute the tool.

        Args:
            input: Input parameters

        Returns:
            ToolResult with output and optional error
        """
        raise NotImplementedError


class ToolRegistry:
    """Registry for tools available to agent."""

    def __init__(self):
        """Initialize empty tool registry."""
        self.tools: dict[str, type[Tool]] = {}
        self.instances: dict[str, Tool] = {}

    def register(self, tool_cls: type[Tool]) -> None:
        """Register a tool class.

        Args:
            tool_cls: Tool class to register (must have name attribute)
        """
        if not hasattr(tool_cls, "name") or not tool_cls.name:
            raise ValueError(f"Tool {tool_cls.__name__} must have a name attribute")

        self.tools[tool_cls.name] = tool_cls
        logger.debug(f"Registered tool: {tool_cls.name}")

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> ToolResult:
        """Execute a registered tool.

        Args:
            tool_name: Name of tool to execute
            tool_input: Input parameters

        Returns:
            ToolResult

        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self.tools:
            return ToolResult(
                output="",
                error=f"Tool not found: {tool_name}",
                exit_code=1,
            )

        try:
            if tool_name not in self.instances:
                self.instances[tool_name] = self.tools[tool_name]()

            tool = self.instances[tool_name]
            return await tool.execute(tool_input)

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return ToolResult(
                output="",
                error=str(e),
                exit_code=1,
            )

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions for LLM.

        Returns:
            List of tool schema definitions
        """
        definitions = []
        for tool_cls in self.tools.values():
            definitions.append(
                {
                    "name": tool_cls.name,
                    "description": tool_cls.description,
                    "input_schema": tool_cls.input_schema,
                }
            )
        return definitions

    def register_builtin_tools(self) -> None:
        """Register all built-in tools."""
        from .bash import BashTool
        from .file_ops import EditTool, GlobTool, GrepTool, ReadTool, WriteTool
        from .web import WebFetchTool, WebSearchTool

        self.register(BashTool)
        self.register(ReadTool)
        self.register(WriteTool)
        self.register(EditTool)
        self.register(GlobTool)
        self.register(GrepTool)
        self.register(WebFetchTool)
        self.register(WebSearchTool)

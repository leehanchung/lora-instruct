"""File operation tools with sandboxing."""

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from . import Tool, ToolResult

logger = logging.getLogger(__name__)

# Default workspace directory
DEFAULT_WORKSPACE = "/workspace"


class FileSandbox:
    """Enforce path sandboxing for file operations."""

    def __init__(self, workspace_dir: str = DEFAULT_WORKSPACE):
        """Initialize sandbox.

        Args:
            workspace_dir: Base directory for operations (default /workspace)
        """
        self.workspace = Path(workspace_dir).resolve()
        # Ensure workspace exists
        self.workspace.mkdir(parents=True, exist_ok=True)
        logger.info(f"File sandbox initialized at {self.workspace}")

    def safe_path(self, requested_path: str) -> Path:
        """Resolve a path safely within sandbox.

        Args:
            requested_path: Requested file path (absolute or relative)

        Returns:
            Safe resolved path

        Raises:
            ValueError: If path escapes sandbox
        """
        if not requested_path:
            raise ValueError("Path cannot be empty")

        # Handle absolute paths
        if os.path.isabs(requested_path):
            path = Path(requested_path).resolve()
        else:
            # Relative paths are relative to workspace
            path = (self.workspace / requested_path).resolve()

        # Prevent directory traversal attacks
        try:
            path.relative_to(self.workspace)
        except ValueError:
            raise ValueError(f"Path {requested_path} escapes sandbox {self.workspace}")

        return path


class ReadTool(Tool):
    """Read file contents."""

    name = "read_file"
    description = "Read the contents of a file."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to read",
            },
            "offset": {
                "type": "integer",
                "description": "Start line number (1-indexed, optional)",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read (optional)",
            },
        },
        "required": ["path"],
    }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Read file contents."""
        path_str = input.get("path", "").strip()
        offset = input.get("offset", 1)
        limit = input.get("limit", None)

        sandbox = FileSandbox()
        try:
            path = sandbox.safe_path(path_str)

            if not path.exists():
                return ToolResult(output="", error=f"File not found: {path_str}", exit_code=1)

            if path.is_dir():
                return ToolResult(output="", error=f"Is a directory: {path_str}", exit_code=1)

            # Read file
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines(keepends=False)

            # Apply offset and limit
            start = max(0, offset - 1)
            if limit:
                end = start + limit
            else:
                end = len(lines)

            result_lines = lines[start:end]
            output = "\n".join(result_lines)

            logger.debug(f"Read {len(result_lines)} lines from {path}")
            return ToolResult(output=output)

        except ValueError as e:
            return ToolResult(output="", error=str(e), exit_code=1)
        except Exception as e:
            logger.error(f"Error reading file: {e}", exc_info=True)
            return ToolResult(output="", error=str(e), exit_code=1)


class WriteTool(Tool):
    """Write file contents."""

    name = "write_file"
    description = "Write contents to a file. Creates file if it doesn't exist."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write",
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Write file contents."""
        path_str = input.get("path", "").strip()
        content = input.get("content", "")

        sandbox = FileSandbox()
        try:
            path = sandbox.safe_path(path_str)

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            path.write_text(content, encoding="utf-8")

            logger.debug(f"Wrote {len(content)} bytes to {path}")
            return ToolResult(output=f"Written to {path_str}")

        except ValueError as e:
            return ToolResult(output="", error=str(e), exit_code=1)
        except Exception as e:
            logger.error(f"Error writing file: {e}", exc_info=True)
            return ToolResult(output="", error=str(e), exit_code=1)


class EditTool(Tool):
    """Find and replace in files."""

    name = "edit_file"
    description = "Find and replace text in a file."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Text to find (exact match)",
            },
            "new_string": {
                "type": "string",
                "description": "Text to replace with",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Find and replace in file."""
        path_str = input.get("path", "").strip()
        old_string = input.get("old_string", "")
        new_string = input.get("new_string", "")

        sandbox = FileSandbox()
        try:
            path = sandbox.safe_path(path_str)

            if not path.exists():
                return ToolResult(output="", error=f"File not found: {path_str}", exit_code=1)

            content = path.read_text(encoding="utf-8", errors="replace")

            # Check if old_string exists
            if old_string not in content:
                return ToolResult(
                    output="",
                    error=f"String not found in file: {repr(old_string[:100])}",
                    exit_code=1,
                )

            # Replace
            new_content = content.replace(old_string, new_string)
            path.write_text(new_content, encoding="utf-8")

            logger.debug(f"Replaced text in {path}")
            return ToolResult(output=f"Replaced in {path_str}")

        except ValueError as e:
            return ToolResult(output="", error=str(e), exit_code=1)
        except Exception as e:
            logger.error(f"Error editing file: {e}", exc_info=True)
            return ToolResult(output="", error=str(e), exit_code=1)


class GlobTool(Tool):
    """File glob pattern matching."""

    name = "glob_files"
    description = "Find files matching a glob pattern."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., '*.py', 'src/**/*.js')",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Glob for files."""
        pattern = input.get("pattern", "").strip()

        sandbox = FileSandbox()
        try:
            # Match patterns from workspace
            matches = sorted(sandbox.workspace.glob(pattern))

            # Convert to relative paths
            result_lines = [str(p.relative_to(sandbox.workspace)) for p in matches if p.is_file()]

            if not result_lines:
                return ToolResult(output="", error=f"No files matching: {pattern}", exit_code=1)

            output = "\n".join(result_lines)
            logger.debug(f"Glob found {len(result_lines)} files matching {pattern}")
            return ToolResult(output=output)

        except Exception as e:
            logger.error(f"Error globbing files: {e}", exc_info=True)
            return ToolResult(output="", error=str(e), exit_code=1)


class GrepTool(Tool):
    """Regex search across files."""

    name = "grep_files"
    description = "Search for regex patterns across files."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory path (. for current workspace)",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Search files with grep."""
        pattern = input.get("pattern", "").strip()
        path_str = input.get("path", ".")

        sandbox = FileSandbox()
        try:
            search_path = sandbox.safe_path(path_str)

            if not search_path.exists():
                return ToolResult(
                    output="", error=f"Path not found: {path_str}", exit_code=1
                )

            # Use grep subprocess for efficiency
            grep_args = ["grep", "-r", "-n", "--include=*"]

            if search_path.is_file():
                # Single file search
                cmd = f"grep -n {re.escape(pattern)} {str(search_path)}"
            else:
                # Directory search
                cmd = f"grep -r -n {re.escape(pattern)} {str(search_path)}"

            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await process.communicate()
            output = stdout.decode("utf-8", errors="replace").strip()

            if process.returncode == 1 and not output:
                return ToolResult(output="", error=f"Pattern not found: {pattern}", exit_code=1)

            logger.debug(f"Grep found matches for {pattern}")
            return ToolResult(output=output, exit_code=0)

        except ValueError as e:
            return ToolResult(output="", error=str(e), exit_code=1)
        except Exception as e:
            logger.error(f"Error grepping files: {e}", exc_info=True)
            return ToolResult(output="", error=str(e), exit_code=1)

"""Bash/shell command execution tool."""

import asyncio
import logging
from typing import Any

from . import Tool, ToolResult

logger = logging.getLogger(__name__)


class BashTool(Tool):
    """Execute shell commands."""

    name = "bash"
    description = "Execute bash shell commands. Use for running scripts, shell utilities, and system commands."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (min 1, max 600, default 120)",
                "default": 120,
            },
        },
        "required": ["command"],
    }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute a bash command.

        Args:
            input: Dict with 'command' and optional 'timeout'

        Returns:
            ToolResult with stdout + stderr and exit code
        """
        command = input.get("command", "").strip()
        if not command:
            return ToolResult(output="", error="No command provided", exit_code=1)

        timeout = input.get("timeout", 120)
        # Enforce limits
        timeout = max(1, min(600, int(timeout)))

        logger.info(f"Executing bash command: {command[:100]}")

        try:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    output="",
                    error=f"Command timed out after {timeout}s",
                    exit_code=124,
                )

            # Decode output
            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()

            # Combine output
            output = stdout_str
            if stderr_str:
                if output:
                    output += "\n" + stderr_str
                else:
                    output = stderr_str

            exit_code = process.returncode

            logger.debug(f"Bash command completed with exit code {exit_code}")
            return ToolResult(output=output, exit_code=exit_code)

        except Exception as e:
            logger.error(f"Error executing bash command: {e}", exc_info=True)
            return ToolResult(output="", error=str(e), exit_code=1)

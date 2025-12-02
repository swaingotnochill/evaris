"""Code executor tool for LLM Judge.

This tool allows the LLM Judge to execute code to verify agent outputs.
Execution is sandboxed using subprocess with resource limits.

WARNING: Code execution carries security risks. Use with caution and
consider running in isolated environments (Docker, etc.).
"""

import asyncio
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional

from evaris.metrics.llm_judge.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class CodeExecutorTool(BaseTool):
    """Execute code to verify agent outputs.

    Supports Python and JavaScript execution with configurable timeouts
    and resource limits.

    Example:
        >>> executor = CodeExecutorTool(timeout=5.0)
        >>> result = executor.execute(
        ...     code="print(2 + 2)",
        ...     language="python"
        ... )
        >>> print(result.output)  # "4"
    """

    name = "run_code"
    description = (
        "Execute code to verify the correctness of an output. "
        "Useful for testing mathematical calculations, data transformations, "
        "or any logic that can be verified programmatically. "
        "Returns the stdout output or error message."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The code to execute",
            },
            "language": {
                "type": "string",
                "enum": ["python", "javascript", "bash"],
                "default": "python",
                "description": "Programming language (python, javascript, or bash)",
            },
            "expected_output": {
                "type": "string",
                "description": "Optional expected output to compare against",
            },
        },
        "required": ["code"],
    }

    def __init__(
        self,
        timeout: float = 10.0,
        max_output_size: int = 10000,
        allowed_languages: Optional[list[str]] = None,
    ):
        """Initialize the code executor.

        Args:
            timeout: Maximum execution time in seconds
            max_output_size: Maximum output size in characters
            allowed_languages: Languages to allow (default: all)
        """
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allowed_languages = allowed_languages or ["python", "javascript", "bash"]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute code and return the result.

        Args:
            **kwargs: Must include 'code'. Optional: 'language' (default: python),
                     'expected_output' for comparison.

        Returns:
            ToolResult with execution output
        """
        code: str = kwargs.get("code", "")
        language: str = kwargs.get("language", "python")
        expected_output: Optional[str] = kwargs.get("expected_output")

        if not code:
            return ToolResult(success=False, error="'code' parameter is required")

        if language not in self.allowed_languages:
            return ToolResult(
                success=False,
                error=f"Language '{language}' not allowed. Allowed: {self.allowed_languages}",
            )

        try:
            if language == "python":
                result = self._execute_python(code)
            elif language == "javascript":
                result = self._execute_javascript(code)
            elif language == "bash":
                result = self._execute_bash(code)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported language: {language}",
                )

            # Check expected output if provided
            if expected_output is not None:
                matches = result.output.strip() == expected_output.strip()
                result.metadata["expected_output"] = expected_output
                result.metadata["output_matches"] = matches
                if not matches:
                    result.metadata["comparison"] = (
                        f"Expected: {expected_output!r}\n" f"Got: {result.output.strip()!r}"
                    )

            return result

        except Exception as e:
            logger.exception(f"Code execution failed for {language}")
            return ToolResult(
                success=False,
                error=f"Execution error: {str(e)}",
                metadata={"language": language, "exception_type": type(e).__name__},
            )

    def _execute_python(self, code: str) -> ToolResult:
        """Execute Python code."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = Path(f.name)

        try:
            result = subprocess.run(
                [sys.executable, str(temp_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout
            if result.returncode != 0:
                error_msg = result.stderr or f"Exit code: {result.returncode}"
                return ToolResult(
                    success=False,
                    output=output[: self.max_output_size] if output else None,
                    error=error_msg[: self.max_output_size],
                    metadata={"exit_code": result.returncode, "language": "python"},
                )

            return ToolResult(
                success=True,
                output=output[: self.max_output_size],
                metadata={"exit_code": 0, "language": "python"},
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Execution timed out after {self.timeout}s",
                metadata={"language": "python"},
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def _execute_javascript(self, code: str) -> ToolResult:
        """Execute JavaScript code using Node.js."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = Path(f.name)

        try:
            result = subprocess.run(
                ["node", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout
            if result.returncode != 0:
                error_msg = result.stderr or f"Exit code: {result.returncode}"
                return ToolResult(
                    success=False,
                    output=output[: self.max_output_size] if output else None,
                    error=error_msg[: self.max_output_size],
                    metadata={"exit_code": result.returncode, "language": "javascript"},
                )

            return ToolResult(
                success=True,
                output=output[: self.max_output_size],
                metadata={"exit_code": 0, "language": "javascript"},
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Node.js not found. Install Node.js to execute JavaScript.",
                metadata={"language": "javascript"},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Execution timed out after {self.timeout}s",
                metadata={"language": "javascript"},
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def _execute_bash(self, code: str) -> ToolResult:
        """Execute Bash code."""
        try:
            result = subprocess.run(
                ["bash", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout
            if result.returncode != 0:
                error_msg = result.stderr or f"Exit code: {result.returncode}"
                return ToolResult(
                    success=False,
                    output=output[: self.max_output_size] if output else None,
                    error=error_msg[: self.max_output_size],
                    metadata={"exit_code": result.returncode, "language": "bash"},
                )

            return ToolResult(
                success=True,
                output=output[: self.max_output_size],
                metadata={"exit_code": 0, "language": "bash"},
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Execution timed out after {self.timeout}s",
                metadata={"language": "bash"},
            )

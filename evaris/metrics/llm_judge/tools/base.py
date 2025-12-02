"""Base tool class and orchestrator for LLM Judge.

This module provides:
- BaseTool: Abstract base class for all tools
- ToolResult: Result from tool execution
- ToolOrchestrator: Manages tool calls and execution loops
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Result from a tool execution.

    Attributes:
        success: Whether the tool executed successfully
        output: The tool's output (string, dict, or any serializable type)
        error: Error message if execution failed
        metadata: Additional tool-specific data
    """

    success: bool = Field(..., description="Whether the tool executed successfully")
    output: Any = Field(None, description="The tool's output")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional tool-specific data"
    )

    def to_string(self) -> str:
        """Convert result to string for LLM consumption."""
        if self.success:
            if isinstance(self.output, str):
                return self.output
            return json.dumps(self.output, indent=2, default=str)
        return f"Error: {self.error}"


class BaseTool(ABC):
    """Abstract base class for LLM Judge tools.

    Tools extend the LLM Judge's capabilities by allowing it to
    execute code, search the web, read files, or perform other
    actions to verify agent outputs.

    Subclasses must implement:
    - name: Unique identifier for the tool
    - description: Human-readable description for the LLM
    - parameters_schema: JSON schema for tool parameters
    - execute(): Synchronous execution method

    Example:
        >>> class MyTool(BaseTool):
        ...     name = "my_tool"
        ...     description = "Does something useful"
        ...     parameters_schema = {
        ...         "type": "object",
        ...         "properties": {"input": {"type": "string"}},
        ...         "required": ["input"]
        ...     }
        ...
        ...     def execute(self, input: str) -> ToolResult:
        ...         return ToolResult(success=True, output=f"Processed: {input}")
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters matching parameters_schema

        Returns:
            ToolResult: The result of tool execution
        """
        pass

    async def a_execute(self, **kwargs: Any) -> ToolResult:
        """Asynchronously execute the tool.

        Default implementation wraps sync execute() in a thread pool.
        Override for truly async implementations.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult: The result of tool execution
        """
        return await asyncio.to_thread(self.execute, **kwargs)

    def to_function_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI-compatible function schema.

        Returns:
            dict: Function schema for LLM function calling
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert tool to Anthropic-compatible tool schema.

        Returns:
            dict: Tool schema for Anthropic's tool use
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema,
        }

    def validate_parameters(self, **kwargs: Any) -> bool:
        """Validate parameters against schema.

        Args:
            **kwargs: Parameters to validate

        Returns:
            bool: True if valid

        Note:
            Basic validation only. Override for complex validation.
        """
        required = self.parameters_schema.get("required", [])
        for param in required:
            if param not in kwargs:
                return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class ToolOrchestrator:
    """Manages tool execution for LLM Judge.

    The orchestrator handles:
    - Tool registration and lookup
    - Parsing tool calls from LLM responses
    - Executing tools and formatting results
    - Managing execution loops with limits

    Example:
        >>> orchestrator = ToolOrchestrator(
        ...     tools=[CodeExecutorTool(), WebSearchTool()],
        ...     max_iterations=5
        ... )
        >>> result = await orchestrator.execute_tool("run_code", code="print('hello')")
    """

    def __init__(
        self,
        tools: list[BaseTool],
        max_iterations: int = 5,
        timeout_seconds: float = 30.0,
    ):
        """Initialize the orchestrator.

        Args:
            tools: List of tools to make available
            max_iterations: Maximum number of tool calls per evaluation
            timeout_seconds: Timeout for individual tool executions
        """
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self._call_count = 0

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """List available tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_function_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible function schemas for all tools.

        Returns:
            List of function schemas
        """
        return [tool.to_function_schema() for tool in self.tools.values()]

    def get_anthropic_schemas(self) -> list[dict[str, Any]]:
        """Get Anthropic-compatible tool schemas for all tools.

        Returns:
            List of tool schemas
        """
        return [tool.to_anthropic_schema() for tool in self.tools.values()]

    def execute_tool(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool synchronously.

        Args:
            name: Tool name
            **kwargs: Tool parameters

        Returns:
            ToolResult: The result of tool execution
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found. Available: {self.list_tools()}",
            )

        if self._call_count >= self.max_iterations:
            return ToolResult(
                success=False,
                error=f"Maximum tool calls ({self.max_iterations}) exceeded",
            )

        self._call_count += 1

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            logger.exception(f"Tool '{name}' execution failed")
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
                metadata={"exception_type": type(e).__name__},
            )

    async def a_execute_tool(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool asynchronously.

        Args:
            name: Tool name
            **kwargs: Tool parameters

        Returns:
            ToolResult: The result of tool execution
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found. Available: {self.list_tools()}",
            )

        if self._call_count >= self.max_iterations:
            return ToolResult(
                success=False,
                error=f"Maximum tool calls ({self.max_iterations}) exceeded",
            )

        self._call_count += 1

        try:
            return await asyncio.wait_for(
                tool.a_execute(**kwargs),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Tool execution timed out after {self.timeout_seconds}s",
            )
        except Exception as e:
            logger.exception(f"Tool '{name}' execution failed")
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
                metadata={"exception_type": type(e).__name__},
            )

    def parse_tool_calls(
        self, response: Union[dict[str, Any], str], provider: str = "openai"
    ) -> list[dict[str, Any]]:
        """Parse tool calls from LLM response.

        Args:
            response: LLM response (dict or string)
            provider: LLM provider ("openai", "anthropic")

        Returns:
            List of tool call dicts with 'name' and 'arguments' keys
        """
        tool_calls = []

        if isinstance(response, str):
            # Try to parse as JSON
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return []

        if provider == "openai":
            # OpenAI format: response.choices[0].message.tool_calls
            if isinstance(response, dict):
                message = response.get("choices", [{}])[0].get("message", {})
                for call in message.get("tool_calls", []):
                    func = call.get("function", {})
                    tool_calls.append(
                        {
                            "id": call.get("id"),
                            "name": func.get("name"),
                            "arguments": json.loads(func.get("arguments", "{}")),
                        }
                    )

        elif provider == "anthropic":
            # Anthropic format: response.content with type="tool_use"
            if isinstance(response, dict):
                for block in response.get("content", []):
                    if block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id"),
                                "name": block.get("name"),
                                "arguments": block.get("input", {}),
                            }
                        )

        return tool_calls

    def format_tool_results(
        self, results: list[tuple[str, ToolResult]], provider: str = "openai"
    ) -> Union[list[dict[str, Any]], str]:
        """Format tool results for LLM consumption.

        Args:
            results: List of (tool_call_id, ToolResult) tuples
            provider: LLM provider

        Returns:
            Formatted results for the provider
        """
        if provider == "openai":
            return [
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result.to_string(),
                }
                for call_id, result in results
            ]

        elif provider == "anthropic":
            return [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": result.to_string(),
                }
                for call_id, result in results
            ]

        # Default: concatenate as string
        return "\n\n".join(f"[{call_id}] {result.to_string()}" for call_id, result in results)

    def reset(self) -> None:
        """Reset the call counter for a new evaluation."""
        self._call_count = 0

    @property
    def calls_remaining(self) -> int:
        """Get the number of tool calls remaining."""
        return max(0, self.max_iterations - self._call_count)

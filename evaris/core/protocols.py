"""Protocol definitions for Evaris evaluation framework.

This module defines the abstract interfaces (protocols) that components
must implement to work with Evaris:
- BaseMetric: Interface for evaluation metrics
- BaseTool: Interface for LLM Judge tools
- AgentProtocol: Interface for agents (structural typing)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.types import MetricResult, TestCase


class ToolResult(BaseModel):
    """Result from a tool execution.

    Tools used by LLM Judge return this structure to communicate
    their results back to the judge.

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


class BaseTool(ABC):
    """Abstract base class for LLM Judge tools.

    Tools extend the LLM Judge's capabilities by allowing it to
    execute code, search the web, read files, or perform other
    actions to verify agent outputs.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description for the LLM
        parameters_schema: JSON schema for tool parameters

    Example:
        >>> class CodeExecutorTool(BaseTool):
        ...     name = "run_code"
        ...     description = "Execute Python code to verify output"
        ...     parameters_schema = {
        ...         "type": "object",
        ...         "properties": {
        ...             "code": {"type": "string"},
        ...             "language": {"type": "string", "default": "python"}
        ...         },
        ...         "required": ["code"]
        ...     }
        ...
        ...     def execute(self, code: str, language: str = "python") -> ToolResult:
        ...         # Implementation
        ...         pass
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
        import asyncio

        return await asyncio.to_thread(self.execute, **kwargs)

    def to_function_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI-compatible function schema.

        Returns:
            dict: Function schema for LLM function calling
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics.

    Metrics evaluate agent outputs and produce scores between 0 and 1.
    All metrics must implement the async a_measure() method.

    Metrics can optionally:
    - Implement measure() for synchronous evaluation
    - Provide reasoning_steps for transparency
    - Support tool calling via tools attribute

    Attributes:
        name: Unique identifier for the metric (auto-set from class name)
        threshold: Score threshold for pass/fail (default 0.5)

    Example:
        >>> class MyMetric(BaseMetric):
        ...     threshold = 0.8
        ...
        ...     async def a_measure(self, test_case, actual_output):
        ...         score = compute_score(test_case, actual_output)
        ...         return MetricResult(
        ...             name=self.__class__.__name__,
        ...             score=score,
        ...             passed=score >= self.threshold
        ...         )
    """

    threshold: float = 0.5

    @property
    def name(self) -> str:
        """Get the metric name (defaults to class name)."""
        return self.__class__.__name__

    @abstractmethod
    async def a_measure(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Asynchronously evaluate a test case.

        This is the primary evaluation method. All metrics must implement this.

        Args:
            test_case: The test case to evaluate (contains input, expected)
            actual_output: The actual output from the agent

        Returns:
            MetricResult: The evaluation result with score and pass/fail status
        """
        pass

    def measure(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Synchronously evaluate a test case.

        Default implementation wraps a_measure() using asyncio.run().
        Override with a native sync implementation for better performance.

        WARNING: This method uses asyncio.run() which creates a new event loop.
        - Performance overhead when called repeatedly
        - Cannot be called from async context (use a_measure() instead)

        Args:
            test_case: The test case to evaluate
            actual_output: The actual output from the agent

        Returns:
            MetricResult: The evaluation result

        Raises:
            RuntimeError: If called from an async context
        """
        import asyncio

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "measure() cannot be called from async context. "
                "Use a_measure() instead, or implement a native sync measure() method."
            )
        except RuntimeError as e:
            if "no running event loop" not in str(e):
                raise

        return asyncio.run(self.a_measure(test_case, actual_output))

    def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Legacy synchronous evaluation method.

        This is kept for backward compatibility. New code should use
        measure() or a_measure() instead.

        Args:
            test_case: The test case to evaluate
            actual_output: The actual output from the agent

        Returns:
            MetricResult: The evaluation result

        Raises:
            NotImplementedError: If metric doesn't implement legacy interface
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement the legacy score() method. "
            "Use measure() or a_measure() instead."
        )

    def validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        """Validate inputs before evaluation.

        Override to add custom validation logic. Raises ValueError
        if inputs are invalid.

        Args:
            test_case: The test case to validate
            actual_output: The actual output to validate

        Raises:
            ValueError: If inputs are invalid
        """
        pass

    def __repr__(self) -> str:
        """String representation of the metric."""
        return f"{self.__class__.__name__}(threshold={self.threshold})"

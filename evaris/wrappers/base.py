"""Base wrapper class for agent integration.

Provides common functionality for all agent wrappers:
- Tracing integration
- Input/output normalization
- Error handling
- Cost tracking (optional)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from evaris.tracing import EvarisTracer
from evaris.types import MultiModalInput, MultiModalOutput


class BaseAgentWrapper(ABC):
    """Abstract base class for agent wrappers.

    Provides common functionality for wrapping agents from different frameworks:
    - Automatic tracing of agent execution
    - Input/output normalization
    - Error handling and recording
    - Optional cost tracking

    Subclasses must implement execute() and a_execute() methods.

    Example:
        >>> class MyAgentWrapper(BaseAgentWrapper):
        ...     def execute(self, input: MultiModalInput) -> MultiModalOutput:
        ...         with self.tracer.start_span("my_agent_execute"):
        ...             normalized_input = self._normalize_input(input)
        ...             output = self.agent.run(normalized_input)
        ...             return self._normalize_output(output)
        ...
        ...     async def a_execute(self, input: MultiModalInput) -> MultiModalOutput:
        ...         # Similar async implementation
        ...         pass
    """

    def __init__(
        self,
        agent: Any,
        tracer: Optional[EvarisTracer] = None,
        trace_llm_calls: bool = False,
        trace_tool_calls: bool = False,
        track_cost: bool = False,
    ):
        """Initialize base wrapper.

        Args:
            agent: The underlying agent to wrap
            tracer: Optional tracer for capturing execution (defaults to global tracer)
            trace_llm_calls: Whether to trace LLM API calls (default: False)
            trace_tool_calls: Whether to trace tool executions (default: False)
            track_cost: Whether to track token costs (default: False)
        """
        self.agent = agent
        self.tracer = tracer or EvarisTracer()
        self.trace_llm_calls = trace_llm_calls
        self.trace_tool_calls = trace_tool_calls
        self.track_cost = track_cost

    @abstractmethod
    def execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute agent synchronously.

        Args:
            input: Input to agent

        Returns:
            Output from agent

        Raises:
            NotImplementedError: If agent doesn't support sync execution
        """
        ...

    @abstractmethod
    async def a_execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute agent asynchronously.

        Args:
            input: Input to agent

        Returns:
            Output from agent

        Raises:
            NotImplementedError: If agent doesn't support async execution
        """
        ...

    def reset(self) -> None:
        """Reset agent state between test cases.

        Default implementation does nothing. Override in subclass if agent
        maintains state (e.g., conversation memory).
        """
        pass

    def _normalize_input(self, input: MultiModalInput) -> Any:
        """Normalize evaris input to agent-specific format.

        Default implementation:
        - str → str (pass through)
        - dict → dict (pass through)
        - list → list (pass through)
        - Path → str (convert to string)
        - other → str (convert to string)

        Override in subclass for agent-specific input format.

        Args:
            input: Input in evaris format

        Returns:
            Input in agent-specific format
        """
        if isinstance(input, (str, dict, list)):
            return input
        else:
            # Convert Path and other types to string
            return str(input)

    def _normalize_output(self, output: Any) -> MultiModalOutput:
        """Normalize agent output to evaris format.

        Default implementation:
        - str → str (pass through)
        - dict → dict (pass through)
        - list → list (pass through)
        - other → str (convert to string)

        Override in subclass for agent-specific output parsing.

        Args:
            output: Output from agent

        Returns:
            Output in evaris format
        """
        if isinstance(output, (str, dict, list)):
            return output
        else:
            # Convert other types to string
            return str(output)

    def _extract_cost(self, response: Any) -> Optional[float]:
        """Extract cost from agent response if available.

        Override in subclass to extract cost information from agent responses.

        Args:
            response: Agent response

        Returns:
            Cost in USD or None
        """
        return None

    def _extract_tokens(self, response: Any) -> Optional[int]:
        """Extract token count from agent response if available.

        Override in subclass to extract token information from agent responses.

        Args:
            response: Agent response

        Returns:
            Total tokens used or None
        """
        return None

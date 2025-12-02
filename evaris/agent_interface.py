"""Universal agent interface using Protocol for structural typing.

This module provides a Protocol-based interface for AI agents, allowing any object
with the correct methods to be used as an agent without requiring inheritance.

The AgentInterface supports:
- Synchronous execution (execute)
- Asynchronous execution (a_execute)
- State reset between test cases (reset)
- Multi-modal inputs and outputs
"""

from typing import Any, Protocol, runtime_checkable

from evaris.types import MultiModalInput, MultiModalOutput


@runtime_checkable
class AgentInterface(Protocol):
    """Universal agent interface using structural typing.

    Any object with execute() and/or a_execute() methods is considered an agent.
    No inheritance required - this uses Python's Protocol for duck typing.

    Agents can implement:
    - Only execute() for sync-only agents
    - Only a_execute() for async-only agents
    - Both execute() and a_execute() for full support
    - Optionally reset() for stateful agents

    Example:
        >>> # Simple sync agent
        >>> class MyAgent:
        ...     def execute(self, input: str) -> str:
        ...         return f"Processed: {input}"
        >>>
        >>> agent = MyAgent()
        >>> isinstance(agent, AgentInterface)  # True
        >>>
        >>> # Async agent
        >>> class AsyncAgent:
        ...     async def a_execute(self, input: str) -> str:
        ...         return f"Async processed: {input}"
        >>>
        >>> async_agent = AsyncAgent()
        >>> isinstance(async_agent, AgentInterface)  # True
    """


def is_async_agent(agent: Any) -> bool:
    """Check if agent supports async execution.

    Args:
        agent: Agent to check

    Returns:
        True if agent has a_execute() method and it's callable
    """
    return hasattr(agent, "a_execute") and callable(getattr(agent, "a_execute"))


def is_sync_agent(agent: Any) -> bool:
    """Check if agent supports sync execution.

    Args:
        agent: Agent to check

    Returns:
        True if agent has execute() method and it's callable
    """
    return hasattr(agent, "execute") and callable(getattr(agent, "execute"))


def is_stateful_agent(agent: Any) -> bool:
    """Check if agent has state that needs reset.

    Args:
        agent: Agent to check

    Returns:
        True if agent has reset() method
    """
    return hasattr(agent, "reset") and callable(getattr(agent, "reset"))


def validate_agent(agent: Any) -> None:
    """Validate that an object implements AgentInterface.

    Args:
        agent: Agent to validate

    Raises:
        TypeError: If agent doesn't implement execute() or a_execute()
    """
    if not is_sync_agent(agent) and not is_async_agent(agent):
        raise TypeError(
            f"Agent must implement at least one of execute() or a_execute() methods. "
            f"Got: {type(agent).__name__}"
        )


class SimpleAgent:
    """Simple example agent for testing and demonstration.

    This is a basic implementation that shows the minimal interface.

    Example:
        >>> agent = SimpleAgent(lambda x: f"Response: {x}")
        >>> output = agent.execute("Hello")
        >>> print(output)  # "Response: Hello"
    """

    def __init__(self, handler: Any = None):
        """Initialize simple agent.

        Args:
            handler: Callable that processes input (defaults to echo)
        """
        self.handler = handler or (lambda x: str(x))

    def execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute agent synchronously."""
        return self.handler(input)

    async def a_execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute agent asynchronously."""
        # Simple agents just run sync handler
        return self.execute(input)

    def reset(self) -> None:
        """Reset agent state (no-op for stateless agent)."""
        pass

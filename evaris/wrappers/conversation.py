"""Conversation agent wrapper for multi-turn dialogue evaluation.

This module provides wrappers for evaluating conversation agents that maintain
dialogue history across multiple turns.
"""

from typing import Any, Callable, Optional


class ConversationAgentWrapper:
    """Wrapper for multi-turn conversation agents.

    Handles conversation state management and multi-turn input parsing
    for agents that maintain dialogue history.

    Example:
        >>> from evaris.wrappers.conversation import ConversationAgentWrapper
        >>> from evaris import evaluate
        >>> from evaris.types import Golden
        >>>
        >>> class MyConversationAgent:
        ...     def __init__(self):
        ...         self.history = []
        ...     def __call__(self, input_str: str) -> str:
        ...         self.history.append(input_str)
        ...         return f"Response to: {input_str}"
        >>>
        >>> wrapped = ConversationAgentWrapper(
        ...     agent_class=MyConversationAgent,
        ...     turn_separator="|TURN|"
        ... )
        >>>
        >>> goldens = [
        ...     Golden(input="Hello |TURN| How are you?", expected="Response to: How are you?")
        ... ]
        >>> result = evaluate(task=wrapped, data=goldens, metrics=[...])
    """

    def __init__(
        self,
        agent_class: type,
        turn_separator: str = "|TURN|",
        reset_per_test: bool = True,
        init_args: Optional[tuple] = None,
        init_kwargs: Optional[dict] = None,
    ):
        """Initialize conversation wrapper.

        Args:
            agent_class: Class to instantiate for conversation agent
            turn_separator: String that separates conversation turns in input
            reset_per_test: Create fresh agent instance for each test case
            init_args: Positional arguments for agent initialization
            init_kwargs: Keyword arguments for agent initialization
        """
        self.agent_class = agent_class
        self.turn_separator = turn_separator
        self.reset_per_test = reset_per_test
        self.init_args = init_args or ()
        self.init_kwargs = init_kwargs or {}
        self.agent: Optional[Any] = None

        if not reset_per_test:
            # Create persistent agent
            self.agent = agent_class(*self.init_args, **self.init_kwargs)

    def __call__(self, input_str: str) -> str:
        """Process conversation input and return response.

        Args:
            input_str: Input string, may contain multiple turns separated by turn_separator

        Returns:
            Response from the last conversation turn
        """
        # Create fresh agent if reset_per_test is enabled
        if self.reset_per_test:
            agent = self.agent_class(*self.init_args, **self.init_kwargs)
        else:
            agent = self.agent
            assert agent is not None, "Agent must be initialized when reset_per_test is False"

        # Parse multi-turn input
        if self.turn_separator in input_str:
            turns = input_str.split(self.turn_separator)
            responses = []

            for turn in turns:
                turn = turn.strip()
                if not turn:
                    continue

                # Process turn
                response = str(agent(turn))
                responses.append(response)

            # Return last response
            return responses[-1] if responses else ""
        else:
            # Single turn
            return str(agent(input_str))


class ConversationAgentFactory:
    """Factory for creating fresh conversation agents per test.

    Alternative to ConversationAgentWrapper for simpler use cases.

    Example:
        >>> from evaris.wrappers.conversation import ConversationAgentFactory
        >>>
        >>> factory = ConversationAgentFactory(MyConversationAgent)
        >>> result = evaluate(task=factory, data=goldens, metrics=[...])
    """

    def __init__(
        self,
        agent_class: type,
        init_args: Optional[tuple] = None,
        init_kwargs: Optional[dict] = None,
    ):
        """Initialize agent factory.

        Args:
            agent_class: Class to instantiate
            init_args: Positional arguments for initialization
            init_kwargs: Keyword arguments for initialization
        """
        self.agent_class = agent_class
        self.init_args = init_args or ()
        self.init_kwargs = init_kwargs or {}

    def __call__(self, input_str: str) -> str:
        """Create fresh agent and process input.

        Args:
            input_str: Input string

        Returns:
            Agent response
        """
        # Create fresh agent for this call
        agent = self.agent_class(*self.init_args, **self.init_kwargs)
        return str(agent(input_str))


def wrap_conversation_agent(
    agent_class: type,
    turn_separator: str = "|TURN|",
    reset_per_test: bool = True,
) -> ConversationAgentWrapper:
    """Convenience function to wrap conversation agent.

    Args:
        agent_class: Conversation agent class
        turn_separator: String that separates turns
        reset_per_test: Reset agent state per test

    Returns:
        Wrapped conversation agent

    Example:
        >>> from evaris.wrappers.conversation import wrap_conversation_agent
        >>>
        >>> wrapped = wrap_conversation_agent(MyConversationAgent)
        >>> result = evaluate(task=wrapped, data=goldens, metrics=[...])
    """
    return ConversationAgentWrapper(
        agent_class=agent_class,
        turn_separator=turn_separator,
        reset_per_test=reset_per_test,
    )


def create_agent_factory(
    agent_fn: Callable[[], Any],
) -> Callable[[str], str]:
    """Create factory function for stateful agents.

    Ensures fresh agent instance per test case to avoid state pollution.

    Args:
        agent_fn: Callable that creates and returns agent instance

    Returns:
        Factory function suitable for evaris evaluation

    Example:
        >>> from evaris.wrappers.conversation import create_agent_factory
        >>>
        >>> def make_agent():
        ...     return MyStatefulAgent()
        >>>
        >>> factory = create_agent_factory(make_agent)
        >>> result = evaluate(task=factory, data=goldens, metrics=[...])
    """

    def factory_fn(input_str: str) -> str:
        """Create fresh agent and process input."""
        agent = agent_fn()
        return str(agent(input_str))

    return factory_fn

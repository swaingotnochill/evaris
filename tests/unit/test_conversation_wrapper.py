"""Tests for conversation agent wrappers."""

from evaris import evaluate
from evaris.metrics.exact_match import ExactMatchMetric
from evaris.types import Golden
from evaris.wrappers.conversation import (
    ConversationAgentFactory,
    ConversationAgentWrapper,
    create_agent_factory,
    wrap_conversation_agent,
)


class MockConversationAgent:
    """Mock conversation agent for testing."""

    def __init__(self, name: str = "Agent"):
        self.name = name
        self.history = []
        self.call_count = 0

    def __call__(self, input_str: str) -> str:
        """Process input and return response."""
        self.call_count += 1
        self.history.append({"role": "user", "content": input_str})

        # Simple response logic
        if "remember" in input_str.lower() and len(self.history) > 1:
            prev_content = self.history[-2]["content"]
            response = f"Yes, you said: {prev_content}"
        elif "name" in input_str.lower():
            response = f"My name is {self.name}"
        elif "count" in input_str.lower():
            response = f"Call count: {self.call_count}"
        else:
            response = f"I heard: {input_str}"

        self.history.append({"role": "assistant", "content": response})
        return response


class TestConversationAgentWrapper:
    """Tests for ConversationAgentWrapper."""

    def test_single_turn(self):
        """Test single turn conversation."""
        wrapped = ConversationAgentWrapper(agent_class=MockConversationAgent, reset_per_test=True)

        response = wrapped("Hello")
        assert response == "I heard: Hello"

    def test_multi_turn(self):
        """Test multi-turn conversation."""
        wrapped = ConversationAgentWrapper(
            agent_class=MockConversationAgent, turn_separator="|TURN|", reset_per_test=True
        )

        # Multi-turn input
        response = wrapped("Hello |TURN| What's your name?")
        assert response == "My name is Agent"

    def test_multi_turn_with_memory(self):
        """Test multi-turn with conversation memory."""
        wrapped = ConversationAgentWrapper(
            agent_class=MockConversationAgent, turn_separator="|TURN|", reset_per_test=True
        )

        # Agent should remember previous turn
        response = wrapped("Tell me a joke |TURN| Do you remember what I asked?")
        assert "Tell me a joke" in response

    def test_reset_per_test_enabled(self):
        """Test that reset_per_test creates fresh agent each time."""
        wrapped = ConversationAgentWrapper(agent_class=MockConversationAgent, reset_per_test=True)

        # Each call should have call_count = 1
        response1 = wrapped("What's the count?")
        assert "Call count: 1" in response1

        response2 = wrapped("What's the count?")
        assert "Call count: 1" in response2  # Fresh agent, count reset

    def test_reset_per_test_disabled(self):
        """Test that reset_per_test=False maintains state."""
        wrapped = ConversationAgentWrapper(agent_class=MockConversationAgent, reset_per_test=False)

        # State should persist
        response1 = wrapped("What's the count?")
        assert "Call count: 1" in response1

        response2 = wrapped("What's the count?")
        assert "Call count: 2" in response2  # Same agent, count incremented

    def test_custom_separator(self):
        """Test custom turn separator."""
        wrapped = ConversationAgentWrapper(
            agent_class=MockConversationAgent, turn_separator="[NEXT]", reset_per_test=True
        )

        response = wrapped("Hello [NEXT] What's your name?")
        assert response == "My name is Agent"

    def test_with_init_kwargs(self):
        """Test passing initialization kwargs to agent."""
        wrapped = ConversationAgentWrapper(
            agent_class=MockConversationAgent,
            reset_per_test=True,
            init_kwargs={"name": "CustomBot"},
        )

        response = wrapped("What's your name?")
        assert "CustomBot" in response

    def test_empty_turns(self):
        """Test handling of empty turns."""
        wrapped = ConversationAgentWrapper(
            agent_class=MockConversationAgent, turn_separator="|TURN|", reset_per_test=True
        )

        # Empty turn should be skipped
        response = wrapped("Hello |TURN|  |TURN| What's your name?")
        assert response == "My name is Agent"


class TestConversationAgentFactory:
    """Tests for ConversationAgentFactory."""

    def test_creates_fresh_agent(self):
        """Test that factory creates fresh agent each time."""
        factory = ConversationAgentFactory(MockConversationAgent)

        # Each call should have call_count = 1
        response1 = factory("What's the count?")
        assert "Call count: 1" in response1

        response2 = factory("What's the count?")
        assert "Call count: 1" in response2

    def test_with_init_kwargs(self):
        """Test factory with initialization kwargs."""
        factory = ConversationAgentFactory(
            agent_class=MockConversationAgent, init_kwargs={"name": "FactoryBot"}
        )

        response = factory("What's your name?")
        assert "FactoryBot" in response


class TestWrapConversationAgent:
    """Tests for wrap_conversation_agent helper."""

    def test_wrap_convenience_function(self):
        """Test convenience wrapper function."""
        wrapped = wrap_conversation_agent(MockConversationAgent)

        response = wrapped("Hello")
        assert response == "I heard: Hello"

    def test_wrap_with_custom_separator(self):
        """Test wrapper with custom separator."""
        wrapped = wrap_conversation_agent(
            MockConversationAgent, turn_separator=">>", reset_per_test=True
        )

        response = wrapped("Hello >> What's your name?")
        assert response == "My name is Agent"


class TestCreateAgentFactory:
    """Tests for create_agent_factory helper."""

    def test_factory_function(self):
        """Test agent factory function."""

        def make_agent():
            return MockConversationAgent(name="FactoryAgent")

        factory = create_agent_factory(make_agent)

        response = factory("What's your name?")
        assert "FactoryAgent" in response

    def test_factory_ensures_fresh_state(self):
        """Test that factory ensures fresh state."""

        def make_agent():
            return MockConversationAgent()

        factory = create_agent_factory(make_agent)

        response1 = factory("What's the count?")
        response2 = factory("What's the count?")

        assert "Call count: 1" in response1
        assert "Call count: 1" in response2


class TestIntegrationWithEvaluate:
    """Integration tests with evaris.evaluate()."""

    def test_evaluate_with_conversation_wrapper(self):
        """Test evaluation with conversation wrapper."""
        goldens = [
            Golden(input="What's your name?", expected="My name is Agent"),
            Golden(input="Hello |TURN| What's your name?", expected="My name is Agent"),
        ]

        wrapped = ConversationAgentWrapper(
            agent_class=MockConversationAgent, turn_separator="|TURN|", reset_per_test=True
        )

        result = evaluate(
            task=wrapped, data=goldens, metrics=[ExactMatchMetric()], name="Conversation Agent Test"
        )

        assert result.total == 2
        assert result.passed == 2
        assert result.accuracy == 1.0

    def test_evaluate_with_factory(self):
        """Test evaluation with agent factory."""
        goldens = [
            Golden(input="What's the count?", expected="Call count: 1"),
            Golden(input="What's the count?", expected="Call count: 1"),
        ]

        factory = ConversationAgentFactory(MockConversationAgent)

        result = evaluate(
            task=factory, data=goldens, metrics=[ExactMatchMetric()], name="Factory Test"
        )

        assert result.total == 2
        assert result.passed == 2
        assert result.accuracy == 1.0

    def test_state_pollution_prevented(self):
        """Test that state pollution is prevented with wrapper."""
        wrapped = ConversationAgentWrapper(agent_class=MockConversationAgent, reset_per_test=True)

        # Include call count in expected to verify fresh state
        goldens_with_count = [
            Golden(input="What's the count?", expected="Call count: 1"),
            Golden(input="What's the count?", expected="Call count: 1"),
            Golden(input="What's the count?", expected="Call count: 1"),
        ]

        result = evaluate(
            task=wrapped,
            data=goldens_with_count,
            metrics=[ExactMatchMetric()],
            name="State Pollution Test",
        )

        # All should pass if state is properly reset
        assert result.passed == 3
        assert result.accuracy == 1.0

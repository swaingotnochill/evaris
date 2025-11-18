"""Practical examples for evaluating different AI agent patterns with evaris.

This script demonstrates how to evaluate various agent architectures:
1. Simple function-calling agents (OpenAI/Claude style)
2. Stateful agents (requires factory pattern)
3. Multi-turn conversation agents (using ConversationAgentWrapper)
4. Async agents
5. Agents with JSON output
"""

import asyncio
import json
from typing import Any

from evaris import evaluate, evaluate_async
from evaris.metrics.exact_match import ExactMatchMetric
from evaris.types import Golden, MetricResult, TestCase
from evaris.wrappers.conversation import ConversationAgentFactory, ConversationAgentWrapper


# Example 1: Simple Function-Calling Agent
def example_function_calling():
    """Demonstrate evaluation of function-calling agent."""
    print("\n=== Example 1: Function-Calling Agent ===")

    def calculator_agent(input_str: str) -> str:
        """Agent with simulated function calling."""
        if "2+2" in input_str:
            return "4"
        elif "10*5" in input_str:
            return "50"
        elif "100/4" in input_str:
            return "25"
        return "I cannot calculate that"

    goldens = [
        Golden(input="Calculate 2+2", expected="4"),
        Golden(input="What is 10*5?", expected="50"),
        Golden(input="Compute 100/4", expected="25"),
    ]

    result = evaluate(
        task=calculator_agent,
        data=goldens,
        metrics=[ExactMatchMetric()],
        name="Calculator Agent",
    )

    print(f"Result: {result.passed}/{result.total} passed ({result.accuracy:.1%})")
    return result


# Example 2: Stateful Agent (WRONG and RIGHT approaches)
def example_stateful_agent():
    """Demonstrate state pollution issue and solution."""
    print("\n=== Example 2: Stateful Agent (State Management) ===")

    class StatefulAgent:
        """Agent that maintains state (simulates session/memory)."""

        def __init__(self):
            self.call_count = 0

        def __call__(self, input_str: str) -> str:
            self.call_count += 1
            return f"Call #{self.call_count}"

    goldens = [
        Golden(input="Test 1", expected="Call #1"),
        Golden(input="Test 2", expected="Call #1"),  # Should be #1, not #2
    ]

    # WRONG: Reusing agent instance (state pollution)
    print("\n[WRONG] Reusing agent instance:")
    agent_wrong = StatefulAgent()
    result_wrong = evaluate(
        task=agent_wrong, data=goldens, metrics=[ExactMatchMetric()], name="Wrong"
    )
    print(f"  Result: {result_wrong.passed}/{result_wrong.total} passed (EXPECTED TO FAIL)")

    # RIGHT: Using factory pattern
    print("\n[RIGHT] Using factory pattern:")
    factory = ConversationAgentFactory(StatefulAgent)
    result_right = evaluate(task=factory, data=goldens, metrics=[ExactMatchMetric()], name="Right")
    print(f"  Result: {result_right.passed}/{result_right.total} passed (SHOULD PASS)")

    return result_right


# Example 3: Multi-Turn Conversation Agent
def example_conversation_agent():
    """Demonstrate ConversationAgentWrapper for multi-turn dialogue."""
    print("\n=== Example 3: Multi-Turn Conversation Agent ===")

    class ConversationBot:
        """Conversation agent with memory."""

        def __init__(self):
            self.history = []

        def __call__(self, input_str: str) -> str:
            self.history.append(input_str)

            if "remember" in input_str.lower() and len(self.history) > 1:
                prev_input = self.history[-2]
                return f"Yes, you said: {prev_input}"
            elif "name" in input_str.lower():
                return "My name is ConversationBot"
            else:
                return f"I heard: {input_str}"

    # Test multi-turn conversations
    goldens = [
        Golden(input="What's your name?", expected="My name is ConversationBot"),
        Golden(
            input="Tell me a joke |TURN| Do you remember what I asked?",
            expected="Yes, you said: Tell me a joke",
        ),
    ]

    # Use ConversationAgentWrapper for multi-turn support
    wrapped = ConversationAgentWrapper(
        agent_class=ConversationBot, turn_separator="|TURN|", reset_per_test=True
    )

    result = evaluate(
        task=wrapped,
        data=goldens,
        metrics=[ExactMatchMetric()],
        name="Conversation Agent",
    )

    print(f"Result: {result.passed}/{result.total} passed ({result.accuracy:.1%})")
    return result


# Example 4: Async Agent
async def example_async_agent():
    """Demonstrate async agent evaluation."""
    print("\n=== Example 4: Async Agent ===")

    async def async_agent(input_str: str) -> str:
        """Async agent with simulated network delay."""
        await asyncio.sleep(0.01)  # Simulate API call
        return f"Processed: {input_str}"

    goldens = [
        Golden(input="Hello", expected="Processed: Hello"),
        Golden(input="World", expected="Processed: World"),
    ]

    result = await evaluate_async(
        task=async_agent,
        data=goldens,
        metrics=[ExactMatchMetric()],
        name="Async Agent",
    )

    print(f"Result: {result.passed}/{result.total} passed ({result.accuracy:.1%})")
    return result


# Example 5: Agent with JSON Output
def example_json_agent():
    """Demonstrate custom metric for JSON output."""
    print("\n=== Example 5: Agent with JSON Output ===")

    def json_agent(input_str: str) -> str:
        """Agent returning structured JSON."""
        result = {
            "answer": f"Response to: {input_str}",
            "confidence": 0.95,
            "sources": ["source1"],
        }
        return json.dumps(result)

    # Custom metric to extract answer from JSON
    class JSONAnswerMetric(ExactMatchMetric):
        """Extract 'answer' field from JSON response."""

        def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
            try:
                data = json.loads(str(actual_output))
                actual_answer = data.get("answer", "")

                modified_tc = TestCase(
                    input=test_case.input,
                    actual_output=actual_answer,
                    expected=test_case.expected,
                )

                return super().score(modified_tc, actual_answer)
            except json.JSONDecodeError:
                return MetricResult(
                    name="json_answer",
                    score=0.0,
                    passed=False,
                    metadata={"error": "Invalid JSON"},
                )

    goldens = [
        Golden(input="Hello", expected="Response to: Hello"),
        Golden(input="Test", expected="Response to: Test"),
    ]

    result = evaluate(
        task=json_agent,
        data=goldens,
        metrics=[JSONAnswerMetric()],
        name="JSON Agent",
    )

    print(f"Result: {result.passed}/{result.total} passed ({result.accuracy:.1%})")
    return result


# Main runner
def run_all_sync_examples():
    """Run all synchronous examples."""
    example_function_calling()
    example_stateful_agent()
    example_conversation_agent()
    example_json_agent()


async def run_all_examples():
    """Run all examples (sync and async)."""
    print("=" * 60)
    print("EVARIS: AI Agent Evaluation Examples")
    print("=" * 60)

    # Run sync examples in thread pool to avoid async context issues
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_all_sync_examples)
        future.result()

    # Run async example
    await example_async_agent()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_examples())

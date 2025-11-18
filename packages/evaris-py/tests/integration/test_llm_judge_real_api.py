"""Integration tests for LLM judge with real API calls.

These tests use actual API keys and make real API calls to verify
the Evaris evaluation framework works with production LLM services.

Uses the complete evaluate() workflow to test end-to-end integration.

To run these tests, set the appropriate environment variables:
- OPENAI_API_KEY for OpenAI tests
- ANTHROPIC_API_KEY for Anthropic tests
- DASHSCOPE_API_KEY for QWEN tests
- GEMINI_API_KEY for GEMINI tests

Tests are skipped if API keys are not available.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from evaris import evaluate
from evaris.metrics.llm_judge import LLMJudgeConfig, LLMJudgeMetric

# Load .env file from package root
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Check which API keys are available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OPENAI_AVAILABLE = OPENAI_API_KEY is not None
ANTHROPIC_AVAILABLE = ANTHROPIC_API_KEY is not None
QWEN_AVAILABLE = DASHSCOPE_API_KEY is not None
GEMINI_AVAILABLE = GEMINI_API_KEY is not None


# Helper function to get metric by name
def get_metric_result(test_result, metric_name: str):
    """Extract metric result by name from test result."""
    metrics = [m for m in test_result.metrics if m.name == metric_name]
    if not metrics:
        raise ValueError(f"Metric '{metric_name}' not found in test result")
    return metrics[0]


# Test agent functions
def qa_agent_correct(query: str) -> str:
    """Agent that answers questions correctly."""
    if "2+2" in query.lower():
        return "2+2 equals 4"
    elif "capital of france" in query.lower():
        return "The capital of France is Paris"
    elif "largest planet" in query.lower():
        return "Jupiter is the largest planet in our solar system"
    elif "10 * 5" in query or "10*5" in query:
        return "10 times 5 equals 50"
    elif "sky" in query.lower() and "blue" in query.lower():
        return "The sky looks blue due to scattering of light by air molecules"
    return "I don't know"


def qa_agent_incorrect(query: str) -> str:
    """Agent that answers questions incorrectly."""
    if "2+2" in query.lower():
        return "The answer is 5"
    elif "capital of france" in query.lower():
        return "London"
    elif "largest planet" in query.lower():
        return "Mars is the largest planet"
    return "Wrong answer"


class TestLLMJudgeIntegration:
    """Integration tests using Evaris evaluate() with real LLM APIs."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    def test_evaluate_with_openai_gpt4(self) -> None:
        """Test complete evaluation workflow with OpenAI GPT-4 as judge."""
        # Configure LLM judge with threshold
        llm_judge_config = LLMJudgeConfig(
            provider="openai",
            model="gpt-4",
            enable_self_consistency=False,
            threshold=0.7,
        )
        llm_judge = LLMJudgeMetric(llm_judge_config)

        # Run evaluation using Evaris framework
        result = evaluate(
            name="openai-gpt4-judge-test",
            task=qa_agent_correct,
            data=[
                {"input": "What is 2+2?", "expected": "4"},
                {"input": "What is the capital of France?", "expected": "Paris"},
            ],
            metrics=[llm_judge],
        )

        # Verify framework behavior (not individual scores)
        assert result.name == "openai-gpt4-judge-test"
        assert result.total == 2
        assert len(result.results) == 2

        # Verify framework calculates accuracy correctly
        assert result.accuracy == result.passed / result.total

        # Verify each test result has required structure
        for test_result in result.results:
            # Get llm_judge metric result
            metric_result = get_metric_result(test_result, "llm_judge")

            # Verify metric returned proper structure
            assert 0.0 <= metric_result.score <= 1.0
            assert isinstance(metric_result.passed, bool)
            assert "reasoning" in metric_result.metadata
            assert metric_result.metadata["provider"] == "openai"
            assert metric_result.metadata["model"] == "gpt-4"
            assert metric_result.metadata["threshold"] == 0.7

            # Verify metric applied threshold correctly
            expected_passed = metric_result.score >= 0.7
            assert metric_result.passed == expected_passed

        print("\n[OK] OpenAI GPT-4 Evaluation Results:")
        print(f"  Total: {result.total}")
        print(f"  Passed: {result.passed}")
        print(f"  Accuracy: {result.accuracy:.2%}")

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    def test_evaluate_distinguishes_correct_vs_incorrect(self) -> None:
        """Test that LLM judge distinguishes correct from incorrect answers."""
        llm_judge_config = LLMJudgeConfig(
            provider="openai",
            model="gpt-4",
            enable_self_consistency=False,
            threshold=0.6,
        )
        llm_judge = LLMJudgeMetric(llm_judge_config)

        # Test with correct agent
        result_correct = evaluate(
            name="correct-agent",
            task=qa_agent_correct,
            data=[{"input": "What is 2+2?", "expected": "4"}],
            metrics=[llm_judge],
        )

        # Test with incorrect agent
        result_incorrect = evaluate(
            name="incorrect-agent",
            task=qa_agent_incorrect,
            data=[{"input": "What is 2+2?", "expected": "4"}],
            metrics=[llm_judge],
        )

        # Correct agent should score higher than incorrect agent
        correct_score = get_metric_result(result_correct.results[0], "llm_judge").score
        incorrect_score = get_metric_result(result_incorrect.results[0], "llm_judge").score

        assert correct_score > incorrect_score, (
            f"Correct agent should score higher than incorrect agent. "
            f"Got correct={correct_score:.2f}, incorrect={incorrect_score:.2f}"
        )

        print("\n[OK] Distinguishing correct vs incorrect:")
        print(f"  Correct agent score: {correct_score:.2f}")
        print(f"  Incorrect agent score: {incorrect_score:.2f}")
        print(
            f"  Correct passed: {
              result_correct.passed}/{result_correct.total}"
        )
        print(
            f"  Incorrect passed: {
              result_incorrect.passed}/{result_incorrect.total}"
        )

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    def test_threshold_behavior(self) -> None:
        """Test that threshold configuration affects pass/fail behavior."""
        test_data = [{"input": "What is 2+2?", "expected": "4"}]

        # Test with strict threshold
        strict_config = LLMJudgeConfig(
            provider="openai",
            model="gpt-4",
            enable_self_consistency=False,
            threshold=0.9,  # Very strict
        )
        strict_metric = LLMJudgeMetric(strict_config)

        # Test with lenient threshold
        lenient_config = LLMJudgeConfig(
            provider="openai",
            model="gpt-4",
            enable_self_consistency=False,
            threshold=0.3,  # Very lenient
        )
        lenient_metric = LLMJudgeMetric(lenient_config)

        # Run evaluations
        strict_result = evaluate(
            name="strict-threshold",
            task=qa_agent_correct,
            data=test_data,
            metrics=[strict_metric],
        )

        lenient_result = evaluate(
            name="lenient-threshold",
            task=qa_agent_correct,
            data=test_data,
            metrics=[lenient_metric],
        )

        # Verify thresholds were applied
        strict_metric_result = get_metric_result(strict_result.results[0], "llm_judge")
        lenient_metric_result = get_metric_result(lenient_result.results[0], "llm_judge")

        assert strict_metric_result.metadata["threshold"] == 0.9
        assert lenient_metric_result.metadata["threshold"] == 0.3

        # Verify threshold logic
        assert strict_metric_result.passed == (strict_metric_result.score >= 0.9)
        assert lenient_metric_result.passed == (lenient_metric_result.score >= 0.3)

        print("\n[OK] Threshold behavior test:")
        print(f"  Score: {strict_metric_result.score:.2f}")
        print(
            f"  Strict (0.9): {
              strict_result.passed}/{strict_result.total} passed"
        )
        print(
            f"  Lenient (0.3): {
              lenient_result.passed}/{lenient_result.total} passed"
        )

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="ANTHROPIC_API_KEY not set")
    def test_evaluate_with_anthropic_claude(self) -> None:
        """Test complete evaluation workflow with Anthropic Claude as judge."""
        llm_judge_config = LLMJudgeConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            enable_self_consistency=False,
            threshold=0.7,
        )
        llm_judge = LLMJudgeMetric(llm_judge_config)

        result = evaluate(
            name="anthropic-claude-judge-test",
            task=qa_agent_correct,
            data=[{"input": "What is the capital of France?", "expected": "Paris"}],
            metrics=[llm_judge],
        )

        # Verify framework behavior
        assert result.name == "anthropic-claude-judge-test"
        assert result.total == 1
        assert result.accuracy == result.passed / result.total

        metric_result = get_metric_result(result.results[0], "llm_judge")
        assert "reasoning" in metric_result.metadata
        assert metric_result.metadata["provider"] == "anthropic"
        assert metric_result.metadata["threshold"] == 0.7

        # Verify threshold was applied correctly
        assert metric_result.passed == (metric_result.score >= 0.7)

        print("\n[OK] Anthropic Claude Evaluation Results:")
        print(f"  Score: {metric_result.score:.2f}")
        print(f"  Passed: {result.passed}/{result.total}")

    @pytest.mark.skipif(not QWEN_AVAILABLE, reason="DASHSCOPE_API_KEY not set")
    def test_evaluate_with_qwen(self) -> None:
        """Test complete evaluation workflow with QWEN as judge."""
        llm_judge_config = LLMJudgeConfig(
            provider="qwen",
            model="qwen-plus",
            enable_self_consistency=False,
            threshold=0.7,
        )
        llm_judge = LLMJudgeMetric(llm_judge_config)

        result = evaluate(
            name="qwen-judge-test",
            task=qa_agent_correct,
            data=[
                {"input": "What is the largest planet in our solar system?", "expected": "Jupiter"}
            ],
            metrics=[llm_judge],
        )

        # Verify framework behavior
        assert result.name == "qwen-judge-test"
        assert result.total == 1
        assert result.accuracy == result.passed / result.total

        metric_result = get_metric_result(result.results[0], "llm_judge")
        assert "reasoning" in metric_result.metadata
        assert metric_result.metadata["provider"] == "qwen"
        assert metric_result.metadata["model"] == "qwen-plus"
        assert metric_result.metadata["threshold"] == 0.7

        # Verify threshold was applied correctly
        assert metric_result.passed == (metric_result.score >= 0.7)

        print("\n[OK] QWEN Evaluation Results:")
        print(f"  Score: {metric_result.score:.2f}")
        print(f"  Passed: {result.passed}/{result.total}")

    @pytest.mark.skipif(not GEMINI_AVAILABLE, reason="GEMINI_API_KEY not set")
    def test_evaluate_with_gemini(self) -> None:
        """Test complete evaluation workflow with GEMINI as judge."""
        llm_judge_config = LLMJudgeConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            enable_self_consistency=False,
            threshold=0.7,
        )
        llm_judge = LLMJudgeMetric(llm_judge_config)

        result = evaluate(
            name="gemini-judge-test",
            task=qa_agent_correct,
            data=[
                {"input": "What is the largest planet in our solar system?", "expected": "Jupiter"}
            ],
            metrics=[llm_judge],
        )

        # Verify framework behavior
        assert result.name == "gemini-judge-test"
        assert result.total == 1
        assert result.accuracy == result.passed / result.total

        metric_result = get_metric_result(result.results[0], "llm_judge")
        assert "reasoning" in metric_result.metadata
        assert metric_result.metadata["provider"] == "gemini"
        assert metric_result.metadata["model"] == "gemini-2.5-flash"
        assert metric_result.metadata["threshold"] == 0.7

        # Verify threshold was applied correctly
        assert metric_result.passed == (metric_result.score >= 0.7)

        print("\n[OK] GEMINI Evaluation Results:")
        print(f"  Score: {metric_result.score:.2f}")
        print(f"  Passed: {result.passed}/{result.total}")
        print(
            f"  Reasoning: {metric_result.metadata.get(
            'reasoning', 'N/A')[:100]}..."
        )

    @pytest.mark.skipif(not QWEN_AVAILABLE, reason="DASHSCOPE_API_KEY not set")
    def test_evaluate_qwen_with_self_consistency(self) -> None:
        """Test QWEN with self-consistency check (ABC O.c.1 compliance)."""
        llm_judge_config = LLMJudgeConfig(
            provider="qwen",
            model="qwen-plus",
            enable_self_consistency=True,
            self_consistency_samples=3,
            threshold=0.7,
        )
        llm_judge = LLMJudgeMetric(llm_judge_config)

        result = evaluate(
            name="qwen-self-consistency-test",
            task=qa_agent_correct,
            data=[{"input": "What is 10 * 5?", "expected": "50"}],
            metrics=[llm_judge],
        )

        # Verify framework behavior
        assert result.total == 1

        metric_result = get_metric_result(result.results[0], "llm_judge")

        # Verify self-consistency metadata exists
        assert "self_consistency_scores" in metric_result.metadata
        assert "self_consistency_variance" in metric_result.metadata
        assert len(metric_result.metadata["self_consistency_scores"]) == 3

        # Verify threshold was applied to mean score
        assert metric_result.passed == (metric_result.score >= 0.7)

        print("\n[OK] QWEN Self-Consistency Test:")
        print(f"  Mean score: {metric_result.score:.2f}")
        print(
            f"  Variance: {
              metric_result.metadata['self_consistency_variance']:.4f}"
        )
        print(
            f"  Individual scores: {
              metric_result.metadata['self_consistency_scores']}"
        )
        print(f"  Passed: {result.passed}/{result.total}")

    @pytest.mark.skipif(
        not QWEN_AVAILABLE or not OPENAI_AVAILABLE,
        reason="Both QWEN and OpenAI API keys required",
    )
    def test_compare_qwen_vs_openai_evaluation(self) -> None:
        """Compare QWEN and OpenAI judgments on the same task."""
        test_data = [
            {
                "input": "Explain why the sky is blue in one sentence",
                "expected": (
                    "The sky appears blue because of Rayleigh scattering of sunlight "
                    "by the atmosphere"
                ),
            }
        ]

        # Evaluate with QWEN
        qwen_config = LLMJudgeConfig(
            provider="qwen",
            model="qwen-plus",
            enable_self_consistency=False,
            threshold=0.6,
        )
        qwen_metric = LLMJudgeMetric(qwen_config)
        qwen_result = evaluate(
            name="qwen-eval", task=qa_agent_correct, data=test_data, metrics=[qwen_metric]
        )

        # Evaluate with OpenAI
        openai_config = LLMJudgeConfig(
            provider="openai",
            model="gpt-4",
            enable_self_consistency=False,
            threshold=0.6,
        )
        openai_metric = LLMJudgeMetric(openai_config)
        openai_result = evaluate(
            name="openai-eval", task=qa_agent_correct, data=test_data, metrics=[openai_metric]
        )

        # Verify framework behavior for both
        assert qwen_result.total == 1
        assert openai_result.total == 1

        qwen_metric = get_metric_result(qwen_result.results[0], "llm_judge")
        openai_metric = get_metric_result(openai_result.results[0], "llm_judge")

        qwen_score = qwen_metric.score
        openai_score = openai_metric.score

        # Verify thresholds were applied
        qwen_passed = qwen_metric.passed
        openai_passed = openai_metric.passed

        assert qwen_passed == (qwen_score >= 0.6)
        assert openai_passed == (openai_score >= 0.6)

        print("\n[OK] QWEN vs OpenAI Comparison:")
        print(
            f"  QWEN score: {qwen_score:.2f} (passed: {
              qwen_result.passed}/{qwen_result.total})"
        )
        print(
            f"  OpenAI score: {openai_score:.2f} "
            f"(passed: {openai_result.passed}/{openai_result.total})"
        )
        print(f"  QWEN accuracy: {qwen_result.accuracy:.2%}")
        print(f"  OpenAI accuracy: {openai_result.accuracy:.2%}")

    @pytest.mark.skipif(not QWEN_AVAILABLE, reason="DASHSCOPE_API_KEY not set")
    def test_multiple_metrics_on_same_task(self) -> None:
        """Test that multiple metrics can evaluate the same task."""
        # Create two judges with different thresholds
        strict_judge = LLMJudgeMetric(
            LLMJudgeConfig(
                provider="qwen", model="qwen-plus", threshold=0.9, enable_self_consistency=False
            )
        )
        lenient_judge = LLMJudgeMetric(
            LLMJudgeConfig(
                provider="qwen", model="qwen-plus", threshold=0.3, enable_self_consistency=False
            )
        )

        # Note: Currently the framework doesn't support multiple metrics with same name
        # This test documents current behavior - may need framework enhancement
        test_data = [{"input": "What is 2+2?", "expected": "4"}]

        result_strict = evaluate(
            name="multi-metric-test",
            task=qa_agent_correct,
            data=test_data,
            metrics=[strict_judge],
        )

        result_lenient = evaluate(
            name="multi-metric-test",
            task=qa_agent_correct,
            data=test_data,
            metrics=[lenient_judge],
        )

        # Both should run successfully
        assert result_strict.total == 1
        assert result_lenient.total == 1

        # Different thresholds should potentially produce different pass/fail results
        strict_score = get_metric_result(result_strict.results[0], "llm_judge").score

        print("\n[OK] Multiple metrics test:")
        print(f"  Score: {strict_score:.2f}")
        print(
            f"  Strict judge (0.9): {
              result_strict.passed}/{result_strict.total}"
        )
        print(
            f"  Lenient judge (0.3): {
              result_lenient.passed}/{result_lenient.total}"
        )

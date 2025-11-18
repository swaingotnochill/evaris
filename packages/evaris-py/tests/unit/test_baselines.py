"""Tests for baselines module."""

import pytest

from evaris.baselines import (
    BaselineAgent,
    BaselineComparison,
    BaselineComparisonReport,
    BaselineConfig,
    BaselineManager,
)
from evaris.types import EvalResult, MetricResult, TestCase, TestResult


class TestBaselineConfig:
    """Tests for BaselineConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = BaselineConfig()

        assert config.require_baselines is True
        assert "random" in config.baseline_types
        assert "do_nothing" in config.baseline_types
        assert "trivial" in config.baseline_types
        assert config.min_improvement_threshold == 0.05

    def test_custom_config(self):
        """Test custom configuration."""
        config = BaselineConfig(
            require_baselines=False,
            baseline_types=["random"],
            min_improvement_threshold=0.1,
        )

        assert config.require_baselines is False
        assert config.baseline_types == ["random"]
        assert config.min_improvement_threshold == 0.1


class TestBaselineAgent:
    """Tests for BaselineAgent."""

    def test_baseline_agent_creation(self):
        """Test creating baseline agent."""

        def simple_fn(tc):
            return "output"

        agent = BaselineAgent(name="test", description="Test baseline", agent_fn=simple_fn)

        assert agent.name == "test"
        assert agent.description == "Test baseline"
        assert agent.agent_fn == simple_fn


class TestBaselineComparison:
    """Tests for BaselineComparison."""

    def test_comparison_creation(self):
        """Test creating baseline comparison."""
        comp = BaselineComparison(
            baseline_name="random",
            agent_score=0.8,
            baseline_score=0.5,
            absolute_improvement=0.3,
            relative_improvement=0.6,
            is_meaningful=True,
            interpretation="Agent outperforms baseline",
        )

        assert comp.baseline_name == "random"
        assert comp.agent_score == 0.8
        assert comp.baseline_score == 0.5
        assert comp.absolute_improvement == 0.3
        assert comp.is_meaningful is True


class TestBaselineManager:
    """Tests for BaselineManager."""

    def test_manager_default_config(self):
        """Test manager with default config."""
        manager = BaselineManager()

        assert manager.config.require_baselines is True
        # Should have registered default baselines
        assert "random" in manager.baselines
        assert "do_nothing" in manager.baselines
        assert "trivial" in manager.baselines

    def test_manager_custom_config(self):
        """Test manager with custom config."""
        config = BaselineConfig(baseline_types=["random"])
        manager = BaselineManager(config)

        assert "random" in manager.baselines
        assert "do_nothing" not in manager.baselines
        assert "trivial" not in manager.baselines

    def test_register_baseline(self):
        """Test registering custom baseline."""
        manager = BaselineManager()

        def custom_baseline(tc):
            return "custom"

        manager.register_baseline("custom", "Custom baseline", custom_baseline)

        assert "custom" in manager.baselines
        assert manager.baselines["custom"].name == "custom"

    def test_run_baseline_random(self):
        """Test running random baseline."""
        manager = BaselineManager()
        test_cases = [
            TestCase(input=f"test{i}", expected=f"output{i}", actual_output=f"output{i}")
            for i in range(5)
        ]

        outputs = manager.run_baseline("random", test_cases)

        assert len(outputs) == 5
        # Random baseline should return valid options
        for output in outputs:
            assert isinstance(output, str)

    def test_run_baseline_do_nothing(self):
        """Test running do-nothing baseline."""
        manager = BaselineManager()
        test_cases = [
            TestCase(input=f"test{i}", expected=f"output{i}", actual_output=f"output{i}")
            for i in range(3)
        ]

        outputs = manager.run_baseline("do_nothing", test_cases)

        assert len(outputs) == 3
        # Do-nothing should return empty strings
        assert all(output == "" for output in outputs)

    def test_run_baseline_trivial(self):
        """Test running trivial baseline."""
        manager = BaselineManager()
        test_cases = [
            TestCase(input=f"test{i}", expected=f"output{i}", actual_output=f"output{i}")
            for i in range(3)
        ]

        outputs = manager.run_baseline("trivial", test_cases)

        assert len(outputs) == 3
        # Trivial should echo input
        assert outputs[0] == "test0"
        assert outputs[1] == "test1"
        assert outputs[2] == "test2"

    def test_run_baseline_not_found(self):
        """Test running non-existent baseline."""
        manager = BaselineManager()
        test_cases = [TestCase(input="test", expected="output", actual_output="output")]

        with pytest.raises(ValueError) as exc_info:
            manager.run_baseline("nonexistent", test_cases)

        assert "not found" in str(exc_info.value)

    def test_run_baseline_handles_errors(self):
        """Test baseline handles errors gracefully."""
        manager = BaselineManager()

        def failing_baseline(tc):
            raise RuntimeError("Baseline failed")

        manager.register_baseline("failing", "Failing baseline", failing_baseline)

        test_cases = [TestCase(input="test", expected="output", actual_output="output")]
        outputs = manager.run_baseline("failing", test_cases)

        # Should return empty output on failure
        assert len(outputs) == 1
        assert outputs[0] == ""

    def test_evaluate_baseline(self):
        """Test evaluating baseline."""
        from evaris.metrics.exact_match import ExactMatchMetric

        manager = BaselineManager()
        # Don't include actual_output - let baseline generate it
        test_cases = [
            TestCase(input="test1", expected="test1", actual_output=""),
            TestCase(input="test2", expected="test2", actual_output=""),
        ]

        # Use actual metric instance
        metrics = [ExactMatchMetric()]

        # Trivial baseline should get perfect score (echoes input)
        score = manager.evaluate_baseline("trivial", test_cases, metrics)

        assert score == 1.0

    def test_evaluate_baseline_metric_error(self):
        """Test baseline evaluation handles metric errors."""
        from evaris.types import BaseMetric

        # Create a metric that always fails
        class FailingMetric(BaseMetric):
            async def a_measure(self, test_case):
                raise RuntimeError("Metric failed")

            def measure(self, test_case):
                raise RuntimeError("Metric failed")

        manager = BaselineManager()
        test_cases = [TestCase(input="test", expected="output", actual_output="output")]

        metrics = [FailingMetric()]
        score = manager.evaluate_baseline("trivial", test_cases, metrics)

        # Should return 0 on metric failure
        assert score == 0.0

    def test_compare_with_baseline_improvement(self):
        """Test comparison with improvement."""
        manager = BaselineManager()

        comparison = manager.compare_with_baseline(
            agent_score=0.8, baseline_name="random", baseline_score=0.5
        )

        assert isinstance(comparison, BaselineComparison)
        assert comparison.agent_score == 0.8
        assert comparison.baseline_score == 0.5
        assert comparison.absolute_improvement == pytest.approx(0.3, abs=0.001)
        assert comparison.relative_improvement == pytest.approx(0.6, abs=0.01)
        assert comparison.is_meaningful is True  # 0.3 > 0.05 threshold
        assert "outperforms" in comparison.interpretation

    def test_compare_with_baseline_slight_improvement(self):
        """Test comparison with slight improvement."""
        manager = BaselineManager()

        comparison = manager.compare_with_baseline(
            agent_score=0.52, baseline_name="random", baseline_score=0.5
        )

        assert comparison.absolute_improvement == pytest.approx(0.02, abs=0.001)
        assert comparison.is_meaningful is False  # 0.02 < 0.05 threshold
        assert "slightly" in comparison.interpretation

    def test_compare_with_baseline_underperform(self):
        """Test comparison with underperformance."""
        manager = BaselineManager()

        comparison = manager.compare_with_baseline(
            agent_score=0.4, baseline_name="random", baseline_score=0.5
        )

        assert comparison.absolute_improvement == pytest.approx(-0.1, abs=0.001)
        assert comparison.is_meaningful is False
        assert "underperforms" in comparison.interpretation

    def test_compare_with_baseline_equal(self):
        """Test comparison with equal performance."""
        manager = BaselineManager()

        comparison = manager.compare_with_baseline(
            agent_score=0.5, baseline_name="random", baseline_score=0.5
        )

        assert comparison.absolute_improvement == 0.0
        assert "equally" in comparison.interpretation

    def test_compare_with_baseline_zero_baseline(self):
        """Test comparison with zero baseline score."""
        manager = BaselineManager()

        comparison = manager.compare_with_baseline(
            agent_score=0.5, baseline_name="random", baseline_score=0.0
        )

        # Should handle division by zero
        assert comparison.relative_improvement == float("inf")

    def test_compare_with_baselines(self):
        """Test comparing with all baselines."""
        from evaris.metrics.exact_match import ExactMatchMetric

        manager = BaselineManager()

        # Create evaluation result
        results = []
        for i in range(30):
            tc = TestCase(input=f"test{i}", expected=f"test{i}", actual_output=f"test{i}")
            mr = MetricResult(name="test", score=1.0, passed=True, metadata={})
            results.append(
                TestResult(test_case=tc, output=f"test{i}", metrics=[mr], latency_ms=10.0)
            )

        eval_result = EvalResult(
            name="test",
            results=results,
            passed=30,
            failed=0,
            total=30,
            accuracy=1.0,
            avg_latency_ms=10.0,
        )

        # Pass metrics that were used for agent evaluation
        metrics = [ExactMatchMetric()]
        report = manager.compare_with_baselines(eval_result, metrics, "TestAgent")

        assert isinstance(report, BaselineComparisonReport)
        assert report.agent_name == "TestAgent"
        assert report.agent_mean_score == 1.0
        assert (
            len(report.comparisons) == 5
        )  # 5 default baselines now (random, do_nothing, trivial, majority, constant)
        assert report.best_baseline in ["random", "do_nothing", "trivial", "majority", "constant"]
        # Agent equals trivial baseline (both echo input), so doesn't beat ALL
        # But it should beat random and do_nothing

    def test_compare_with_baselines_no_baselines(self):
        """Test comparison with no baselines registered."""
        from evaris.metrics.exact_match import ExactMatchMetric

        config = BaselineConfig(baseline_types=[])
        manager = BaselineManager(config)

        results = [
            TestResult(
                test_case=TestCase(input="test", expected="output", actual_output="output"),
                output="output",
                metrics=[MetricResult(name="test", score=0.8, passed=True, metadata={})],
                latency_ms=10.0,
            )
        ]
        eval_result = EvalResult(
            name="test",
            results=results,
            passed=1,
            failed=0,
            total=1,
            accuracy=1.0,
            avg_latency_ms=10.0,
        )

        metrics = [ExactMatchMetric()]
        with pytest.raises(ValueError) as exc_info:
            manager.compare_with_baselines(eval_result, metrics)

        assert "No baselines" in str(exc_info.value)

    def test_format_comparison_report(self):
        """Test formatting comparison report."""
        manager = BaselineManager()

        # Create simple report
        comparisons = [
            BaselineComparison(
                baseline_name="baseline1",
                agent_score=0.8,
                baseline_score=0.5,
                absolute_improvement=0.3,
                relative_improvement=0.6,
                is_meaningful=True,
                interpretation="Good",
            )
        ]

        report = BaselineComparisonReport(
            agent_name="TestAgent",
            agent_mean_score=0.8,
            comparisons=comparisons,
            best_baseline="baseline1",
            best_baseline_score=0.5,
            improvement_over_best=0.3,
            passes_all_baselines=True,
        )

        formatted = manager.format_comparison_report(report)

        assert "TestAgent" in formatted
        assert "0.8" in formatted
        assert "baseline1" in formatted
        assert "Yes" in formatted  # Passes all baselines


class TestABCCompliance:
    """Tests for ABC compliance (R.12, R.13)."""

    def test_abc_r_12_baseline_comparison(self):
        """Test ABC R.12: Reports results relative to baselines."""
        from evaris.metrics.exact_match import ExactMatchMetric

        manager = BaselineManager()

        # Create evaluation result
        results = [
            TestResult(
                test_case=TestCase(
                    input=f"test{i}", expected=f"output{i}", actual_output=f"output{i}"
                ),
                output=f"output{i}",
                metrics=[MetricResult(name="test", score=0.8, passed=True, metadata={})],
                latency_ms=10.0,
            )
            for i in range(30)
        ]

        eval_result = EvalResult(
            name="test",
            results=results,
            passed=30,
            failed=0,
            total=30,
            accuracy=1.0,
            avg_latency_ms=10.0,
        )

        metrics = [ExactMatchMetric()]
        report = manager.compare_with_baselines(eval_result, metrics)

        # Should compare with all baselines (now 5: random, do_nothing, trivial, majority, constant)
        assert len(report.comparisons) >= 5
        assert report.best_baseline is not None

    def test_abc_r_13_meaningful_improvement(self):
        """Test ABC R.13: Validates meaningful improvement over baselines."""
        manager = BaselineManager()

        # Small improvement (not meaningful)
        comp1 = manager.compare_with_baseline(
            agent_score=0.52, baseline_name="baseline", baseline_score=0.5
        )

        assert comp1.is_meaningful is False  # Below threshold

        # Large improvement (meaningful)
        comp2 = manager.compare_with_baseline(
            agent_score=0.8, baseline_name="baseline", baseline_score=0.5
        )

        assert comp2.is_meaningful is True  # Above threshold

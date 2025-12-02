"""Tests for statistics module."""

import pytest

from evaris.statistics import (
    ConfidenceInterval,
    StatisticalAnalyzer,
    StatisticalConfig,
    StatisticalReport,
    StatisticalSignificance,
)
from evaris.types import EvalResult, MetricResult, TestCase, TestResult


class TestStatisticalConfig:
    """Tests for StatisticalConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = StatisticalConfig()

        assert config.confidence_level == 0.95
        assert config.min_sample_size == 30
        assert config.bootstrap_samples == 1000
        assert config.report_effect_size is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = StatisticalConfig(
            confidence_level=0.99,
            min_sample_size=50,
            bootstrap_samples=5000,
            report_effect_size=False,
        )

        assert config.confidence_level == 0.99
        assert config.min_sample_size == 50
        assert config.bootstrap_samples == 5000
        assert config.report_effect_size is False


class TestConfidenceInterval:
    """Tests for ConfidenceInterval."""

    def test_confidence_interval_creation(self):
        """Test creating confidence interval."""
        ci = ConfidenceInterval(
            lower=0.7,
            upper=0.9,
            point_estimate=0.8,
            confidence_level=0.95,
            method="bootstrap",
        )

        assert ci.lower == 0.7
        assert ci.upper == 0.9
        assert ci.point_estimate == 0.8
        assert ci.confidence_level == 0.95
        assert ci.method == "bootstrap"


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer."""

    def test_analyzer_default_config(self):
        """Test analyzer with default config."""
        analyzer = StatisticalAnalyzer()

        assert analyzer.config.confidence_level == 0.95

    def test_analyzer_custom_config(self):
        """Test analyzer with custom config."""
        config = StatisticalConfig(confidence_level=0.99)
        analyzer = StatisticalAnalyzer(config)

        assert analyzer.config.confidence_level == 0.99

    def test_compute_mean_normal(self):
        """Test computing mean of scores."""
        analyzer = StatisticalAnalyzer()
        scores = [0.8, 0.9, 0.7, 0.85, 0.95]

        mean = analyzer._compute_mean(scores)

        assert mean == pytest.approx(0.84, abs=0.01)

    def test_compute_mean_empty(self):
        """Test computing mean of empty list."""
        analyzer = StatisticalAnalyzer()

        mean = analyzer._compute_mean([])

        assert mean == 0.0

    def test_compute_median_odd_count(self):
        """Test computing median with odd count."""
        analyzer = StatisticalAnalyzer()
        scores = [0.7, 0.8, 0.9]

        median = analyzer._compute_median(scores)

        assert median == 0.8

    def test_compute_median_even_count(self):
        """Test computing median with even count."""
        analyzer = StatisticalAnalyzer()
        scores = [0.7, 0.8, 0.85, 0.9]

        median = analyzer._compute_median(scores)

        assert median == pytest.approx(0.825, abs=0.001)

    def test_compute_median_empty(self):
        """Test computing median of empty list."""
        analyzer = StatisticalAnalyzer()

        median = analyzer._compute_median([])

        assert median == 0.0

    def test_compute_std(self):
        """Test computing standard deviation."""
        analyzer = StatisticalAnalyzer()
        scores = [0.8, 0.9, 0.7, 0.85, 0.95]
        mean = analyzer._compute_mean(scores)

        std = analyzer._compute_std(scores, mean)

        assert std > 0.0
        assert std < 0.2  # Should be reasonably small

    def test_compute_std_single_value(self):
        """Test std with single value."""
        analyzer = StatisticalAnalyzer()
        scores = [0.8]
        mean = 0.8

        std = analyzer._compute_std(scores, mean)

        assert std == 0.0

    def test_compute_quartiles(self):
        """Test computing quartiles."""
        analyzer = StatisticalAnalyzer()
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        quartiles = analyzer._compute_quartiles(scores)

        assert "Q1" in quartiles
        assert "Q2" in quartiles
        assert "Q3" in quartiles
        assert quartiles["Q1"] < quartiles["Q2"] < quartiles["Q3"]

    def test_compute_quartiles_empty(self):
        """Test quartiles with empty list."""
        analyzer = StatisticalAnalyzer()

        quartiles = analyzer._compute_quartiles([])

        assert quartiles == {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0}

    def test_analyze_scores_normal(self):
        """Test analyzing normal score distribution."""
        analyzer = StatisticalAnalyzer()
        scores = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95] * 10  # 60 samples

        report = analyzer.analyze_scores(scores)

        assert isinstance(report, StatisticalReport)
        assert report.sample_size == 60
        assert 0.7 <= report.mean <= 0.95
        assert 0.7 <= report.median <= 0.95
        assert report.std >= 0.0
        assert report.min_score == 0.7
        assert report.max_score == 0.95
        assert report.ci.lower <= report.mean <= report.ci.upper

    def test_analyze_scores_empty(self):
        """Test analyzing empty scores."""
        analyzer = StatisticalAnalyzer()

        report = analyzer.analyze_scores([])

        assert report.sample_size == 0
        assert report.mean == 0.0
        assert report.median == 0.0
        assert report.std == 0.0

    def test_analyze_scores_small_sample(self):
        """Test analyzing small sample (uses t-distribution)."""
        analyzer = StatisticalAnalyzer()
        scores = [0.8, 0.85, 0.9]

        report = analyzer.analyze_scores(scores)

        assert report.sample_size == 3
        assert report.ci.method == "t_distribution"

    def test_analyze_scores_large_sample(self):
        """Test analyzing large sample (uses bootstrap)."""
        config = StatisticalConfig(min_sample_size=30)
        analyzer = StatisticalAnalyzer(config)
        scores = [0.8 + i * 0.01 for i in range(50)]

        report = analyzer.analyze_scores(scores)

        assert report.sample_size == 50
        assert report.ci.method == "bootstrap"

    def test_analyze_evaluation(self):
        """Test analyzing evaluation result."""
        analyzer = StatisticalAnalyzer()

        # Create eval result
        results = []
        for i in range(30):
            tc = TestCase(input=f"input{i}", expected=f"output{i}", actual_output=f"output{i}")
            score = min(0.8 + i * 0.005, 1.0)  # Keep scores <= 1.0
            mr = MetricResult(name="test", score=score, passed=True, metadata={})
            results.append(
                TestResult(test_case=tc, output=f"output{i}", metrics=[mr], latency_ms=10.0)
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

        report = analyzer.analyze_evaluation(eval_result)

        assert isinstance(report, StatisticalReport)
        assert report.sample_size == 30

    def test_compute_cohens_d(self):
        """Test computing Cohen's d effect size."""
        analyzer = StatisticalAnalyzer()
        scores1 = [0.9, 0.95, 0.85, 0.92, 0.88]
        scores2 = [0.5, 0.55, 0.45, 0.52, 0.48]

        effect_size = analyzer._compute_cohens_d(scores1, scores2)

        assert effect_size > 0.0  # Positive effect (scores1 > scores2)
        assert effect_size > 2.0  # Large effect size

    def test_compute_cohens_d_no_difference(self):
        """Test Cohen's d with no difference."""
        analyzer = StatisticalAnalyzer()
        scores1 = [0.8, 0.85, 0.9]
        scores2 = [0.8, 0.85, 0.9]

        effect_size = analyzer._compute_cohens_d(scores1, scores2)

        assert effect_size == pytest.approx(0.0, abs=0.01)

    def test_compute_cohens_d_empty(self):
        """Test Cohen's d with empty lists."""
        analyzer = StatisticalAnalyzer()

        effect_size = analyzer._compute_cohens_d([], [0.8])

        assert effect_size == 0.0

    def test_compare_groups_significant_difference(self):
        """Test comparing groups with significant difference."""
        config = StatisticalConfig(report_effect_size=True)
        analyzer = StatisticalAnalyzer(config)

        scores1 = [0.9, 0.95, 0.85, 0.92, 0.88] * 10  # High scores
        scores2 = [0.5, 0.55, 0.45, 0.52, 0.48] * 10  # Low scores

        result = analyzer.compare_groups(scores1, scores2, "Agent A", "Agent B")

        assert isinstance(result, StatisticalSignificance)
        assert result.is_significant is True
        assert result.p_value < 0.05
        assert result.effect_size is not None
        assert result.effect_size > 0.0
        assert "Agent A" in result.interpretation

    def test_compare_groups_no_difference(self):
        """Test comparing groups with no difference."""
        analyzer = StatisticalAnalyzer()
        scores1 = [0.8, 0.85, 0.9] * 10
        scores2 = [0.8, 0.85, 0.9] * 10

        result = analyzer.compare_groups(scores1, scores2)

        assert result.is_significant is False
        assert "No significant difference" in result.interpretation

    def test_compare_groups_empty(self):
        """Test comparing empty groups."""
        analyzer = StatisticalAnalyzer()

        result = analyzer.compare_groups([], [0.8, 0.9])

        assert result.is_significant is False
        assert result.test_name == "insufficient_data"

    def test_compare_groups_without_effect_size(self):
        """Test comparison without effect size."""
        config = StatisticalConfig(report_effect_size=False)
        analyzer = StatisticalAnalyzer(config)

        scores1 = [0.9, 0.95]
        scores2 = [0.5, 0.55]

        result = analyzer.compare_groups(scores1, scores2)

        assert result.effect_size is None

    def test_format_report(self):
        """Test formatting statistical report."""
        analyzer = StatisticalAnalyzer()
        scores = [0.8, 0.85, 0.9, 0.95]

        report = analyzer.analyze_scores(scores)
        formatted = analyzer.format_report(report)

        assert "Statistical Report" in formatted
        assert "Mean:" in formatted
        assert "Median:" in formatted
        assert "CI:" in formatted
        assert str(report.sample_size) in formatted


class TestABCCompliance:
    """Tests for ABC compliance (R.10)."""

    def test_abc_r_10_confidence_intervals(self):
        """Test ABC R.10: Reports statistical significance with confidence intervals."""
        analyzer = StatisticalAnalyzer()
        scores = [0.8, 0.85, 0.9, 0.95] * 15  # 60 samples

        report = analyzer.analyze_scores(scores)

        # Should have confidence interval
        assert report.ci is not None
        assert report.ci.lower <= report.mean <= report.ci.upper
        assert report.ci.confidence_level == 0.95

    def test_abc_r_10_statistical_tests(self):
        """Test ABC R.10: Performs proper statistical tests."""
        analyzer = StatisticalAnalyzer()

        # Two significantly different groups
        high_scores = [0.9, 0.95, 0.92, 0.88] * 10
        low_scores = [0.5, 0.55, 0.52, 0.48] * 10

        result = analyzer.compare_groups(high_scores, low_scores)

        # Should detect significance
        assert result.is_significant is True
        assert result.p_value < 0.05
        assert result.test_name == "welch_t_test"

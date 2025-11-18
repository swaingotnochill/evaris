"""Statistical significance reporting and confidence intervals.

This module provides tools for rigorous statistical analysis of evaluation results.
Implements ABC check R.10 for statistical significance reporting.
"""

import math
from typing import Optional

from pydantic import BaseModel, Field

from evaris.types import EvalResult


class StatisticalConfig(BaseModel):
    """Configuration for statistical analysis."""

    confidence_level: float = Field(
        default=0.95, description="Confidence level for intervals (0.0-1.0)"
    )
    min_sample_size: int = Field(default=30, description="Minimum sample size for valid statistics")
    bootstrap_samples: int = Field(
        default=1000, description="Number of bootstrap samples for CI estimation"
    )
    report_effect_size: bool = Field(
        default=True, description="Report effect size (Cohen's d) for comparisons"
    )


class ConfidenceInterval(BaseModel):
    """Confidence interval for a statistic."""

    lower: float = Field(description="Lower bound of confidence interval")
    upper: float = Field(description="Upper bound of confidence interval")
    point_estimate: float = Field(description="Point estimate (mean/median)")
    confidence_level: float = Field(description="Confidence level (e.g., 0.95)")
    method: str = Field(description="Method used to compute CI (e.g., 'bootstrap')")


class StatisticalSignificance(BaseModel):
    """Statistical significance test result."""

    p_value: float = Field(description="P-value from statistical test")
    is_significant: bool = Field(description="Whether result is statistically significant")
    test_name: str = Field(description="Name of statistical test used")
    effect_size: Optional[float] = Field(default=None, description="Effect size (e.g., Cohen's d)")
    interpretation: str = Field(description="Human-readable interpretation")


class StatisticalReport(BaseModel):
    """Comprehensive statistical report for evaluation results."""

    mean: float = Field(description="Mean score")
    median: float = Field(description="Median score")
    std: float = Field(description="Standard deviation")
    ci: ConfidenceInterval = Field(description="Confidence interval for mean")
    sample_size: int = Field(description="Number of samples")
    min_score: float = Field(description="Minimum score")
    max_score: float = Field(description="Maximum score")
    quartiles: dict[str, float] = Field(description="Quartiles (Q1, Q2, Q3)")


class StatisticalAnalyzer:
    """Statistical analyzer for evaluation results.

    Computes confidence intervals, statistical significance tests, and
    comprehensive statistical reports for evaluation results.

    ABC Compliance:
    - R.10: Reports statistical significance with confidence intervals

    Example:
        >>> from evaris.statistics import StatisticalAnalyzer, StatisticalConfig
        >>> from evaris.types import EvalResult, MetricResult
        >>>
        >>> config = StatisticalConfig(confidence_level=0.95)
        >>> analyzer = StatisticalAnalyzer(config)
        >>>
        >>> # Analyze scores
        >>> scores = [0.8, 0.9, 0.7, 0.85, 0.95]
        >>> report = analyzer.analyze_scores(scores)
        >>> print(f"Mean: {report.mean:.2f} [{report.ci.lower:.2f}, {report.ci.upper:.2f}]")
    """

    def __init__(self, config: Optional[StatisticalConfig] = None):
        """Initialize statistical analyzer.

        Args:
            config: Configuration for statistical analysis. If None, uses defaults.
        """
        self.config = config or StatisticalConfig()

    def _compute_mean(self, scores: list[float]) -> float:
        """Compute mean of scores.

        Args:
            scores: List of scores

        Returns:
            Mean score
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _compute_median(self, scores: list[float]) -> float:
        """Compute median of scores.

        Args:
            scores: List of scores

        Returns:
            Median score
        """
        if not scores:
            return 0.0

        sorted_scores = sorted(scores)
        n = len(sorted_scores)

        if n % 2 == 0:
            return (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
        else:
            return sorted_scores[n // 2]

    def _compute_std(self, scores: list[float], mean: float) -> float:
        """Compute standard deviation of scores.

        Args:
            scores: List of scores
            mean: Mean of scores

        Returns:
            Standard deviation
        """
        if len(scores) < 2:
            return 0.0

        variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
        return math.sqrt(variance)

    def _compute_quartiles(self, scores: list[float]) -> dict[str, float]:
        """Compute quartiles of scores.

        Args:
            scores: List of scores

        Returns:
            Dictionary with Q1, Q2 (median), Q3
        """
        if not scores:
            return {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0}

        sorted_scores = sorted(scores)
        n = len(sorted_scores)

        def percentile(p: float) -> float:
            """Compute percentile using linear interpolation."""
            k = (n - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_scores[int(k)]
            d0 = sorted_scores[int(f)] * (c - k)
            d1 = sorted_scores[int(c)] * (k - f)
            return d0 + d1

        return {
            "Q1": percentile(0.25),
            "Q2": percentile(0.50),
            "Q3": percentile(0.75),
        }

    def _bootstrap_ci(self, scores: list[float], confidence_level: float) -> ConfidenceInterval:
        """Compute confidence interval using bootstrap.

        ABC R.10: Computes confidence intervals for mean.

        Args:
            scores: List of scores
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            Confidence interval
        """
        import random

        if len(scores) < self.config.min_sample_size:
            # Not enough samples, use t-distribution approximation
            return self._t_distribution_ci(scores, confidence_level)

        # Bootstrap resampling
        bootstrap_means: list[float] = []
        for _ in range(self.config.bootstrap_samples):
            sample = random.choices(scores, k=len(scores))
            bootstrap_means.append(self._compute_mean(sample))

        # Compute percentiles for CI
        sorted_means = sorted(bootstrap_means)
        alpha = 1 - confidence_level
        lower_idx = int(alpha / 2 * len(sorted_means))
        upper_idx = int((1 - alpha / 2) * len(sorted_means))

        return ConfidenceInterval(
            lower=sorted_means[lower_idx],
            upper=sorted_means[upper_idx],
            point_estimate=self._compute_mean(scores),
            confidence_level=confidence_level,
            method="bootstrap",
        )

    def _t_distribution_ci(
        self, scores: list[float], confidence_level: float
    ) -> ConfidenceInterval:
        """Compute confidence interval using t-distribution.

        Args:
            scores: List of scores
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            Confidence interval
        """
        if len(scores) < 2:
            mean = self._compute_mean(scores)
            return ConfidenceInterval(
                lower=mean,
                upper=mean,
                point_estimate=mean,
                confidence_level=confidence_level,
                method="single_sample",
            )

        mean = self._compute_mean(scores)
        std = self._compute_std(scores, mean)
        n = len(scores)

        # Use t-distribution critical value
        # Approximation for common confidence levels
        if confidence_level == 0.95:
            # t-critical for 95% CI (approximate)
            if n >= 30:
                t_critical = 1.96  # z-score for large samples
            else:
                # Conservative estimate
                t_critical = 2.0 + (30 - n) * 0.05
        elif confidence_level == 0.99:
            t_critical = 2.576 if n >= 30 else 3.0
        else:
            # Default approximation
            t_critical = 2.0

        margin = t_critical * (std / math.sqrt(n))

        return ConfidenceInterval(
            lower=mean - margin,
            upper=mean + margin,
            point_estimate=mean,
            confidence_level=confidence_level,
            method="t_distribution",
        )

    def analyze_scores(self, scores: list[float]) -> StatisticalReport:
        """Analyze list of scores and compute statistics.

        ABC R.10: Computes comprehensive statistics with confidence intervals.

        Args:
            scores: List of scores to analyze

        Returns:
            StatisticalReport with comprehensive statistics
        """
        if not scores:
            # Return empty report
            empty_ci = ConfidenceInterval(
                lower=0.0,
                upper=0.0,
                point_estimate=0.0,
                confidence_level=self.config.confidence_level,
                method="empty",
            )
            return StatisticalReport(
                mean=0.0,
                median=0.0,
                std=0.0,
                ci=empty_ci,
                sample_size=0,
                min_score=0.0,
                max_score=0.0,
                quartiles={"Q1": 0.0, "Q2": 0.0, "Q3": 0.0},
            )

        mean = self._compute_mean(scores)
        median = self._compute_median(scores)
        std = self._compute_std(scores, mean)
        ci = self._bootstrap_ci(scores, self.config.confidence_level)
        quartiles = self._compute_quartiles(scores)

        return StatisticalReport(
            mean=mean,
            median=median,
            std=std,
            ci=ci,
            sample_size=len(scores),
            min_score=min(scores),
            max_score=max(scores),
            quartiles=quartiles,
        )

    def analyze_evaluation(self, result: EvalResult) -> StatisticalReport:
        """Analyze evaluation result and compute statistics.

        ABC R.10: Provides statistical analysis of evaluation results.

        Args:
            result: Evaluation result to analyze

        Returns:
            StatisticalReport with statistics
        """
        # Extract scores from results - average metrics per test
        scores = []
        for r in result.results:
            if r.metrics:
                test_score = sum(m.score for m in r.metrics) / len(r.metrics)
                scores.append(test_score)
        return self.analyze_scores(scores)

    def _compute_cohens_d(self, scores1: list[float], scores2: list[float]) -> float:
        """Compute Cohen's d effect size.

        Args:
            scores1: First group of scores
            scores2: Second group of scores

        Returns:
            Cohen's d effect size
        """
        if not scores1 or not scores2:
            return 0.0

        mean1 = self._compute_mean(scores1)
        mean2 = self._compute_mean(scores2)
        std1 = self._compute_std(scores1, mean1)
        std2 = self._compute_std(scores2, mean2)

        # Pooled standard deviation
        n1, n2 = len(scores1), len(scores2)
        if n1 + n2 <= 2:
            return 0.0
        pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    def compare_groups(
        self,
        scores1: list[float],
        scores2: list[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
    ) -> StatisticalSignificance:
        """Compare two groups of scores for statistical significance.

        ABC R.10: Tests statistical significance of differences.

        Args:
            scores1: First group of scores
            scores2: Second group of scores
            group1_name: Name of first group
            group2_name: Name of second group

        Returns:
            StatisticalSignificance with test results
        """
        if not scores1 or not scores2:
            return StatisticalSignificance(
                p_value=1.0,
                is_significant=False,
                test_name="insufficient_data",
                interpretation="Insufficient data for comparison",
            )

        # Compute means and standard deviations
        mean1 = self._compute_mean(scores1)
        mean2 = self._compute_mean(scores2)
        std1 = self._compute_std(scores1, mean1)
        std2 = self._compute_std(scores2, mean2)
        n1, n2 = len(scores1), len(scores2)

        # Welch's t-test (unequal variances)
        if std1 == 0 and std2 == 0:
            # Perfect scores, no variance
            p_value = 1.0 if mean1 == mean2 else 0.0
            t_stat = 0.0
        else:
            se = math.sqrt((std1**2 / n1) + (std2**2 / n2))
            if se == 0:
                t_stat = 0.0
                p_value = 1.0
            else:
                t_stat = (mean1 - mean2) / se

                # Approximate p-value using standard normal (for large samples)
                # For small samples, this is conservative
                z_score = abs(t_stat)
                # Two-tailed p-value approximation
                if z_score > 3.5:
                    p_value = 0.0005
                elif z_score > 2.576:
                    p_value = 0.01
                elif z_score > 1.96:
                    p_value = 0.05
                elif z_score > 1.645:
                    p_value = 0.10
                else:
                    # Very rough approximation
                    p_value = 0.5

        # Compute effect size
        effect_size = None
        if self.config.report_effect_size:
            effect_size = self._compute_cohens_d(scores1, scores2)

        # Determine significance
        is_significant = p_value < 0.05

        # Generate interpretation
        if is_significant:
            direction = "higher" if mean1 > mean2 else "lower"
            effect_size_str = f"{effect_size:.2f}" if effect_size is not None else "N/A"
            interpretation = (
                f"{group1_name} scored significantly {direction} than {group2_name} "
                f"(p={p_value:.4f}, effect size={effect_size_str})"
            )
        else:
            interpretation = (
                f"No significant difference between {group1_name} and {group2_name} "
                f"(p={p_value:.4f})"
            )

        return StatisticalSignificance(
            p_value=p_value,
            is_significant=is_significant,
            test_name="welch_t_test",
            effect_size=effect_size,
            interpretation=interpretation,
        )

    def format_report(self, report: StatisticalReport) -> str:
        """Format statistical report as human-readable string.

        Args:
            report: Statistical report to format

        Returns:
            Formatted string
        """
        ci_pct = int(report.ci.confidence_level * 100)
        return f"""Statistical Report (n={report.sample_size}):
  Mean: {report.mean:.4f} ({ci_pct}% CI: [{report.ci.lower:.4f}, {report.ci.upper:.4f}])
  Median: {report.median:.4f}
  Std Dev: {report.std:.4f}
  Range: [{report.min_score:.4f}, {report.max_score:.4f}]
  Quartiles: Q1={report.quartiles['Q1']:.4f}, Q2={report.quartiles['Q2']:.4f}, """
        f"""Q3={report.quartiles['Q3']:.4f}"""

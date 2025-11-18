"""Baseline comparison utilities for agent evaluation.

This module provides tools for comparing agent performance against baselines
and reporting relative improvements. Implements ABC checks R.12 and R.13.
"""

from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from evaris.types import BaseMetric, EvalResult, TestCase


class BaselineConfig(BaseModel):
    """Configuration for ABC baseline comparisons.

    It configures how agent's perfomance is compared against simple
    baseline agents (like random output, do-nothing, echo input).
    This is part of ABC (Agent-Computer Benchmark) compliance -
    specifically checks R.12 and R.13.
    """

    # Whether to mandate baseline comparisons in evaluation reports.
    # Set to False to skip baseline validation (not recommended for production).
    require_baselines: bool = Field(
        default=True, description="Require baseline comparisons in reports"
    )

    # List of baseline agent types to compare against.
    # Available types: "random" (random outputs), "do_nothing" (empty output),
    # "trivial" (echo input), "majority" (most common expected output),
    # "constant" (first expected output). Custom baselines can be added
    # via register_baseline().
    baseline_types: list[str] = Field(
        default_factory=lambda: ["random", "do_nothing", "trivial", "majority", "constant"],
        description="Types of baselines to include",
    )

    # Minimum absolute score improvement (0-1 scale) required to consider
    # the agent's performance "meaningful". Default 0.05 = 5% improvement.
    # ABC R.13 compliance: ensures statistically significant improvement.
    min_improvement_threshold: float = Field(
        default=0.05, description="Minimum improvement over baseline to be meaningful"
    )


class BaselineAgent(BaseModel):
    """Baseline agent for comparison.

    Represents a simple agent strategy used as a performance baseline.
    Each baseline has a name, description, and callable function.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Unique identifier for this baseline (e.g., "random", "do_nothing")
    name: str = Field(description="Name of baseline agent")

    description: str = Field(description="Description of baseline strategy")

    # Callable that takes TestCase and returns output. Signature: (TestCase) -> Any
    agent_fn: Any = Field(description="Baseline agent function")


class BaselineComparison(BaseModel):
    """Comparison between agent and baseline.

    Contains metrics comparing a single agent's performance against one baseline.
    Includes both absolute and relative improvement calculations.
    """

    baseline_name: str = Field(description="Name of baseline")
    agent_score: float = Field(description="Agent's score")
    baseline_score: float = Field(description="Baseline's score")
    absolute_improvement: float = Field(description="Absolute improvement (agent - baseline)")
    relative_improvement: float = Field(
        description="Relative improvement ((agent - baseline) / baseline)"
    )
    is_meaningful: bool = Field(description="Whether improvement is meaningful (above threshold)")
    interpretation: str = Field(description="Human-readable interpretation")


class BaselineComparisonReport(BaseModel):
    """Comprehensive baseline comparison report.

    Aggregates comparisons across all baselines to provide a complete
    picture of agent performance relative to simple strategies.
    """

    agent_name: str = Field(description="Name of agent being evaluated")
    agent_mean_score: float = Field(description="Agent's mean score")
    comparisons: list[BaselineComparison] = Field(description="Comparisons with each baseline")
    best_baseline: str = Field(description="Name of best-performing baseline")
    best_baseline_score: float = Field(description="Score of best baseline")
    improvement_over_best: float = Field(description="Improvement over best baseline")
    passes_all_baselines: bool = Field(description="Whether agent beats all baselines")


# Rebuild EvalResult model now that BaselineComparisonReport is defined
# This resolves the forward reference from types.py
EvalResult.model_rebuild()


class BaselineManager:
    """Manager for baseline agents and comparisons.

    Provides tools for running baseline agents, comparing results,
    and generating comparison reports.

    ABC Compliance:
    - R.12: Reports results relative to relevant baseline agents
    - R.13: Ensures agent significantly improves over baselines

    Example:
        >>> from evaris.baselines import BaselineManager, BaselineConfig
        >>> from evaris.types import TestCase
        >>>
        >>> config = BaselineConfig(require_baselines=True)
        >>> manager = BaselineManager(config)
        >>>
        >>> # Register baselines
        >>> manager.register_baseline("random", "Random output", lambda tc: "random")
        >>>
        >>> # Compare agent with baselines
        >>> agent_result = EvaluationResult(...)
        >>> report = manager.compare_with_baselines(agent_result, "MyAgent")
    """

    def __init__(self, config: Optional[BaselineConfig] = None):
        """Initialize baseline manager.

        Args:
            config: Configuration for baseline comparisons. If None, uses defaults.
        """
        self.config = config or BaselineConfig()
        self.baselines: dict[str, BaselineAgent] = {}
        self._register_default_baselines()

    def _register_default_baselines(self) -> None:
        """Register default baseline agents.

        ABC R.12: Provides standard baselines for comparison.
        """
        # Random baseline - returns random output
        if "random" in self.config.baseline_types:

            def random_baseline(test_case: TestCase) -> str:
                """Random baseline agent."""
                import random

                options = ["yes", "no", "maybe", "unknown", "42", ""]
                return random.choice(options)

            self.register_baseline(
                "random",
                "Outputs random response from common options",
                random_baseline,
            )

        # Do-nothing baseline - returns empty output
        if "do_nothing" in self.config.baseline_types:

            def do_nothing_baseline(test_case: TestCase) -> str:
                """Do-nothing baseline agent."""
                return ""

            self.register_baseline(
                "do_nothing",
                "Returns empty output (does nothing)",
                do_nothing_baseline,
            )

        # Trivial baseline - returns input as output
        if "trivial" in self.config.baseline_types:

            def trivial_baseline(test_case: TestCase) -> str:
                """Trivial baseline agent."""
                return str(test_case.input)

            self.register_baseline(
                "trivial",
                "Returns input as output (echo)",
                trivial_baseline,
            )

    def _register_smart_baselines(self, test_cases: list[TestCase]) -> None:
        """Register smart baselines that depend on test case data.

        These baselines need to analyze the test set to determine their strategy.

        Args:
            test_cases: Test cases to analyze for smart baseline strategies
        """
        # Majority baseline - returns most common expected output
        if "majority" in self.config.baseline_types:
            # Find the most common expected output
            from collections import Counter

            expected_outputs = [tc.expected for tc in test_cases if tc.expected is not None]
            if expected_outputs:
                # Get most common output
                counter = Counter(str(exp) for exp in expected_outputs)
                majority_output = counter.most_common(1)[0][0]
            else:
                majority_output = ""

            def majority_baseline(test_case: TestCase) -> Any:
                """Majority baseline agent."""
                return majority_output

            self.register_baseline(
                "majority",
                f"Returns most common expected output: '{majority_output}'",
                majority_baseline,
            )

        # Constant baseline - returns first expected output
        if "constant" in self.config.baseline_types:
            # Find first non-None expected output
            constant_output = ""
            for tc in test_cases:
                if tc.expected is not None:
                    constant_output = tc.expected
                    break

            def constant_baseline(test_case: TestCase) -> Any:
                """Constant baseline agent."""
                return constant_output

            self.register_baseline(
                "constant",
                f"Returns constant value: '{constant_output}'",
                constant_baseline,
            )

    def register_baseline(
        self, name: str, description: str, agent_fn: Callable[[TestCase], Any]
    ) -> None:
        """Register a baseline agent.

        Args:
            name: Name of baseline
            description: Description of baseline strategy
            agent_fn: Function that takes TestCase and returns output
        """
        self.baselines[name] = BaselineAgent(name=name, description=description, agent_fn=agent_fn)

    def run_baseline(self, baseline_name: str, test_cases: list[TestCase]) -> list[Any]:
        """Run a baseline agent on test cases.

        Args:
            baseline_name: Name of baseline to run
            test_cases: Test cases to evaluate

        Returns:
            List of baseline outputs

        Raises:
            ValueError: If baseline not found
        """
        if baseline_name not in self.baselines:
            raise ValueError(
                f"Baseline '{baseline_name}' not found. "
                f"Available: {list(self.baselines.keys())}"
            )

        baseline = self.baselines[baseline_name]
        outputs = []

        for test_case in test_cases:
            try:
                output = baseline.agent_fn(test_case)
                outputs.append(output)
            except Exception as e:
                # Baseline failed, use empty output and log error
                import logging

                logging.warning(
                    f"Baseline '{baseline_name}' failed on test case: {type(e).__name__}: {e}"
                )
                outputs.append("")

        return outputs

    def evaluate_baseline(
        self,
        baseline_name: str,
        test_cases: list[TestCase],
        metrics: list[BaseMetric],
    ) -> float:
        """Evaluate a baseline agent using the same metrics as the agent.

        Args:
            baseline_name: Name of baseline to evaluate
            test_cases: Test cases to evaluate
            metrics: List of metrics to evaluate (same as used for agent)

        Returns:
            Mean score across all test cases and metrics
        """
        outputs = self.run_baseline(baseline_name, test_cases)

        all_scores = []
        for test_case, output in zip(test_cases, outputs):
            baseline_test_case = TestCase(
                input=test_case.input,
                actual_output=output,
                expected=test_case.expected,
                metadata=test_case.metadata,
            )

            for metric in metrics:
                try:
                    # Prefer score() method to avoid asyncio.run() overhead in measure()
                    # score() is the legacy sync interface - more efficient for baseline evaluation
                    if hasattr(metric, "score"):
                        result = metric.score(baseline_test_case, output)
                        all_scores.append(result.score)
                    # Fallback to measure() if score() not available
                    elif hasattr(metric, "measure"):
                        result = metric.measure(baseline_test_case)
                        all_scores.append(result.score)
                    else:
                        # Metric doesn't have expected interface
                        all_scores.append(0.0)
                except Exception as e:
                    # Metric failed, assume 0 score and log error
                    import logging

                    logging.warning(
                        f"Metric '{metric.__class__.__name__}' failed during baseline "
                        f"'{baseline_name}' evaluation: {type(e).__name__}: {e}"
                    )
                    all_scores.append(0.0)

        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    def compare_with_baseline(
        self,
        agent_score: float,
        baseline_name: str,
        baseline_score: float,
    ) -> BaselineComparison:
        """Compare agent score with baseline score.

        ABC R.13: Validates meaningful improvement over baseline.

        Args:
            agent_score: Agent's mean score
            baseline_name: Name of baseline
            baseline_score: Baseline's mean score

        Returns:
            BaselineComparison with comparison details
        """
        absolute_improvement = agent_score - baseline_score

        if baseline_score == 0:
            if agent_score > 0:
                relative_improvement = float("inf")
            else:
                relative_improvement = 0.0
        else:
            relative_improvement = absolute_improvement / baseline_score

        is_meaningful = absolute_improvement >= self.config.min_improvement_threshold

        if absolute_improvement > 0:
            if is_meaningful:
                interpretation = (
                    f"Agent significantly outperforms {
                        baseline_name} baseline "
                    f"(+{absolute_improvement:.2%} absolute, "
                    f"+{relative_improvement:.1%} relative)"
                )
            else:
                interpretation = (
                    f"Agent slightly outperforms {baseline_name} baseline "
                    f"(+{absolute_improvement:.2%}, below threshold)"
                )
        elif absolute_improvement < 0:
            interpretation = (
                f"Agent underperforms {
                    baseline_name} baseline "
                f"({absolute_improvement:.2%})"
            )
        else:
            interpretation = f"Agent performs equally to {
                baseline_name} baseline"

        return BaselineComparison(
            baseline_name=baseline_name,
            agent_score=agent_score,
            baseline_score=baseline_score,
            absolute_improvement=absolute_improvement,
            relative_improvement=relative_improvement,
            is_meaningful=is_meaningful,
            interpretation=interpretation,
        )

    def compare_with_baselines(
        self,
        agent_result: EvalResult,
        metrics: list[BaseMetric],
        agent_name: str = "Agent",
    ) -> BaselineComparisonReport:
        """Compare agent with all registered baselines using the same metrics.

        ABC Compliance:
        - R.12: Reports results relative to baselines
        - R.13: Validates improvement over baselines

        Args:
            agent_result: Agent's evaluation result
            metrics: List of metrics used to evaluate the agent (baselines use same metrics)
            agent_name: Name of agent

        Returns:
            BaselineComparisonReport with all comparisons

        Raises:
            ValueError: If no baselines registered or no metrics provided
        """
        if not metrics:
            raise ValueError("No metrics provided for baseline comparison")

        # Extract test cases from evaluation result
        test_cases = [r.test_case for r in agent_result.results]

        # Register smart baselines that depend on test case data
        self._register_smart_baselines(test_cases)

        if not self.baselines:
            raise ValueError("No baselines registered")

        # Compute agent's mean score from metrics
        agent_scores = []
        for r in agent_result.results:
            if r.metrics:
                # Average of all metric scores for this test
                test_score = sum(m.score for m in r.metrics) / len(r.metrics)
                agent_scores.append(test_score)
        agent_mean_score = sum(agent_scores) / len(agent_scores) if agent_scores else 0.0

        # Evaluate each baseline using the same metrics as the agent
        comparisons: list[BaselineComparison] = []
        baseline_scores: dict[str, float] = {}

        for baseline_name in self.baselines:
            baseline_score = self.evaluate_baseline(baseline_name, test_cases, metrics)
            baseline_scores[baseline_name] = baseline_score

            comparison = self.compare_with_baseline(agent_mean_score, baseline_name, baseline_score)
            comparisons.append(comparison)

        if not baseline_scores:
            raise ValueError(
                "No baselines available for comparison. "
                "Register baselines using register_baseline() or configure baseline_types."
            )

        best_baseline_name = max(baseline_scores, key=baseline_scores.get)  # type: ignore
        best_baseline_score = baseline_scores[best_baseline_name]

        # Check if agent beats all baselines
        passes_all = all(c.absolute_improvement > 0 for c in comparisons)

        improvement_over_best = agent_mean_score - best_baseline_score

        return BaselineComparisonReport(
            agent_name=agent_name,
            agent_mean_score=agent_mean_score,
            comparisons=comparisons,
            best_baseline=best_baseline_name,
            best_baseline_score=best_baseline_score,
            improvement_over_best=improvement_over_best,
            passes_all_baselines=passes_all,
        )

    def format_comparison_report(self, report: BaselineComparisonReport) -> str:
        """Format baseline comparison report as human-readable string.

        Args:
            report: Baseline comparison report

        Returns:
            Formatted string
        """
        lines = [
            f"Baseline Comparison Report: {report.agent_name}",
            f"Agent Mean Score: {report.agent_mean_score:.4f}",
            "",
            "Comparisons with Baselines:",
        ]

        for comp in report.comparisons:
            if comp.is_meaningful:
                status = "[PASS]"
            elif comp.absolute_improvement > 0:
                status = "[WEAK]"
            else:
                status = "[FAIL]"
            lines.append(
                f"  {status} {comp.baseline_name}: {comp.baseline_score:.4f} "
                f"(Δ = {comp.absolute_improvement:+.4f})"
            )

        lines.extend(
            [
                "",
                f"Best Baseline: {
                    report.best_baseline} ({report.best_baseline_score:.4f})",
                f"Improvement over Best: {report.improvement_over_best:+.4f}",
                f"Passes All Baselines: {
                    'Yes' if report.passes_all_baselines else 'No'}",
            ]
        )

        return "\n".join(lines)

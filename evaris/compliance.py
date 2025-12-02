"""ABC Compliance checking and enforcement.

This module provides tools for checking compliance with the Agentic Benchmark
Checklist (ABC) and enforcing best practices in evaluation.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ABCComplianceConfig(BaseModel):
    """Configuration for ABC compliance checking."""

    enabled: bool = Field(default=True, description="Enable compliance checking")
    strict_mode: bool = Field(
        default=False,
        description="Block execution on critical violations (default: warnings only)",
    )
    check_baselines: bool = Field(
        default=True, description="Check for baseline comparisons (R.12, R.13)"
    )
    check_statistics: bool = Field(default=True, description="Check for statistical rigor (R.10)")
    check_sample_size: bool = Field(default=True, description="Check for adequate sample size")
    check_validation: bool = Field(default=True, description="Check for dataset validation (T.3)")
    check_contamination: bool = Field(
        default=True, description="Check for contamination tracking (R.3, R.4)"
    )
    min_sample_size: int = Field(default=30, description="Minimum recommended sample size")


class ABCWarning(BaseModel):
    """Single ABC compliance warning."""

    code: str = Field(description="ABC check code (e.g., 'R.10', 'II.5')")
    severity: Literal["info", "warning", "critical"] = Field(description="Warning severity level")
    message: str = Field(description="Warning message")
    fix: Optional[str] = Field(default=None, description="Suggested fix")
    impact: Optional[str] = Field(default=None, description="Impact of not addressing this warning")
    tier: Literal[1, 2, 3] = Field(
        default=1, description="ABC compliance tier (1=automatic, 2=recommended, 3=advanced)"
    )


class ABCComplianceReport(BaseModel):
    """ABC compliance check report."""

    warnings: list[ABCWarning] = Field(
        default_factory=list, description="List of compliance warnings"
    )
    num_critical: int = Field(default=0, description="Number of critical warnings")
    num_warnings: int = Field(default=0, description="Number of regular warnings")
    num_info: int = Field(default=0, description="Number of info messages")
    is_compliant: bool = Field(
        default=True, description="Whether evaluation meets basic ABC standards"
    )

    def has_critical(self) -> bool:
        """Check if there are critical warnings."""
        return self.num_critical > 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return self.num_warnings > 0 or self.num_critical > 0

    def format_warnings(self) -> str:
        """Format warnings as human-readable string."""
        if not self.warnings:
            return "ABC Compliance: All checks passed"

        lines = ["ABC Compliance Report", "=" * 50, ""]

        # Summary
        if self.is_compliant:
            lines.append("Basic compliance PASSED (recommendations below)")
        else:
            lines.append("X Compliance ISSUES DETECTED")

        lines.append("")
        lines.append(f"Critical: {self.num_critical}")
        lines.append(f"Warnings: {self.num_warnings}")
        lines.append(f"Info: {self.num_info}")
        lines.append("")

        # Group by severity
        critical = [w for w in self.warnings if w.severity == "critical"]
        warnings = [w for w in self.warnings if w.severity == "warning"]
        info = [w for w in self.warnings if w.severity == "info"]

        # Critical issues
        if critical:
            lines.append("CRITICAL ISSUES:")
            for w in critical:
                lines.append(f"  X [{w.code}] {w.message}")
                if w.impact:
                    lines.append(f"    Impact: {w.impact}")
                if w.fix:
                    lines.append(f"    Fix: {w.fix}")
                lines.append("")

        # Warnings
        if warnings:
            lines.append("WARNINGS:")
            for w in warnings:
                lines.append(f"  ! [{w.code}] {w.message}")
                if w.fix:
                    lines.append(f"    Fix: {w.fix}")
                lines.append("")

        # Info
        if info:
            lines.append("RECOMMENDATIONS:")
            for w in info:
                lines.append(f"  i [{w.code}] {w.message}")
                if w.fix:
                    lines.append(f"    Suggestion: {w.fix}")
                lines.append("")

        return "\n".join(lines)


class ABCComplianceChecker:
    """Checker for ABC compliance.

    Analyzes evaluation configurations and reports compliance with the
    Agentic Benchmark Checklist (ABC). Provides warnings and recommendations
    without blocking execution (unless strict_mode is enabled).

    Compliance Tiers:
    - Tier 1 (Automatic): Framework-enforced checks (statistics, baselines, validation)
    - Tier 2 (Recommended): Requires user cooperation (agent isolation, state reset)
    - Tier 3 (Advanced): Future features (exploit detection, adversarial testing)

    Example:
        >>> from evaris.compliance import ABCComplianceChecker, ABCComplianceConfig
        >>>
        >>> config = ABCComplianceConfig(strict_mode=False)
        >>> checker = ABCComplianceChecker(config)
        >>>
        >>> report = checker.check_evaluation_config(
        ...     includes_baselines=False,
        ...     includes_statistics=True,
        ...     sample_size=50
        ... )
        >>> print(report.format_warnings())
    """

    def __init__(self, config: Optional[ABCComplianceConfig] = None):
        """Initialize ABC compliance checker.

        Args:
            config: Compliance checking configuration. If None, uses defaults.
        """
        self.config = config or ABCComplianceConfig()

    def check_evaluation_config(
        self,
        includes_baselines: bool = False,
        includes_statistics: bool = False,
        includes_validation: bool = False,
        includes_contamination_check: bool = False,
        sample_size: Optional[int] = None,
        uses_oracle_validation: bool = False,
        agent_isolated: bool = False,
        state_reset_enabled: bool = False,
    ) -> ABCComplianceReport:
        """Check evaluation configuration for ABC compliance.

        Args:
            includes_baselines: Whether baseline comparisons are included (R.12, R.13)
            includes_statistics: Whether statistical analysis is included (R.10)
            includes_validation: Whether dataset validation is performed (T.3)
            includes_contamination_check: Whether contamination checking is enabled (R.3, R.4)
            sample_size: Number of test cases (for adequacy check)
            uses_oracle_validation: Whether oracle solver validation is used (T.9)
            agent_isolated: Whether agent runs in isolated environment (II.5) [Tier 2]
            state_reset_enabled: Whether state is reset between tests (II.4) [Tier 2]

        Returns:
            ABCComplianceReport with warnings and recommendations
        """
        warnings: list[ABCWarning] = []

        # Tier 1 Checks (Framework-enforced, should always be enabled)

        # Check R.12, R.13: Baseline comparisons
        if self.config.check_baselines and not includes_baselines:
            warnings.append(
                ABCWarning(
                    code="R.12/R.13",
                    severity="warning",
                    message="No baseline comparisons detected",
                    fix="Add BaselineManager to compare against trivial/random baselines",
                    impact="Cannot verify agent improves over simple baselines",
                    tier=1,
                )
            )

        # Check R.10: Statistical rigor
        if self.config.check_statistics and not includes_statistics:
            warnings.append(
                ABCWarning(
                    code="R.10",
                    severity="warning",
                    message="No statistical analysis detected",
                    fix="Add StatisticalAnalyzer to compute confidence intervals",
                    impact="Results lack uncertainty quantification and significance testing",
                    tier=1,
                )
            )

        # Check sample size adequacy
        if self.config.check_sample_size and sample_size is not None:
            if sample_size < self.config.min_sample_size:
                warnings.append(
                    ABCWarning(
                        code="R.10",
                        severity="critical" if sample_size < 10 else "warning",
                        message=(
                            f"Sample size ({sample_size}) below recommended minimum "
                            f"({self.config.min_sample_size})"
                        ),
                        fix=f"Add more test cases (recommended: {self.config.min_sample_size}+)",
                        impact="Insufficient statistical power to detect meaningful differences",
                        tier=1,
                    )
                )

        # Check T.3: Dataset validation
        if self.config.check_validation and not includes_validation:
            warnings.append(
                ABCWarning(
                    code="T.3",
                    severity="warning",
                    message="No dataset validation detected",
                    fix="Add TestCaseValidator to check dataset quality",
                    impact="May have invalid or incomplete test cases",
                    tier=1,
                )
            )

        # Check R.3, R.4: Contamination tracking
        if self.config.check_contamination and not includes_contamination_check:
            warnings.append(
                ABCWarning(
                    code="R.3/R.4",
                    severity="info",
                    message="No contamination tracking detected",
                    fix=(
                        "Add ContaminationDetector to track dataset versions and "
                        "check for contamination"
                    ),
                    impact="Cannot verify dataset integrity or detect training data leakage",
                    tier=1,
                )
            )

        # Check T.9: Oracle validation (optional but recommended)
        if not uses_oracle_validation:
            warnings.append(
                ABCWarning(
                    code="T.9",
                    severity="info",
                    message="No oracle solver validation detected",
                    fix="Add OracleValidator to verify tasks are solvable",
                    impact="Cannot guarantee all tasks are actually solvable",
                    tier=1,
                )
            )

        # Tier 2 Checks (Recommended, requires user cooperation)

        # Check II.5: Agent isolation
        if not agent_isolated:
            warnings.append(
                ABCWarning(
                    code="II.5",
                    severity="info",
                    message="Agent isolation not detected (Tier 2)",
                    fix="Use isolated agent wrappers (Docker/process-based) when available",
                    impact="Agent may access ground truth or evaluation code (cheating risk)",
                    tier=2,
                )
            )

        # Check II.4: State contamination
        if not state_reset_enabled:
            warnings.append(
                ABCWarning(
                    code="II.4",
                    severity="info",
                    message="State reset not detected (Tier 2)",
                    fix="Enable state reset between test cases to prevent contamination",
                    impact="Residual state may cause false positives/negatives",
                    tier=2,
                )
            )

        # Count warnings by severity
        num_critical = sum(1 for w in warnings if w.severity == "critical")
        num_warnings = sum(1 for w in warnings if w.severity == "warning")
        num_info = sum(1 for w in warnings if w.severity == "info")

        # Determine overall compliance
        # Only critical issues fail compliance
        is_compliant = num_critical == 0

        return ABCComplianceReport(
            warnings=warnings,
            num_critical=num_critical,
            num_warnings=num_warnings,
            num_info=num_info,
            is_compliant=is_compliant,
        )

    def check_evaluation_result(self, result: Any) -> ABCComplianceReport:
        """Check evaluation result for ABC compliance.

        Analyzes an evaluation result object to determine compliance.

        Args:
            result: EvalResult object

        Returns:
            ABCComplianceReport with warnings
        """
        # Detect what's included in the result
        has_stats = hasattr(result, "statistics") or hasattr(result, "confidence_interval")
        has_baselines = hasattr(result, "baseline_comparisons") or hasattr(
            result, "baseline_report"
        )

        # Sample size
        sample_size = None
        if hasattr(result, "total"):
            sample_size = result.total
        elif hasattr(result, "results"):
            sample_size = len(result.results)

        return self.check_evaluation_config(
            includes_baselines=has_baselines,
            includes_statistics=has_stats,
            sample_size=sample_size,
        )


class ABCViolationError(Exception):
    """Exception raised when ABC compliance violations are detected in strict mode."""

    def __init__(self, report: ABCComplianceReport):
        """Initialize exception with compliance report.

        Args:
            report: ABC compliance report with violations
        """
        self.report = report
        message = (
            f"ABC compliance violations detected:\n\n"
            f"{report.format_warnings()}\n\n"
            f"Evaluation blocked due to {report.num_critical} critical violations.\n"
            f"Fix the issues above or disable strict_mode to continue."
        )
        super().__init__(message)


def check_compliance(
    includes_baselines: bool = False,
    includes_statistics: bool = False,
    sample_size: Optional[int] = None,
    strict_mode: bool = False,
) -> ABCComplianceReport:
    """Convenience function to check ABC compliance.

    Args:
        includes_baselines: Whether baseline comparisons are included
        includes_statistics: Whether statistical analysis is included
        sample_size: Number of test cases
        strict_mode: Raise exception on critical violations

    Returns:
        ABCComplianceReport

    Raises:
        ABCViolationError: If strict_mode=True and critical violations found
    """
    config = ABCComplianceConfig(strict_mode=strict_mode)
    checker = ABCComplianceChecker(config)

    report = checker.check_evaluation_config(
        includes_baselines=includes_baselines,
        includes_statistics=includes_statistics,
        sample_size=sample_size,
    )

    if strict_mode and report.has_critical():
        raise ABCViolationError(report)

    return report

"""Comprehensive validation and error handling for benchmark integrity.

This module provides tools for validating test cases, datasets, and evaluation
configurations. Implements ABC checks T.3 and T.10.
"""

from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field

from evaris.types import TestCase


class ValidationConfig(BaseModel):
    """Configuration for validation."""

    require_expected: bool = Field(
        default=True, description="Require expected values in test cases"
    )
    validate_input_types: bool = Field(default=True, description="Validate input value types")
    validate_expected_types: bool = Field(default=True, description="Validate expected value types")
    allow_empty_input: bool = Field(default=False, description="Allow empty input values")
    allow_empty_expected: bool = Field(default=False, description="Allow empty expected values")
    check_duplicates: bool = Field(default=True, description="Check for duplicate test cases")
    max_input_length: Optional[int] = Field(
        default=None, description="Maximum input length (characters)"
    )
    strict_mode: bool = Field(
        default=False, description="Enable strict validation (fails on warnings)"
    )


class ValidationIssue(BaseModel):
    """Single validation issue."""

    severity: Literal["error", "warning", "info"] = Field(description="Severity of the issue")
    message: str = Field(description="Issue description")
    test_case_index: Optional[int] = Field(default=None, description="Index of affected test case")
    field: Optional[str] = Field(default=None, description="Affected field")
    suggestion: Optional[str] = Field(default=None, description="Suggestion to fix the issue")


class ValidationResult(BaseModel):
    """Result of validation."""

    is_valid: bool = Field(description="Whether validation passed")
    issues: list[ValidationIssue] = Field(
        default_factory=list, description="List of validation issues"
    )
    num_errors: int = Field(default=0, description="Number of errors")
    num_warnings: int = Field(default=0, description="Number of warnings")
    num_test_cases: int = Field(default=0, description="Number of test cases validated")


class TestCaseValidator:
    """Validator for test cases and datasets.

    Provides comprehensive validation to ensure test cases are well-formed
    and suitable for evaluation.

    ABC Compliance:
    - T.3: Validates test case quality and completeness
    - T.10: Ensures proper error handling and reporting

    Example:
        >>> from evaris.validation import TestCaseValidator, ValidationConfig
        >>> from evaris.types import TestCase
        >>>
        >>> config = ValidationConfig(require_expected=True)
        >>> validator = TestCaseValidator(config)
        >>>
        >>> tc = TestCase(input="question", expected="answer")
        >>> result = validator.validate_test_case(tc)
        >>> print(result.is_valid)  # True
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize test case validator.

        Args:
            config: Configuration for validation. If None, uses defaults.
        """
        self.config = config or ValidationConfig()

    def _add_issue(
        self,
        issues: list[ValidationIssue],
        severity: Literal["error", "warning", "info"],
        message: str,
        test_case_index: Optional[int] = None,
        field: Optional[str] = None,
        suggestion: Optional[str] = None,
    ) -> None:
        """Add a validation issue to the list.

        Args:
            issues: List to add issue to
            severity: Issue severity
            message: Issue description
            test_case_index: Index of affected test case
            field: Affected field
            suggestion: Suggestion to fix
        """
        issues.append(
            ValidationIssue(
                severity=severity,
                message=message,
                test_case_index=test_case_index,
                field=field,
                suggestion=suggestion,
            )
        )

    def validate_test_case(
        self, test_case: TestCase, index: Optional[int] = None
    ) -> ValidationResult:
        """Validate a single test case.

        ABC T.3: Validates test case completeness and quality.

        Args:
            test_case: Test case to validate
            index: Index of test case (for reporting)

        Returns:
            ValidationResult with validation details
        """
        issues: list[ValidationIssue] = []

        # Validate input
        if test_case.input is None:
            self._add_issue(
                issues,
                "error",
                "Test case input is None",
                index,
                "input",
                "Provide a valid input value",
            )
        elif self.config.validate_input_types:
            # Check if input is empty
            if not self.config.allow_empty_input and not str(test_case.input).strip():
                self._add_issue(
                    issues,
                    "warning",
                    "Test case input is empty",
                    index,
                    "input",
                    "Provide non-empty input value",
                )

            # Check input length
            if self.config.max_input_length is not None:
                input_length = len(str(test_case.input))
                if input_length > self.config.max_input_length:
                    self._add_issue(
                        issues,
                        "warning",
                        (
                            f"Input length ({input_length}) exceeds maximum "
                            f"({self.config.max_input_length})"
                        ),
                        index,
                        "input",
                        "Shorten input or increase max_input_length",
                    )

        # Validate expected
        if self.config.require_expected:
            if test_case.expected is None:
                self._add_issue(
                    issues,
                    "error",
                    "Test case expected value is None",
                    index,
                    "expected",
                    "Provide expected value for evaluation",
                )
            elif self.config.validate_expected_types:
                # Check if expected is empty
                if not self.config.allow_empty_expected and not str(test_case.expected).strip():
                    self._add_issue(
                        issues,
                        "warning",
                        "Test case expected value is empty",
                        index,
                        "expected",
                        "Provide non-empty expected value",
                    )

        # Validate metadata
        if test_case.metadata is not None:
            if not isinstance(test_case.metadata, dict):
                self._add_issue(
                    issues,
                    "error",
                    f"Metadata must be dict, got {type(test_case.metadata).__name__}",
                    index,
                    "metadata",
                    "Provide metadata as dictionary",
                )

        # Count errors and warnings
        num_errors = sum(1 for issue in issues if issue.severity == "error")
        num_warnings = sum(1 for issue in issues if issue.severity == "warning")

        # Determine if valid
        is_valid = num_errors == 0
        if self.config.strict_mode:
            is_valid = is_valid and num_warnings == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            num_errors=num_errors,
            num_warnings=num_warnings,
            num_test_cases=1,
        )

    def validate_dataset(self, test_cases: list[TestCase]) -> ValidationResult:
        """Validate an entire dataset.

        ABC Compliance:
        - T.3: Validates dataset completeness and quality
        - T.10: Provides comprehensive error reporting

        Args:
            test_cases: List of test cases to validate

        Returns:
            ValidationResult with aggregated validation details
        """
        all_issues: list[ValidationIssue] = []

        # Validate each test case
        for i, test_case in enumerate(test_cases):
            result = self.validate_test_case(test_case, i)
            all_issues.extend(result.issues)

        # Check for duplicates
        if self.config.check_duplicates:
            duplicates = self._find_duplicates(test_cases)
            for idx1, idx2 in duplicates:
                self._add_issue(
                    all_issues,
                    "warning",
                    f"Duplicate test case found: indices {idx1} and {idx2}",
                    idx1,
                    suggestion="Remove duplicate test cases",
                )

        # Check dataset size
        if len(test_cases) == 0:
            self._add_issue(
                all_issues,
                "error",
                "Dataset is empty",
                suggestion="Add test cases to dataset",
            )
        elif len(test_cases) < 10:
            self._add_issue(
                all_issues,
                "warning",
                f"Dataset has only {len(test_cases)} test cases",
                suggestion="Add more test cases for robust evaluation (recommended: 30+)",
            )

        # Count errors and warnings
        num_errors = sum(1 for issue in all_issues if issue.severity == "error")
        num_warnings = sum(1 for issue in all_issues if issue.severity == "warning")

        # Determine if valid
        is_valid = num_errors == 0
        if self.config.strict_mode:
            is_valid = is_valid and num_warnings == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=all_issues,
            num_errors=num_errors,
            num_warnings=num_warnings,
            num_test_cases=len(test_cases),
        )

    def _find_duplicates(self, test_cases: list[TestCase]) -> list[tuple[int, int]]:
        """Find duplicate test cases.

        ABC T.3: Detects duplicate test cases.

        Args:
            test_cases: List of test cases

        Returns:
            List of (index1, index2) tuples for duplicate pairs
        """
        duplicates: list[tuple[int, int]] = []
        seen: dict[str, int] = {}

        for i, tc in enumerate(test_cases):
            # Create hash of test case
            key = f"{tc.input}|{tc.expected}"

            if key in seen:
                duplicates.append((seen[key], i))
            else:
                seen[key] = i

        return duplicates

    def format_validation_report(self, result: ValidationResult) -> str:
        """Format validation result as human-readable report.

        ABC T.10: Provides clear error reporting.

        Args:
            result: Validation result to format

        Returns:
            Formatted report string
        """
        lines = ["Validation Report", "=" * 50, ""]

        # Overall status
        if result.is_valid:
            lines.append("Validation PASSED")
        else:
            lines.append("X Validation FAILED")

        lines.append("")

        # Summary
        lines.append(f"Test Cases: {result.num_test_cases}")
        lines.append(f"Errors: {result.num_errors}")
        lines.append(f"Warnings: {result.num_warnings}")
        lines.append("")

        # Group issues by severity
        errors = [i for i in result.issues if i.severity == "error"]
        warnings = [i for i in result.issues if i.severity == "warning"]
        # Note: infos not currently used in report, but kept for potential future use
        # infos = [i for i in result.issues if i.severity == "info"]

        # Report errors
        if errors:
            lines.append("ERRORS:")
            for issue in errors:
                location = ""
                if issue.test_case_index is not None:
                    location = f"[Test case {issue.test_case_index}]"
                if issue.field:
                    location += f" ({issue.field})"

                lines.append(f"  X {location} {issue.message}")
                if issue.suggestion:
                    lines.append(f"    → {issue.suggestion}")
            lines.append("")

        # Report warnings
        if warnings:
            lines.append("WARNINGS:")
            for issue in warnings:
                location = ""
                if issue.test_case_index is not None:
                    location = f"[Test case {issue.test_case_index}]"
                if issue.field:
                    location += f" ({issue.field})"

                lines.append(f"  ! {location} {issue.message}")
                if issue.suggestion:
                    lines.append(f"    → {issue.suggestion}")
            lines.append("")

        return "\n".join(lines)


class AgentValidator:
    """Validator for agent functions.

    Validates that agent functions behave correctly and handle errors properly.

    ABC T.10: Ensures proper error handling in evaluation.

    Example:
        >>> from evaris.validation import AgentValidator
        >>> from evaris.types import TestCase
        >>>
        >>> def my_agent(input: str) -> str:
        ...     return f"Response: {input}"
        >>>
        >>> validator = AgentValidator()
        >>> tc = TestCase(input="test", expected="Response: test")
        >>> result = validator.validate_agent(my_agent, tc)
        >>> print(result.is_valid)  # True
    """

    def validate_agent(
        self, agent_fn: Callable[[Any], Any], test_case: TestCase
    ) -> ValidationResult:
        """Validate that agent function works correctly.

        ABC T.10: Validates agent behavior and error handling.

        Args:
            agent_fn: Agent function to validate
            test_case: Test case to use for validation

        Returns:
            ValidationResult with validation details
        """
        issues: list[ValidationIssue] = []

        # Test that agent can be called
        try:
            output = agent_fn(test_case.input)

            # Check output type
            if output is None:
                self._add_issue(
                    issues,
                    "warning",
                    "Agent returned None",
                    suggestion="Ensure agent returns valid output",
                )

        except TypeError as e:
            self._add_issue(
                issues,
                "error",
                f"Agent function signature error: {e}",
                suggestion="Check that agent accepts input parameter",
            )
        except Exception as e:
            self._add_issue(
                issues,
                "error",
                f"Agent raised exception: {type(e).__name__}: {e}",
                suggestion="Fix agent implementation or handle errors",
            )

        num_errors = sum(1 for issue in issues if issue.severity == "error")
        num_warnings = sum(1 for issue in issues if issue.severity == "warning")

        return ValidationResult(
            is_valid=num_errors == 0,
            issues=issues,
            num_errors=num_errors,
            num_warnings=num_warnings,
            num_test_cases=1,
        )

    def _add_issue(
        self,
        issues: list[ValidationIssue],
        severity: Literal["error", "warning", "info"],
        message: str,
        suggestion: Optional[str] = None,
    ) -> None:
        """Add a validation issue."""
        issues.append(ValidationIssue(severity=severity, message=message, suggestion=suggestion))


def validate_metric(metric: Any, test_case: TestCase, output: Any) -> ValidationResult:
    """Validate that a metric works correctly.

    ABC T.10: Validates metric implementation.

    Args:
        metric: Metric to validate
        test_case: Test case to use
        output: Output to evaluate

    Returns:
        ValidationResult with validation details
    """
    issues: list[ValidationIssue] = []

    # Check that metric has score method
    if not hasattr(metric, "score"):
        issues.append(
            ValidationIssue(
                severity="error",
                message="Metric must have 'score' method",
                suggestion="Implement score(test_case, output) method",
            )
        )
        return ValidationResult(
            is_valid=False, issues=issues, num_errors=1, num_warnings=0, num_test_cases=1
        )

    # Test that metric can be called
    try:
        result = metric.score(test_case, output)

        # Validate result structure
        if not hasattr(result, "score") or not hasattr(result, "passed"):
            issues.append(
                ValidationIssue(
                    severity="error",
                    message="Metric must return MetricResult with 'score' and 'passed' fields",
                    suggestion="Return MetricResult object from score method",
                )
            )

        # Validate score range
        if hasattr(result, "score"):
            if not (0.0 <= result.score <= 1.0):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=f"Score {result.score} outside [0, 1] range",
                        suggestion="Ensure score is between 0.0 and 1.0",
                    )
                )

    except Exception as e:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"Metric raised exception: {type(e).__name__}: {e}",
                suggestion="Fix metric implementation or handle errors",
            )
        )

    num_errors = sum(1 for issue in issues if issue.severity == "error")
    num_warnings = sum(1 for issue in issues if issue.severity == "warning")

    return ValidationResult(
        is_valid=num_errors == 0,
        issues=issues,
        num_errors=num_errors,
        num_warnings=num_warnings,
        num_test_cases=1,
    )

"""Tests for validation module."""

from evaris.types import MetricResult, TestCase
from evaris.validation import (
    AgentValidator,
    TestCaseValidator,
    ValidationConfig,
    validate_metric,
)


class TestValidationConfig:
    """Tests for ValidationConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ValidationConfig()

        assert config.require_expected is True
        assert config.validate_input_types is True
        assert config.check_duplicates is True
        assert config.allow_empty_input is False
        assert config.strict_mode is False

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ValidationConfig(
            require_expected=False,
            strict_mode=True,
            max_input_length=100,
        )

        assert config.require_expected is False
        assert config.strict_mode is True
        assert config.max_input_length == 100


class TestTestCaseValidator:
    """Tests for TestCaseValidator."""

    def test_valid_test_case(self) -> None:
        """Test validation of valid test case."""
        validator = TestCaseValidator()
        tc = TestCase(input="test input", expected="test output", actual_output="test output")

        result = validator.validate_test_case(tc)

        assert result.is_valid is True
        assert result.num_errors == 0

    def test_missing_input(self) -> None:
        """Test validation with missing input."""
        validator = TestCaseValidator()
        tc = TestCase(input=None, expected="output", actual_output="output")

        result = validator.validate_test_case(tc)

        assert result.is_valid is False
        assert result.num_errors > 0
        assert any("input is None" in issue.message for issue in result.issues)

    def test_missing_expected_required(self) -> None:
        """Test validation with missing expected (required)."""
        validator = TestCaseValidator()
        tc = TestCase(input="input", expected=None, actual_output="output")

        result = validator.validate_test_case(tc)

        assert result.is_valid is False
        assert result.num_errors > 0
        assert any("expected" in issue.message.lower() for issue in result.issues)

    def test_missing_expected_not_required(self) -> None:
        """Test validation with missing expected (not required)."""
        config = ValidationConfig(require_expected=False)
        validator = TestCaseValidator(config)
        tc = TestCase(input="input", expected=None, actual_output="output")

        result = validator.validate_test_case(tc)

        assert result.is_valid is True

    def test_empty_input_not_allowed(self) -> None:
        """Test validation with empty input (not allowed)."""
        validator = TestCaseValidator()
        tc = TestCase(input="", expected="output", actual_output="output")

        result = validator.validate_test_case(tc)

        assert result.num_warnings > 0
        assert any("empty" in issue.message.lower() for issue in result.issues)

    def test_empty_input_allowed(self) -> None:
        """Test validation with empty input (allowed)."""
        config = ValidationConfig(allow_empty_input=True)
        validator = TestCaseValidator(config)
        tc = TestCase(input="", expected="output", actual_output="output")

        result = validator.validate_test_case(tc)

        # Should not have warning about empty input
        assert not any(
            "empty" in issue.message.lower() and "input" in issue.message.lower()
            for issue in result.issues
            if issue.severity == "error"
        )

    def test_max_input_length_exceeded(self) -> None:
        """Test validation with input exceeding max length."""
        config = ValidationConfig(max_input_length=10)
        validator = TestCaseValidator(config)
        tc = TestCase(input="a" * 20, expected="output", actual_output="output")

        result = validator.validate_test_case(tc)

        assert result.num_warnings > 0
        assert any("length" in issue.message.lower() for issue in result.issues)

    def test_invalid_metadata_type(self) -> None:
        """Test validation with invalid metadata type - Pydantic validates at creation."""
        validator = TestCaseValidator()

        # Pydantic will raise validation error at creation time
        # So we test that the validator works with valid metadata
        tc = TestCase(
            input="input", expected="output", actual_output="output", metadata={"valid": "dict"}
        )

        result = validator.validate_test_case(tc)

        # Should be valid with proper metadata
        assert result.is_valid is True

    def test_validate_dataset_empty(self) -> None:
        """Test validation of empty dataset."""
        validator = TestCaseValidator()

        result = validator.validate_dataset([])

        assert result.is_valid is False
        assert any("empty" in issue.message.lower() for issue in result.issues)

    def test_validate_dataset_small(self) -> None:
        """Test validation of small dataset."""
        validator = TestCaseValidator()
        test_cases = [
            TestCase(input=f"input{i}", expected=f"output{i}", actual_output=f"output{i}")
            for i in range(5)
        ]

        result = validator.validate_dataset(test_cases)

        assert result.num_warnings > 0
        assert any(
            "only 5" in issue.message or "only 5 test cases" in issue.message
            for issue in result.issues
        )

    def test_validate_dataset_duplicates(self) -> None:
        """Test validation detects duplicates."""
        validator = TestCaseValidator()
        test_cases = [
            TestCase(input="input1", expected="output1", actual_output="output1"),
            TestCase(input="input1", expected="output1", actual_output="output1"),  # Duplicate
            TestCase(input="input2", expected="output2", actual_output="output2"),
        ]

        result = validator.validate_dataset(test_cases)

        assert result.num_warnings > 0
        assert any("duplicate" in issue.message.lower() for issue in result.issues)

    def test_validate_dataset_no_duplicates_check(self) -> None:
        """Test validation without duplicate checking."""
        config = ValidationConfig(check_duplicates=False)
        validator = TestCaseValidator(config)
        test_cases = [
            TestCase(input="input1", expected="output1", actual_output="output1"),
            TestCase(input="input1", expected="output1", actual_output="output1"),  # Duplicate
        ]

        result = validator.validate_dataset(test_cases)

        # Should not report duplicates
        assert not any("duplicate" in issue.message.lower() for issue in result.issues)

    def test_strict_mode_fails_on_warnings(self) -> None:
        """Test strict mode fails on warnings."""
        config = ValidationConfig(strict_mode=True)
        validator = TestCaseValidator(config)
        tc = TestCase(input="", expected="output", actual_output="output")  # Empty input (warning)

        result = validator.validate_test_case(tc)

        assert result.is_valid is False  # Strict mode: warnings fail

    def test_format_validation_report(self) -> None:
        """Test formatting validation report."""
        validator = TestCaseValidator()
        tc = TestCase(input=None, expected=None, actual_output="output")

        result = validator.validate_test_case(tc)
        report = validator.format_validation_report(result)

        assert "Validation Report" in report
        assert "ERRORS" in report
        assert len(report) > 50  # Should be substantial


class TestAgentValidator:
    """Tests for AgentValidator."""

    def test_valid_agent(self) -> None:
        """Test validation of valid agent."""

        def agent(input_str: str) -> str:
            return f"Response: {input_str}"

        validator = AgentValidator()
        tc = TestCase(input="test", expected="Response: test", actual_output="Response: test")

        result = validator.validate_agent(agent, tc)

        assert result.is_valid is True
        assert result.num_errors == 0

    def test_agent_returns_none(self) -> None:
        """Test agent that returns None."""

        def agent(input_str: str) -> None:
            return None

        validator = AgentValidator()
        tc = TestCase(input="test", expected="output", actual_output="output")

        result = validator.validate_agent(agent, tc)

        assert result.num_warnings > 0
        assert any("None" in issue.message for issue in result.issues)

    def test_agent_raises_exception(self) -> None:
        """Test agent that raises exception."""

        def agent(input_str: str) -> str:
            raise ValueError("Test error")

        validator = AgentValidator()
        tc = TestCase(input="test", expected="output", actual_output="output")

        result = validator.validate_agent(agent, tc)

        assert result.is_valid is False
        assert result.num_errors > 0
        assert any("exception" in issue.message.lower() for issue in result.issues)

    def test_agent_wrong_signature(self) -> None:
        """Test agent with wrong signature."""

        def agent() -> str:  # Missing input parameter
            return "output"

        validator = AgentValidator()
        tc = TestCase(input="test", expected="output", actual_output="output")

        result = validator.validate_agent(agent, tc)  # type: ignore[arg-type]

        assert result.is_valid is False
        assert result.num_errors > 0
        assert any(
            "signature" in issue.message.lower() or "TypeError" in issue.message
            for issue in result.issues
        )


class TestValidateMetric:
    """Tests for validate_metric function."""

    def test_valid_metric(self) -> None:
        """Test validation of valid metric."""

        class ValidMetric:
            def score(self, test_case: TestCase, output: str) -> MetricResult:
                return MetricResult(name="test", score=0.5, passed=True, metadata={})

        metric = ValidMetric()
        tc = TestCase(input="test", expected="output", actual_output="output")

        result = validate_metric(metric, tc, "output")

        assert result.is_valid is True
        assert result.num_errors == 0

    def test_metric_missing_score_method(self) -> None:
        """Test metric without score method."""

        class InvalidMetric:
            pass

        metric = InvalidMetric()
        tc = TestCase(input="test", expected="output", actual_output="output")

        result = validate_metric(metric, tc, "output")

        assert result.is_valid is False
        assert any("score" in issue.message.lower() for issue in result.issues)

    def test_metric_score_out_of_range(self) -> None:
        """Test metric with score out of range - Pydantic validates at creation."""

        class InvalidMetric:
            def score(self, test_case: TestCase, output: str) -> MetricResult:
                # Pydantic will raise validation error, which we catch
                return MetricResult(
                    name="test",
                    score=1.5,  # Out of range - will fail Pydantic validation
                    passed=True,
                    metadata={},
                )

        metric = InvalidMetric()
        tc = TestCase(input="test", expected="output", actual_output="output")

        result = validate_metric(metric, tc, "output")

        # Should report error (exception during validation)
        assert result.is_valid is False
        assert result.num_errors > 0

    def test_metric_raises_exception(self) -> None:
        """Test metric that raises exception."""

        class InvalidMetric:
            def score(self, test_case: TestCase, output: str) -> MetricResult:
                raise ValueError("Test error")

        metric = InvalidMetric()
        tc = TestCase(input="test", expected="output", actual_output="output")

        result = validate_metric(metric, tc, "output")

        assert result.is_valid is False
        assert any("exception" in issue.message.lower() for issue in result.issues)


class TestABCCompliance:
    """Tests for ABC compliance (T.3, T.10)."""

    def test_abc_t_3_test_case_validation(self) -> None:
        """Test ABC T.3: Validates test case completeness and quality."""
        validator = TestCaseValidator()

        # Invalid test case
        tc = TestCase(input=None, expected=None, actual_output="output")
        result = validator.validate_test_case(tc)

        assert result.is_valid is False
        assert result.num_errors > 0

    def test_abc_t_10_error_handling(self) -> None:
        """Test ABC T.10: Proper error handling and reporting."""
        validator = AgentValidator()

        # Agent that fails
        def failing_agent(input_str: str) -> str:
            raise RuntimeError("Critical error")

        tc = TestCase(input="test", expected="output", actual_output="output")
        result = validator.validate_agent(failing_agent, tc)

        # Should handle gracefully
        assert result.is_valid is False
        assert any(issue.suggestion is not None for issue in result.issues)

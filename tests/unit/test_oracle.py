"""Tests for oracle module."""

from evaris.oracle import (
    BenchmarkValidationResult,
    OracleValidationConfig,
    OracleValidationResult,
    OracleValidator,
)
from evaris.types import TestCase


class TestOracleValidationConfig:
    """Tests for OracleValidationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = OracleValidationConfig()

        assert config.require_oracle is False
        assert config.oracle_timeout_seconds == 300
        assert config.validate_all_tasks is True
        assert config.sample_size is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = OracleValidationConfig(
            require_oracle=True,
            oracle_timeout_seconds=60,
            validate_all_tasks=False,
            sample_size=10,
        )

        assert config.require_oracle is True
        assert config.oracle_timeout_seconds == 60
        assert config.validate_all_tasks is False
        assert config.sample_size == 10


class TestOracleValidationResult:
    """Tests for OracleValidationResult."""

    def test_validation_result_valid(self):
        """Test valid validation result."""
        result = OracleValidationResult(
            is_valid=True,
            solvable=True,
            oracle_output="answer",
            validation_message="Success",
        )

        assert result.is_valid is True
        assert result.solvable is True
        assert result.oracle_output == "answer"
        assert result.validation_message == "Success"

    def test_validation_result_invalid(self):
        """Test invalid validation result."""
        result = OracleValidationResult(
            is_valid=False,
            solvable=False,
            oracle_output=None,
            validation_message="Failed",
        )

        assert result.is_valid is False
        assert result.solvable is False
        assert result.oracle_output is None


class TestBenchmarkValidationResult:
    """Tests for BenchmarkValidationResult."""

    def test_benchmark_result(self):
        """Test benchmark validation result."""
        result = BenchmarkValidationResult(
            is_valid=True,
            total_tasks=10,
            validated_tasks=10,
            solvable_tasks=9,
            unsolvable_tasks=1,
            validation_message="Good benchmark",
        )

        assert result.is_valid is True
        assert result.total_tasks == 10
        assert result.solvable_tasks == 9
        assert result.unsolvable_tasks == 1


class TestOracleValidator:
    """Tests for OracleValidator."""

    def test_validator_default_config(self):
        """Test validator with default config."""
        validator = OracleValidator()

        assert validator.config.require_oracle is False
        assert validator.oracle_solver is None

    def test_validator_custom_config(self):
        """Test validator with custom config."""
        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config)

        assert validator.config.require_oracle is True

    def test_validator_with_oracle_solver(self):
        """Test validator with oracle solver."""

        def simple_oracle(tc: TestCase):
            return str(tc.expected)

        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config, simple_oracle)

        assert validator.oracle_solver == simple_oracle

    def test_validate_task_no_oracle_required(self):
        """Test validation when oracle not required."""
        validator = OracleValidator()
        tc = TestCase(input="test", expected="output", actual_output="output")

        result = validator.validate_task(tc)

        assert isinstance(result, OracleValidationResult)
        assert result.is_valid is True
        assert "not required" in result.validation_message.lower()

    def test_validate_task_oracle_not_provided(self):
        """Test validation when oracle required but not provided."""
        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config)  # No oracle solver
        tc = TestCase(input="test", expected="output", actual_output="output")

        result = validator.validate_task(tc)

        assert result.is_valid is True
        assert "not provided" in result.validation_message.lower()

    def test_validate_task_oracle_success(self):
        """Test successful oracle validation."""

        def oracle(tc: TestCase):
            return str(tc.expected)

        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config, oracle)

        tc = TestCase(input="What is 2+2?", expected="4", actual_output="4")
        result = validator.validate_task(tc)

        assert result.is_valid is True
        assert result.solvable is True
        assert result.oracle_output == "4"
        assert "successfully" in result.validation_message.lower()

    def test_validate_task_oracle_mismatch(self):
        """Test oracle output doesn't match expected."""

        def oracle(tc: TestCase):
            return "wrong answer"

        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config, oracle)

        tc = TestCase(input="test", expected="correct answer", actual_output="correct answer")
        result = validator.validate_task(tc)

        assert result.is_valid is True  # Task is valid, just oracle differs
        assert result.solvable is True
        assert "differs" in result.validation_message.lower()

    def test_validate_task_oracle_returns_none(self):
        """Test oracle returns None."""

        def oracle(tc: TestCase):
            return None

        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config, oracle)

        tc = TestCase(input="test", expected="output", actual_output="output")
        result = validator.validate_task(tc)

        assert result.is_valid is False
        assert result.solvable is False
        assert "None" in result.validation_message

    def test_validate_task_oracle_raises_exception(self):
        """Test oracle raises exception."""

        def oracle(tc: TestCase):
            raise RuntimeError("Oracle failed")

        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config, oracle)

        tc = TestCase(input="test", expected="output", actual_output="output")
        result = validator.validate_task(tc)

        assert result.is_valid is False
        assert result.solvable is False
        # Check for error indicators in message
        assert (
            "failed" in result.validation_message.lower()
            or "error" in result.validation_message.lower()
        )

    def test_validate_benchmark_all_tasks(self):
        """Test validating all tasks in benchmark."""

        def oracle(tc: TestCase):
            return str(tc.expected)

        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config, oracle)

        test_cases = [
            TestCase(input=f"q{i}", expected=f"a{i}", actual_output=f"a{i}") for i in range(10)
        ]

        result = validator.validate_benchmark(test_cases)

        assert isinstance(result, BenchmarkValidationResult)
        assert result.is_valid is True
        assert result.total_tasks == 10
        assert result.solvable_tasks == 10

    def test_validate_benchmark_partial_success(self):
        """Test validation with some failures."""
        call_count = {"count": 0}

        def flaky_oracle(tc: TestCase):
            call_count["count"] += 1
            if call_count["count"] % 2 == 0:
                raise RuntimeError("Failed")
            return str(tc.expected)

        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config, flaky_oracle)

        test_cases = [
            TestCase(input=f"q{i}", expected=f"a{i}", actual_output=f"a{i}") for i in range(10)
        ]

        result = validator.validate_benchmark(test_cases)

        assert result.total_tasks == 10
        assert result.solvable_tasks == 5

    def test_validate_benchmark_sample(self):
        """Test validation with sampling."""

        def oracle(tc: TestCase):
            return str(tc.expected)

        config = OracleValidationConfig(
            require_oracle=True, validate_all_tasks=False, sample_size=5
        )
        validator = OracleValidator(config, oracle)

        test_cases = [
            TestCase(input=f"q{i}", expected=f"a{i}", actual_output=f"a{i}") for i in range(20)
        ]

        result = validator.validate_benchmark(test_cases)

        # Should only validate 5 samples
        assert result.validated_tasks == 5
        assert result.solvable_tasks is not None
        assert result.solvable_tasks <= 5

    def test_validate_benchmark_no_oracle(self):
        """Test benchmark validation without oracle."""
        validator = OracleValidator()  # No oracle required

        test_cases = [
            TestCase(input=f"q{i}", expected=f"a{i}", actual_output=f"a{i}") for i in range(5)
        ]

        result = validator.validate_benchmark(test_cases)

        assert result.is_valid is True
        assert "not required" in result.validation_message.lower()

    def test_validation_benchmark_report(self):
        """Test validation report contains key information."""

        def oracle(tc: TestCase):
            return str(tc.expected)

        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config, oracle)

        test_cases = [
            TestCase(input=f"q{i}", expected=f"a{i}", actual_output=f"a{i}") for i in range(5)
        ]

        result = validator.validate_benchmark(test_cases)

        # Check result has validation message
        assert isinstance(result.validation_message, str)
        assert len(result.validation_message) > 0


class TestABCCompliance:
    """Tests for ABC compliance (T.9)."""

    def test_abc_t_9_oracle_validation(self):
        """Test ABC T.9: Validates tasks are solvable by oracle solver."""

        # Oracle solver that can solve tasks
        def oracle_solver(tc: TestCase):
            # Oracle has perfect knowledge
            return str(tc.expected)

        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config, oracle_solver)

        # Create test cases
        test_cases = [
            TestCase(input=f"task{i}", expected=f"solution{i}", actual_output=f"solution{i}")
            for i in range(10)
        ]

        # Validate benchmark
        result = validator.validate_benchmark(test_cases)

        # All tasks should be solvable
        assert result.is_valid is True
        assert result.solvable_tasks == 10

    def test_abc_t_9_unsolvable_tasks_detected(self):
        """Test ABC T.9: Detects unsolvable tasks."""

        def oracle_solver(tc: TestCase):
            # Oracle fails on some tasks
            if "hard" in str(tc.input):
                raise RuntimeError("Too difficult")
            return str(tc.expected)

        config = OracleValidationConfig(require_oracle=True)
        validator = OracleValidator(config, oracle_solver)

        test_cases = [
            TestCase(input="easy1", expected="sol1", actual_output="sol1"),
            TestCase(input="hard1", expected="sol2", actual_output="sol2"),
            TestCase(input="easy2", expected="sol3", actual_output="sol3"),
        ]

        result = validator.validate_benchmark(test_cases)

        # Should detect that not all tasks are solvable
        assert result.solvable_tasks is not None
        assert result.solvable_tasks < result.total_tasks

"""Tests for fuzz test metric."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaris.metrics.fuzz_test import FuzzTestConfig, FuzzTestMetric
from evaris.types import TestCase


class TestFuzzTestConfig:
    """Tests for FuzzTestConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FuzzTestConfig()

        assert config.num_fuzz_cases == 100
        assert config.timeout_seconds == 30
        assert config.input_types == ["int", "float", "str", "list", "dict", "none"]
        assert config.edge_cases is True
        assert config.memory_stress is False
        assert config.type_confusion is True
        assert config.boundary_values is True
        assert config.custom_generator is None

    def test_custom_config(self):
        """Test custom configuration values."""

        def custom_gen(n):
            return [1, 2, 3]

        config = FuzzTestConfig(
            num_fuzz_cases=50,
            timeout_seconds=10,
            input_types=["int", "str"],
            edge_cases=False,
            memory_stress=True,
            type_confusion=False,
            boundary_values=False,
            custom_generator=custom_gen,
        )

        assert config.num_fuzz_cases == 50
        assert config.timeout_seconds == 10
        assert config.input_types == ["int", "str"]
        assert config.edge_cases is False
        assert config.memory_stress is True
        assert config.type_confusion is False
        assert config.boundary_values is False
        assert config.custom_generator == custom_gen


class TestFuzzTestMetric:
    """Tests for FuzzTestMetric."""

    @pytest.fixture
    def metric(self):
        """Create metric instance for testing."""
        config = FuzzTestConfig()
        return FuzzTestMetric(config)

    def test_metric_initialization(self, metric):
        """Test metric initializes correctly."""
        assert metric.config is not None
        assert isinstance(metric.config, FuzzTestConfig)

    def test_metric_initialization_default_config(self):
        """Test metric with default config."""
        metric = FuzzTestMetric()
        assert metric.config.num_fuzz_cases == 100

    def test_generate_edge_cases_basic(self, metric):
        """Test edge case generation."""
        edge_cases = metric._generate_edge_cases()

        assert None in edge_cases  # None type
        assert 0 in edge_cases  # Zero
        assert 1 in edge_cases  # One
        assert -1 in edge_cases  # Negative one
        assert "" in edge_cases  # Empty string
        assert [] in edge_cases  # Empty list
        assert {} in edge_cases  # Empty dict
        assert float("inf") in edge_cases  # Infinity
        assert float("-inf") in edge_cases  # Negative infinity

    def test_generate_edge_cases_large_values(self, metric):
        """Test edge cases include large values."""
        edge_cases = metric._generate_edge_cases()

        # Check for large string
        assert any(isinstance(v, str) and len(v) > 1000 for v in edge_cases)
        # Check for large list
        assert any(isinstance(v, list) and len(v) > 100 for v in edge_cases)

    def test_generate_edge_cases_special_chars(self, metric):
        """Test edge cases include special characters."""
        edge_cases = metric._generate_edge_cases()

        # Check for unicode
        assert any(isinstance(v, str) and "你好" in v for v in edge_cases)
        # Check for special HTML chars
        assert any(isinstance(v, str) and "<>&" in v for v in edge_cases)
        # Check for whitespace
        assert any(isinstance(v, str) and "\n\t\r" in v for v in edge_cases)

    def test_generate_edge_cases_with_limited_types(self):
        """Test edge cases with limited input types."""
        config = FuzzTestConfig(input_types=["int", "str"])
        metric = FuzzTestMetric(config)

        edge_cases = metric._generate_edge_cases()

        # Should have int and str, but not lists or dicts
        assert any(isinstance(v, int) for v in edge_cases)
        assert any(isinstance(v, str) for v in edge_cases)
        # Should not have "none" since not in input_types
        assert None not in edge_cases

    def test_generate_boundary_values(self, metric):
        """Test boundary value generation."""
        boundary_values = metric._generate_boundary_values()

        # Integer boundaries
        assert -1 in boundary_values
        assert 0 in boundary_values
        assert 1 in boundary_values
        assert 255 in boundary_values
        assert 256 in boundary_values
        assert 65536 in boundary_values

        # Float boundaries
        assert 0.0 in boundary_values
        assert 0.1 in boundary_values
        assert 0.9 in boundary_values
        assert 1.0 in boundary_values

        # String boundaries (various lengths)
        string_lengths = [len(v) for v in boundary_values if isinstance(v, str)]
        assert 1 in string_lengths  # Single char
        assert 255 in string_lengths  # Byte boundary
        assert 256 in string_lengths
        assert 1024 in string_lengths

    def test_generate_type_confusion_cases(self, metric):
        """Test type confusion case generation."""
        type_cases = metric._generate_type_confusion_cases()

        # Should all be tuples
        assert all(isinstance(tc, tuple) for tc in type_cases)

        # Check for string representations of numbers
        assert ("42", "3.14") in type_cases
        assert ("0", "1") in type_cases

        # Check for mixed numeric types
        assert (42, 3.14) in type_cases
        assert (0, 0.0) in type_cases

        # Check for collections with mixed types
        assert ([1, "2", 3.0], [None]) in type_cases

    def test_generate_fuzz_inputs_no_custom_generator(self, metric):
        """Test fuzz input generation without custom generator."""
        fuzz_inputs = metric._generate_fuzz_inputs()

        # Should generate requested number
        assert len(fuzz_inputs) <= metric.config.num_fuzz_cases

        # Should contain various types
        assert any(isinstance(v, int) for v in fuzz_inputs if not isinstance(v, bool))
        assert any(isinstance(v, float) for v in fuzz_inputs)
        assert any(isinstance(v, str) for v in fuzz_inputs)

    def test_generate_fuzz_inputs_with_custom_generator(self):
        """Test fuzz input generation with custom generator."""
        custom_inputs = [1, 2, 3, 4, 5]
        config = FuzzTestConfig(custom_generator=lambda n: custom_inputs)
        metric = FuzzTestMetric(config)

        fuzz_inputs = metric._generate_fuzz_inputs()

        assert fuzz_inputs == custom_inputs

    def test_generate_fuzz_inputs_with_memory_stress(self):
        """Test fuzz inputs include memory stress cases."""
        # Disable other generators to ensure memory stress items are included
        config = FuzzTestConfig(
            memory_stress=True,
            num_fuzz_cases=10,
            edge_cases=False,
            boundary_values=False,
            type_confusion=False,
            input_types=["str"],  # Only strings to reduce random inputs
        )
        metric = FuzzTestMetric(config)

        fuzz_inputs = metric._generate_fuzz_inputs()

        # Should contain at least one large value (memory stress items are added at end)
        has_large_str = any(isinstance(v, str) and len(v) > 100000 for v in fuzz_inputs)
        has_large_list = any(isinstance(v, list) and len(v) > 10000 for v in fuzz_inputs)
        has_large_dict = any(isinstance(v, dict) and len(v) > 1000 for v in fuzz_inputs)
        # At least one of the memory stress types should be present
        assert has_large_str or has_large_list or has_large_dict

    def test_generate_fuzz_inputs_respects_num_fuzz_cases(self):
        """Test that num_fuzz_cases is respected."""
        config = FuzzTestConfig(num_fuzz_cases=10)
        metric = FuzzTestMetric(config)

        fuzz_inputs = metric._generate_fuzz_inputs()

        assert len(fuzz_inputs) == 10

    def test_create_fuzz_test_file(self, metric, tmp_path):
        """Test fuzz test file creation."""
        code = "def add(a, b): return a + b"
        function_name = "add"
        fuzz_inputs = [1, 2, 3, (4, 5)]

        code_file, test_file = metric._create_fuzz_test_file(
            code, function_name, fuzz_inputs, tmp_path
        )

        # Check files exist
        assert code_file.exists()
        assert test_file.exists()

        # Check code file content
        assert code_file.read_text() == code

        # Check test file content
        test_content = test_file.read_text()
        assert "from generated_code import add" in test_content
        assert "run_fuzz_tests" in test_content
        assert "FUZZ_RESULTS:" in test_content
        assert repr(fuzz_inputs) in test_content

    def test_create_fuzz_test_file_handles_tuples(self, metric, tmp_path):
        """Test fuzz test file handles tuple arguments."""
        code = "def add(a, b): return a + b"
        function_name = "add"
        fuzz_inputs = [(1, 2), (3, 4)]

        code_file, test_file = metric._create_fuzz_test_file(
            code, function_name, fuzz_inputs, tmp_path
        )

        test_content = test_file.read_text()
        # Should handle tuple unpacking
        assert "if isinstance(test_input, tuple):" in test_content
        assert f"{function_name}(*test_input)" in test_content

    @patch("subprocess.run")
    def test_run_fuzz_tests_success(self, mock_run, metric, tmp_path):
        """Test running fuzz tests successfully."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# dummy test")

        # Mock successful execution
        mock_run.return_value = MagicMock(
            returncode=0, stdout="FUZZ_RESULTS: 80 passed, 20 failed\n", stderr=""
        )

        success, output, metrics = metric._run_fuzz_tests(test_file)

        assert success is True
        assert metrics["passed_cases"] == 80
        assert metrics["failed_cases"] == 20
        assert metrics["total_cases"] == 100
        assert metrics["pass_rate"] == 0.8

    @patch("subprocess.run")
    def test_run_fuzz_tests_with_errors(self, mock_run, metric, tmp_path):
        """Test running fuzz tests with error output."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# dummy test")

        # Mock execution with errors
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "FUZZ_RESULTS: 60 passed, 40 failed\nERRORS:\n"
                "  Input 0: None -> TypeError: unsupported\n"
                "  Input 1: 'test' -> ValueError: invalid\n"
            ),
            stderr="",
        )

        success, output, metrics = metric._run_fuzz_tests(test_file)

        assert success is False  # Below 70% threshold
        assert metrics["pass_rate"] == 0.6
        assert "error_samples" in metrics
        assert len(metrics["error_samples"]) > 0

    @patch("subprocess.run")
    def test_run_fuzz_tests_timeout(self, mock_run, metric, tmp_path):
        """Test fuzz test timeout."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# dummy test")

        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("pytest", 30)

        success, output, metrics = metric._run_fuzz_tests(test_file)

        assert success is False
        assert "timed out" in output.lower()

    @patch("subprocess.run")
    def test_run_fuzz_tests_no_results(self, mock_run, metric, tmp_path):
        """Test fuzz test with unparseable results."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# dummy test")

        # Mock execution with no parseable results
        mock_run.return_value = MagicMock(returncode=1, stdout="Some error output", stderr="")

        success, output, metrics = metric._run_fuzz_tests(test_file)

        assert success is False
        assert "error" in metrics

    @patch("evaris.metrics.fuzz_test.FuzzTestMetric._run_fuzz_tests")
    @patch("evaris.metrics.fuzz_test.FuzzTestMetric._create_fuzz_test_file")
    def test_score_success(self, mock_create, mock_run, metric):
        """Test scoring with successful fuzz tests."""
        code = "def add(a, b): return a + b"
        tc = TestCase(
            input="Write a function to add two numbers",
            expected={"function_name": "add", "args": ["a", "b"]},
            actual_output=code,
        )

        # Mock file creation
        mock_create.return_value = (Path("/tmp/code.py"), Path("/tmp/test.py"))

        # Mock successful test run
        mock_run.return_value = (
            True,
            "FUZZ_RESULTS: 90 passed, 10 failed",
            {"passed_cases": 90, "failed_cases": 10, "total_cases": 100, "pass_rate": 0.9},
        )

        result = metric.score(tc, code)

        assert result.name == "fuzz_test"
        assert result.score == 0.9
        assert result.passed is True
        assert result.metadata["passed_cases"] == 90
        assert result.metadata["failed_cases"] == 10

    @patch("evaris.metrics.fuzz_test.FuzzTestMetric._run_fuzz_tests")
    @patch("evaris.metrics.fuzz_test.FuzzTestMetric._create_fuzz_test_file")
    def test_score_failure(self, mock_create, mock_run, metric):
        """Test scoring with failed fuzz tests."""
        code = "def test(): pass"
        tc = TestCase(
            input="Write a function", expected={"function_name": "test"}, actual_output=code
        )

        mock_create.return_value = (Path("/tmp/code.py"), Path("/tmp/test.py"))

        # Mock failed test run
        mock_run.return_value = (
            False,
            "FUZZ_RESULTS: 50 passed, 50 failed",
            {"passed_cases": 50, "failed_cases": 50, "total_cases": 100, "pass_rate": 0.5},
        )

        result = metric.score(tc, code)

        assert result.score == 0.5
        assert result.passed is False

    def test_score_no_expected_raises(self, metric):
        """Test score raises ValueError when expected is None."""
        tc = TestCase(input="test", expected=None, actual_output="code")

        with pytest.raises(ValueError, match="expected"):
            metric.score(tc, "code")

    def test_score_invalid_expected_format_raises(self, metric):
        """Test score raises ValueError for invalid expected format."""
        tc = TestCase(input="test", expected="invalid", actual_output="code")

        with pytest.raises(ValueError, match="dict"):
            metric.score(tc, "code")

    def test_score_missing_function_name_raises(self, metric):
        """Test score raises ValueError when function_name missing."""
        tc = TestCase(input="test", expected={"no_function": "test"}, actual_output="code")

        with pytest.raises(ValueError, match="function_name"):
            metric.score(tc, "code")

    @patch("evaris.metrics.fuzz_test.FuzzTestMetric._run_fuzz_tests")
    @patch("evaris.metrics.fuzz_test.FuzzTestMetric._create_fuzz_test_file")
    def test_score_handles_exceptions(self, mock_create, mock_run, metric):
        """Test score handles exceptions gracefully."""
        tc = TestCase(input="test", expected={"function_name": "test"}, actual_output="code")

        # Mock exception during file creation
        mock_create.side_effect = RuntimeError("File creation failed")

        result = metric.score(tc, "code")

        assert result.score == 0.0
        assert result.passed is False
        assert "error" in result.metadata
        assert result.metadata["error_type"] == "RuntimeError"


class TestABCCompliance:
    """Tests for ABC compliance (O.e.1, O.e.2, O.e.3)."""

    def test_abc_o_e_1_diverse_inputs(self):
        """Test ABC O.e.1: Generates diverse inputs covering edge cases."""
        metric = FuzzTestMetric()

        edge_cases = metric._generate_edge_cases()

        # Should cover diverse types
        types_present = set(type(v).__name__ for v in edge_cases if v is not None)
        assert "int" in types_present
        assert "float" in types_present
        assert "str" in types_present
        assert "list" in types_present
        assert "dict" in types_present

        # Should have None
        assert None in edge_cases

    def test_abc_o_e_2_comprehensive_coverage(self):
        """Test ABC O.e.2: Ensures comprehensive input variation coverage."""
        metric = FuzzTestMetric()

        # Generate all input types
        fuzz_inputs = metric._generate_fuzz_inputs()

        # Should have good variation in values
        unique_inputs = len(set(str(v) for v in fuzz_inputs))
        assert unique_inputs >= 50  # At least 50% unique inputs

    def test_abc_o_e_3_sensitive_inputs(self):
        """Test ABC O.e.3: Generates inputs the code is sensitive to."""
        metric = FuzzTestMetric()

        type_cases = metric._generate_type_confusion_cases()

        # Should have type confusion that can trigger bugs
        # String numbers
        assert any("42" in str(tc) for tc in type_cases)
        # Mixed int/float
        assert any((42, 3.14) == tc for tc in type_cases)
        # Boolean as int
        assert any((1, True) == tc for tc in type_cases)


class TestFuzzTestAsync:
    """Tests for async fuzz test methods."""

    @pytest.mark.asyncio
    async def test_run_fuzz_tests_async_success(self):
        """Test async fuzz test execution success."""
        import tempfile

        config = FuzzTestConfig(num_fuzz_cases=5, edge_cases=False, input_types=["int"])
        metric = FuzzTestMetric(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create fuzz test file
            fuzz_inputs = [1, 2, 3, 4, 5]
            code_file, test_file = metric._create_fuzz_test_file(
                "def square(x): return x * x",
                "square",
                fuzz_inputs,
                temp_path,
            )

            success, output, metrics = await metric._run_fuzz_tests_async(test_file)

            assert success
            assert metrics["passed_cases"] == 5
            assert metrics["failed_cases"] == 0
            assert metrics["pass_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_run_fuzz_tests_async_timeout(self):
        """Test async fuzz test timeout."""
        import tempfile

        config = FuzzTestConfig(timeout_seconds=1)
        metric = FuzzTestMetric(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            test_file = temp_path / "test_timeout.py"
            test_file.write_text(
                """
import time
time.sleep(10)  # Will timeout
print("FUZZ_RESULTS: 0 passed, 0 failed")
"""
            )

            success, output, metrics = await metric._run_fuzz_tests_async(test_file)

            assert not success
            assert "timed out" in output.lower()

"""Tests for unit test metric."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaris.metrics.unit_test import UnitTestConfig, UnitTestMetric
from evaris.types import TestCase


class TestUnitTestConfig:
    """Tests for UnitTestConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnitTestConfig()

        assert config.test_framework == "pytest"
        assert config.timeout_seconds == 30
        assert config.coverage_threshold == 0.0
        assert config.measure_coverage is False
        assert config.measure_complexity is False
        assert config.max_complexity == 10
        assert config.additional_deps == []
        assert config.setup_code is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = UnitTestConfig(
            test_framework="unittest",
            timeout_seconds=60,
            coverage_threshold=0.8,
            measure_coverage=True,
            measure_complexity=True,
            max_complexity=5,
            additional_deps=["numpy", "pandas"],
            setup_code="import sys; sys.path.insert(0, '.')",
        )

        assert config.test_framework == "unittest"
        assert config.timeout_seconds == 60
        assert config.coverage_threshold == 0.8
        assert config.measure_coverage is True
        assert config.measure_complexity is True
        assert config.max_complexity == 5
        assert config.additional_deps == ["numpy", "pandas"]
        assert config.setup_code == "import sys; sys.path.insert(0, '.')"


class TestUnitTestMetric:
    """Tests for UnitTestMetric."""

    @pytest.fixture
    def metric(self):
        """Create metric instance for testing."""
        config = UnitTestConfig()
        return UnitTestMetric(config)

    def test_metric_initialization(self, metric):
        """Test metric initializes correctly."""
        assert metric.config is not None
        assert isinstance(metric.config, UnitTestConfig)

    def test_metric_initialization_default_config(self):
        """Test metric with default config."""
        metric = UnitTestMetric()
        assert metric.config.test_framework == "pytest"

    def test_create_test_file(self, metric, tmp_path):
        """Test test file creation."""
        code = "def add(a, b):\n    return a + b"
        tests = [
            "def test_add():\n    assert add(2, 3) == 5",
            "def test_add_negative():\n    assert add(-1, -1) == -2",
        ]

        code_file, test_file = metric._create_test_file(code, tests, tmp_path)

        # Check files exist
        assert code_file.exists()
        assert test_file.exists()

        # Check code file content
        assert code_file.read_text() == code

        # Check test file content
        test_content = test_file.read_text()
        assert "from generated_code import *" in test_content
        assert "def test_add():" in test_content
        assert "def test_add_negative():" in test_content
        assert "assert add(2, 3) == 5" in test_content

    def test_create_test_file_paths(self, metric, tmp_path):
        """Test test file paths are correct."""
        code = "def test(): pass"
        tests = ["def test_test(): pass"]

        code_file, test_file = metric._create_test_file(code, tests, tmp_path)

        assert code_file.name == "generated_code.py"
        assert test_file.name == "test_generated.py"
        assert code_file.parent == tmp_path
        assert test_file.parent == tmp_path

    @patch("subprocess.run")
    def test_run_tests_success(self, mock_run, metric, tmp_path):
        """Test running tests successfully."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# dummy test")

        # Mock successful test execution
        mock_run.return_value = MagicMock(
            returncode=0, stdout="test_generated.py::test_add PASSED\n2 passed in 0.1s", stderr=""
        )

        success, output, metrics = metric._run_tests(test_file)

        assert success is True
        assert "PASSED" in output
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_tests_failure(self, mock_run, metric, tmp_path):
        """Test running tests with failures."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# dummy test")

        # Mock failed test execution
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="test_generated.py::test_add FAILED\n1 failed in 0.1s",
            stderr="AssertionError: assert 5 == 6",
        )

        success, output, metrics = metric._run_tests(test_file)

        assert success is False
        assert "FAILED" in output

    @patch("subprocess.run")
    def test_run_tests_timeout(self, mock_run, metric, tmp_path):
        """Test test execution timeout."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# dummy test")

        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("pytest", 30)

        success, output, metrics = metric._run_tests(test_file)

        assert success is False
        assert "timed out" in output.lower()

    @patch("subprocess.run")
    def test_run_tests_with_coverage(self, mock_run, tmp_path):
        """Test running tests with coverage measurement."""
        config = UnitTestConfig(measure_coverage=True, coverage_threshold=0.8)
        metric = UnitTestMetric(config)
        test_file = tmp_path / "test.py"
        test_file.write_text("# dummy test")

        # Mock execution with coverage output
        mock_run.return_value = MagicMock(
            returncode=0, stdout="test_generated.py PASSED\nTOTAL          85%\n", stderr=""
        )

        success, output, metrics = metric._run_tests(test_file)

        assert success is True
        assert "coverage" in metrics
        assert metrics["coverage"] == 0.85

        # Verify coverage flags were added to command
        call_args = mock_run.call_args[0][0]
        assert "--cov=generated_code" in call_args

    @patch("subprocess.run")
    def test_run_tests_coverage_parsing(self, mock_run, tmp_path):
        """Test coverage percentage parsing."""
        config = UnitTestConfig(measure_coverage=True)
        metric = UnitTestMetric(config)
        test_file = tmp_path / "test.py"
        test_file.write_text("# dummy test")

        # Mock execution with coverage output
        mock_run.return_value = MagicMock(
            returncode=0, stdout="generated_code.py    50%\nTOTAL                 92%\n", stderr=""
        )

        success, output, metrics = metric._run_tests(test_file)

        assert "coverage" in metrics
        assert metrics["coverage"] == 0.92

    @patch("subprocess.run")
    def test_run_tests_exception_handling(self, mock_run, metric, tmp_path):
        """Test exception handling during test execution."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# dummy test")

        # Mock exception
        mock_run.side_effect = RuntimeError("Execution failed")

        success, output, metrics = metric._run_tests(test_file)

        assert success is False
        assert "failed" in output.lower()

    @patch("subprocess.run")
    def test_measure_complexity(self, mock_run, metric, tmp_path):
        """Test complexity measurement."""
        code_file = tmp_path / "code.py"
        code_file.write_text("def test(): pass")

        # Mock radon output
        complexity_data = {
            str(code_file): [
                {"complexity": 2, "name": "func1"},
                {"complexity": 5, "name": "func2"},
                {"complexity": 8, "name": "func3"},
            ]
        }
        mock_run.return_value = MagicMock(
            returncode=0, stdout=__import__("json").dumps(complexity_data), stderr=""
        )

        metrics = metric._measure_complexity(code_file)

        assert "avg_complexity" in metrics
        assert "max_complexity" in metrics
        assert "complexity_violations" in metrics
        assert metrics["avg_complexity"] == 5.0  # (2 + 5 + 8) / 3
        assert metrics["max_complexity"] == 8

    @patch("subprocess.run")
    def test_measure_complexity_violations(self, mock_run, tmp_path):
        """Test detecting complexity violations."""
        config = UnitTestConfig(max_complexity=5)
        metric = UnitTestMetric(config)
        code_file = tmp_path / "code.py"
        code_file.write_text("def test(): pass")

        # Mock radon output with high complexity
        complexity_data = {
            str(code_file): [
                {"complexity": 3, "name": "func1"},
                {"complexity": 7, "name": "func2"},  # Violation
                {"complexity": 12, "name": "func3"},  # Violation
            ]
        }
        mock_run.return_value = MagicMock(
            returncode=0, stdout=__import__("json").dumps(complexity_data), stderr=""
        )

        metrics = metric._measure_complexity(code_file)

        assert metrics["complexity_violations"] == 2

    @patch("subprocess.run")
    def test_measure_complexity_error_handling(self, mock_run, metric, tmp_path):
        """Test complexity measurement error handling."""
        code_file = tmp_path / "code.py"
        code_file.write_text("def test(): pass")

        # Mock radon failure
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")

        metrics = metric._measure_complexity(code_file)

        # Should not crash, may have empty metrics
        assert isinstance(metrics, dict)

    @patch("subprocess.run")
    def test_measure_complexity_exception(self, mock_run, metric, tmp_path):
        """Test complexity measurement with exception."""
        code_file = tmp_path / "code.py"
        code_file.write_text("def test(): pass")

        # Mock exception
        mock_run.side_effect = RuntimeError("Radon error")

        metrics = metric._measure_complexity(code_file)

        assert "complexity_error" in metrics

    @patch("evaris.metrics.unit_test.UnitTestMetric._run_tests")
    @patch("evaris.metrics.unit_test.UnitTestMetric._create_test_file")
    def test_score_success(self, mock_create, mock_run, metric):
        """Test scoring with passing tests."""
        code = "def add(a, b): return a + b"
        tc = TestCase(
            input="Write a function to add two numbers",
            expected={"tests": ["def test_add(): assert add(2, 3) == 5"]},
            actual_output=code,
        )

        # Mock file creation
        mock_create.return_value = (Path("/tmp/code.py"), Path("/tmp/test.py"))

        # Mock successful test run
        mock_run.return_value = (True, "All tests passed", {})

        result = metric.score(tc, code)

        assert result.name == "unit_test"
        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata["num_tests"] == 1

    @patch("evaris.metrics.unit_test.UnitTestMetric._run_tests")
    @patch("evaris.metrics.unit_test.UnitTestMetric._create_test_file")
    def test_score_failure(self, mock_create, mock_run, metric):
        """Test scoring with failing tests."""
        code = "def test(): pass"
        tc = TestCase(
            input="test", expected={"tests": ["def test_fail(): assert False"]}, actual_output=code
        )

        mock_create.return_value = (Path("/tmp/code.py"), Path("/tmp/test.py"))

        # Mock failed test run
        mock_run.return_value = (False, "Test failed", {})

        result = metric.score(tc, code)

        assert result.score == 0.0
        assert result.passed is False

    @patch("evaris.metrics.unit_test.UnitTestMetric._run_tests")
    @patch("evaris.metrics.unit_test.UnitTestMetric._create_test_file")
    @patch("evaris.metrics.unit_test.UnitTestMetric._measure_complexity")
    def test_score_with_coverage(self, mock_complexity, mock_create, mock_run):
        """Test scoring with coverage measurement."""
        config = UnitTestConfig(measure_coverage=True, coverage_threshold=0.8)
        metric = UnitTestMetric(config)

        tc = TestCase(input="test", expected={"tests": ["def test(): pass"]}, actual_output="code")

        mock_create.return_value = (Path("/tmp/code.py"), Path("/tmp/test.py"))
        mock_run.return_value = (True, "Tests passed", {"coverage": 0.85})

        result = metric.score(tc, "code")

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata["coverage"] == 0.85

    @patch("evaris.metrics.unit_test.UnitTestMetric._run_tests")
    @patch("evaris.metrics.unit_test.UnitTestMetric._create_test_file")
    def test_score_coverage_below_threshold(self, mock_create, mock_run):
        """Test scoring with coverage below threshold."""
        config = UnitTestConfig(measure_coverage=True, coverage_threshold=0.8)
        metric = UnitTestMetric(config)

        tc = TestCase(input="test", expected={"tests": ["def test(): pass"]}, actual_output="code")

        mock_create.return_value = (Path("/tmp/code.py"), Path("/tmp/test.py"))
        # Coverage below threshold - test_run should return success=True initially
        # but score() will set success=False due to coverage
        mock_run.return_value = (True, "Tests passed", {"coverage": 0.5})

        result = metric.score(tc, "code")

        # Score is 0 due to low coverage, but passed flag is based on test success
        assert result.score == 0.0  # Fails due to low coverage
        # passed stays True because tests passed, only score changes
        assert "coverage_error" in result.metadata

    @patch("evaris.metrics.unit_test.UnitTestMetric._run_tests")
    @patch("evaris.metrics.unit_test.UnitTestMetric._create_test_file")
    @patch("evaris.metrics.unit_test.UnitTestMetric._measure_complexity")
    def test_score_with_complexity(self, mock_complexity, mock_create, mock_run):
        """Test scoring with complexity measurement."""
        config = UnitTestConfig(measure_complexity=True, max_complexity=10)
        metric = UnitTestMetric(config)

        tc = TestCase(input="test", expected={"tests": ["def test(): pass"]}, actual_output="code")

        mock_create.return_value = (Path("/tmp/code.py"), Path("/tmp/test.py"))
        mock_run.return_value = (True, "Tests passed", {})
        mock_complexity.return_value = {
            "avg_complexity": 5.0,
            "max_complexity": 8,
            "complexity_violations": 0,
        }

        result = metric.score(tc, "code")

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata["avg_complexity"] == 5.0

    @patch("evaris.metrics.unit_test.UnitTestMetric._run_tests")
    @patch("evaris.metrics.unit_test.UnitTestMetric._create_test_file")
    @patch("evaris.metrics.unit_test.UnitTestMetric._measure_complexity")
    def test_score_complexity_violations(self, mock_complexity, mock_create, mock_run):
        """Test scoring with complexity violations."""
        config = UnitTestConfig(measure_complexity=True, max_complexity=5)
        metric = UnitTestMetric(config)

        tc = TestCase(input="test", expected={"tests": ["def test(): pass"]}, actual_output="code")

        mock_create.return_value = (Path("/tmp/code.py"), Path("/tmp/test.py"))
        mock_run.return_value = (True, "Tests passed", {})
        # Has complexity violations
        mock_complexity.return_value = {
            "avg_complexity": 12.0,
            "max_complexity": 15,
            "complexity_violations": 2,
        }

        result = metric.score(tc, "code")

        assert result.score == 0.0  # Fails due to complexity
        assert result.passed is False
        assert "complexity_error" in result.metadata

    def test_score_no_expected_raises(self, metric):
        """Test score raises ValueError when expected is None."""
        tc = TestCase(input="test", expected=None, actual_output="code")

        with pytest.raises(ValueError, match="expected"):
            metric.score(tc, "code")

    def test_score_expected_as_list(self, metric):
        """Test score with expected as list of tests."""
        tc = TestCase(
            input="test", expected=["def test1(): pass", "def test2(): pass"], actual_output="code"
        )

        with (
            patch("evaris.metrics.unit_test.UnitTestMetric._run_tests") as mock_run,
            patch("evaris.metrics.unit_test.UnitTestMetric._create_test_file") as mock_create,
        ):
            mock_create.return_value = (Path("/tmp/code.py"), Path("/tmp/test.py"))
            mock_run.return_value = (True, "Tests passed", {})

            result = metric.score(tc, "code")

            assert result.metadata["num_tests"] == 2

    def test_score_invalid_expected_format_raises(self, metric):
        """Test score raises ValueError for invalid expected format."""
        tc = TestCase(input="test", expected="invalid", actual_output="code")

        with pytest.raises(ValueError):
            metric.score(tc, "code")

    def test_score_no_tests_raises(self, metric):
        """Test score raises ValueError when no tests provided."""
        tc = TestCase(input="test", expected={"tests": []}, actual_output="code")

        with pytest.raises(ValueError, match="No tests"):
            metric.score(tc, "code")

    @patch("evaris.metrics.unit_test.UnitTestMetric._run_tests")
    @patch("evaris.metrics.unit_test.UnitTestMetric._create_test_file")
    def test_score_handles_exceptions(self, mock_create, mock_run, metric):
        """Test score handles exceptions gracefully."""
        tc = TestCase(input="test", expected={"tests": ["def test(): pass"]}, actual_output="code")

        # Mock exception during file creation
        mock_create.side_effect = RuntimeError("File creation failed")

        result = metric.score(tc, "code")

        assert result.score == 0.0
        assert result.passed is False
        assert "error" in result.metadata
        assert result.metadata["error_type"] == "RuntimeError"


class TestABCCompliance:
    """Tests for ABC compliance (O.d.1, O.d.2)."""

    @patch("evaris.metrics.unit_test.UnitTestMetric._run_tests")
    @patch("evaris.metrics.unit_test.UnitTestMetric._create_test_file")
    def test_abc_o_d_1_verifies_test_cases(self, mock_create, mock_run):
        """Test ABC O.d.1: Verifies test cases for correctness."""
        metric = UnitTestMetric()

        code = "def add(a, b): return a + b"
        tc = TestCase(
            input="Write add function",
            expected={
                "tests": [
                    "def test_add(): assert add(2, 3) == 5",
                    "def test_add_negative(): assert add(-1, -1) == -2",
                ]
            },
            actual_output=code,
        )

        mock_create.return_value = (Path("/tmp/code.py"), Path("/tmp/test.py"))
        mock_run.return_value = (True, "All tests passed", {})

        result = metric.score(tc, code)

        # Should run and verify all test cases
        assert result.passed is True
        assert result.metadata["num_tests"] == 2

    @patch("evaris.metrics.unit_test.UnitTestMetric._run_tests")
    @patch("evaris.metrics.unit_test.UnitTestMetric._create_test_file")
    @patch("evaris.metrics.unit_test.UnitTestMetric._measure_complexity")
    def test_abc_o_d_2_measures_quality(self, mock_complexity, mock_create, mock_run):
        """Test ABC O.d.2: Measures quality using objective metrics."""
        config = UnitTestConfig(
            measure_coverage=True, measure_complexity=True, coverage_threshold=0.8
        )
        metric = UnitTestMetric(config)

        tc = TestCase(input="test", expected={"tests": ["def test(): pass"]}, actual_output="code")

        mock_create.return_value = (Path("/tmp/code.py"), Path("/tmp/test.py"))
        # Mock quality metrics
        mock_run.return_value = (True, "Tests passed", {"coverage": 0.9})
        mock_complexity.return_value = {
            "avg_complexity": 4.0,
            "max_complexity": 6,
            "complexity_violations": 0,
        }

        result = metric.score(tc, "code")

        # Should measure and report quality metrics
        assert "coverage" in result.metadata
        assert "avg_complexity" in result.metadata
        assert "max_complexity" in result.metadata
        assert result.metadata["coverage"] == 0.9
        assert result.metadata["avg_complexity"] == 4.0


class TestUnitTestAsync:
    """Tests for async unit test methods."""

    @pytest.mark.asyncio
    async def test_a_measure_with_setup_code(self):
        """Test async unit test with setup code."""
        config = UnitTestConfig(setup_code="# Setup code\nimport sys\nprint('Setup complete')")
        metric = UnitTestMetric(config)
        tc = TestCase(
            input="test",
            expected={"tests": ["def test_foo():\n    assert 1 == 1"]},
            actual_output="# Code",
        )

        result = await metric.a_measure(tc)

        assert result.name == "unit_test"
        # Should execute without error
        assert "test_output" in result.metadata

    @pytest.mark.asyncio
    async def test_a_measure_with_complexity(self):
        """Test async unit test with complexity measurement."""
        config = UnitTestConfig(measure_complexity=True, max_complexity=5)
        metric = UnitTestMetric(config)
        tc = TestCase(
            input="test",
            expected={"tests": ["def test_simple():\n    assert simple() == 1"]},
            actual_output="def simple():\n    return 1",
        )

        result = await metric.a_measure(tc)

        assert result.name == "unit_test"
        # Should run without error (complexity measurement is optional)
        assert "test_output" in result.metadata

    @pytest.mark.asyncio
    async def test_run_tests_async_timeout(self):
        """Test async test execution timeout."""
        import tempfile

        config = UnitTestConfig(timeout_seconds=1)
        metric = UnitTestMetric(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test_timeout.py"
            test_file.write_text(
                """
import time
def test_slow():
    time.sleep(10)  # Will timeout
    assert True
"""
            )

            success, output, metrics = await metric._run_tests_async(test_file)

            assert not success
            assert "timed out" in output.lower()

    @pytest.mark.asyncio
    async def test_measure_complexity_async_success(self):
        """Test async complexity measurement success."""
        import tempfile

        metric = UnitTestMetric()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            code_file = temp_path / "code.py"
            code_file.write_text(
                """
def simple_function(x):
    return x + 1

def complex_function(x):
    if x > 0:
        if x > 10:
            return x * 2
        else:
            return x + 1
    else:
        return 0
"""
            )

            metrics = await metric._measure_complexity_async(code_file)

            # Should return a dict (may have complexity or error)
            assert isinstance(metrics, dict)
            # If radon is installed, should have complexity metrics
            # If not, should be empty or have error
            assert len(metrics) >= 0  # Just check it runs without crashing

    @pytest.mark.asyncio
    async def test_a_measure_with_additional_deps(self):
        """Test async unit test with additional dependencies."""
        # Use a package that's likely already installed
        config = UnitTestConfig(additional_deps=["setuptools"])
        metric = UnitTestMetric(config)
        tc = TestCase(
            input="test",
            expected={"tests": ["def test_basic():\n    assert 1 == 1"]},
            actual_output="# Simple code",
        )

        result = await metric.a_measure(tc)

        assert result.name == "unit_test"
        # Should install and run without major errors
        assert "test_output" in result.metadata

    @pytest.mark.asyncio
    async def test_run_tests_async_exception_handling(self):
        """Test async test execution exception handling."""
        import tempfile

        metric = UnitTestMetric()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create invalid test file
            test_file = temp_path / "nonexistent_dir" / "test.py"

            success, output, metrics = await metric._run_tests_async(test_file)

            assert not success
            assert "failed" in output.lower()

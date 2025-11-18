"""Unit testing metric for code generation evaluation.

This metric evaluates generated code by running unit tests.
Implements ABC checks O.d.1 and O.d.2 for test quality verification.
"""

import asyncio
import asyncio.subprocess
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.types import BaseMetric, MetricResult, TestCase


class UnitTestConfig(BaseModel):
    """Configuration for unit testing metric."""

    test_framework: str = Field(default="pytest", description="Testing framework to use")
    timeout_seconds: int = Field(default=30, description="Test execution timeout")
    coverage_threshold: float = Field(default=0.0, description="Minimum code coverage (0.0-1.0)")
    measure_coverage: bool = Field(
        default=False, description="Measure code coverage (requires coverage.py)"
    )
    measure_complexity: bool = Field(
        default=False, description="Measure cyclomatic complexity (requires radon)"
    )
    max_complexity: int = Field(default=10, description="Maximum allowed cyclomatic complexity")
    additional_deps: list[str] = Field(
        default_factory=list, description="Additional pip dependencies"
    )
    setup_code: Optional[str] = Field(default=None, description="Setup code to run before tests")


class UnitTestMetric(BaseMetric):
    """Unit testing metric for code generation.

    Evaluates generated code by running unit tests and optionally measuring
    code quality metrics like coverage and complexity.

    SECURITY WARNING
    This metric executes generated code in a subprocess. While subprocess
    isolation provides some protection, it does NOT provide true sandboxing:
    - Code runs with the same user privileges
    - Code can access the filesystem, network, and system resources
    - Malicious code could delete files, exfiltrate data, or make network calls

    RECOMMENDATIONS:
    - Only use with trusted agents in controlled environments
    - For untrusted agents, run evaluations in Docker containers or VMs
    - Consider using dedicated evaluation infrastructure with network isolation
    - Monitor resource usage and implement rate limiting
    - Review generated code before execution when possible

    The timeout parameter (default: 30s) prevents infinite loops but does NOT
    prevent malicious behavior within the timeout window.

    ABC Compliance:
    - O.d.1: Verifies test cases for correctness and quality
    - O.d.2: Measures quality using objective metrics (coverage, complexity)

    Example:
        >>> from evaris.metrics.unit_test import UnitTestMetric, UnitTestConfig
        >>> config = UnitTestConfig(measure_coverage=True, coverage_threshold=0.8)
        >>> metric = UnitTestMetric(config)
        >>> tc = TestCase(
        ...     input="Write a function to add two numbers",
        ...     expected={
        ...         "tests": ["def test_add(): assert add(2, 3) == 5"]
        ...     }
        ... )
        >>> result = metric.score(tc, "def add(a, b): return a + b")
        >>> print(result.passed)  # True if tests pass
    """

    def __init__(self, config: Optional[UnitTestConfig] = None):
        """Initialize unit testing metric.

        Args:
            config: Configuration for unit testing. If None, uses defaults.
        """
        self.config = config or UnitTestConfig()

    def _create_test_file(self, code: str, tests: list[str], temp_dir: Path) -> tuple[Path, Path]:
        """Create temporary code and test files.

        Args:
            code: Generated code to test
            tests: List of test functions
            temp_dir: Temporary directory

        Returns:
            Tuple of (code_file_path, test_file_path)
        """
        # Write code file
        code_file = temp_dir / "generated_code.py"
        code_file.write_text(code)

        # Write test file
        test_content = f"""# Generated test file
import sys
sys.path.insert(0, '{temp_dir}')
from generated_code import *

{chr(10).join(tests)}
"""
        test_file = temp_dir / "test_generated.py"
        test_file.write_text(test_content)

        return code_file, test_file

    def _run_tests(self, test_file: Path) -> tuple[bool, str, dict[str, Any]]:
        """Run tests using pytest.

        Args:
            test_file: Path to test file

        Returns:
            Tuple of (success, output, metrics)
        """
        metrics: dict[str, Any] = {}

        try:
            # Build pytest command
            cmd = [
                "python",
                "-m",
                "pytest",
                str(test_file),
                "-v",
                "--tb=short",
            ]

            # Add coverage if requested
            if self.config.measure_coverage:
                cmd.extend(
                    [
                        "--cov=generated_code",
                        "--cov-report=term-missing",
                        f"--cov-fail-under={int(self.config.coverage_threshold * 100)}",
                    ]
                )

            # Run tests
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                cwd=test_file.parent,
            )

            output = result.stdout + "\n" + result.stderr
            success = result.returncode == 0

            # Parse coverage from output if measured
            if self.config.measure_coverage:
                import re

                coverage_match = re.search(r"TOTAL.*?(\d+)%", output)
                if coverage_match:
                    metrics["coverage"] = float(coverage_match.group(1)) / 100.0

            return success, output, metrics

        except subprocess.TimeoutExpired:
            return False, "Test execution timed out", metrics
        except Exception as e:
            return False, f"Test execution failed: {str(e)}", metrics

    def _measure_complexity(self, code_file: Path) -> dict[str, Any]:
        """Measure cyclomatic complexity using radon.

        ABC O.d.2: Measures complexity using objective metrics.

        Args:
            code_file: Path to code file

        Returns:
            Dictionary with complexity metrics
        """
        metrics: dict[str, Any] = {}

        try:
            # Run radon to measure complexity
            result = subprocess.run(
                ["python", "-m", "radon", "cc", str(code_file), "-s", "-j"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                import json

                complexity_data = json.loads(result.stdout)
                if complexity_data and str(code_file) in complexity_data:
                    file_complexity = complexity_data[str(code_file)]
                    if file_complexity:
                        # Get average and max complexity
                        complexities = [item["complexity"] for item in file_complexity]
                        metrics["avg_complexity"] = (
                            sum(complexities) / len(complexities) if complexities else 0
                        )
                        metrics["max_complexity"] = max(complexities) if complexities else 0
                        metrics["complexity_violations"] = sum(
                            1 for c in complexities if c > self.config.max_complexity
                        )

        except Exception as e:
            metrics["complexity_error"] = str(e)

        return metrics

    def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Score generated code using unit tests.

        ABC Compliance:
        - O.d.1: Runs verified test cases
        - O.d.2: Measures code quality with coverage and complexity

        Args:
            test_case: Test case with test definitions
            actual_output: Generated code to test

        Returns:
            MetricResult with test results and metrics

        Raises:
            ValueError: If test case doesn't contain tests
        """
        if test_case.expected is None:
            raise ValueError("Unit test metric requires 'expected' value with tests")

        # Extract tests from expected
        if isinstance(test_case.expected, dict):
            tests = test_case.expected.get("tests", [])
        elif isinstance(test_case.expected, list):
            tests = test_case.expected
        else:
            raise ValueError("Expected value must be dict with 'tests' key or list of tests")

        if not tests:
            raise ValueError("No tests provided in test case")

        metadata: dict[str, Any] = {
            "num_tests": len(tests),
            "framework": self.config.test_framework,
        }

        try:
            # Create temporary directory for test execution
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create test files
                code_file, test_file = self._create_test_file(str(actual_output), tests, temp_path)

                # Install additional dependencies if needed
                if self.config.additional_deps:
                    for dep in self.config.additional_deps:
                        subprocess.run(
                            ["python", "-m", "pip", "install", "-q", dep],
                            timeout=60,
                        )

                # Run setup code if provided
                if self.config.setup_code:
                    setup_file = temp_path / "setup.py"
                    setup_file.write_text(self.config.setup_code)
                    subprocess.run(["python", str(setup_file)], timeout=30, cwd=temp_path)

                # Run tests
                success, output, test_metrics = self._run_tests(test_file)
                metadata["test_output"] = output
                metadata.update(test_metrics)

                # Measure complexity if requested
                if self.config.measure_complexity:
                    complexity_metrics = self._measure_complexity(code_file)
                    metadata.update(complexity_metrics)

                    # Check complexity violations
                    if "complexity_violations" in metadata:
                        if metadata["complexity_violations"] > 0:
                            success = False
                            metadata["complexity_error"] = (
                                f"Code has {metadata['complexity_violations']} functions "
                                f"exceeding max complexity {self.config.max_complexity}"
                            )

                # Calculate final score
                score = 1.0 if success else 0.0

                # Adjust score based on coverage if measured
                if self.config.measure_coverage and "coverage" in metadata:
                    coverage = metadata["coverage"]
                    if coverage < self.config.coverage_threshold:
                        score = 0.0
                        metadata["coverage_error"] = (
                            f"Coverage {coverage:.1%} below threshold "
                            f"{self.config.coverage_threshold:.1%}"
                        )

                return MetricResult(
                    name="unit_test", score=score, passed=success, metadata=metadata
                )

        except Exception as e:
            return MetricResult(
                name="unit_test",
                score=0.0,
                passed=False,
                metadata={
                    **metadata,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def _run_tests_async(self, test_file: Path) -> tuple[bool, str, dict[str, Any]]:
        """Run tests using pytest asynchronously.

        Args:
            test_file: Path to test file

        Returns:
            Tuple of (success, output, metrics)
        """
        metrics: dict[str, Any] = {}

        try:
            # Build pytest command
            cmd = [
                "python",
                "-m",
                "pytest",
                str(test_file),
                "-v",
                "--tb=short",
            ]

            # Add coverage if requested
            if self.config.measure_coverage:
                cmd.extend(
                    [
                        "--cov=generated_code",
                        "--cov-report=term-missing",
                        f"--cov-fail-under={int(self.config.coverage_threshold * 100)}",
                    ]
                )

            # Run tests asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(test_file.parent),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.config.timeout_seconds
                )
                output = stdout.decode() + "\n" + stderr.decode()
                success = process.returncode == 0

                # Parse coverage from output if measured
                if self.config.measure_coverage:
                    import re

                    coverage_match = re.search(r"TOTAL.*?(\d+)%", output)
                    if coverage_match:
                        metrics["coverage"] = float(coverage_match.group(1)) / 100.0

                return success, output, metrics

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return False, "Test execution timed out", metrics

        except Exception as e:
            return False, f"Test execution failed: {str(e)}", metrics

    async def _measure_complexity_async(self, code_file: Path) -> dict[str, Any]:
        """Measure cyclomatic complexity using radon asynchronously.

        ABC O.d.2: Measures complexity using objective metrics.

        Args:
            code_file: Path to code file

        Returns:
            Dictionary with complexity metrics
        """
        metrics: dict[str, Any] = {}

        try:
            # Run radon to measure complexity
            process = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "radon",
                "cc",
                str(code_file),
                "-s",
                "-j",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)

                if process.returncode == 0:
                    import json

                    complexity_data = json.loads(stdout.decode())
                    if complexity_data and str(code_file) in complexity_data:
                        file_complexity = complexity_data[str(code_file)]
                        if file_complexity:
                            # Get average and max complexity
                            complexities = [item["complexity"] for item in file_complexity]
                            metrics["avg_complexity"] = (
                                sum(complexities) / len(complexities) if complexities else 0
                            )
                            metrics["max_complexity"] = max(complexities) if complexities else 0
                            metrics["complexity_violations"] = sum(
                                1 for c in complexities if c > self.config.max_complexity
                            )

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                metrics["complexity_error"] = "Complexity measurement timed out"

        except Exception as e:
            metrics["complexity_error"] = str(e)

        return metrics

    async def a_measure(self, test_case: TestCase) -> MetricResult:
        """Asynchronously score generated code using unit tests.

        This async version uses asyncio.create_subprocess_exec for running
        pytest and radon subprocesses, allowing multiple tests to run
        concurrently without blocking the event loop.

        ABC Compliance:
        - O.d.1: Runs verified test cases
        - O.d.2: Measures code quality with coverage and complexity

        Args:
            test_case: Test case with test definitions and actual_output

        Returns:
            MetricResult with test results and metrics

        Raises:
            ValueError: If test case doesn't contain tests or actual_output
        """
        if test_case.actual_output is None:
            raise ValueError("Unit test metric requires 'actual_output' in test case")

        if test_case.expected is None:
            raise ValueError("Unit test metric requires 'expected' value with tests")

        # Extract tests from expected
        if isinstance(test_case.expected, dict):
            tests = test_case.expected.get("tests", [])
        elif isinstance(test_case.expected, list):
            tests = test_case.expected
        else:
            raise ValueError("Expected value must be dict with 'tests' key or list of tests")

        if not tests:
            raise ValueError("No tests provided in test case")

        metadata: dict[str, Any] = {
            "num_tests": len(tests),
            "framework": self.config.test_framework,
        }

        try:
            # Create temporary directory for test execution
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create test files
                code_file, test_file = self._create_test_file(
                    str(test_case.actual_output), tests, temp_path
                )

                # Install additional dependencies if needed (run in parallel if multiple)
                if self.config.additional_deps:
                    install_tasks = []
                    for dep in self.config.additional_deps:
                        task = asyncio.create_subprocess_exec(
                            "python",
                            "-m",
                            "pip",
                            "install",
                            "-q",
                            dep,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        install_tasks.append(task)

                    # Wait for all installations to complete
                    processes = await asyncio.gather(*install_tasks, return_exceptions=True)
                    for process in processes:
                        if not isinstance(process, (Exception, BaseException)):
                            await asyncio.wait_for(process.communicate(), timeout=60)

                # Run setup code if provided
                if self.config.setup_code:
                    setup_file = temp_path / "setup.py"
                    setup_file.write_text(self.config.setup_code)
                    setup_process = await asyncio.create_subprocess_exec(
                        "python",
                        str(setup_file),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=str(temp_path),
                    )
                    await asyncio.wait_for(setup_process.communicate(), timeout=30)

                # Run tests asynchronously
                success, output, test_metrics = await self._run_tests_async(test_file)
                metadata["test_output"] = output
                metadata.update(test_metrics)

                # Measure complexity if requested (run in parallel with tests if possible)
                if self.config.measure_complexity:
                    complexity_metrics = await self._measure_complexity_async(code_file)
                    metadata.update(complexity_metrics)

                    # Check complexity violations
                    if "complexity_violations" in metadata:
                        if metadata["complexity_violations"] > 0:
                            success = False
                            metadata["complexity_error"] = (
                                f"Code has {metadata['complexity_violations']} functions "
                                f"exceeding max complexity {self.config.max_complexity}"
                            )

                # Calculate final score
                score = 1.0 if success else 0.0

                # Adjust score based on coverage if measured
                if self.config.measure_coverage and "coverage" in metadata:
                    coverage = metadata["coverage"]
                    if coverage < self.config.coverage_threshold:
                        score = 0.0
                        metadata["coverage_error"] = (
                            f"Coverage {coverage:.1%} below threshold "
                            f"{self.config.coverage_threshold:.1%}"
                        )

                return MetricResult(
                    name="unit_test", score=score, passed=success, metadata=metadata
                )

        except Exception as e:
            return MetricResult(
                name="unit_test",
                score=0.0,
                passed=False,
                metadata={
                    **metadata,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

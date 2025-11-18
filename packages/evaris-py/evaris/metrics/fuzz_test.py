"""Fuzz testing metric for code generation evaluation.

This metric generates diverse test inputs to evaluate code robustness.
Implements ABC checks O.e.1, O.e.2, and O.e.3 for comprehensive input coverage.
"""

import asyncio
import asyncio.subprocess
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from evaris.types import BaseMetric, MetricResult, TestCase


class FuzzTestConfig(BaseModel):
    """Configuration for fuzz testing metric."""

    num_fuzz_cases: int = Field(default=100, description="Number of fuzz test cases")
    timeout_seconds: int = Field(default=30, description="Test execution timeout")
    input_types: list[str] = Field(
        default_factory=lambda: ["int", "float", "str", "list", "dict", "none"],
        description="Input types to generate",
    )
    edge_cases: bool = Field(default=True, description="Include edge cases (empty, null, max/min)")
    memory_stress: bool = Field(
        default=False, description="Include memory stress tests (large inputs)"
    )
    type_confusion: bool = Field(default=True, description="Test with unexpected types")
    boundary_values: bool = Field(
        default=True, description="Test boundary values (0, -1, max_int, etc.)"
    )
    custom_generator: Optional[Callable[[int], list[Any]]] = Field(
        default=None, description="Custom input generator function"
    )


class FuzzTestMetric(BaseMetric):
    """Fuzz testing metric for code generation.

    Generates diverse test inputs to evaluate code robustness and error handling.
    Tests with edge cases, type confusion, and boundary values.

    ABC Compliance:
    - O.e.1: Generates diverse inputs covering various edge cases
    - O.e.2: Ensures comprehensive input variation coverage
    - O.e.3: Generates inputs the code is sensitive to

    Example:
        >>> from evaris.metrics.fuzz_test import FuzzTestMetric, FuzzTestConfig
        >>> config = FuzzTestConfig(num_fuzz_cases=50, edge_cases=True)
        >>> metric = FuzzTestMetric(config)
        >>> tc = TestCase(
        ...     input="Write a function to divide two numbers",
        ...     expected={
        ...         "function_name": "divide",
        ...         "args": ["a", "b"]
        ...     }
        ... )
        >>> code = "def divide(a, b): return a / b"
        >>> result = metric.score(tc, code)
        >>> print(f"Passed {result.metadata['passed_cases']}/{result.metadata['total_cases']}")
    """

    def __init__(self, config: Optional[FuzzTestConfig] = None):
        """Initialize fuzz testing metric.

        Args:
            config: Configuration for fuzz testing. If None, uses defaults.
        """
        self.config = config or FuzzTestConfig()

    def _generate_edge_cases(self) -> list[Any]:
        """Generate edge case inputs.

        ABC O.e.1: Covers diverse edge cases.

        Returns:
            List of edge case values
        """
        edge_cases: list[Any] = []

        if "none" in self.config.input_types:
            edge_cases.append(None)

        if "int" in self.config.input_types:
            edge_cases.extend([0, 1, -1, 2147483647, -2147483648])  # max/min int32

        if "float" in self.config.input_types:
            edge_cases.extend([0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf")])

        if "str" in self.config.input_types:
            edge_cases.extend(
                [
                    "",
                    " ",
                    "a" * 10000,  # Large string
                    "\n\t\r",
                    "unicode: 你好🎉",
                    "special: <>&\"'",
                ]
            )

        if "list" in self.config.input_types:
            edge_cases.extend([[], [None], list(range(1000))])

        if "dict" in self.config.input_types:
            edge_cases.extend([{}, {"": ""}, {None: None}])

        return edge_cases

    def _generate_boundary_values(self) -> list[Any]:
        """Generate boundary value inputs.

        ABC O.e.2: Ensures comprehensive coverage of value ranges.

        Returns:
            List of boundary values
        """
        boundary_values: list[Any] = []

        if "int" in self.config.input_types:
            boundary_values.extend(
                [
                    -1,
                    0,
                    1,  # Around zero
                    255,
                    256,
                    257,  # Around byte boundary
                    65535,
                    65536,
                    65537,  # Around 16-bit boundary
                ]
            )

        if "float" in self.config.input_types:
            boundary_values.extend(
                [
                    0.0,
                    0.1,
                    0.9,
                    1.0,
                    1e-10,
                    1e10,  # Very small and large
                ]
            )

        if "str" in self.config.input_types:
            boundary_values.extend(
                [
                    "a",  # Single char
                    "a" * 255,  # Near common buffer size
                    "a" * 256,
                    "a" * 1024,
                ]
            )

        return boundary_values

    def _generate_type_confusion_cases(self) -> list[tuple[Any, ...]]:
        """Generate type confusion test cases.

        ABC O.e.3: Tests with inputs the code may be sensitive to.

        Returns:
            List of tuples with mixed types
        """
        type_cases: list[tuple[Any, ...]] = []

        # String representations of numbers
        type_cases.extend(
            [
                ("42", "3.14"),
                ("0", "1"),
                ("true", "false"),
            ]
        )

        # Mixed numeric types
        type_cases.extend(
            [
                (42, 3.14),
                (0, 0.0),
                (1, True),
            ]
        )

        # Collections with mixed types
        type_cases.extend(
            [
                ([1, "2", 3.0], [None]),
                ({"a": 1, "b": "2"}, {}),
            ]
        )

        return type_cases

    def _generate_fuzz_inputs(self) -> list[Any]:
        """Generate comprehensive fuzz test inputs.

        ABC Compliance:
        - O.e.1: Diverse inputs covering edge cases
        - O.e.2: Comprehensive input variation coverage
        - O.e.3: Inputs the code is sensitive to

        Returns:
            List of fuzz test inputs
        """
        # Use custom generator if provided
        if self.config.custom_generator is not None:
            return self.config.custom_generator(self.config.num_fuzz_cases)

        inputs: list[Any] = []

        # Add memory stress tests first if enabled (high priority)
        if self.config.memory_stress:
            inputs.extend(
                [
                    "x" * (10**6),  # 1MB string
                    list(range(10**5)),  # 100k element list
                    {f"key{i}": i for i in range(10**4)},  # 10k element dict
                ]
            )

        # Add edge cases
        if self.config.edge_cases:
            inputs.extend(self._generate_edge_cases())

        # Add boundary values
        if self.config.boundary_values:
            inputs.extend(self._generate_boundary_values())

        # Add type confusion cases
        if self.config.type_confusion:
            inputs.extend(self._generate_type_confusion_cases())

        # Add random valid inputs
        import random

        while len(inputs) < self.config.num_fuzz_cases:
            if "int" in self.config.input_types:
                inputs.append(random.randint(-1000, 1000))
            if "float" in self.config.input_types:
                inputs.append(random.uniform(-1000, 1000))
            if "str" in self.config.input_types:
                length = random.randint(0, 100)
                inputs.append("".join(random.choices("abcdefghijklmnopqrstuvwxyz ", k=length)))
            if "list" in self.config.input_types:
                length = random.randint(0, 10)
                inputs.append([random.randint(0, 100) for _ in range(length)])

        return inputs[: self.config.num_fuzz_cases]

    def _create_fuzz_test_file(
        self, code: str, function_name: str, fuzz_inputs: list[Any], temp_dir: Path
    ) -> tuple[Path, Path]:
        """Create temporary code and fuzz test files.

        Args:
            code: Generated code to test
            function_name: Name of function to fuzz
            fuzz_inputs: List of fuzz test inputs
            temp_dir: Temporary directory

        Returns:
            Tuple of (code_file_path, test_file_path)
        """
        # Write code file
        code_file = temp_dir / "generated_code.py"
        code_file.write_text(code)

        # Write fuzz test file
        # TODO: Replace repr() with json.dumps() for better serialization of complex
        #       objects (e.g., objects with custom __repr__, nested structures).
        #       Current limitation: repr() can fail or produce non-evaluable strings
        #       for certain complex objects. json.dumps(default=str) would be more robust.
        # Note: Using repr() currently to maintain compatibility with existing tests.
        test_content = f"""# Generated fuzz test file
import sys
import traceback
sys.path.insert(0, '{temp_dir}')
from generated_code import {function_name}

def run_fuzz_tests():
    \"\"\"Run fuzz tests and report results.\"\"\"
    inputs = {repr(fuzz_inputs)}
    passed = 0
    failed = 0
    errors = []

    for i, test_input in enumerate(inputs):
        try:
            # Handle single vs multiple arguments
            if isinstance(test_input, tuple):
                result = {function_name}(*test_input)
            else:
                result = {function_name}(test_input)
            passed += 1
        except Exception as e:
            failed += 1
            error_msg = f"Input {{i}}: {{test_input!r}} -> {{type(e).__name__}}: {{e}}"
            errors.append(error_msg)

    print(f"FUZZ_RESULTS: {{passed}} passed, {{failed}} failed")
    if errors:
        print("ERRORS:")
        for error in errors[:10]:  # Limit to first 10 errors
            print(f"  {{error}}")

    return passed, failed, errors

if __name__ == "__main__":
    run_fuzz_tests()
"""
        test_file = temp_dir / "fuzz_test.py"
        test_file.write_text(test_content)

        return code_file, test_file

    def _run_fuzz_tests(self, test_file: Path) -> tuple[bool, str, dict[str, Any]]:
        """Run fuzz tests.

        Args:
            test_file: Path to fuzz test file

        Returns:
            Tuple of (success, output, metrics)
        """
        metrics: dict[str, Any] = {}

        try:
            # Run fuzz tests
            result = subprocess.run(
                ["python", str(test_file)],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                cwd=test_file.parent,
            )

            output = result.stdout + "\n" + result.stderr

            # Parse results from output
            import re

            results_match = re.search(r"FUZZ_RESULTS: (\d+) passed, (\d+) failed", output)
            if results_match:
                passed = int(results_match.group(1))
                failed = int(results_match.group(2))
                metrics["passed_cases"] = passed
                metrics["failed_cases"] = failed
                metrics["total_cases"] = passed + failed
                metrics["pass_rate"] = passed / (passed + failed) if (passed + failed) > 0 else 0.0

                # Extract error details
                errors_section = output.split("ERRORS:")
                if len(errors_section) > 1:
                    error_lines = [
                        line.strip() for line in errors_section[1].split("\n") if line.strip()
                    ]
                    metrics["error_samples"] = error_lines[:10]

                # Success if at least 70% of tests pass
                success = metrics["pass_rate"] >= 0.7
            else:
                success = False
                metrics["error"] = "Failed to parse fuzz test results"

            return success, output, metrics

        except subprocess.TimeoutExpired:
            return False, "Fuzz test execution timed out", metrics
        except Exception as e:
            return False, f"Fuzz test execution failed: {str(e)}", metrics

    def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Score generated code using fuzz testing.

        ABC Compliance:
        - O.e.1: Tests with diverse inputs covering edge cases
        - O.e.2: Ensures comprehensive input variation coverage
        - O.e.3: Tests with inputs the code is sensitive to

        Args:
            test_case: Test case with function information
            actual_output: Generated code to test

        Returns:
            MetricResult with fuzz test results and metrics

        Raises:
            ValueError: If test case doesn't contain function information
        """
        if test_case.expected is None:
            raise ValueError("Fuzz test metric requires 'expected' value with function info")

        # Extract function information from expected
        if isinstance(test_case.expected, dict):
            function_name = test_case.expected.get("function_name")
            if not function_name:
                raise ValueError("Expected dict must contain 'function_name' key")
        else:
            raise ValueError("Expected value must be dict with function information")

        metadata: dict[str, Any] = {
            "function_name": function_name,
            "num_fuzz_cases": self.config.num_fuzz_cases,
        }

        try:
            # Generate fuzz inputs
            fuzz_inputs = self._generate_fuzz_inputs()
            metadata["input_types"] = self.config.input_types

            # Create temporary directory for test execution
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create fuzz test files
                code_file, test_file = self._create_fuzz_test_file(
                    str(actual_output), function_name, fuzz_inputs, temp_path
                )

                # Run fuzz tests
                success, output, test_metrics = self._run_fuzz_tests(test_file)
                metadata["test_output"] = output
                metadata.update(test_metrics)

                # Calculate final score based on pass rate
                if "pass_rate" in metadata:
                    score = metadata["pass_rate"]
                else:
                    score = 0.0

                return MetricResult(
                    name="fuzz_test", score=score, passed=success, metadata=metadata
                )

        except Exception as e:
            return MetricResult(
                name="fuzz_test",
                score=0.0,
                passed=False,
                metadata={
                    **metadata,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def _run_fuzz_tests_async(self, test_file: Path) -> tuple[bool, str, dict[str, Any]]:
        """Run fuzz tests asynchronously.

        Args:
            test_file: Path to fuzz test file

        Returns:
            Tuple of (success, output, metrics)
        """
        metrics: dict[str, Any] = {}

        try:
            # Run fuzz tests asynchronously
            process = await asyncio.create_subprocess_exec(
                "python",
                str(test_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(test_file.parent),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.config.timeout_seconds
                )
                output = stdout.decode() + "\n" + stderr.decode()

                # Parse results from output
                import re

                results_match = re.search(r"FUZZ_RESULTS: (\d+) passed, (\d+) failed", output)
                if results_match:
                    passed = int(results_match.group(1))
                    failed = int(results_match.group(2))
                    metrics["passed_cases"] = passed
                    metrics["failed_cases"] = failed
                    metrics["total_cases"] = passed + failed
                    metrics["pass_rate"] = (
                        passed / (passed + failed) if (passed + failed) > 0 else 0.0
                    )

                    # Extract error details
                    errors_section = output.split("ERRORS:")
                    if len(errors_section) > 1:
                        error_lines = [
                            line.strip() for line in errors_section[1].split("\n") if line.strip()
                        ]
                        metrics["error_samples"] = error_lines[:10]

                    # Success if at least 70% of tests pass
                    success = metrics["pass_rate"] >= 0.7
                else:
                    success = False
                    metrics["error"] = "Failed to parse fuzz test results"

                return success, output, metrics

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return False, "Fuzz test execution timed out", metrics

        except Exception as e:
            return False, f"Fuzz test execution failed: {str(e)}", metrics

    async def a_measure(self, test_case: TestCase) -> MetricResult:
        """Asynchronously score generated code using fuzz testing.

        This async version uses asyncio.create_subprocess_exec for running
        fuzz tests, allowing multiple fuzz tests to run concurrently without
        blocking the event loop.

        ABC Compliance:
        - O.e.1: Tests with diverse inputs covering edge cases
        - O.e.2: Ensures comprehensive input variation coverage
        - O.e.3: Tests with inputs the code is sensitive to

        Args:
            test_case: Test case with function information and actual_output

        Returns:
            MetricResult with fuzz test results and metrics

        Raises:
            ValueError: If test case doesn't contain function information or actual_output
        """
        if test_case.actual_output is None:
            raise ValueError("Fuzz test metric requires 'actual_output' in test case")

        if test_case.expected is None:
            raise ValueError("Fuzz test metric requires 'expected' value with function info")

        # Extract function information from expected
        if isinstance(test_case.expected, dict):
            function_name = test_case.expected.get("function_name")
            if not function_name:
                raise ValueError("Expected dict must contain 'function_name' key")
        else:
            raise ValueError("Expected value must be dict with function information")

        metadata: dict[str, Any] = {
            "function_name": function_name,
            "num_fuzz_cases": self.config.num_fuzz_cases,
        }

        try:
            # Generate fuzz inputs (CPU-bound, but fast)
            fuzz_inputs = self._generate_fuzz_inputs()
            metadata["input_types"] = self.config.input_types

            # Create temporary directory for test execution
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create fuzz test files (I/O-bound, but fast)
                code_file, test_file = self._create_fuzz_test_file(
                    str(test_case.actual_output), function_name, fuzz_inputs, temp_path
                )

                # Run fuzz tests asynchronously
                success, output, test_metrics = await self._run_fuzz_tests_async(test_file)
                metadata["test_output"] = output
                metadata.update(test_metrics)

                # Calculate final score based on pass rate
                if "pass_rate" in metadata:
                    score = metadata["pass_rate"]
                else:
                    score = 0.0

                return MetricResult(
                    name="fuzz_test", score=score, passed=success, metadata=metadata
                )

        except Exception as e:
            return MetricResult(
                name="fuzz_test",
                score=0.0,
                passed=False,
                metadata={
                    **metadata,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

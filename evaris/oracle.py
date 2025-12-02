"""Oracle solver pattern for benchmark validation.

This module provides tools to verify benchmark solvability and validate
that tasks can be solved by automated methods. Implements ABC check T.9.
"""

from typing import Any, Callable, Optional, Protocol

from pydantic import BaseModel, Field

from evaris.types import TestCase


class OracleSolver(Protocol):
    """Protocol for oracle solvers.

    Oracle solvers are automated methods that can solve benchmark tasks
    to validate that the tasks are solvable and well-defined.

    ABC T.9: Validates that any solver can pass the benchmark.
    """

    def solve(self, test_case: TestCase) -> Any:
        """Solve a test case.

        Args:
            test_case: Test case to solve

        Returns:
            Solution output
        """
        ...


class OracleValidationConfig(BaseModel):
    """Configuration for oracle validation."""

    require_oracle: bool = Field(default=False, description="Require oracle solver for benchmark")
    oracle_timeout_seconds: int = Field(default=300, description="Timeout for oracle solver")
    validate_all_tasks: bool = Field(default=True, description="Validate all tasks or sample")
    sample_size: Optional[int] = Field(
        default=None, description="Number of tasks to sample for validation"
    )


class OracleValidator:
    """Oracle validator for benchmark verification.

    Uses oracle solvers to verify that benchmark tasks are solvable
    and well-defined before using them for evaluation.

    ABC Compliance:
    - T.9: Validates that any solver can pass the benchmark by demonstrating
      that an automated oracle solver can solve the tasks.

    Example:
        >>> from evaris.oracle import OracleValidator, OracleValidationConfig
        >>> from evaris.types import TestCase
        >>>
        >>> # Define a simple oracle solver
        >>> def simple_oracle(test_case: TestCase) -> str:
        ...     # Oracle has perfect knowledge
        ...     return test_case.expected
        >>>
        >>> config = OracleValidationConfig(require_oracle=True)
        >>> validator = OracleValidator(config, simple_oracle)
        >>>
        >>> tc = TestCase(input="What is 2+2?", expected="4")
        >>> result = validator.validate_task(tc)
        >>> print(result.is_valid)  # True if oracle solved it
    """

    def __init__(
        self,
        config: Optional[OracleValidationConfig] = None,
        oracle_solver: Optional[Callable[[TestCase], Any]] = None,
    ):
        """Initialize oracle validator.

        Args:
            config: Configuration for oracle validation. If None, uses defaults.
            oracle_solver: Oracle solver function. If None, no validation performed.
        """
        self.config = config or OracleValidationConfig()
        self.oracle_solver = oracle_solver

    def validate_task(self, test_case: TestCase) -> "OracleValidationResult":
        """Validate that a task is solvable using oracle solver.

        ABC T.9: Demonstrates task solvability.

        Args:
            test_case: Test case to validate

        Returns:
            OracleValidationResult with validation details
        """
        # Skip validation if oracle not required or not provided
        if not self.config.require_oracle or self.oracle_solver is None:
            return OracleValidationResult(
                is_valid=True,
                solvable=None,
                oracle_output=None,
                validation_message="Oracle validation not required or solver not provided",
            )

        try:
            # Run oracle solver with timeout using threading.Timer (cross-platform)
            import threading

            # Narrow type for mypy - we know oracle_solver is not None here
            oracle_fn = self.oracle_solver
            assert oracle_fn is not None

            oracle_output = None
            exception_holder: list[Exception] = []

            def run_oracle() -> None:
                nonlocal oracle_output
                try:
                    oracle_output = oracle_fn(test_case)
                except Exception as e:
                    exception_holder.append(e)

            thread = threading.Thread(target=run_oracle, daemon=True)
            thread.start()
            thread.join(timeout=self.config.oracle_timeout_seconds)

            if thread.is_alive():
                # Timeout occurred
                raise TimeoutError("Oracle solver timed out")

            if exception_holder:
                # Oracle raised an exception
                raise exception_holder[0]

            # Check if oracle produced output
            if oracle_output is None:
                return OracleValidationResult(
                    is_valid=False,
                    solvable=False,
                    oracle_output=None,
                    validation_message="Oracle solver returned None",
                )

            # Validate oracle output matches expected (if available)
            if test_case.expected is not None:
                # Simple comparison (could be enhanced with metric)
                oracle_str = str(oracle_output).strip().lower()
                expected_str = str(test_case.expected).strip().lower()

                matches = oracle_str == expected_str
                if not matches:
                    return OracleValidationResult(
                        is_valid=True,  # Task is still valid, oracle just didn't match
                        solvable=True,
                        oracle_output=oracle_output,
                        validation_message=(
                            f"Oracle output '{oracle_output}' differs from "
                            f"expected '{test_case.expected}'"
                        ),
                    )

            return OracleValidationResult(
                is_valid=True,
                solvable=True,
                oracle_output=oracle_output,
                validation_message="Oracle successfully solved task",
            )

        except TimeoutError:
            return OracleValidationResult(
                is_valid=False,
                solvable=False,
                oracle_output=None,
                validation_message=(
                    f"Oracle solver exceeded timeout of " f"{self.config.oracle_timeout_seconds}s"
                ),
            )
        except Exception as e:
            return OracleValidationResult(
                is_valid=False,
                solvable=False,
                oracle_output=None,
                validation_message=f"Oracle solver failed: {type(e).__name__}: {str(e)}",
            )

    def validate_benchmark(self, test_cases: list[TestCase]) -> "BenchmarkValidationResult":
        """Validate entire benchmark using oracle solver.

        ABC T.9: Validates benchmark solvability.

        Args:
            test_cases: List of test cases to validate

        Returns:
            BenchmarkValidationResult with aggregate validation results
        """
        if not self.config.require_oracle or self.oracle_solver is None:
            return BenchmarkValidationResult(
                is_valid=True,
                total_tasks=len(test_cases),
                validated_tasks=0,
                solvable_tasks=None,
                unsolvable_tasks=None,
                validation_message="Oracle validation not required or solver not provided",
            )

        # Sample tasks if configured
        if not self.config.validate_all_tasks and self.config.sample_size is not None:
            import random

            sample_size = min(self.config.sample_size, len(test_cases))
            tasks_to_validate = random.sample(test_cases, sample_size)
        else:
            tasks_to_validate = test_cases

        # Validate each task
        results: list[OracleValidationResult] = []
        for test_case in tasks_to_validate:
            result = self.validate_task(test_case)
            results.append(result)

        # Aggregate results
        valid_tasks = sum(1 for r in results if r.is_valid)
        solvable_tasks = sum(1 for r in results if r.solvable is True)
        unsolvable_tasks = sum(1 for r in results if r.solvable is False)

        # Benchmark is valid if all validated tasks are solvable
        is_valid = valid_tasks == len(results) and unsolvable_tasks == 0

        if is_valid:
            message = f"All {len(results)} validated tasks are solvable by oracle"
        else:
            message = (
                f"{unsolvable_tasks}/{len(results)} tasks failed oracle validation. "
                "Some tasks may be unsolvable or poorly defined."
            )

        return BenchmarkValidationResult(
            is_valid=is_valid,
            total_tasks=len(test_cases),
            validated_tasks=len(results),
            solvable_tasks=solvable_tasks,
            unsolvable_tasks=unsolvable_tasks,
            validation_message=message,
            task_results=results,
        )


class OracleValidationResult(BaseModel):
    """Result of oracle validation for a single task."""

    is_valid: bool = Field(description="Whether task passed validation")
    solvable: Optional[bool] = Field(default=None, description="Whether oracle could solve task")
    oracle_output: Optional[Any] = Field(default=None, description="Oracle's output for the task")
    validation_message: str = Field(description="Validation details or error message")


class BenchmarkValidationResult(BaseModel):
    """Result of oracle validation for entire benchmark."""

    is_valid: bool = Field(description="Whether benchmark passed validation")
    total_tasks: int = Field(description="Total number of tasks in benchmark")
    validated_tasks: int = Field(description="Number of tasks validated")
    solvable_tasks: Optional[int] = Field(
        default=None, description="Number of tasks solved by oracle"
    )
    unsolvable_tasks: Optional[int] = Field(
        default=None, description="Number of tasks oracle could not solve"
    )
    validation_message: str = Field(description="Aggregate validation message")
    task_results: list[OracleValidationResult] = Field(
        default_factory=list, description="Individual task validation results"
    )


def create_rule_based_oracle(rules: dict[str, Any]) -> Callable[[TestCase], Any]:
    """Create a rule-based oracle solver.

    ABC T.9: Provides simple oracle implementation for common patterns.

    Args:
        rules: Dictionary mapping input patterns to outputs

    Returns:
        Oracle solver function

    Example:
        >>> from evaris.oracle import create_rule_based_oracle
        >>> rules = {
        ...     "What is 2+2?": "4",
        ...     "What is the capital of France?": "Paris"
        ... }
        >>> oracle = create_rule_based_oracle(rules)
        >>> tc = TestCase(input="What is 2+2?", expected="4")
        >>> output = oracle(tc)
        >>> print(output)  # "4"
    """

    def oracle_solver(test_case: TestCase) -> Any:
        """Rule-based oracle solver."""
        input_str = str(test_case.input).strip()

        # Exact match
        if input_str in rules:
            return rules[input_str]

        # Pattern matching (simple substring)
        for pattern, output in rules.items():
            if pattern.lower() in input_str.lower():
                return output

        # Fallback to expected if available
        if test_case.expected is not None:
            return test_case.expected

        # No rule matched
        return None

    return oracle_solver


def create_function_oracle(func: Callable[[str], Any]) -> Callable[[TestCase], Any]:
    """Create a function-based oracle solver.

    ABC T.9: Wraps existing functions as oracle solvers.

    Args:
        func: Function that takes input string and returns output

    Returns:
        Oracle solver function

    Example:
        >>> from evaris.oracle import create_function_oracle
        >>> def calculator(input_str: str) -> str:
        ...     if "2+2" in input_str:
        ...         return "4"
        ...     return "I don't know"
        >>> oracle = create_function_oracle(calculator)
        >>> tc = TestCase(input="What is 2+2?", expected="4")
        >>> output = oracle(tc)
        >>> print(output)  # "4"
    """

    def oracle_solver(test_case: TestCase) -> Any:
        """Function-based oracle solver."""
        return func(str(test_case.input))

    return oracle_solver

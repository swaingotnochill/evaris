"""State matching metric for environment interaction evaluation.

This metric compares environment states before and after agent actions.
Implements ABC checks O.g.1, O.g.2, and O.g.3 for state-based evaluation.
"""

import asyncio
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field

from evaris.types import BaseMetric, MetricResult, TestCase


class StateMatchConfig(BaseModel):
    """Configuration for state matching metric."""

    comparison_mode: Literal["exact", "subset", "custom"] = Field(
        default="exact", description="State comparison mode"
    )
    ignore_keys: list[str] = Field(
        default_factory=list, description="Keys to ignore in state comparison"
    )
    normalize_types: bool = Field(
        default=True, description="Normalize types before comparison (e.g., int/float)"
    )
    tolerance: float = Field(default=1e-6, description="Tolerance for numeric comparisons")
    check_side_effects: bool = Field(default=True, description="Check for unintended side effects")
    allowed_side_effects: list[str] = Field(
        default_factory=list, description="Explicitly allowed side effect keys"
    )
    custom_comparator: Optional[Callable[[Any, Any], tuple[bool, str]]] = Field(
        default=None, description="Custom state comparison function"
    )


class StateMatchMetric(BaseMetric):
    """State matching metric for environment interaction.

    Compares environment states before and after agent actions to verify
    correct state transitions and detect unintended side effects.

    ABC Compliance:
    - O.g.1: Verifies state changes match expected outcomes
    - O.g.2: Compares final states with goal states
    - O.g.3: Detects unintended side effects in environment

    Example:
        >>> from evaris.metrics.state_match import StateMatchMetric, StateMatchConfig
        >>> config = StateMatchConfig(check_side_effects=True)
        >>> metric = StateMatchMetric(config)
        >>> tc = TestCase(
        ...     input="Move file from /tmp/a.txt to /tmp/b.txt",
        ...     expected={
        ...         "state": {
        ...             "files": {"/tmp/b.txt": "content"},
        ...             "removed": ["/tmp/a.txt"]
        ...         }
        ...     }
        ... )
        >>> actual_state = {
        ...     "files": {"/tmp/b.txt": "content"},
        ...     "removed": ["/tmp/a.txt"]
        ... }
        >>> result = metric.score(tc, actual_state)
        >>> print(result.passed)  # True if states match
    """

    def __init__(self, config: Optional[StateMatchConfig] = None):
        """Initialize state matching metric.

        Args:
            config: Configuration for state matching. If None, uses defaults.
        """
        self.config = config or StateMatchConfig()

    def _normalize_value(self, value: Any) -> Any:
        """Normalize value for comparison.

        ABC O.g.1: Handles type variations in state representation.

        Args:
            value: Value to normalize

        Returns:
            Normalized value
        """
        if not self.config.normalize_types:
            return value

        # Normalize numeric types
        if isinstance(value, bool):
            return value  # Keep booleans as-is
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, dict):
            return {k: self._normalize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._normalize_value(v) for v in value]
        else:
            return value

    def _compare_numeric(self, expected: float, actual: float) -> bool:
        """Compare numeric values with tolerance.

        Args:
            expected: Expected numeric value
            actual: Actual numeric value

        Returns:
            True if values are within tolerance, False otherwise
        """
        return abs(expected - actual) <= self.config.tolerance

    def _compare_values(self, expected: Any, actual: Any, path: str = "") -> tuple[bool, list[str]]:
        """Recursively compare two values.

        Args:
            expected: Expected value
            actual: Actual value
            path: Current path in nested structure

        Returns:
            Tuple of (match, differences)
        """
        differences: list[str] = []

        # Normalize values
        expected_norm = self._normalize_value(expected)
        actual_norm = self._normalize_value(actual)

        # Type check
        if type(expected_norm) is not type(actual_norm):
            differences.append(
                f"{path}: Type mismatch - expected {type(expected_norm).__name__}, "
                f"got {type(actual_norm).__name__}"
            )
            return False, differences

        # Compare based on type
        if isinstance(expected_norm, dict):
            # Compare dictionaries
            all_keys = set(expected_norm.keys()) | set(actual_norm.keys())
            for key in all_keys:
                key_path = f"{path}.{key}" if path else str(key)

                # Skip ignored keys
                if key in self.config.ignore_keys:
                    continue

                if key not in expected_norm:
                    if self.config.comparison_mode == "exact":
                        differences.append(f"{key_path}: Unexpected key in actual state")
                elif key not in actual_norm:
                    differences.append(f"{key_path}: Missing key in actual state")
                else:
                    match, subdiffs = self._compare_values(
                        expected_norm[key], actual_norm[key], key_path
                    )
                    if not match:
                        differences.extend(subdiffs)

        elif isinstance(expected_norm, list):
            # Compare lists
            if len(expected_norm) != len(actual_norm):
                differences.append(
                    f"{path}: List length mismatch - expected {len(expected_norm)}, "
                    f"got {len(actual_norm)}"
                )
            else:
                for i, (exp_item, act_item) in enumerate(zip(expected_norm, actual_norm)):
                    item_path = f"{path}[{i}]"
                    match, subdiffs = self._compare_values(exp_item, act_item, item_path)
                    if not match:
                        differences.extend(subdiffs)

        elif isinstance(expected_norm, float):
            # Compare with tolerance
            if not self._compare_numeric(expected_norm, actual_norm):
                differences.append(
                    f"{path}: Numeric mismatch - expected {expected_norm}, "
                    f"got {actual_norm} (tolerance: {self.config.tolerance})"
                )

        else:
            # Direct comparison
            if expected_norm != actual_norm:
                differences.append(
                    f"{path}: Value mismatch - expected {expected_norm!r}, got {actual_norm!r}"
                )

        return len(differences) == 0, differences

    def _check_side_effects(
        self, expected_state: dict[str, Any], actual_state: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Check for unintended side effects.

        ABC O.g.3: Detects unintended changes in environment state.

        Args:
            expected_state: Expected state
            actual_state: Actual state

        Returns:
            Tuple of (no_violations, side_effects)
        """
        if not self.config.check_side_effects:
            return True, []

        side_effects: list[str] = []

        # Check for unexpected keys in actual state
        expected_keys = set(expected_state.keys())
        actual_keys = set(actual_state.keys())

        unexpected_keys = actual_keys - expected_keys - set(self.config.ignore_keys)
        for key in unexpected_keys:
            # Check if this is an allowed side effect
            if key not in self.config.allowed_side_effects:
                side_effects.append(
                    f"Unintended side effect: unexpected key '{key}' "
                    f"with value {actual_state[key]!r}"
                )

        return len(side_effects) == 0, side_effects

    def _compare_states(
        self, expected_state: Any, actual_state: Any
    ) -> tuple[bool, float, list[str]]:
        """Compare expected and actual states.

        ABC Compliance:
        - O.g.1: Verifies state changes match expected
        - O.g.2: Compares final states with goal states
        - O.g.3: Detects unintended side effects

        Args:
            expected_state: Expected state
            actual_state: Actual state

        Returns:
            Tuple of (match, score, differences)
        """
        # Use custom comparator if provided
        if self.config.custom_comparator is not None:
            match, explanation = self.config.custom_comparator(expected_state, actual_state)
            return match, 1.0 if match else 0.0, [] if match else [explanation]

        # Convert to dicts if needed
        if not isinstance(expected_state, dict):
            expected_state = {"value": expected_state}
        if not isinstance(actual_state, dict):
            actual_state = {"value": actual_state}

        differences: list[str] = []

        # Compare states
        if self.config.comparison_mode == "exact":
            # Exact match required
            match, diffs = self._compare_values(expected_state, actual_state)
            differences.extend(diffs)

        elif self.config.comparison_mode == "subset":
            # Actual state must contain at least expected keys
            for key, expected_value in expected_state.items():
                if key in self.config.ignore_keys:
                    continue

                if key not in actual_state:
                    differences.append(f"Missing expected key: {key}")
                else:
                    match, diffs = self._compare_values(expected_value, actual_state[key], key)
                    if not match:
                        differences.extend(diffs)

            match = len(differences) == 0

        else:  # custom mode handled above
            match = False
            differences.append("Invalid comparison mode")

        # Check for side effects
        has_side_effects = False
        if isinstance(expected_state, dict) and isinstance(actual_state, dict):
            no_violations, side_effects = self._check_side_effects(expected_state, actual_state)
            if not no_violations:
                differences.extend(side_effects)
                match = False
                has_side_effects = True

        # Calculate score
        if match:
            score = 1.0
        elif has_side_effects:
            # No partial credit when side effects detected
            score = 0.0
        else:
            # Partial credit based on number of matching fields
            if isinstance(expected_state, dict) and isinstance(actual_state, dict):
                expected_keys = set(expected_state.keys()) - set(self.config.ignore_keys)
                if expected_keys:
                    matching_keys = sum(
                        1
                        for key in expected_keys
                        if key in actual_state
                        and self._compare_values(expected_state[key], actual_state[key])[0]
                    )
                    score = matching_keys / len(expected_keys)
                else:
                    score = 0.0
            else:
                score = 0.0

        return match, score, differences

    def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Score environment state using state matching.

        ABC Compliance:
        - O.g.1: Verifies state changes match expected outcomes
        - O.g.2: Compares final states with goal states
        - O.g.3: Detects unintended side effects

        Args:
            test_case: Test case with expected state
            actual_output: Agent's actual output state

        Returns:
            MetricResult with score and metadata

        Raises:
            ValueError: If expected state is missing
        """
        if test_case.expected is None:
            raise ValueError("State match metric requires 'expected' value in test case")

        # Extract expected state
        if isinstance(test_case.expected, dict) and "state" in test_case.expected:
            expected_state = test_case.expected["state"]
        else:
            expected_state = test_case.expected

        metadata: dict[str, Any] = {
            "expected_state": expected_state,
            "actual_state": actual_output,
            "comparison_mode": self.config.comparison_mode,
            "check_side_effects": self.config.check_side_effects,
        }

        try:
            # Compare states
            match, score, differences = self._compare_states(expected_state, actual_output)

            metadata["differences"] = differences
            metadata["num_differences"] = len(differences)
            metadata["match"] = match

            return MetricResult(name="state_match", score=score, passed=match, metadata=metadata)

        except Exception as e:
            # Handle errors gracefully
            return MetricResult(
                name="state_match",
                score=0.0,
                passed=False,
                metadata={
                    **metadata,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def a_measure(self, test_case: TestCase) -> MetricResult:
        """Asynchronously score state changes using state matching.

        Since state matching is CPU-bound (dict comparison), this runs the sync version
        in a thread pool to avoid blocking the event loop.

        ABC Compliance:
        - O.g.1: Validates state changes
        - O.g.2: Supports partial state matching
        - O.g.3: Handles numeric tolerance

        Args:
            test_case: Test case with expected state and actual_output containing state

        Returns:
            MetricResult with score and metadata

        Raises:
            ValueError: If expected output or actual_output is missing
        """
        if test_case.actual_output is None:
            raise ValueError("State matching metric requires 'actual_output' in test case")

        # Run CPU-bound state comparison in thread pool
        return await asyncio.to_thread(self.score, test_case, test_case.actual_output)

"""JSON Correctness metric for LLM outputs.

This metric evaluates whether the LLM output is valid JSON
and optionally validates against a provided schema.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase


class JsonCorrectnessConfig(BaseModel):
    """Configuration for JSON correctness metric."""

    threshold: float = Field(
        default=1.0,
        description="Score threshold for passing. Default 1.0 means JSON must be perfect.",
    )
    schema_key: str = Field(
        default="json_schema",
        description="Key in metadata containing expected JSON schema",
    )
    expected_keys_key: str = Field(
        default="expected_keys",
        description="Key in metadata containing list of required keys",
    )


class JsonCorrectnessMetric(BaseMetric):
    """JSON Correctness metric.

    Validates that LLM output is valid JSON and optionally
    matches expected schema or contains required keys.

    This is a deterministic metric - no LLM call needed.

    Scoring:
    - 1.0: Valid JSON with all expected keys/schema match
    - 0.5: Valid JSON but missing some expected keys
    - 0.0: Invalid JSON
    """

    threshold: float = 1.0

    def __init__(self, config: Optional[JsonCorrectnessConfig] = None):
        self.config = config or JsonCorrectnessConfig()
        self.threshold = self.config.threshold

    def validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        if not actual_output:
            raise ValueError("JSON correctness metric requires 'actual_output'")

    def _extract_json(self, text: str) -> tuple[Optional[Any], str]:
        """Extract and parse JSON from text.

        Handles cases where JSON is embedded in markdown code blocks.

        Returns:
            Tuple of (parsed_json, error_message)
        """
        text = text.strip()

        # Try to extract from code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        try:
            return json.loads(text), ""
        except json.JSONDecodeError as e:
            return None, str(e)

    def _check_keys(
        self,
        parsed: Any,
        expected_keys: list[str],
    ) -> tuple[float, list[str], list[str]]:
        """Check if parsed JSON contains expected keys.

        Returns:
            Tuple of (score, found_keys, missing_keys)
        """
        if not isinstance(parsed, dict):
            return 0.0, [], expected_keys

        found = [k for k in expected_keys if k in parsed]
        missing = [k for k in expected_keys if k not in parsed]

        score = len(found) / len(expected_keys) if expected_keys else 1.0
        return score, found, missing

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        output_str = str(actual_output)
        parsed, error = self._extract_json(output_str)

        if parsed is None:
            return MetricResult(
                name="json_correctness",
                score=0.0,
                passed=False,
                metadata={
                    "valid_json": False,
                    "parse_error": error,
                },
            )

        # Check expected keys if provided
        expected_keys = None
        if test_case.metadata:
            expected_keys = test_case.metadata.get(self.config.expected_keys_key)

        if expected_keys:
            key_score, found, missing = self._check_keys(parsed, expected_keys)
            score = key_score
            passed = score >= self.threshold

            return MetricResult(
                name="json_correctness",
                score=score,
                passed=passed,
                metadata={
                    "valid_json": True,
                    "expected_keys": expected_keys,
                    "found_keys": found,
                    "missing_keys": missing,
                },
            )

        # Just validate JSON syntax
        return MetricResult(
            name="json_correctness",
            score=1.0,
            passed=True,
            metadata={
                "valid_json": True,
                "parsed_type": type(parsed).__name__,
            },
        )

"""Answer matching metric with format specification.

This metric parses agent outputs according to specified formats and validates answers.
Implements ABC checks O.h.1 and O.h.2 for format specification and guessing prevention.
"""

import asyncio
import re
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field

from evaris.types import BaseMetric, MetricResult, TestCase


class AnswerMatchConfig(BaseModel):
    """Configuration for answer matching metric."""

    answer_format: Literal["json", "xml", "regex", "delimited"] = Field(
        default="delimited", description="Expected answer format"
    )
    delimiter: str = Field(default="Answer:", description="Delimiter for answer extraction")
    regex_pattern: Optional[str] = Field(
        default=None, description="Regex pattern for answer extraction"
    )
    case_sensitive: bool = Field(default=False, description="Case sensitive matching")
    strip_whitespace: bool = Field(default=True, description="Strip whitespace")
    allow_substring: bool = Field(default=False, description="Allow substring matching")
    custom_parser: Optional[Callable[[str], Any]] = Field(
        default=None, description="Custom parsing function"
    )


class AnswerMatchMetric(BaseMetric):
    """Answer matching metric with format specification.

    Parses agent outputs according to specified formats and validates against
    expected answers. Prevents success by guessing through format requirements.

    ABC Compliance:
    - O.h.1: Specifies required answer formats in challenge descriptions
    - O.h.2: Minimizes possibility of success by random guessing

    Example:
        >>> from evaris.metrics.answer_match import AnswerMatchMetric, AnswerMatchConfig
        >>> config = AnswerMatchConfig(
        ...     answer_format="delimited",
        ...     delimiter="Answer:"
        ... )
        >>> metric = AnswerMatchMetric(config)
        >>> tc = TestCase(
        ...     input="What is 2+2?",
        ...     expected="4"
        ... )
        >>> result = metric.score(tc, "Let me think... Answer: 4")
        >>> print(result.passed)  # True
    """

    def __init__(self, config: Optional[AnswerMatchConfig] = None):
        """Initialize answer matching metric.

        Args:
            config: Configuration for answer matching. If None, uses defaults.
        """
        self.config = config or AnswerMatchConfig()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        text = str(text)

        if self.config.strip_whitespace:
            text = text.strip()

        if not self.config.case_sensitive:
            text = text.lower()

        return text

    def _extract_json_answer(self, output: str) -> Optional[Any]:
        """Extract answer from JSON format.

        Args:
            output: Agent output

        Returns:
            Extracted answer or None if parsing fails
        """
        import json

        try:
            data = json.loads(output)
            # Try common keys
            for key in ["answer", "result", "output", "value"]:
                if key in data:
                    return data[key]
            return data  # Return whole object if no standard key
        except json.JSONDecodeError:
            return None

    def _extract_xml_answer(self, output: str) -> Optional[str]:
        """Extract answer from XML format.

        Args:
            output: Agent output

        Returns:
            Extracted answer or None if parsing fails
        """
        # Look for <answer>...</answer> tags
        match = re.search(r"<answer>(.*?)</answer>", output, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_regex_answer(self, output: str) -> Optional[str]:
        """Extract answer using regex pattern.

        Args:
            output: Agent output

        Returns:
            Extracted answer or None if no match
        """
        if self.config.regex_pattern is None:
            return None

        match = re.search(self.config.regex_pattern, output, re.IGNORECASE | re.DOTALL)
        if match:
            # Return first group if groups exist, otherwise whole match
            return match.group(1) if match.groups() else match.group(0)
        return None

    def _extract_delimited_answer(self, output: str) -> Optional[str]:
        """Extract answer using delimiter.

        ABC O.h.1: Requires specific format with delimiter.

        Args:
            output: Agent output

        Returns:
            Extracted answer or None if delimiter not found
        """
        delimiter = self.config.delimiter
        if delimiter not in output:
            return None

        # Get text after delimiter
        parts = output.split(delimiter, 1)
        if len(parts) < 2:
            return None

        answer = parts[1].strip()

        # Handle common patterns like "Answer: X\n..."
        # Take only the first line or sentence
        lines = answer.split("\n")
        return lines[0].strip()

    def _extract_answer(self, output: str) -> Optional[Any]:
        """Extract answer from agent output based on format.

        ABC O.h.1: Enforces format specification.

        Args:
            output: Agent output

        Returns:
            Extracted answer or None if extraction fails
        """
        # Use custom parser if provided
        if self.config.custom_parser is not None:
            try:
                return self.config.custom_parser(output)
            except Exception:
                return None

        # Use format-specific extraction
        if self.config.answer_format == "json":
            return self._extract_json_answer(output)
        elif self.config.answer_format == "xml":
            return self._extract_xml_answer(output)
        elif self.config.answer_format == "regex":
            return self._extract_regex_answer(output)
        elif self.config.answer_format == "delimited":
            return self._extract_delimited_answer(output)
        else:
            return None

    def _compare_answers(self, expected: Any, actual: Any) -> bool:
        """Compare expected and actual answers.

        Args:
            expected: Expected answer
            actual: Actual answer

        Returns:
            True if answers match, False otherwise
        """
        # Normalize both values
        expected_normalized = self._normalize_text(str(expected))
        actual_normalized = self._normalize_text(str(actual))

        # Check for match
        if self.config.allow_substring:
            return expected_normalized in actual_normalized
        else:
            return expected_normalized == actual_normalized

    def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Score agent output using answer matching.

        ABC Compliance:
        - O.h.1: Enforces specific answer format
        - O.h.2: Format requirement minimizes guessing success

        Args:
            test_case: Test case with expected answer
            actual_output: Agent's actual output

        Returns:
            MetricResult with score and metadata

        Raises:
            ValueError: If expected output is missing
        """
        if test_case.expected is None:
            raise ValueError("Answer matching metric requires 'expected' value in test case")

        metadata: dict[str, Any] = {
            "expected": test_case.expected,
            "actual_raw": actual_output,
            "answer_format": self.config.answer_format,
            "delimiter": (
                self.config.delimiter if self.config.answer_format == "delimited" else None
            ),
        }

        try:
            # Extract answer from output
            extracted_answer = self._extract_answer(str(actual_output))
            metadata["extracted_answer"] = extracted_answer

            if extracted_answer is None:
                # Failed to extract answer in required format
                metadata["error"] = "Failed to extract answer in required format"
                return MetricResult(name="answer_match", score=0.0, passed=False, metadata=metadata)

            # Compare extracted answer with expected
            matches = self._compare_answers(test_case.expected, extracted_answer)
            score = 1.0 if matches else 0.0

            metadata["match"] = matches

            return MetricResult(name="answer_match", score=score, passed=matches, metadata=metadata)

        except Exception as e:
            # Handle errors gracefully
            return MetricResult(
                name="answer_match",
                score=0.0,
                passed=False,
                metadata={
                    **metadata,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def a_measure(self, test_case: TestCase) -> MetricResult:
        """Asynchronously score agent output using answer matching.

        Since answer matching is CPU-bound (parsing), this runs the sync version
        in a thread pool to avoid blocking the event loop.

        ABC Compliance:
        - O.h.1: Enforces specific answer format
        - O.h.2: Format requirement minimizes guessing success

        Args:
            test_case: Test case with expected answer and actual_output

        Returns:
            MetricResult with score and metadata

        Raises:
            ValueError: If expected output or actual_output is missing
        """
        if test_case.actual_output is None:
            raise ValueError("Answer matching metric requires 'actual_output' in test case")

        # Run CPU-bound parsing in thread pool
        return await asyncio.to_thread(self.score, test_case, test_case.actual_output)

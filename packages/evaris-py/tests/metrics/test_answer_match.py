"""Tests for answer matching metric."""

import pytest

from evaris.metrics.answer_match import AnswerMatchConfig, AnswerMatchMetric
from evaris.types import TestCase


class TestAnswerMatchConfig:
    """Tests for AnswerMatchConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AnswerMatchConfig()

        assert config.answer_format == "delimited"
        assert config.delimiter == "Answer:"
        assert config.regex_pattern is None
        assert config.case_sensitive is False
        assert config.strip_whitespace is True
        assert config.allow_substring is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = AnswerMatchConfig(
            answer_format="json",
            case_sensitive=True,
            allow_substring=True,
        )

        assert config.answer_format == "json"
        assert config.case_sensitive is True
        assert config.allow_substring is True


class TestAnswerMatchMetric:
    """Tests for AnswerMatchMetric."""

    def test_delimited_format_basic(self):
        """Test basic delimited format extraction."""
        config = AnswerMatchConfig(answer_format="delimited", delimiter="Answer:")
        metric = AnswerMatchMetric(config)

        tc = TestCase(input="test", expected="Paris", actual_output="Let me think... Answer: Paris")
        result = metric.score(tc, "Let me think... Answer: Paris")

        assert result.passed is True
        assert result.metadata["extracted_answer"] == "Paris"

    def test_delimited_format_not_found(self):
        """Test delimited format when delimiter not found."""
        metric = AnswerMatchMetric()

        tc = TestCase(input="test", expected="Paris", actual_output="Let me think... Paris")
        result = metric.score(tc, "Let me think... Paris")  # No delimiter

        assert result.passed is False
        assert result.metadata["extracted_answer"] is None

    def test_json_format(self):
        """Test JSON format extraction."""
        config = AnswerMatchConfig(answer_format="json")
        metric = AnswerMatchMetric(config)

        tc = TestCase(
            input="test", expected="Paris", actual_output='{"answer": "Paris", "confidence": 0.95}'
        )
        result = metric.score(tc, '{"answer": "Paris", "confidence": 0.95}')

        assert result.passed is True
        assert result.metadata["extracted_answer"] == "Paris"

    def test_xml_format(self):
        """Test XML format extraction."""
        config = AnswerMatchConfig(answer_format="xml")
        metric = AnswerMatchMetric(config)

        tc = TestCase(
            input="test",
            expected="Paris",
            actual_output="<response><answer>Paris</answer></response>",
        )
        result = metric.score(tc, "<response><answer>Paris</answer></response>")

        assert result.passed is True
        assert result.metadata["extracted_answer"] == "Paris"

    def test_regex_format(self):
        """Test regex format extraction."""
        config = AnswerMatchConfig(answer_format="regex", regex_pattern=r"Final answer: (.*?)$")
        metric = AnswerMatchMetric(config)

        tc = TestCase(
            input="test", expected="Paris", actual_output="Thinking... Final answer: Paris"
        )
        result = metric.score(tc, "Thinking... Final answer: Paris")

        assert result.passed is True
        assert result.metadata["extracted_answer"] == "Paris"

    def test_case_insensitive_matching(self):
        """Test case insensitive matching."""
        config = AnswerMatchConfig(case_sensitive=False)
        metric = AnswerMatchMetric(config)

        tc = TestCase(input="test", expected="paris", actual_output="Answer: PARIS")
        result = metric.score(tc, "Answer: PARIS")

        assert result.passed is True

    def test_case_sensitive_matching(self):
        """Test case sensitive matching."""
        config = AnswerMatchConfig(case_sensitive=True)
        metric = AnswerMatchMetric(config)

        tc = TestCase(input="test", expected="paris", actual_output="Answer: PARIS")
        result = metric.score(tc, "Answer: PARIS")

        assert result.passed is False

    def test_substring_matching(self):
        """Test substring matching."""
        config = AnswerMatchConfig(allow_substring=True)
        metric = AnswerMatchMetric(config)

        tc = TestCase(
            input="test", expected="Paris", actual_output="Answer: The city of Paris in France"
        )
        result = metric.score(tc, "Answer: The city of Paris in France")

        assert result.passed is True

    def test_no_expected_raises(self):
        """Test that missing expected value raises error."""
        metric = AnswerMatchMetric()

        tc = TestCase(input="test", expected=None, actual_output="output")

        with pytest.raises(ValueError, match="expected"):
            metric.score(tc, "output")

    def test_abc_compliance_o_h_1(self):
        """Test ABC O.h.1: Specifies required answer format."""
        metric = AnswerMatchMetric()

        tc = TestCase(input="test", expected="Paris", actual_output="Paris")

        # Without delimiter, should fail
        result1 = metric.score(tc, "Paris")
        assert result1.passed is False

        # With delimiter, should pass
        tc2 = TestCase(input="test", expected="Paris", actual_output="Answer: Paris")
        result2 = metric.score(tc2, "Answer: Paris")
        assert result2.passed is True

    def test_abc_compliance_o_h_2(self):
        """Test ABC O.h.2: Minimizes guessing through format requirements."""
        metric = AnswerMatchMetric()

        tc = TestCase(input="test", expected="A", actual_output="A")

        # Simple guess without format should fail
        result = metric.score(tc, "A")
        assert result.passed is False

        # Must follow format
        tc2 = TestCase(input="test", expected="A", actual_output="Answer: A")
        result2 = metric.score(tc2, "Answer: A")
        assert result2.passed is True

    def test_json_format_extraction(self):
        """Test JSON format answer extraction."""
        config = AnswerMatchConfig(answer_format="json")
        metric = AnswerMatchMetric(config)

        tc = TestCase(input="test", expected="42", actual_output='{"answer": "42"}')
        result = metric.score(tc, '{"answer": "42"}')
        assert result.passed is True

    def test_xml_format_extraction(self):
        """Test XML format answer extraction."""
        config = AnswerMatchConfig(answer_format="xml")
        metric = AnswerMatchMetric(config)

        tc = TestCase(input="test", expected="42", actual_output="<answer>42</answer>")
        result = metric.score(tc, "<answer>42</answer>")
        assert result.passed is True

    def test_regex_format_extraction(self):
        """Test regex format answer extraction."""
        config = AnswerMatchConfig(answer_format="regex", regex_pattern=r"Result:\s*(\d+)")
        metric = AnswerMatchMetric(config)

        tc = TestCase(input="test", expected="42", actual_output="Result: 42")
        result = metric.score(tc, "Result: 42")
        assert result.passed is True

    def test_xml_format_no_match(self):
        """Test XML format when answer tag is missing."""
        config = AnswerMatchConfig(answer_format="xml")
        metric = AnswerMatchMetric(config)

        tc = TestCase(input="test", expected="42", actual_output="No answer tag here")
        result = metric.score(tc, "No answer tag here")
        assert result.passed is False
        assert "error" in result.metadata

    def test_regex_format_no_match(self):
        """Test regex format when pattern doesn't match."""
        config = AnswerMatchConfig(answer_format="regex", regex_pattern=r"Result:\s*(\d+)")
        metric = AnswerMatchMetric(config)

        tc = TestCase(input="test", expected="42", actual_output="No match here")
        result = metric.score(tc, "No match here")
        assert result.passed is False
        assert "error" in result.metadata

    def test_json_invalid_format(self):
        """Test JSON format with invalid JSON."""
        config = AnswerMatchConfig(answer_format="json")
        metric = AnswerMatchMetric(config)

        tc = TestCase(input="test", expected="42", actual_output="{invalid json}")
        result = metric.score(tc, "{invalid json}")
        assert result.passed is False
        assert "error" in result.metadata

    def test_allow_substring_matching(self):
        """Test substring matching when enabled."""
        config = AnswerMatchConfig(allow_substring=True)
        metric = AnswerMatchMetric(config)

        tc = TestCase(input="test", expected="Paris", actual_output="Answer: Paris is the capital")
        result = metric.score(tc, "Answer: Paris is the capital")
        assert result.passed is True

    def test_custom_parser(self):
        """Test custom parser function."""

        def custom_parser(text: str) -> str:
            return text.split(":")[-1].strip()

        config = AnswerMatchConfig(custom_parser=custom_parser)
        metric = AnswerMatchMetric(config)

        tc = TestCase(input="test", expected="42", actual_output="Custom: 42")
        result = metric.score(tc, "Custom: 42")
        assert result.passed is True

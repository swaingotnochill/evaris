"""Tests for Agentic evaluation metrics.

This module tests:
- ToolCorrectnessMetric: Evaluates if agent used correct tools
- TaskCompletionMetric: Evaluates if agent completed the task

Following TDD: tests written before implementation.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaris.types import TestCase


class TestToolCorrectnessMetric:
    """Tests for ToolCorrectnessMetric.

    Formula: Number of Correctly Used Tools / Total Tools Called
    Required inputs: input, actual_output, tools_called, expected_tools
    """

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        from evaris.metrics.agentic import ToolCorrectnessConfig

        config = ToolCorrectnessConfig()
        assert config.threshold == 0.5
        assert config.tools_called_key == "tools_called"
        assert config.expected_tools_key == "expected_tools"

    def test_metric_name(self) -> None:
        """Test metric has correct name."""
        from evaris.metrics.agentic import ToolCorrectnessMetric

        metric = ToolCorrectnessMetric()
        assert metric.name == "ToolCorrectnessMetric"

    def test_validate_inputs_missing_tools_called(self) -> None:
        """Test validation fails when tools_called is missing."""
        from evaris.metrics.agentic import ToolCorrectnessMetric

        metric = ToolCorrectnessMetric()
        test_case = TestCase(
            input="Get the weather", expected=None, metadata={"expected_tools": ["get_weather"]}
        )

        with pytest.raises(ValueError, match="tools_called"):
            metric.validate_inputs(test_case, "some output")

    def test_validate_inputs_missing_expected_tools(self) -> None:
        """Test validation fails when expected_tools is missing."""
        from evaris.metrics.agentic import ToolCorrectnessMetric

        metric = ToolCorrectnessMetric()
        test_case = TestCase(
            input="Get the weather", expected=None, metadata={"tools_called": ["get_weather"]}
        )

        with pytest.raises(ValueError, match="expected_tools"):
            metric.validate_inputs(test_case, "some output")

    def test_calculate_correctness_perfect_match(self) -> None:
        """Test correctness calculation with perfect tool match."""
        from evaris.metrics.agentic import ToolCorrectnessMetric

        metric = ToolCorrectnessMetric()
        tools_called = ["get_weather", "send_notification"]
        expected_tools = ["get_weather", "send_notification"]

        score = metric._calculate_correctness(tools_called, expected_tools)

        assert score == 1.0

    def test_calculate_correctness_partial_match(self) -> None:
        """Test correctness calculation with partial tool match."""
        from evaris.metrics.agentic import ToolCorrectnessMetric

        metric = ToolCorrectnessMetric()
        tools_called = ["get_weather", "wrong_tool"]
        expected_tools = ["get_weather", "send_notification"]

        score = metric._calculate_correctness(tools_called, expected_tools)

        assert score == 0.5  # 1 correct out of 2 called

    def test_calculate_correctness_no_match(self) -> None:
        """Test correctness calculation with no tool match."""
        from evaris.metrics.agentic import ToolCorrectnessMetric

        metric = ToolCorrectnessMetric()
        tools_called = ["wrong_tool1", "wrong_tool2"]
        expected_tools = ["get_weather", "send_notification"]

        score = metric._calculate_correctness(tools_called, expected_tools)

        assert score == 0.0

    def test_calculate_correctness_no_tools_called(self) -> None:
        """Test correctness calculation when no tools were called."""
        from evaris.metrics.agentic import ToolCorrectnessMetric

        metric = ToolCorrectnessMetric()
        tools_called = []
        expected_tools = ["get_weather"]

        score = metric._calculate_correctness(tools_called, expected_tools)

        assert score == 0.0  # No tools called is failure

    @pytest.mark.asyncio
    async def test_measure_perfect_tool_usage(self) -> None:
        """Test scoring with perfect tool usage."""
        from evaris.metrics.agentic import ToolCorrectnessMetric

        metric = ToolCorrectnessMetric()
        test_case = TestCase(
            input="What's the weather in NYC?",
            expected=None,
            metadata={"tools_called": ["get_weather"], "expected_tools": ["get_weather"]},
        )

        result = await metric.a_measure(test_case, "The weather in NYC is sunny.")

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_measure_wrong_tool_usage(self) -> None:
        """Test scoring with incorrect tool usage."""
        from evaris.metrics.agentic import ToolCorrectnessMetric

        metric = ToolCorrectnessMetric()
        test_case = TestCase(
            input="What's the weather in NYC?",
            expected=None,
            metadata={
                "tools_called": ["send_email"],  # Wrong tool
                "expected_tools": ["get_weather"],
            },
        )

        result = await metric.a_measure(test_case, "Email sent.")

        assert result.score == 0.0
        assert result.passed is False


class TestTaskCompletionMetric:
    """Tests for TaskCompletionMetric.

    Evaluates how well the agent completed its intended task.
    Required inputs: input, actual_output, task description (optional)
    """

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        from evaris.metrics.agentic import TaskCompletionConfig

        config = TaskCompletionConfig()
        assert config.threshold == 0.5
        assert config.task_key == "task"

    def test_metric_name(self) -> None:
        """Test metric has correct name."""
        from evaris.metrics.agentic import TaskCompletionMetric

        metric = TaskCompletionMetric()
        assert metric.name == "TaskCompletionMetric"

    def test_validate_inputs_missing_output(self) -> None:
        """Test validation fails when actual_output is missing."""
        from evaris.metrics.agentic import TaskCompletionMetric

        metric = TaskCompletionMetric()
        test_case = TestCase(input="Do something", expected=None)

        with pytest.raises(ValueError, match="actual_output"):
            metric.validate_inputs(test_case, None)

    @pytest.mark.asyncio
    @patch("evaris.metrics.agentic.task_completion.get_provider")
    async def test_measure_task_completed(self, mock_get_provider: Any) -> None:
        """Test scoring when task is successfully completed."""
        from evaris.metrics.agentic import TaskCompletionMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.9, "task_completed": true, "reasoning": "The agent successfully retrieved the weather information for NYC as requested."}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = TaskCompletionMetric()
        test_case = TestCase(
            input="What's the weather in NYC?",
            expected=None,
            metadata={"task": "Get weather information for a location"},
        )

        result = await metric.a_measure(test_case, "The weather in NYC is 72F and sunny.")

        assert result.score == 0.9
        assert result.passed is True
        assert result.metadata["task_completed"] is True

    @pytest.mark.asyncio
    @patch("evaris.metrics.agentic.task_completion.get_provider")
    async def test_measure_task_not_completed(self, mock_get_provider: Any) -> None:
        """Test scoring when task is not completed."""
        from evaris.metrics.agentic import TaskCompletionMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.2, "task_completed": false, "reasoning": "The agent failed to provide weather information and instead gave an unrelated response."}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = TaskCompletionMetric()
        test_case = TestCase(
            input="What's the weather in NYC?",
            expected=None,
        )

        result = await metric.a_measure(test_case, "I cannot help with that.")

        assert result.score == 0.2
        assert result.passed is False
        assert result.metadata["task_completed"] is False

    @pytest.mark.asyncio
    @patch("evaris.metrics.agentic.task_completion.get_provider")
    async def test_measure_with_trace_data(self, mock_get_provider: Any) -> None:
        """Test scoring with agent trace data in metadata."""
        from evaris.metrics.agentic import TaskCompletionMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 1.0, "task_completed": true, "reasoning": "Task fully completed with all required steps executed."}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = TaskCompletionMetric()
        test_case = TestCase(
            input="Book a flight to NYC",
            expected=None,
            metadata={
                "task": "Book a flight",
                "trace": [
                    {"step": "search_flights", "status": "success"},
                    {"step": "select_flight", "status": "success"},
                    {"step": "book_flight", "status": "success"},
                ],
            },
        )

        result = await metric.a_measure(test_case, "Flight booked successfully!")

        assert result.score == 1.0
        assert result.passed is True


class TestAgenticMetricsIntegration:
    """Integration tests for Agentic metrics."""

    def test_all_agentic_metrics_importable(self) -> None:
        """Test all agentic metrics can be imported."""
        from evaris.metrics.agentic import (
            TaskCompletionConfig,
            TaskCompletionMetric,
            ToolCorrectnessConfig,
            ToolCorrectnessMetric,
        )

        assert ToolCorrectnessMetric is not None
        assert TaskCompletionMetric is not None

    def test_metrics_share_base_class(self) -> None:
        """Test all agentic metrics inherit from BaseMetric."""
        from evaris.core.protocols import BaseMetric
        from evaris.metrics.agentic import TaskCompletionMetric, ToolCorrectnessMetric

        assert issubclass(ToolCorrectnessMetric, BaseMetric)
        assert issubclass(TaskCompletionMetric, BaseMetric)

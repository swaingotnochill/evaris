"""Unit tests for Golden and EvaluationDataset functionality."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from evaris.dataset import EvaluationDataset, load_dataset
from evaris.types import Golden, TestCase

# Get the fixtures directory path
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestLoadDatasetAsGoldens:
    """Tests for loading datasets as Goldens."""

    def test_load_jsonl_as_goldens(self) -> None:
        """Test loading JSONL file as Golden objects."""
        path = FIXTURES_DIR / "valid.jsonl"
        goldens = load_dataset(str(path), as_goldens=True)

        assert len(goldens) == 3
        assert all(isinstance(g, Golden) for g in goldens)

        # Check that Golden objects don't have actual_output
        assert goldens[0].input == "What is 2+2?"
        assert goldens[0].expected == "4"
        assert not hasattr(goldens[0], "actual_output")

    def test_load_jsonl_as_goldens_no_expected(self) -> None:
        """Test loading JSONL as Goldens without expected field."""
        path = FIXTURES_DIR / "valid_no_expected.jsonl"
        goldens = load_dataset(str(path), as_goldens=True)

        assert len(goldens) == 3
        assert goldens[0].input == "What is 2+2?"
        assert goldens[0].expected is None

    def test_load_json_as_goldens(self) -> None:
        """Test loading JSON file as Golden objects."""
        path = FIXTURES_DIR / "valid.json"
        goldens = load_dataset(str(path), as_goldens=True)

        assert len(goldens) == 3
        assert all(isinstance(g, Golden) for g in goldens)
        assert goldens[0].input == "What is 2+2?"
        assert goldens[0].expected == "4"

    def test_load_csv_as_goldens(self) -> None:
        """Test loading CSV file as Golden objects."""
        path = FIXTURES_DIR / "valid.csv"
        goldens = load_dataset(str(path), as_goldens=True)

        assert len(goldens) == 3
        assert all(isinstance(g, Golden) for g in goldens)
        assert goldens[0].input == "What is 2+2?"
        assert goldens[0].expected == "4"


class TestLoadDatasetAsTestCases:
    """Tests for loading datasets with actual_output as TestCases."""

    def test_load_jsonl_with_actual_output(self) -> None:
        """Test loading JSONL file with actual_output field as TestCases."""
        # This test requires a fixture with actual_output
        # We'll create it on the fly for now
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"input": "2+2?", "expected": "4", "actual_output": "4"}) + "\n")
            f.write(
                json.dumps({"input": "Capital?", "expected": "Paris", "actual_output": "Paris"})
                + "\n"
            )
            temp_path = f.name

        try:
            test_cases = load_dataset(temp_path, as_goldens=False)

            assert len(test_cases) == 2
            assert all(isinstance(tc, TestCase) for tc in test_cases)

            # Type narrowing: we know test_cases[0] is TestCase
            first_tc = test_cases[0]
            assert isinstance(first_tc, TestCase)
            assert first_tc.input == "2+2?"
            assert first_tc.expected == "4"
            assert first_tc.actual_output == "4"
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_without_actual_output_raises_error(self) -> None:
        """Test that loading as TestCases without actual_output raises error."""
        path = FIXTURES_DIR / "valid.jsonl"  # This file doesn't have actual_output

        with pytest.raises(Exception):  # Should raise ValidationError
            load_dataset(str(path), as_goldens=False)


class TestEvaluationDataset:
    """Tests for EvaluationDataset class."""

    def test_create_empty_dataset(self) -> None:
        """Test creating an empty EvaluationDataset."""
        dataset = EvaluationDataset(name="test-dataset")

        assert dataset.name == "test-dataset"
        assert dataset.goldens == []
        assert dataset.test_cases == []
        assert dataset.metadata == {}

    def test_create_dataset_with_goldens(self) -> None:
        """Test creating dataset with initial goldens."""
        goldens = [
            Golden(input="What is 2+2?", expected="4"),
            Golden(input="Capital of France?", expected="Paris"),
        ]
        dataset = EvaluationDataset(name="test-dataset", goldens=goldens)

        assert len(dataset.goldens) == 2
        assert dataset.goldens[0].input == "What is 2+2?"

    def test_add_golden(self) -> None:
        """Test adding a golden to the dataset."""
        dataset = EvaluationDataset(name="test")
        golden = Golden(input="test input", expected="test output")

        dataset.add_golden(golden)

        assert len(dataset.goldens) == 1
        assert dataset.goldens[0].input == "test input"

    def test_add_test_case(self) -> None:
        """Test adding a test case to the dataset."""
        dataset = EvaluationDataset(name="test")
        test_case = TestCase(input="test", actual_output="output", expected="output")

        dataset.add_test_case(test_case)

        assert len(dataset.test_cases) == 1
        assert dataset.test_cases[0].input == "test"

    def test_from_goldens_classmethod(self) -> None:
        """Test creating dataset from goldens using classmethod."""
        goldens = [
            Golden(input="q1", expected="a1"),
            Golden(input="q2", expected="a2"),
        ]
        dataset = EvaluationDataset.from_goldens(name="my-dataset", goldens=goldens)

        assert dataset.name == "my-dataset"
        assert len(dataset.goldens) == 2
        assert len(dataset.test_cases) == 0

    def test_from_file_as_goldens(self) -> None:
        """Test loading dataset from file as goldens."""
        path = FIXTURES_DIR / "valid.jsonl"
        dataset = EvaluationDataset.from_file(str(path), as_goldens=True)

        assert isinstance(dataset, EvaluationDataset)
        assert len(dataset.goldens) == 3
        assert dataset.goldens[0].input == "What is 2+2?"

    async def test_generate_test_cases_async(self) -> None:
        """Test generating test cases asynchronously from goldens."""

        async def mock_agent(input: Any) -> str:
            """Mock agent that echoes input."""
            await asyncio.sleep(0.01)  # Simulate async work
            return f"Response to: {input}"

        goldens = [
            Golden(input="q1", expected="a1"),
            Golden(input="q2", expected="a2"),
            Golden(input="q3", expected="a3"),
        ]
        dataset = EvaluationDataset(name="test", goldens=goldens)

        await dataset.generate_test_cases_async(mock_agent, max_concurrency=2)

        assert len(dataset.test_cases) == 3
        assert all(isinstance(tc, TestCase) for tc in dataset.test_cases)
        assert dataset.test_cases[0].input == "q1"
        assert dataset.test_cases[0].actual_output == "Response to: q1"
        assert dataset.test_cases[0].expected == "a1"

    async def test_generate_test_cases_async_with_progress(self) -> None:
        """Test that progress callback is called during generation."""

        async def mock_agent(input: Any) -> str:
            await asyncio.sleep(0.01)
            return "output"

        progress_calls = []

        def progress_callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        goldens = [Golden(input=f"q{i}") for i in range(5)]
        dataset = EvaluationDataset(name="test", goldens=goldens)

        await dataset.generate_test_cases_async(
            mock_agent, max_concurrency=2, progress_callback=progress_callback
        )

        assert len(progress_calls) == 5
        assert progress_calls[0] == (1, 5)
        assert progress_calls[4] == (5, 5)

    async def test_generate_test_cases_respects_concurrency_limit(self) -> None:
        """Test that concurrency limit is respected."""
        active_count = 0
        max_active = 0

        async def mock_agent(input: Any) -> str:
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.05)
            active_count -= 1
            return "output"

        goldens = [Golden(input=f"q{i}") for i in range(10)]
        dataset = EvaluationDataset(name="test", goldens=goldens)

        await dataset.generate_test_cases_async(mock_agent, max_concurrency=3)

        # Max active should not exceed concurrency limit
        assert max_active <= 3
        assert len(dataset.test_cases) == 10

    def test_dataset_with_metadata(self) -> None:
        """Test that dataset metadata is preserved."""
        dataset = EvaluationDataset(name="test", metadata={"version": "1.0", "author": "test_user"})

        assert dataset.metadata["version"] == "1.0"
        assert dataset.metadata["author"] == "test_user"


class TestEvaluationDatasetEdgeCases:
    """Test edge cases for EvaluationDataset."""

    async def test_generate_with_empty_goldens(self) -> None:
        """Test generating test cases with no goldens."""

        async def mock_agent(input: Any) -> str:
            return "output"

        dataset = EvaluationDataset(name="test", goldens=[])

        await dataset.generate_test_cases_async(mock_agent)

        assert len(dataset.test_cases) == 0

    async def test_generate_preserves_metadata(self) -> None:
        """Test that metadata from goldens is preserved in test cases."""

        async def mock_agent(input: Any) -> str:
            return "output"

        goldens = [
            Golden(input="q1", expected="a1", metadata={"category": "math", "difficulty": "easy"}),
        ]
        dataset = EvaluationDataset(name="test", goldens=goldens)

        await dataset.generate_test_cases_async(mock_agent)

        assert dataset.test_cases[0].metadata["category"] == "math"
        assert dataset.test_cases[0].metadata["difficulty"] == "easy"

    async def test_generate_handles_agent_errors(self) -> None:
        """Test that agent errors are propagated."""

        async def failing_agent(input: Any) -> str:
            raise ValueError("Agent failed")

        goldens = [Golden(input="q1")]
        dataset = EvaluationDataset(name="test", goldens=goldens)

        with pytest.raises(ValueError, match="Agent failed"):
            await dataset.generate_test_cases_async(failing_agent)

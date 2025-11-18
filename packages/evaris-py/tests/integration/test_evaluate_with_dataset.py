"""Integration tests for evaluate() with dataset loading."""

from pathlib import Path

import pytest

from evaris import evaluate, load_dataset

# Get the fixtures directory path
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestEvaluateWithLoadedDataset:
    """Tests for integrating load_dataset with evaluate()."""

    def test_evaluate_with_jsonl_dataset(self) -> None:
        """Test end-to-end evaluation with JSONL dataset."""
        # Load dataset
        test_cases = load_dataset(str(FIXTURES_DIR / "valid.jsonl"))

        # Define a simple agent
        def simple_agent(query: str) -> str:
            # Simple pattern matching for test cases
            if "2+2" in query:
                return "4"
            elif "capital of France" in query:
                return "Paris"
            elif "10*5" in query:
                return "50"
            return "unknown"

        # Run evaluation
        result = evaluate(
            name="jsonl-test",
            task=simple_agent,
            data=test_cases,
            metrics=["exact_match", "latency"],
        )

        assert result.name == "jsonl-test"
        assert result.total == 3
        assert result.passed == 3
        assert result.failed == 0
        assert result.accuracy == 1.0
        assert result.avg_latency_ms >= 0

    def test_evaluate_with_csv_dataset(self) -> None:
        """Test end-to-end evaluation with CSV dataset."""
        # Load dataset
        test_cases = load_dataset(str(FIXTURES_DIR / "valid.csv"))

        def simple_agent(query: str) -> str:
            if "2+2" in query:
                return "4"
            elif "capital of France" in query:
                return "Paris"
            elif "10*5" in query:
                return "50"
            return "unknown"

        result = evaluate(
            name="csv-test", task=simple_agent, data=test_cases, metrics=["exact_match"]
        )

        assert result.total == 3
        assert result.passed == 3
        assert result.accuracy == 1.0

    def test_evaluate_with_json_dataset(self) -> None:
        """Test end-to-end evaluation with JSON dataset."""
        # Load dataset
        test_cases = load_dataset(str(FIXTURES_DIR / "valid.json"))

        def simple_agent(query: str) -> str:
            if "2+2" in query:
                return "4"
            elif "capital of France" in query:
                return "Paris"
            elif "10*5" in query:
                return "50"
            return "unknown"

        result = evaluate(
            name="json-test", task=simple_agent, data=test_cases, metrics=["exact_match"]
        )

        assert result.total == 3
        assert result.passed == 3

    def test_evaluate_with_dataset_no_expected_values(self) -> None:
        """Test evaluation with dataset that has no expected values."""
        # Load dataset without expected values
        test_cases = load_dataset(str(FIXTURES_DIR / "valid_no_expected.jsonl"))

        def simple_agent(query: str) -> str:
            return f"Response to: {query}"

        # Run evaluation with only latency metric (doesn't need expected)
        result = evaluate(
            name="no-expected-test", task=simple_agent, data=test_cases, metrics=["latency"]
        )

        assert result.total == 3
        # All should pass since latency always passes
        assert result.passed == 3
        assert result.accuracy == 1.0

    def test_evaluate_with_dataset_and_failures(self) -> None:
        """Test evaluation with dataset where some tests fail."""
        test_cases = load_dataset(str(FIXTURES_DIR / "valid.jsonl"))

        # Agent that only answers math questions correctly
        def partial_agent(query: str) -> str:
            if "2+2" in query or "10*5" in query:
                if "2+2" in query:
                    return "4"
                else:
                    return "50"
            return "I don't know"

        result = evaluate(
            name="partial-test", task=partial_agent, data=test_cases, metrics=["exact_match"]
        )

        assert result.total == 3
        assert result.passed == 2  # Only math questions pass
        assert result.failed == 1  # Geography question fails
        assert result.accuracy == pytest.approx(0.666666, rel=0.01)

    def test_evaluate_with_custom_key_mapping(self) -> None:
        """Test loading dataset with custom keys and evaluating."""
        test_cases = load_dataset(
            str(FIXTURES_DIR / "custom_keys.jsonl"), input_key="query", expected_key="answer"
        )

        def simple_agent(query: str) -> str:
            if "2+2" in query:
                return "4"
            elif "capital of France" in query:
                return "Paris"
            return "unknown"

        result = evaluate(
            name="custom-keys-test", task=simple_agent, data=test_cases, metrics=["exact_match"]
        )

        assert result.total == 2
        assert result.passed == 2

    def test_evaluate_with_metadata_preserved(self) -> None:
        """Test that metadata from dataset is preserved in results."""
        test_cases = load_dataset(str(FIXTURES_DIR / "valid.jsonl"))

        def simple_agent(query: str) -> str:
            if "10*5" in query:
                return "50"
            return "unknown"

        result = evaluate(
            name="metadata-test", task=simple_agent, data=test_cases, metrics=["exact_match"]
        )

        # Find the test case with metadata
        test_with_metadata = [
            r
            for r in result.results
            if r.test_case.metadata is not None and "difficulty" in r.test_case.metadata
        ]

        assert len(test_with_metadata) == 1
        assert test_with_metadata[0].test_case.metadata["difficulty"] == "easy"
        assert test_with_metadata[0].test_case.metadata["category"] == "math"

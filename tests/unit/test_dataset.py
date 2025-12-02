"""Unit tests for dataset loading functionality.

These tests check loading datasets as Goldens (default) or TestCases (when actual_output present).
See test_dataset_golden.py for tests of EvaluationDataset.
"""

from pathlib import Path

import pytest

from evaris.dataset import load_dataset
from evaris.types import Golden

# Get the fixtures directory path
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestLoadDatasetJSONL:
    """Tests for loading JSONL datasets."""

    def test_load_valid_jsonl(self) -> None:
        """Test loading a valid JSONL file with input and expected fields."""
        path = FIXTURES_DIR / "valid.jsonl"
        goldens = load_dataset(str(path))

        assert len(goldens) == 3
        assert all(isinstance(g, Golden) for g in goldens)

        assert goldens[0].input == "What is 2+2?"
        assert goldens[0].expected == "4"
        assert goldens[0].metadata == {}

        assert goldens[1].input == "What is the capital of France?"
        assert goldens[1].expected == "Paris"

        assert goldens[2].input == "What is 10*5?"
        assert goldens[2].expected == "50"
        assert goldens[2].metadata == {"difficulty": "easy", "category": "math"}

    def test_load_jsonl_without_expected(self) -> None:
        """Test loading JSONL file with only input field."""
        path = FIXTURES_DIR / "valid_no_expected.jsonl"
        test_cases = load_dataset(str(path))

        assert len(test_cases) == 3
        assert test_cases[0].input == "What is 2+2?"
        assert test_cases[0].expected is None

        assert test_cases[1].input == "What is the capital of France?"
        assert test_cases[1].expected is None

    def test_load_empty_jsonl(self) -> None:
        """Test loading an empty JSONL file returns empty list."""
        path = FIXTURES_DIR / "empty.jsonl"
        test_cases = load_dataset(str(path))

        assert test_cases == []

    def test_load_jsonl_with_invalid_line(self) -> None:
        """Test loading JSONL with invalid JSON line skips that line."""
        path = FIXTURES_DIR / "invalid_json_line.jsonl"

        # Should skip the invalid line and load the valid ones
        test_cases = load_dataset(str(path))

        assert len(test_cases) == 2
        assert test_cases[0].input == "What is 2+2?"
        assert test_cases[1].input == "What is the capital of France?"

    def test_load_jsonl_with_custom_keys(self) -> None:
        """Test loading JSONL with custom key mapping."""
        path = FIXTURES_DIR / "custom_keys.jsonl"
        test_cases = load_dataset(str(path), input_key="query", expected_key="answer")

        assert len(test_cases) == 2
        assert test_cases[0].input == "What is 2+2?"
        assert test_cases[0].expected == "4"

    def test_load_jsonl_auto_detect_format(self) -> None:
        """Test that .jsonl extension is auto-detected."""
        path = FIXTURES_DIR / "valid.jsonl"
        test_cases = load_dataset(str(path))  # No format specified

        assert len(test_cases) == 3

    def test_load_jsonl_explicit_format(self) -> None:
        """Test loading with explicit file_format='jsonl'."""
        path = FIXTURES_DIR / "valid.jsonl"
        test_cases = load_dataset(str(path), file_format="jsonl")

        assert len(test_cases) == 3


class TestLoadDatasetJSON:
    """Tests for loading JSON datasets."""

    def test_load_valid_json_array(self) -> None:
        """Test loading a valid JSON array."""
        path = FIXTURES_DIR / "valid.json"
        test_cases = load_dataset(str(path))

        assert len(test_cases) == 3
        assert test_cases[0].input == "What is 2+2?"
        assert test_cases[0].expected == "4"

    def test_load_json_single_object(self) -> None:
        """Test loading a single JSON object (wrapped in list)."""
        path = FIXTURES_DIR / "valid_single_object.json"
        test_cases = load_dataset(str(path))

        assert len(test_cases) == 1
        assert test_cases[0].input == "What is 2+2?"
        assert test_cases[0].expected == "4"

    def test_load_empty_json_array(self) -> None:
        """Test loading an empty JSON array returns empty list."""
        path = FIXTURES_DIR / "empty.json"
        test_cases = load_dataset(str(path))

        assert test_cases == []

    def test_load_json_auto_detect_format(self) -> None:
        """Test that .json extension is auto-detected."""
        path = FIXTURES_DIR / "valid.json"
        test_cases = load_dataset(str(path))  # No format specified

        assert len(test_cases) == 3

    def test_load_json_explicit_format(self) -> None:
        """Test loading with explicit file_format='json'."""
        path = FIXTURES_DIR / "valid.json"
        test_cases = load_dataset(str(path), file_format="json")

        assert len(test_cases) == 3


class TestLoadDatasetCSV:
    """Tests for loading CSV datasets."""

    def test_load_valid_csv(self) -> None:
        """Test loading a valid CSV file with input and expected columns."""
        path = FIXTURES_DIR / "valid.csv"
        test_cases = load_dataset(str(path))

        assert len(test_cases) == 3
        assert test_cases[0].input == "What is 2+2?"
        assert test_cases[0].expected == "4"

        assert test_cases[1].input == "What is the capital of France?"
        assert test_cases[1].expected == "Paris"

    def test_load_csv_without_expected(self) -> None:
        """Test loading CSV with only input column."""
        path = FIXTURES_DIR / "valid_no_expected.csv"
        test_cases = load_dataset(str(path))

        assert len(test_cases) == 3
        assert test_cases[0].input == "What is 2+2?"
        assert test_cases[0].expected is None

    def test_load_csv_with_extra_columns(self) -> None:
        """Test loading CSV with extra columns (stored in metadata)."""
        path = FIXTURES_DIR / "valid_extra_columns.csv"
        test_cases = load_dataset(str(path))

        assert len(test_cases) == 2
        assert test_cases[0].input == "What is 2+2?"
        assert test_cases[0].expected == "4"
        assert test_cases[0].metadata == {"difficulty": "easy", "category": "math"}

        assert test_cases[1].metadata == {"difficulty": "easy", "category": "geography"}

    def test_load_csv_with_custom_columns(self) -> None:
        """Test loading CSV with custom column mapping."""
        path = FIXTURES_DIR / "custom_columns.csv"
        test_cases = load_dataset(str(path), input_key="query", expected_key="answer")

        assert len(test_cases) == 2
        assert test_cases[0].input == "What is 2+2?"
        assert test_cases[0].expected == "4"

    def test_load_empty_csv(self) -> None:
        """Test loading an empty CSV file returns empty list."""
        path = FIXTURES_DIR / "empty.csv"
        test_cases = load_dataset(str(path))

        assert test_cases == []

    def test_load_csv_auto_detect_format(self) -> None:
        """Test that .csv extension is auto-detected."""
        path = FIXTURES_DIR / "valid.csv"
        test_cases = load_dataset(str(path))  # No format specified

        assert len(test_cases) == 3

    def test_load_csv_explicit_format(self) -> None:
        """Test loading with explicit file_format='csv'."""
        path = FIXTURES_DIR / "valid.csv"
        test_cases = load_dataset(str(path), file_format="csv")

        assert len(test_cases) == 3


class TestLoadDatasetEdgeCases:
    """Tests for edge cases and error handling."""

    def test_load_missing_file_raises_error(self) -> None:
        """Test that loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_file.jsonl")

    def test_load_unsupported_format_raises_error(self) -> None:
        """Test that unsupported format raises ValueError."""
        path = FIXTURES_DIR / "valid.jsonl"
        with pytest.raises(ValueError, match="Unsupported format"):
            load_dataset(str(path), file_format="xml")

    def test_load_unsupported_extension_raises_error(self) -> None:
        """Test that unsupported file extension raises ValueError."""
        # Create a file with unsupported extension
        unsupported_file = FIXTURES_DIR / "test.txt"
        unsupported_file.write_text("some data")

        try:
            with pytest.raises(ValueError, match="Cannot determine format"):
                load_dataset(str(unsupported_file))
        finally:
            # Cleanup
            if unsupported_file.exists():
                unsupported_file.unlink()

    def test_load_missing_input_key_raises_error(self) -> None:
        """Test that missing input key in data raises ValueError."""
        path = FIXTURES_DIR / "custom_keys.jsonl"

        # Try to load with default 'input' key, but file has 'query'
        with pytest.raises(ValueError, match="Missing required key"):
            load_dataset(str(path))

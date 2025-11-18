"""Dataset loading functionality for Evaris."""

import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, Field

from evaris.types import AsyncAgentFunction, Golden, TestCase

logger = logging.getLogger(__name__)


class EvaluationDataset(BaseModel):
    """Container for managing goldens and test cases.

    This class separates static test data (goldens) from dynamic evaluation data (test cases).
    The typical workflow is:
    1. Load goldens from file or create them programmatically
    2. Generate test cases by running your agent on the goldens
    3. Evaluate the test cases with metrics
    """

    name: str = Field(..., description="Dataset name")
    goldens: list[Golden] = Field(default_factory=list, description="Static test data")
    test_cases: list[TestCase] = Field(
        default_factory=list, description="Test cases with actual outputs"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional dataset metadata"
    )

    def add_golden(self, golden: Golden) -> None:
        """Add a golden to the dataset.

        Args:
            golden: The golden test data to add
        """
        self.goldens.append(golden)

    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the dataset.

        Args:
            test_case: The test case to add
        """
        self.test_cases.append(test_case)

    async def generate_test_cases_async(
        self,
        agent_fn: AsyncAgentFunction,
        max_concurrency: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Generate test cases by running agent on all goldens in parallel.

        Args:
            agent_fn: Async function that takes input and returns actual output
            max_concurrency: Maximum number of concurrent agent calls
            progress_callback: Optional callback(current, total) for progress tracking
        """
        if not self.goldens:
            return

        semaphore = asyncio.Semaphore(max_concurrency)

        async def generate_one(golden: Golden, index: int) -> TestCase:
            async with semaphore:
                actual_output = await agent_fn(golden.input)
                test_case = TestCase.from_golden(golden, actual_output)

                if progress_callback:
                    progress_callback(index + 1, len(self.goldens))

                return test_case

        test_cases = await asyncio.gather(
            *[generate_one(golden, i) for i, golden in enumerate(self.goldens)]
        )
        self.test_cases.extend(test_cases)

    @classmethod
    def from_goldens(cls, name: str, goldens: list[Golden]) -> "EvaluationDataset":
        """Create dataset from a list of goldens.

        Args:
            name: Dataset name
            goldens: List of golden test data

        Returns:
            EvaluationDataset with the goldens loaded
        """
        return cls(name=name, goldens=goldens)

    @classmethod
    def from_file(cls, path: str, as_goldens: bool = True, **kwargs: Any) -> "EvaluationDataset":
        """Load dataset from file.

        Args:
            path: Path to dataset file
            as_goldens: If True, load as goldens. If False, load as test cases.
            **kwargs: Additional arguments passed to load_dataset()

        Returns:
            EvaluationDataset with data loaded
        """
        dataset_name = Path(path).stem
        data = load_dataset(path, as_goldens=as_goldens, **kwargs)

        if as_goldens:
            return cls(name=dataset_name, goldens=data)  # type: ignore
        else:
            return cls(name=dataset_name, test_cases=data)  # type: ignore


def load_dataset(
    path: str,
    *,
    as_goldens: bool = True,
    file_format: Optional[str] = None,
    input_key: str = "input",
    expected_key: str = "expected",
    actual_output_key: str = "actual_output",
    metadata_key: Optional[str] = "metadata",
    encoding: str = "utf-8",
) -> Union[list[Golden], list[TestCase]]:
    """Load dataset from file.

    Supports JSONL (JSON Lines), JSON, and CSV formats. The format is auto-detected
    from the file extension if not explicitly specified.

    Args:
        path: Path to dataset file (.jsonl, .json, .csv)
        as_goldens: If True, load as Golden objects (no actual_output required).
                   If False, load as TestCase objects (actual_output required).
        file_format: File format override ("jsonl", "json", "csv"). Auto-detected if None.
        input_key: Key/column name for input field (default: "input")
        expected_key: Key/column name for expected field (default: "expected")
        actual_output_key: Key/column name for actual_output field (default: "actual_output")
        metadata_key: Key/column name for metadata field (default: "metadata")
        encoding: File encoding (default: "utf-8")

    Returns:
        List of Golden or TestCase objects depending on as_goldens parameter

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is invalid or file is malformed

    Examples:
        >>> # Load JSONL file as goldens (static test data)
        >>> goldens = load_dataset("data.jsonl", as_goldens=True)
        >>> len(goldens)
        10

        >>> # Load JSONL file as test cases (with actual outputs)
        >>> test_cases = load_dataset("data_with_outputs.jsonl", as_goldens=False)

        >>> # Load CSV with custom column names as goldens
        >>> goldens = load_dataset(
        ...     "data.csv",
        ...     as_goldens=True,
        ...     input_key="query",
        ...     expected_key="answer"
        ... )
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Auto-detect format from extension if not specified
    if file_format is None:
        file_format = _detect_format(file_path)

    if file_format == "jsonl":
        return _load_jsonl(
            file_path,
            as_goldens=as_goldens,
            input_key=input_key,
            expected_key=expected_key,
            actual_output_key=actual_output_key,
            metadata_key=metadata_key,
            encoding=encoding,
        )
    elif file_format == "json":
        return _load_json(
            file_path,
            as_goldens=as_goldens,
            input_key=input_key,
            expected_key=expected_key,
            actual_output_key=actual_output_key,
            metadata_key=metadata_key,
            encoding=encoding,
        )
    elif file_format == "csv":
        return _load_csv(
            file_path,
            as_goldens=as_goldens,
            input_key=input_key,
            expected_key=expected_key,
            actual_output_key=actual_output_key,
            metadata_key=metadata_key,
            encoding=encoding,
        )
    else:
        raise ValueError(f"Unsupported format: {file_format}. Supported: jsonl, json, csv")


def _detect_format(file_path: Path) -> str:
    """Detect file format from extension."""
    suffix = file_path.suffix.lower()

    if suffix in [".jsonl", ".ndjson"]:
        return "jsonl"
    elif suffix == ".json":
        return "json"
    elif suffix == ".csv":
        return "csv"
    else:
        raise ValueError(
            f"Cannot determine format from extension '{suffix}'. "
            "Supported: .jsonl, .ndjson, .json, .csv"
        )


def _load_jsonl(
    file_path: Path,
    as_goldens: bool,
    input_key: str,
    expected_key: str,
    actual_output_key: str,
    metadata_key: Optional[str],
    encoding: str,
) -> Union[list[Golden], list[TestCase]]:
    """Load dataset from JSONL file."""
    results: list[Union[Golden, TestCase]] = []

    with open(file_path, encoding=encoding) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                item = _dict_to_item(
                    data,
                    as_golden=as_goldens,
                    input_key=input_key,
                    expected_key=expected_key,
                    actual_output_key=actual_output_key,
                    metadata_key=metadata_key,
                )
                results.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num} in {file_path}: {e}")
                continue
            except ValueError as e:
                logger.error(f"Error at line {line_num} in {file_path}: {e}")
                raise

    return results  # type: ignore


def _load_json(
    file_path: Path,
    as_goldens: bool,
    input_key: str,
    expected_key: str,
    actual_output_key: str,
    metadata_key: Optional[str],
    encoding: str,
) -> Union[list[Golden], list[TestCase]]:
    """Load dataset from JSON file."""
    with open(file_path, encoding=encoding) as f:
        data = json.load(f)

    # Handle single object (wrap in list)
    if isinstance(data, dict):
        data = [data]

    # Handle empty array
    if not data:
        return []

    # Convert each dict to Golden or TestCase
    results: list[Union[Golden, TestCase]] = []
    for item in data:
        result = _dict_to_item(
            item,
            as_golden=as_goldens,
            input_key=input_key,
            expected_key=expected_key,
            actual_output_key=actual_output_key,
            metadata_key=metadata_key,
        )
        results.append(result)

    return results  # type: ignore


def _load_csv(
    file_path: Path,
    as_goldens: bool,
    input_key: str,
    expected_key: str,
    actual_output_key: str,
    metadata_key: Optional[str],
    encoding: str,
) -> Union[list[Golden], list[TestCase]]:
    """Load dataset from CSV file."""
    results: list[Union[Golden, TestCase]] = []

    with open(file_path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)

        # Start at 2 (after header)
        for row_num, row in enumerate(reader, start=2):
            try:
                item = _dict_to_item(
                    row,
                    as_golden=as_goldens,
                    input_key=input_key,
                    expected_key=expected_key,
                    actual_output_key=actual_output_key,
                    metadata_key=metadata_key,
                )
                results.append(item)
            except ValueError as e:
                logger.error(f"Error at row {row_num} in {file_path}: {e}")
                raise

    return results  # type: ignore


def _dict_to_item(
    data: dict[str, Any],
    as_golden: bool,
    input_key: str,
    expected_key: str,
    actual_output_key: str,
    metadata_key: Optional[str],
) -> Union[Golden, TestCase]:
    """Convert a dictionary to a Golden or TestCase object.

    Args:
        data: Dictionary containing test data
        as_golden: If True, create Golden. If False, create TestCase.
        input_key: Key name for input field
        expected_key: Key name for expected field
        actual_output_key: Key name for actual_output field
        metadata_key: Key name for metadata field

    Returns:
        Golden or TestCase object

    Raises:
        ValueError: If required keys are missing
    """
    if input_key not in data:
        raise ValueError(
            f"Missing required key '{input_key}' in data. Available keys: {list(data.keys())}"
        )

    input_value = data[input_key]
    expected_value = data.get(expected_key)

    # Handle metadata
    metadata: dict[str, Any] = {}
    if metadata_key and metadata_key in data:
        # If there's an explicit metadata field, use it
        metadata_value = data[metadata_key]
        if isinstance(metadata_value, dict):
            metadata = metadata_value
        else:
            logger.warning(f"Metadata field '{metadata_key}' is not a dict, ignoring")
    else:
        # Store any extra fields in metadata
        reserved_keys = {input_key, expected_key, actual_output_key}
        if metadata_key:
            reserved_keys.add(metadata_key)

        for key, value in data.items():
            if key not in reserved_keys:
                metadata[key] = value

    if as_golden:
        # Create Golden (no actual_output required)
        return Golden(
            input=input_value,
            expected=expected_value,
            metadata=metadata if metadata else {},
        )
    else:
        # Create TestCase (actual_output required)
        if actual_output_key not in data:
            raise ValueError(
                f"Missing required key '{actual_output_key}' for TestCase. "
                f"Available keys: {list(data.keys())}. "
                f"Use as_goldens=True if loading data without actual outputs."
            )

        actual_output_value = data[actual_output_key]
        return TestCase(
            input=input_value,
            actual_output=actual_output_value,
            expected=expected_value,
            metadata=metadata if metadata else {},
        )

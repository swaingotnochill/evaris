# Evaris Python SDK - API Documentation

## Installation

```bash
uv pip install evaris
```

## Quick Start

```python
from evaris import evaluate

def my_agent(query: str) -> str:
    """Your AI agent implementation."""
    return f"Answer: {query}"

result = evaluate(
    name="my-first-eval",
    task=my_agent,
    data=[
        {"input": "What is 2+2?", "expected": "Answer: What is 2+2?"}
    ],
    metrics=["exact_match", "latency"]
)

print(result)
# Output:
# Evaluation: my-first-eval
# Total: 1 | Passed: 1 | Failed: 0
# Accuracy: 100.00% | Avg Latency: 0.15ms
```

---

## Core API

### `load_dataset()`

Load test datasets from files in JSONL, JSON, or CSV formats.

#### Signature

```python
def load_dataset(
    path: str,
    *,
    file_format: Optional[str] = None,
    input_key: str = "input",
    expected_key: str = "expected",
    metadata_key: Optional[str] = "metadata",
    encoding: str = "utf-8",
) -> list[TestCase]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | **Required** | Path to dataset file (.jsonl, .json, .csv) |
| `file_format` | `Optional[str]` | `None` | File format override ("jsonl", "json", "csv"). Auto-detected from extension if not specified. |
| `input_key` | `str` | `"input"` | Key/column name for input field |
| `expected_key` | `str` | `"expected"` | Key/column name for expected output field |
| `metadata_key` | `Optional[str]` | `"metadata"` | Key/column name for metadata field |
| `encoding` | `str` | `"utf-8"` | File encoding |

#### Returns

`list[TestCase]` - List of test case objects ready to pass to `evaluate()`.

#### Supported Formats

**JSONL (JSON Lines)**
- Each line contains a single JSON object
- Memory efficient - processes line-by-line
- Skips invalid JSON lines with warning
- Extensions: `.jsonl`, `.ndjson`

```jsonl
{"input": "What is 2+2?", "expected": "4"}
{"input": "Capital of France?", "expected": "Paris", "metadata": {"difficulty": "easy"}}
```

**JSON**
- Array of objects or single object (auto-wrapped)
- Extension: `.json`

```json
[
  {"input": "What is 2+2?", "expected": "4"},
  {"input": "Capital of France?", "expected": "Paris"}
]
```

**CSV**
- First row is header with column names
- Extra columns are stored in metadata
- Extension: `.csv`

```csv
input,expected
What is 2+2?,4
Capital of France?,Paris
```

#### Examples

**Load JSONL dataset**
```python
from evaris import load_dataset, evaluate

test_cases = load_dataset("data/eval.jsonl")

result = evaluate(
    name="my-eval",
    task=my_agent,
    data=test_cases,
    metrics=["exact_match"]
)
```

**Load CSV with custom column names**
```python
test_cases = load_dataset(
    "data/questions.csv",
    input_key="question",
    expected_key="answer"
)
```

**Load dataset without expected values** (for latency-only testing)
```python
test_cases = load_dataset("data/inputs.jsonl")

result = evaluate(
    name="latency-test",
    task=my_agent,
    data=test_cases,
    metrics=["latency"]  # Only latency, no comparison needed
)
```

**Handle metadata from CSV**
```python
# CSV file with extra columns:
# input,expected,difficulty,category
# What is 2+2?,4,easy,math

test_cases = load_dataset("data/tests.csv")

# Extra columns are automatically stored in metadata
print(test_cases[0].metadata)  # {"difficulty": "easy", "category": "math"}
```

#### Raises

- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If format is unsupported or required keys are missing
- `json.JSONDecodeError`: Logged and skipped for invalid JSONL lines

---

### `evaluate()`

Main function for running evaluations on your AI agent.

#### Signature

```python
def evaluate(
    name: str,
    task: Callable[[Any], Any],
    data: list[dict[str, Any]] | list[TestCase],
    metrics: list[str],
) -> EvalResult
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | **Required.** Unique identifier for this evaluation run. Used for tracking and comparison. |
| `task` | `Callable[[Any], Any]` | **Required.** Your agent function. Takes test input, returns agent output. |
| `data` | `list[dict]` or `list[TestCase]` | **Required.** Test cases to evaluate. See [Test Data Format](#test-data-format). |
| `metrics` | `list[str]` | **Required.** Metrics to apply. See [Available Metrics](#available-metrics). |

#### Returns

`EvalResult` object with:
- `name`: Evaluation name
- `total`: Total test cases run
- `passed`: Number of passed tests
- `failed`: Number of failed tests
- `accuracy`: Overall accuracy (0.0 to 1.0)
- `avg_latency_ms`: Average execution time
- `results`: List of individual `TestResult` objects

#### Raises

- `ValueError`: If `data` is empty or `metrics` contains unknown metric names
- `TypeError`: If required parameters are missing

#### Examples

**Basic Usage**
```python
from evaris import evaluate

def greet(name: str) -> str:
    return f"Hello {name}"

result = evaluate(
    name="greeting-test",
    task=greet,
    data=[
        {"input": "Alice", "expected": "Hello Alice"},
        {"input": "Bob", "expected": "Hello Bob"},
    ],
    metrics=["exact_match"]
)

print(f"Accuracy: {result.accuracy:.2%}")  # 100.00%
print(f"Latency: {result.avg_latency_ms:.2f}ms")  # 0.05ms
```

**Multiple Metrics**
```python
result = evaluate(
    name="multi-metric-test",
    task=my_agent,
    data=[...],
    metrics=["exact_match", "latency"]
)

# Access individual test results
for test_result in result.results:
    print(f"Input: {test_result.test_case.input}")
    print(f"Output: {test_result.output}")
    for metric in test_result.metrics:
        print(f"  {metric.name}: {metric.score}")
```

**Using TestCase Objects**
```python
from evaris import evaluate, TestCase

test_cases = [
    TestCase(
        input="What is AI?",
        expected="Artificial Intelligence",
        metadata={"difficulty": "easy"}
    ),
    TestCase(
        input="Explain quantum computing",
        expected="Complex explanation...",
        metadata={"difficulty": "hard"}
    )
]

result = evaluate(
    name="knowledge-test",
    task=my_agent,
    data=test_cases,
    metrics=["exact_match"]
)
```

---

## Test Data Format

### Dictionary Format

Each test case is a dictionary with the following keys:

```python
{
    "input": Any,              # Required: Input to your agent
    "expected": Any,           # Optional: Expected output (required for exact_match)
    "metadata": dict[str, Any] # Optional: Additional context
}
```

**Example:**
```python
test_data = [
    {
        "input": "Translate 'hello' to Spanish",
        "expected": "hola",
        "metadata": {"language": "es", "difficulty": "easy"}
    },
    {
        "input": {"query": "weather", "location": "NYC"},
        "expected": {"temp": 72, "condition": "sunny"}
    }
]
```

### TestCase Objects

For more structure, use `TestCase` Pydantic models:

```python
from evaris import TestCase

tc = TestCase(
    input="Your input here",
    expected="Expected output",
    metadata={"key": "value"}
)
```

---

## Available Metrics

### `exact_match`

Checks if agent output exactly matches expected output.

**Configuration:**
```python
from evaris.metrics import ExactMatchMetric

# Default: case-sensitive, whitespace-sensitive
metric = ExactMatchMetric()

# Case-insensitive
metric = ExactMatchMetric(case_sensitive=False)

# Strip whitespace
metric = ExactMatchMetric(strip_whitespace=True)

# Both
metric = ExactMatchMetric(case_sensitive=False, strip_whitespace=True)
```

**Behavior:**
- Returns score `1.0` if exact match, `0.0` otherwise
- Supports strings, numbers, dicts, lists
- For lists, order matters
- Raises `ValueError` if `expected` is `None`

**Examples:**
```python
# Exact match
TestCase(input="test", expected="output") -> score=1.0 for output="output"
TestCase(input="test", expected="output") -> score=0.0 for output="different"

# Case-insensitive
ExactMatchMetric(case_sensitive=False)
TestCase(input="test", expected="Hello") -> score=1.0 for output="hello"

# Whitespace stripping
ExactMatchMetric(strip_whitespace=True)
TestCase(input="test", expected="hello") -> score=1.0 for output="  hello  "
```

### `latency`

Measures agent execution time in milliseconds.

**Behavior:**
- Always passes (`score=1.0`)
- Records execution time in `metadata["latency_ms"]`
- Useful for performance tracking

**Example:**
```python
result = evaluate(
    name="perf-test",
    task=slow_agent,
    data=[{"input": "test"}],
    metrics=["latency"]
)

print(result.avg_latency_ms)  # e.g., 245.67
```

---

## Data Models

### TestCase

```python
class TestCase(BaseModel):
    """A single test case for evaluation."""

    input: Any
    """Input to the agent (any type)."""

    expected: Optional[Any] = None
    """Expected output from the agent."""

    metadata: Optional[dict[str, Any]] = {}
    """Additional context or tags."""
```

### MetricResult

```python
class MetricResult(BaseModel):
    """Result from a single metric evaluation."""

    name: str
    """Metric name (e.g., 'exact_match')."""

    score: float
    """Score between 0.0 and 1.0."""

    passed: bool
    """Whether the test passed."""

    metadata: Optional[dict[str, Any]] = {}
    """Metric-specific data (e.g., error messages, reasoning)."""
```

### TestResult

```python
class TestResult(BaseModel):
    """Result from evaluating a single test case."""

    test_case: TestCase
    """The original test case."""

    output: Any
    """Actual output from the agent."""

    metrics: list[MetricResult]
    """Results from each metric."""

    latency_ms: float
    """Execution time in milliseconds."""

    error: Optional[str] = None
    """Error message if agent failed."""
```

### EvalResult

```python
class EvalResult(BaseModel):
    """Aggregated results from an evaluation run."""

    name: str
    """Evaluation identifier."""

    total: int
    """Total number of test cases."""

    passed: int
    """Number of passed test cases."""

    failed: int
    """Number of failed test cases."""

    accuracy: float
    """Overall accuracy (passed/total)."""

    avg_latency_ms: float
    """Average execution time across all tests."""

    results: list[TestResult]
    """Individual test results."""

    metadata: Optional[dict[str, Any]] = {}
    """Additional evaluation metadata."""
```

---

## Common Patterns

### Handling Agent Errors

Evaris gracefully handles agent exceptions:

```python
def flaky_agent(input_data):
    if "error" in input_data:
        raise RuntimeError("Agent failed!")
    return "success"

result = evaluate(
    name="error-handling",
    task=flaky_agent,
    data=[
        {"input": "normal", "expected": "success"},
        {"input": "error trigger", "expected": "success"}
    ],
    metrics=["exact_match"]
)

# Check for errors
for test_result in result.results:
    if test_result.error:
        print(f"Test failed: {test_result.error}")
```

### Filtering Results

```python
result = evaluate(...)

# Get only failed tests
failed_tests = [r for r in result.results if not all(m.passed for m in r.metrics)]

# Get tests by latency
slow_tests = [r for r in result.results if r.latency_ms > 100]

# Get tests by metric score
low_accuracy = [
    r for r in result.results
    if any(m.name == "exact_match" and m.score < 0.8 for m in r.metrics)
]
```

### Complex Input Types

```python
# JSON/Dict inputs
data = [
    {
        "input": {"query": "weather", "location": "NYC", "units": "F"},
        "expected": {"temperature": 72, "condition": "sunny"}
    }
]

# List inputs
data = [
    {
        "input": ["item1", "item2", "item3"],
        "expected": ["processed1", "processed2", "processed3"]
    }
]
```

### Metadata Usage

```python
result = evaluate(
    name="categorized-test",
    data=[
        TestCase(input="easy q", expected="a", metadata={"category": "easy"}),
        TestCase(input="hard q", expected="a", metadata={"category": "hard"}),
    ],
    task=my_agent,
    metrics=["exact_match"]
)

# Group by category
from collections import defaultdict
by_category = defaultdict(list)
for test_result in result.results:
    category = test_result.test_case.metadata.get("category")
    by_category[category].append(test_result)

# Calculate accuracy per category
for category, tests in by_category.items():
    passed = sum(1 for t in tests if all(m.passed for m in t.metrics))
    accuracy = passed / len(tests)
    print(f"{category}: {accuracy:.2%}")
```

---

## Error Reference

### ValueError: Data must contain at least one test case

**Cause:** Empty `data` list passed to `evaluate()`

**Solution:**
```python
# Wrong
evaluate(name="test", task=agent, data=[], metrics=["exact_match"])

# Correct
evaluate(name="test", task=agent, data=[{"input": "test"}], metrics=["latency"])
```

### ValueError: Unknown metric 'metric_name'

**Cause:** Metric name not recognized

**Solution:**
```python
# Wrong
evaluate(name="test", task=agent, data=[...], metrics=["invalid_metric"])

# Correct
evaluate(name="test", task=agent, data=[...], metrics=["exact_match", "latency"])
```

### ValueError: ExactMatchMetric requires test case to have an 'expected' value

**Cause:** Using `exact_match` metric without providing `expected` field

**Solution:**
```python
# Wrong
data = [{"input": "test"}]  # No expected value
evaluate(name="test", task=agent, data=data, metrics=["exact_match"])

# Correct - provide expected value
data = [{"input": "test", "expected": "output"}]
evaluate(name="test", task=agent, data=data, metrics=["exact_match"])

# Or use latency metric which doesn't need expected
data = [{"input": "test"}]
evaluate(name="test", task=agent, data=data, metrics=["latency"])
```

---

## Type Hints

Evaris is fully typed. Import types for better IDE support:

```python
from evaris import evaluate, EvalResult, TestCase, MetricResult
from typing import Any

def my_agent(query: str) -> str:
    return f"Response to {query}"

def run_evaluation() -> EvalResult:
    return evaluate(
        name="typed-eval",
        task=my_agent,
        data=[{"input": "test", "expected": "Response to test"}],
        metrics=["exact_match"]
    )

# Type checking will catch errors
result: EvalResult = run_evaluation()
accuracy: float = result.accuracy
```

---

## Next Steps

- See [DEVELOPMENT.md](./DEVELOPMENT.md) for contributing to the SDK
- Check [README.md](./README.md) for installation and quick start
- Review test files in `tests/` for more examples

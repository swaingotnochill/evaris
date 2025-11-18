# Test Structure

This directory contains tests organized by type and scope.

## Directory Structure

```
tests/
├── unit/                    # Unit tests for individual functions/classes
│   ├── test_types.py       # Tests for Pydantic models
│   ├── test_metrics.py     # Tests for metric implementations
│   └── test_evaluate_helpers.py  # Tests for internal helper functions
├── integration/             # Integration tests for component interactions
│   └── test_evaluate.py    # End-to-end tests for evaluate() function
└── system/                  # System tests (future: CLI, file I/O, etc.)
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual functions, classes, or modules in isolation.

**Characteristics:**
- Fast (< 10ms per test)
- No external dependencies
- Mock/stub external calls
- Test single units of code
- High coverage of edge cases

**Examples:**
- Pydantic model validation
- Metric scoring logic
- Helper function behavior

**Naming Convention:** `test_<module>.py` or `test_<module>_<component>.py`

### Integration Tests (`tests/integration/`)

Test interactions between multiple components.

**Characteristics:**
- Moderate speed (10-100ms per test)
- Test component integration
- May use real implementations
- Focus on workflows and data flow

**Examples:**
- Complete evaluate() workflow
- Metric application to test cases
- Data transformation pipeline

**Naming Convention:** `test_<feature>.py`

### System Tests (`tests/system/`)

Test the entire system end-to-end (future).

**Characteristics:**
- Slower (> 100ms per test)
- Test real-world scenarios
- Use actual files, APIs, etc.
- Verify system behavior

**Examples (future):**
- CLI commands
- File loading (JSONL, CSV)
- API integrations
- Multi-step workflows

**Naming Convention:** `test_<scenario>.py`

## Running Tests

### All Tests
```bash
pytest
```

### By Category
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# System tests only (future)
pytest tests/system/
```

### Specific Test File
```bash
pytest tests/unit/test_types.py
```

### Specific Test Class
```bash
pytest tests/unit/test_metrics.py::TestExactMatchMetric
```

### Specific Test Method
```bash
pytest tests/unit/test_metrics.py::TestExactMatchMetric::test_exact_match_strings
```

### With Coverage
```bash
# All tests with coverage
pytest --cov=evaris --cov-report=term-missing

# Unit tests with coverage
pytest tests/unit/ --cov=evaris --cov-report=term-missing
```

### Fast Feedback Loop
```bash
# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run tests that failed, then all
pytest --ff
```

## Test Guidelines

### 1. Test Naming

- **File names:** `test_<what_is_tested>.py`
- **Class names:** `Test<ComponentName>` or `Test<FeatureName>`
- **Method names:** `test_<what_it_tests>`

Good examples:
```python
# tests/unit/test_metrics.py
class TestExactMatchMetric:
    def test_exact_match_strings(self) -> None:
        """Test exact string matching."""
        pass

    def test_case_insensitive_mode(self) -> None:
        """Test case-insensitive matching when enabled."""
        pass
```

### 2. Test Structure

Follow the AAA pattern: **Arrange, Act, Assert**

```python
def test_example(self) -> None:
    """Test description."""
    # Arrange: Setup test data
    test_case = TestCase(input="test", expected="output")
    metric = ExactMatchMetric()

    # Act: Execute the code under test
    result = metric.score(test_case, "output")

    # Assert: Verify the results
    assert result.score == 1.0
    assert result.passed is True
```

### 3. Docstrings

Every test must have a docstring explaining what it tests:

```python
def test_exact_match_strings(self) -> None:
    """Test that identical strings result in score of 1.0."""
    pass
```

### 4. Type Hints

All test functions must have return type hint:

```python
def test_example(self) -> None:  # Good
    pass

def test_example():  # Bad - missing type hint
    pass
```

### 5. Fixtures

Use pytest fixtures for shared setup:

```python
import pytest

@pytest.fixture
def sample_agent():
    """Create a sample agent for testing."""
    def agent(query: str) -> str:
        return f"Response: {query}"
    return agent

def test_with_fixture(sample_agent) -> None:
    """Test using the fixture."""
    result = sample_agent("test")
    assert result == "Response: test"
```

### 6. Parametrized Tests

Use `@pytest.mark.parametrize` for testing multiple inputs:

```python
@pytest.mark.parametrize("input_text,expected_output", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("", ""),
])
def test_uppercase(input_text: str, expected_output: str) -> None:
    """Test uppercase conversion."""
    assert input_text.upper() == expected_output
```

### 7. Test Coverage

- **Target:** 90% minimum, 100% ideal
- **Focus:** Cover edge cases, not just happy paths
- **Check:** `pytest --cov=evaris --cov-report=term-missing`

```python
# Good: Tests both success and failure
def test_valid_input(self) -> None:
    pass

def test_invalid_input_raises_error(self) -> None:
    with pytest.raises(ValueError):
        pass
```

## Adding New Tests

### For New Feature

1. **Start with unit tests** for individual components
2. **Add integration tests** for component interactions
3. **Consider system tests** if feature touches external systems

### Example: Adding LLM Judge Metric

```
tests/
├── unit/
│   └── test_llm_judge.py        # Unit tests for LLMJudge class
├── integration/
│   └── test_evaluate_llm.py     # Integration with evaluate()
└── system/
    └── test_openai_integration.py  # Real API calls (optional)
```

## Continuous Integration

GitHub Actions runs all tests on every push:

```yaml
# .github/workflows/python-ci.yml
- name: Run tests with coverage
  run: uv run pytest --cov=evaris --cov-fail-under=90
```

## TODO

- [ ] Add system tests for file loading (JSONL, CSV)
- [ ] Add integration tests for LLM judge when implemented
- [ ] Add performance benchmarks in `tests/benchmarks/`
- [ ] Add property-based tests with Hypothesis

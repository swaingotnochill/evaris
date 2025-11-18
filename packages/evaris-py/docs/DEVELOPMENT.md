# Evaris Python SDK - Development Guide

## Development Workflow

### Setup Development Environment

```bash
# Create virtual environment
uv venv

# Install package in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

### Tools Used

- **uv**: Fast Python package manager
- **pytest**: Testing framework
- **black**: Code formatting
- **ruff**: Fast Python linter
- **mypy**: Static type checking
- **ty**: Fast Rust-based type checker (Astral)
- **pytest-cov**: Code coverage

### Development Cycle

```bash
# 1. Write tests first (TDD approach)
#    Create test file in tests/test_*.py

# 2. Run tests (watch them fail)
pytest -v

# 3. Implement feature

# 4. Run tests again (watch them pass)
pytest -v

# 5. Check coverage (must be >90%)
pytest --cov=evaris --cov-report=term-missing --cov-fail-under=90

# 6. Format code
black evaris tests

# 7. Lint code
ruff check evaris tests

# 8. Type check (mypy) [depreceated in support of ty]
mypy evaris

# 9. Type check (ty - fast Rust-based checker)
ty check

# Ensuring 90% test coverage.
pytest --cov=evaris --cov-fail-under=90

# 10. Run all checks together
pytest --cov=evaris && \
black evaris tests && \
ruff check evaris tests && \
mypy evaris && \
ty check

```

### Pre-Commit Checklist

Before committing, ensure:
- [ ] All tests pass (`pytest -v`)
- [ ] Coverage ≥ 90% (`pytest --cov=evaris --cov-fail-under=90`)
- [ ] Code is formatted (`black evaris tests`)
- [ ] No linting errors (`ruff check evaris tests`)
- [ ] Type checking passes (`mypy evaris` and `ty check`)
- [ ] Documentation is updated
- [ ] No unnecessary files staged (check `.gitignore`)

### Project Structure

```
evaris-py/
├── evaris/                  # Source code
│   ├── __init__.py         # Public API exports
│   ├── types.py            # Pydantic models
│   ├── evaluate.py         # Core evaluation logic
│   └── metrics/            # Built-in metrics
│       ├── __init__.py
│       └── exact_match.py
├── tests/                   # Test suite
│   ├── test_types.py       # Unit tests for models
│   ├── test_metrics.py     # Unit tests for metrics
│   └── test_evaluate.py    # Integration tests
├── pyproject.toml          # Package configuration
├── README.md               # User documentation
└── DEVELOPMENT.md          # This file
```

## Code Standards

### Documentation

All public APIs must have Google-style docstrings:

```python
def evaluate(
    name: str,
    task: AgentFunction,
    data: DatasetInput,
    metrics: list[str],
) -> EvalResult:
    """Evaluate an agent on a dataset using specified metrics.

    Args:
        name: Name of the evaluation (for identification)
        task: The agent function to evaluate
        data: Dataset to evaluate on
        metrics: List of metric names to apply

    Returns:
        EvalResult containing aggregated results

    Raises:
        ValueError: If data is empty or metrics are invalid

    Example:
        >>> def my_agent(query: str) -> str:
        ...     return f"Answer: {query}"
        >>> result = evaluate(
        ...     name="test",
        ...     task=my_agent,
        ...     data=[{"input": "Q", "expected": "Answer: Q"}],
        ...     metrics=["exact_match"]
        ... )
        >>> print(result.accuracy)
        1.0
    """
```

### Type Hints

- Use modern type hints (`list[str]` not `List[str]`)
- All functions must have complete type annotations
- Use `typing.Any` only when truly necessary
- Prefer specific types over generic ones

### Comments

- **DO**: Use docstrings for all public functions/classes
- **DO**: Use TODO comments for future work
- **DON'T**: Add inline comments explaining obvious code
- **DON'T**: Leave commented-out code

Good:
```python
# TODO: Add support for async agents in evaluate() function

def calculate_accuracy(passed: int, total: int) -> float:
    """Calculate accuracy as percentage of passed tests."""
    return passed / total if total > 0 else 0.0
```

Bad:
```python
# This function calculates the accuracy
def calculate_accuracy(passed, total):
    # Divide passed by total
    result = passed / total  # Get the result
    return result  # Return it
```

### Testing

#### Test Organization

- Unit tests: `test_<module>.py` (e.g., `test_types.py`)
- Integration tests: Include in `test_evaluate.py`
- Test class naming: `TestClassName`
- Test method naming: `test_<what_it_tests>`

#### Coverage Requirements

- Minimum 90% code coverage
- Test happy paths AND edge cases
- Test error handling
- Test input validation

#### Test Structure

```python
class TestExactMatchMetric:
    """Tests for ExactMatchMetric."""

    def setup_method(self) -> None:
        """Setup for each test."""
        self.metric = ExactMatchMetric()

    def test_exact_match_strings(self) -> None:
        """Test exact string matching."""
        tc = TestCase(input="test", expected="output")
        result = self.metric.score(tc, "output")

        assert result.score == 1.0
        assert result.passed is True
```

### Git Workflow

1. **Feature Branches**: Create branch for each feature
2. **Atomic Commits**: One logical change per commit
3. **Commit Messages**: Use conventional commits format

```bash
# Format: <type>: <description>
feat: add semantic similarity metric
fix: handle None values in exact match
docs: update API documentation
test: add edge cases for evaluate()
refactor: simplify metric resolution logic
```

## API Reference (For Package Developers)

### Core Types (`evaris/types.py`)

#### TestCase
```python
class TestCase(BaseModel):
    """A single test case for evaluation."""
    input: Any                           # Input to the agent
    expected: Optional[Any]              # Expected output
    metadata: Optional[dict[str, Any]]   # Additional metadata
```

#### MetricResult
```python
class MetricResult(BaseModel):
    """Result from a single metric evaluation."""
    name: str                            # Metric name
    score: float                         # Score [0.0, 1.0]
    passed: bool                         # Whether test passed
    metadata: Optional[dict[str, Any]]   # Metric-specific data
```

#### TestResult
```python
class TestResult(BaseModel):
    """Result from evaluating a single test case."""
    test_case: TestCase                  # Original test case
    output: Any                          # Agent's output
    metrics: list[MetricResult]          # Metric results
    latency_ms: float                    # Execution time
    error: Optional[str]                 # Error if failed
```

#### EvalResult
```python
class EvalResult(BaseModel):
    """Aggregated results from an evaluation run."""
    name: str                            # Evaluation name
    total: int                           # Total test cases
    passed: int                          # Number passed
    failed: int                          # Number failed
    accuracy: float                      # Overall accuracy [0.0, 1.0]
    avg_latency_ms: float                # Average latency
    results: list[TestResult]            # Individual results
    metadata: Optional[dict[str, Any]]   # Additional data
```

### Main API (`evaris/evaluate.py`)

#### evaluate()
```python
def evaluate(
    name: str,
    task: AgentFunction,
    data: DatasetInput,
    metrics: list[str],
) -> EvalResult:
    """Main evaluation entry point.

    Args:
        name: Evaluation identifier
        task: Agent function (Any -> Any)
        data: List of test cases (dicts or TestCase objects)
        metrics: Metric names ["exact_match", "latency"]

    Returns:
        EvalResult with aggregated statistics

    Implementation Notes:
        - Runs each test case sequentially
        - Catches agent exceptions and stores in TestResult.error
        - Metric errors are caught and stored in MetricResult.metadata
    """
```

### Metrics API

#### Creating Custom Metrics

```python
class CustomMetric:
    """Custom metric implementation."""

    def score(
        self,
        test_case: TestCase,
        actual_output: Any
    ) -> MetricResult:
        """Score the agent output.

        Args:
            test_case: Test case with expected value
            actual_output: Agent's actual output

        Returns:
            MetricResult with score and metadata

        Raises:
            ValueError: If test_case.expected is required but missing
        """
        # Implementation
        return MetricResult(
            name="custom_metric",
            score=1.0,
            passed=True,
            metadata={}
        )
```

#### Registering Custom Metrics

```python
# TODO: Add custom metric registration in evaluate.py
# from evaris.evaluate import register_metric
#
# register_metric("custom", CustomMetric)
```

## Extending the SDK

### Adding a New Metric

1. Create file in `evaris/metrics/<metric_name>.py`
2. Implement metric class with `score()` method
3. Add tests in `tests/test_metrics.py`
4. Register in `evaris/evaluate.py:BUILTIN_METRICS`
5. Export from `evaris/metrics/__init__.py`
6. Update documentation

Example:
```python
# evaris/metrics/semantic_similarity.py
from evaris.types import MetricResult, TestCase

class SemanticSimilarityMetric:
    """Measures semantic similarity using embeddings."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with embedding model."""
        self.model = model
        # TODO: Load model

    def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Calculate cosine similarity between embeddings."""
        # TODO: Implementation
        pass
```

### Adding Dataset Loading

```python
# TODO: Implement in evaris/dataset.py
def load_dataset(path: str) -> list[TestCase]:
    """Load dataset from file (JSONL, CSV, JSON)."""
    pass
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Reinstall in editable mode
uv pip install -e ".[dev]"
```

**Test Failures**
```bash
# Run with verbose output and stop on first failure
pytest -vsx

# Run specific test
pytest tests/test_evaluate.py::TestEvaluateBasic::test_evaluate_single_test_case_exact_match
```

**Type Checking Errors**
```bash
# Run mypy with verbose output
mypy --show-error-codes evaris

# Run ty for fast type checking
ty check
```

**Coverage Below 90%**
```bash
# Generate HTML coverage report
pytest --cov=evaris --cov-report=html
# Open htmlcov/index.html in browser
```

## Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [pytest Documentation](https://docs.pytest.org/)
- [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## TODO

### Core Features (v0.2.0)
- [ ] Add async support for evaluate()
- [ ] Add custom metric registration API
- [ ] Add progress bars for long evaluations
- [ ] Add result export (JSON, Markdown, HTML)
- [ ] Add CLI tool (`evaris run config.yaml`)

### ABC Tier 2 Enforcement (v0.2.0-v0.3.0)

#### Agent Isolation (ABC II.5) - High Priority
- [ ] Implement `DockerAgent` wrapper for container-based isolation
  - Prevents agent from accessing evaluation code
  - Prevents agent from reading test case files
  - Blocks network access to prevent data leakage
- [ ] Implement `ProcessAgent` wrapper for process-based isolation
  - Runs agent in separate Python process
  - Limits file system access
  - Configurable timeout and resource limits
- [ ] Add isolation detection in `ABCComplianceChecker`
  - Detect if agent runs in same process
  - Warn if isolation not enabled
  - Block in strict mode if isolation required

#### State Contamination Prevention (ABC II.4) - High Priority
- [ ] Add state reset verification
  - Detect if agent has persistent state across tests
  - Verify state is cleared between evaluations
  - Check for residual side effects
- [ ] Implement state reset helpers
  - Automatic database cleanup
  - File system reset utilities
  - Browser state clearing
- [ ] Add state contamination detection
  - Correlate results with test order
  - Test with shuffled test cases
  - Warn if order-dependent behavior detected

#### Shortcut & Exploit Detection (ABC II.10) - Medium Priority
- [ ] Automated shortcut detection
  - Run trivial baselines (do-nothing, random) automatically
  - Flag if baseline scores > 10% (indicates broken dataset)
  - Detect exhaustive listing attacks (dumping all data)
- [ ] Test modification detection
  - Monitor if agent modifies test files
  - Detect if agent overwrites evaluation logic
  - Flag suspicious file system operations
- [ ] Ground truth access detection
  - Monitor file system access patterns
  - Detect if agent reads answer files
  - Block access to expected values

#### Environment Setup Stability (ABC II.6) - Medium Priority
- [ ] External dependency tracking
  - Snapshot API versions and responses
  - Detect when external services change
  - Warn if setup changes affect results
- [ ] Reproducibility verification
  - Hash environment configurations
  - Track Python/package versions
  - Freeze external dependencies

### Advanced Features (v0.4.0+)

#### Adversarial Testing (ABC Level 3)
- [ ] Automated adversarial test generation
  - Prompt injection detection
  - Data exfiltration attempts
  - Jailbreak pattern testing
- [ ] Security vulnerability scanner
  - SQL injection in tool calls
  - Command injection in shell tools
  - Path traversal attempts

#### Continuous Monitoring
- [ ] Real-time benchmark health tracking
  - Monitor baseline performance over time
  - Detect dataset drift
  - Alert on anomalous results
- [ ] Automated contamination detection
  - Check for data leakage to web
  - Monitor agent training data sources
  - Track public dataset releases

### Framework Integrations (v0.5.0+)
- [ ] LangChain integration
  - Native LangChain agent support
  - Chain evaluation utilities
  - Tool execution tracking
- [ ] CrewAI integration
  - Multi-agent evaluation
  - Crew performance metrics
  - Agent coordination analysis
- [ ] AutoGen integration
  - Conversation evaluation
  - Multi-turn dialogue metrics
  - Agent collaboration scoring

### Notes on Tier 2 Implementation

**Why Tier 2 is Complex**:
- Requires significant architectural changes to agent systems
- Depends on user's agent implementation details
- May conflict with existing agent frameworks
- Performance overhead from isolation/sandboxing

**Implementation Strategy**:
1. Start with **detection** (v0.2.0): Warn users about potential issues
2. Add **optional helpers** (v0.2.0): Provide isolation wrappers users can adopt
3. Make **enforcement configurable** (v0.3.0): Strict mode can require isolation
4. Build **full automation** (v0.4.0): Framework handles isolation automatically

**User Adoption Path**:
```python
# Phase 1: Detection (v0.2.0) - warns but doesn't block
config = ABCComplianceConfig(check_isolation=True, strict_mode=False)
result = evaluate(..., compliance_config=config)
# Warning: Agent runs in same process - isolation recommended

# Phase 2: Opt-in helpers (v0.2.0)
from evaris.isolation import ProcessAgent
isolated_agent = ProcessAgent(my_agent)
result = evaluate(..., task=isolated_agent)
# No warnings - isolation detected

# Phase 3: Strict enforcement (v0.3.0)
config = ABCComplianceConfig(strict_mode=True, require_isolation=True)
result = evaluate(..., compliance_config=config)
# Raises ABCViolationError if agent not isolated

# Phase 4: Automatic (v0.4.0)
result = evaluate(..., auto_isolate=True)  # Framework handles it
```

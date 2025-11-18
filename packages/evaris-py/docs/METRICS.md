# Evaris Metrics Guide

Comprehensive guide to all evaluation metrics available in Evaris, organized by use case and ABC compliance.

## Quick Reference

| Metric | Use Case | ABC Checks | File |
|--------|----------|------------|------|
| **ExactMatch** | Exact string comparison | - | `metrics/exact_match.py` |
| **Latency** | Performance measurement | - | `evaluate.py` |
| **SemanticSimilarity** | Natural language outputs | O.a.1, O.a.2 | `metrics/semantic_similarity.py` |
| **LLMJudge** | Complex semantic evaluation | O.c.1, O.c.2 | `metrics/llm_judge.py` |
| **AnswerMatch** | Structured output parsing | O.h.1, O.h.2 | `metrics/answer_match.py` |
| **UnitTest** | Code generation | O.d.1, O.d.2 | `metrics/unit_test.py` |
| **FuzzTest** | Code robustness | O.e.1, O.e.2, O.e.3 | `metrics/fuzz_test.py` |
| **StateMatch** | Environment interaction | O.g.1, O.g.2, O.g.3 | `metrics/state_match.py` |

---

## Built-in Metrics

### ExactMatch

Simple exact string comparison between expected and actual outputs.

```python
from evaris.metrics.exact_match import ExactMatchMetric

metric = ExactMatchMetric()
result = metric.score(test_case, actual_output)
# Returns MetricResult with score 1.0 if exact match, 0.0 otherwise
```

**Configuration**: None
**When to use**: When outputs must match exactly
**Limitations**: No handling of semantic equivalence or formatting

---

### Latency

Measures agent execution time.

```python
from evaris import evaluate

result = evaluate(
    name="latency-test",
    task=my_agent,
    data=test_cases,
    metrics=["latency"]  # Built-in latency tracking
)
print(f"Avg latency: {result.avg_latency_ms}ms")
```

**Configuration**: Automatic
**When to use**: Always (included by default)
**Output**: Latency in milliseconds in metadata

---

## Semantic Evaluation Metrics

### SemanticSimilarity

**ABC Compliance**: O.a.1, O.a.2

Uses sentence embeddings to measure semantic similarity between outputs.

```python
from evaris.metrics.semantic_similarity import (
    SemanticSimilarityMetric,
    SemanticSimilarityConfig
)

config = SemanticSimilarityConfig(
    model="sentence-transformers/all-MiniLM-L6-v2",
    threshold=0.8,
    normalize=True,
    case_sensitive=False
)
metric = SemanticSimilarityMetric(config)

result = metric.score(test_case, actual_output)
print(f"Similarity: {result.score:.2f}")
print(f"Passed: {result.passed}")
```

**Configuration Options**:
- `model`: Sentence transformer model name (default: "all-MiniLM-L6-v2")
- `threshold`: Similarity threshold for passing (0.0-1.0, default: 0.8)
- `normalize`: Normalize text before comparison (default: True)
- `case_sensitive`: Case-sensitive comparison (default: False)

**When to use**:
- Natural language outputs
- When exact wording varies but meaning is consistent
- Question answering, summarization, translation

**Example**:
```python
tc = TestCase(input="What is the capital of France?", expected="Paris")
result = metric.score(tc, "The capital is Paris")
# Score: ~0.85, Passed: True (above threshold)
```

**Requirements**: `pip install sentence-transformers`

---

### LLMJudge

**ABC Compliance**: O.c.1, O.c.2

Uses an LLM to evaluate semantic correctness with self-consistency validation.

```python
from evaris.metrics.llm_judge import LLMJudgeMetric, LLMJudgeConfig

config = LLMJudgeConfig(
    provider="openai",  # or "anthropic"
    model="gpt-4",
    temperature=0.0,
    enable_self_consistency=True,
    self_consistency_samples=3
)
metric = LLMJudgeMetric(config)

result = metric.score(test_case, actual_output)
print(f"Score: {result.score:.2f}")
print(f"Reasoning: {result.metadata['reasoning']}")
```

**Configuration Options**:
- `provider`: "openai" or "anthropic" (default: "openai")
- `model`: Model name (default: "gpt-4")
- `api_key`: API key (optional, reads from env)
- `temperature`: Generation temperature (default: 0.0)
- `max_tokens`: Max response tokens (default: 500)
- `custom_prompt`: Custom judge prompt template (optional)
- `self_consistency_samples`: Number of samples for validation (default: 3)
- `enable_self_consistency`: Enable self-consistency check (default: True)

**When to use**:
- Complex semantic evaluation
- When exact matching and embeddings are insufficient
- Subjective tasks (creativity, style, coherence)

**Example**:
```python
tc = TestCase(
    input="Explain quantum computing",
    expected="Quantum computing uses quantum mechanics principles..."
)
result = metric.score(tc, "Quantum computers use qubits that can be in superposition...")
# Score: ~0.9 (semantically correct)
# Metadata includes reasoning and self-consistency variance
```

**Requirements**: `pip install openai` or `pip install anthropic`

---

## Structured Output Metrics

### AnswerMatch

**ABC Compliance**: O.h.1, O.h.2

Parses structured outputs according to specified formats to prevent guessing.

```python
from evaris.metrics.answer_match import AnswerMatchMetric, AnswerMatchConfig

config = AnswerMatchConfig(
    answer_format="delimited",  # or "json", "xml", "regex"
    delimiter="Answer:",
    case_sensitive=False,
    strip_whitespace=True
)
metric = AnswerMatchMetric(config)

result = metric.score(test_case, "Let me think... Answer: 42")
# Extracts "42" from output and compares with expected
```

**Configuration Options**:
- `answer_format`: "json", "xml", "regex", or "delimited" (default: "delimited")
- `delimiter`: Delimiter for answer extraction (default: "Answer:")
- `regex_pattern`: Regex pattern for extraction (when format="regex")
- `case_sensitive`: Case-sensitive matching (default: False)
- `strip_whitespace`: Strip whitespace (default: True)
- `allow_substring`: Allow substring matching (default: False)
- `custom_parser`: Custom parsing function (optional)

**Format Examples**:

**Delimited**:
```python
output = "Let me think... Answer: Paris\nThat's the capital."
# Extracts: "Paris"
```

**JSON**:
```python
output = '{"answer": "Paris", "confidence": 0.95}'
# Extracts: "Paris" (looks for common keys: answer, result, output, value)
```

**XML**:
```python
output = "<response><answer>Paris</answer></response>"
# Extracts: "Paris"
```

**Regex**:
```python
config = AnswerMatchConfig(
    answer_format="regex",
    regex_pattern=r"Final answer: (.*?)$"
)
# Extracts content after "Final answer:"
```

**When to use**:
- When agents must follow specific output formats
- To prevent success by random guessing
- Structured tasks with format requirements

---

## Code Generation Metrics

### UnitTest

**ABC Compliance**: O.d.1, O.d.2

Evaluates generated code by running unit tests and measuring quality metrics.

```python
from evaris.metrics.unit_test import UnitTestMetric, UnitTestConfig

config = UnitTestConfig(
    test_framework="pytest",
    timeout_seconds=30,
    measure_coverage=True,
    coverage_threshold=0.8,
    measure_complexity=True,
    max_complexity=10
)
metric = UnitTestMetric(config)

tc = TestCase(
    input="Write a function to add two numbers",
    expected={
        "tests": [
            "def test_add(): assert add(2, 3) == 5",
            "def test_add_negative(): assert add(-1, 1) == 0"
        ]
    }
)

code = "def add(a, b): return a + b"
result = metric.score(tc, code)
print(f"Pass rate: {result.metadata.get('pass_rate', 0)}")
print(f"Coverage: {result.metadata.get('coverage', 0)}")
```

**Configuration Options**:
- `test_framework`: Testing framework (default: "pytest")
- `timeout_seconds`: Test execution timeout (default: 30)
- `coverage_threshold`: Minimum coverage (0.0-1.0, default: 0.0)
- `measure_coverage`: Measure code coverage (default: False)
- `measure_complexity`: Measure cyclomatic complexity (default: False)
- `max_complexity`: Maximum allowed complexity (default: 10)
- `additional_deps`: Additional pip dependencies (default: [])
- `setup_code`: Setup code to run before tests (optional)

**When to use**:
- Code generation tasks
- When correctness must be verified
- To measure code quality

**Requirements**: `pip install pytest pytest-cov radon`

---

### FuzzTest

**ABC Compliance**: O.e.1, O.e.2, O.e.3

Tests generated code with diverse inputs including edge cases.

```python
from evaris.metrics.fuzz_test import FuzzTestMetric, FuzzTestConfig

config = FuzzTestConfig(
    num_fuzz_cases=100,
    timeout_seconds=30,
    input_types=["int", "float", "str", "list", "dict", "none"],
    edge_cases=True,
    type_confusion=True,
    boundary_values=True,
    memory_stress=False
)
metric = FuzzTestMetric(config)

tc = TestCase(
    input="Write a function to divide two numbers",
    expected={
        "function_name": "divide",
        "args": ["a", "b"]
    }
)

code = "def divide(a, b): return a / b if b != 0 else None"
result = metric.score(tc, code)
print(f"Pass rate: {result.metadata['pass_rate']}")
print(f"Passed: {result.metadata['passed_cases']}/{result.metadata['total_cases']}")
```

**Configuration Options**:
- `num_fuzz_cases`: Number of test cases to generate (default: 100)
- `timeout_seconds`: Execution timeout (default: 30)
- `input_types`: Types to test (default: all types)
- `edge_cases`: Include edge cases (empty, null, max/min) (default: True)
- `memory_stress`: Include large inputs (default: False)
- `type_confusion`: Test with unexpected types (default: True)
- `boundary_values`: Test boundary values (default: True)
- `custom_generator`: Custom input generator (optional)

**Generated Inputs Include**:
- Edge cases: None, empty strings, max/min integers
- Boundary values: 0, -1, 255, 256, 65535, 65536
- Type confusion: "42" (string) vs 42 (int)
- Random valid inputs
- Memory stress: 1MB strings, 100k element lists

**When to use**:
- Code robustness testing
- Ensuring error handling
- Finding edge case bugs

---

## Environment Interaction Metrics

### StateMatch

**ABC Compliance**: O.g.1, O.g.2, O.g.3

Compares environment states and detects unintended side effects.

```python
from evaris.metrics.state_match import StateMatchMetric, StateMatchConfig

config = StateMatchConfig(
    comparison_mode="exact",  # or "subset", "custom"
    ignore_keys=["timestamp", "session_id"],
    normalize_types=True,
    tolerance=1e-6,
    check_side_effects=True,
    allowed_side_effects=["log_entries"]
)
metric = StateMatchMetric(config)

tc = TestCase(
    input="Move file from /tmp/a.txt to /tmp/b.txt",
    expected={
        "state": {
            "files": {"/tmp/b.txt": "content"},
            "removed": ["/tmp/a.txt"]
        }
    }
)

actual_state = {
    "files": {"/tmp/b.txt": "content"},
    "removed": ["/tmp/a.txt"]
}

result = metric.score(tc, actual_state)
print(f"Matched: {result.passed}")
print(f"Differences: {result.metadata['differences']}")
```

**Configuration Options**:
- `comparison_mode`: "exact", "subset", or "custom" (default: "exact")
- `ignore_keys`: Keys to ignore in comparison (default: [])
- `normalize_types`: Normalize types (int/float) (default: True)
- `tolerance`: Tolerance for numeric comparisons (default: 1e-6)
- `check_side_effects`: Check for unintended side effects (default: True)
- `allowed_side_effects`: Explicitly allowed side effect keys (default: [])
- `custom_comparator`: Custom comparison function (optional)

**Comparison Modes**:
- **exact**: All keys must match exactly
- **subset**: Actual must contain at least expected keys
- **custom**: Use custom_comparator function

**When to use**:
- Environment manipulation tasks
- File system operations
- Database modifications
- API state changes

**Example with side effects**:
```python
expected = {"user_count": 5}
actual = {"user_count": 5, "cache_cleared": True}  # Unintended side effect!

result = metric.score(tc, actual)
# passed: False
# differences: ["Unintended side effect: unexpected key 'cache_cleared'"]
```

---

## Choosing the Right Metric

### Decision Tree

```
Is your task about...

├─ Exact string matching?
│  └─ Use: ExactMatch
│
├─ Natural language understanding?
│  ├─ Simple semantic similarity?
│  │  └─ Use: SemanticSimilarity
│  └─ Complex semantic evaluation?
│     └─ Use: LLMJudge
│
├─ Structured output parsing?
│  └─ Use: AnswerMatch
│
├─ Code generation?
│  ├─ Correctness verification?
│  │  └─ Use: UnitTest
│  └─ Robustness testing?
│     └─ Use: FuzzTest
│
└─ Environment interaction?
   └─ Use: StateMatch
```

### Combining Metrics

Use multiple metrics for comprehensive evaluation:

```python
from evaris import evaluate
from evaris.metrics.semantic_similarity import SemanticSimilarityMetric
from evaris.metrics.llm_judge import LLMJudgeMetric

# Combine semantic similarity and LLM judge
result = evaluate(
    name="comprehensive-eval",
    task=my_agent,
    data=test_cases,
    metrics=[
        SemanticSimilarityMetric(),
        LLMJudgeMetric(),
        "latency"  # Built-in
    ]
)

# Each test result has multiple metric scores
for test_result in result.results:
    for metric_result in test_result.metrics:
        print(f"{metric_result.name}: {metric_result.score}")
```

---

## Custom Metrics

Create custom metrics by implementing the metric interface:

```python
from evaris.types import MetricResult, TestCase

class CustomMetric:
    """Custom metric implementation."""

    def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Score the output.

        Args:
            test_case: Test case with input and expected
            actual_output: Agent's actual output

        Returns:
            MetricResult with score and metadata
        """
        # Your scoring logic here
        score = 1.0 if self.my_logic(actual_output, test_case.expected) else 0.0

        return MetricResult(
            name="custom_metric",
            score=score,
            passed=score >= 0.5,
            metadata={"additional": "info"}
        )
```

---

## Metric Best Practices

### 1. Match Metric to Task
- Don't use ExactMatch for natural language
- Don't use SemanticSimilarity for code generation
- Use multiple metrics for comprehensive evaluation

### 2. Configure Appropriately
- Set reasonable thresholds (0.7-0.9 for semantic similarity)
- Enable self-consistency for LLM judges (3-5 samples)
- Adjust timeouts based on task complexity

### 3. Validate Metrics
```python
from evaris.validation import validate_metric

result = validate_metric(my_metric, test_case, output)
if not result.is_valid:
    print("Metric validation failed!")
```

### 4. Monitor Performance
- Track latency for all metrics
- Use caching for expensive operations (LLM calls, embeddings)
- Batch evaluations when possible

### 5. Report Comprehensively
```python
from evaris.statistics import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
stats = analyzer.analyze_evaluation(eval_result)
print(f"Mean: {stats.mean:.2f} [{stats.ci.lower:.2f}, {stats.ci.upper:.2f}]")
```

---

## See Also

- [ABC Compliance Report](./ABC_COMPLIANCE.md) - Full ABC checklist coverage
- [API Documentation](./API.md) - Core API reference
- [Examples](../examples/) - Metric usage examples

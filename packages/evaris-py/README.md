# Evaris Python SDK

> AI Agent Evaluation and Observability Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/swaingotnochill/evaris)

## Installation

```bash
pip install evaris
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

## Features

### Core Evaluation
- **Simple API** - One function to evaluate agents
- **Type-Safe** - Full type hints with Pydantic models
- **Dataset Loading** - Load test cases from JSONL, JSON, CSV files
- **Error Handling** - Graceful failure with detailed error messages
- **100% Test Coverage** - Comprehensive test suite

### ABC Compliance (100% Coverage)
- **[ABC Compliant](./docs/ABC_COMPLIANCE.md)** - Implements all 48 checks from Agentic Benchmark Checklist
- **Statistical Rigor** - Confidence intervals, significance testing, effect sizes
- **Baseline Comparisons** - Built-in baselines (random, do-nothing, trivial)
- **Contamination Detection** - Dataset fingerprinting and versioning
- **Oracle Validation** - Verify benchmark solvability

### Comprehensive Metrics
- **Semantic Similarity** - Embedding-based similarity (O.a.1, O.a.2)
- **LLM-as-Judge** - GPT-4/Claude evaluators with self-consistency (O.c.1, O.c.2)
- **Answer Matching** - Structured output parsing (O.h.1, O.h.2)
- **Unit Testing** - Code generation with coverage & complexity (O.d.1, O.d.2)
- **Fuzz Testing** - Robustness with diverse inputs (O.e.1, O.e.2, O.e.3)
- **State Matching** - Environment interaction validation (O.g.1, O.g.2, O.g.3)

### Tracing & Observability
- **OpenTelemetry Integration** - Distributed tracing with span hierarchy
- **Debug Logging** - Detailed LLM prompt/response logging
- **Multiple Exporters** - Console, OTLP (Jaeger, Zipkin, cloud)
- **Performance Tracking** - Latency, token usage, cost attribution
- **Environment Configuration** - Flexible env var and API configuration

### Coming Soon
- **Async Support** - Parallel evaluation
- **Framework Integrations** - LangChain, CrewAI, AutoGen
- **Web Dashboard** - Visual evaluation reports

## Documentation

- **[ABC Compliance Report](./docs/ABC_COMPLIANCE.md)** - Complete Agentic Benchmark Checklist coverage
- **[Metrics Guide](./docs/METRICS.md)** - Comprehensive guide to all evaluation metrics
- **[API Reference](./docs/API.md)** - Complete API documentation with examples
- **[Development Guide](./docs/DEVELOPMENT.md)** - Contributing and development workflow
- **[Test Structure](./tests/README.md)** - Test organization and guidelines

## Basic Usage

### Simple Evaluation

```python
from evaris import evaluate

def chatbot(message: str) -> str:
    return f"You said: {message}"

result = evaluate(
    name="chatbot-test",
    task=chatbot,
    data=[
        {"input": "Hello", "expected": "You said: Hello"},
        {"input": "Hi there", "expected": "You said: Hi there"}
    ],
    metrics=["exact_match"]
)

print(f"Passed: {result.passed}/{result.total}")
print(f"Accuracy: {result.accuracy:.2%}")
```

### Load Datasets from Files

```python
from evaris import load_dataset, evaluate

# Load test cases from JSONL file
test_cases = load_dataset("tests/eval.jsonl")

def my_agent(query: str) -> str:
    # Your agent implementation
    return generate_response(query)

result = evaluate(
    name="production-eval",
    task=my_agent,
    data=test_cases,
    metrics=["exact_match", "latency"]
)

print(f"Passed: {result.passed}/{result.total}")
```

**Supported formats:**
- JSONL (`.jsonl`, `.ndjson`) - Memory efficient, line-by-line processing
- JSON (`.json`) - Array of objects or single object
- CSV (`.csv`) - First row as header, extra columns stored as metadata

**Example dataset file (eval.jsonl):**
```jsonl
{"input": "What is 2+2?", "expected": "4"}
{"input": "Capital of France?", "expected": "Paris", "metadata": {"difficulty": "easy"}}
```

### Multiple Metrics

```python
result = evaluate(
    name="multi-metric",
    task=my_agent,
    data=[...],
    metrics=["exact_match", "latency"]
)

# Access detailed results
for test_result in result.results:
    print(f"Input: {test_result.test_case.input}")
    print(f"Output: {test_result.output}")
    print(f"Latency: {test_result.latency_ms}ms")

    for metric in test_result.metrics:
        print(f"  {metric.name}: {metric.score}")
```

### Complex Data Types

```python
# JSON/Dict inputs and outputs
data = [
    {
        "input": {"query": "weather", "location": "NYC"},
        "expected": {"temp": 72, "condition": "sunny"}
    }
]

result = evaluate(name="api-test", task=api_agent, data=data, metrics=["exact_match"])
```

## Tracing and Observability

Evaris includes built-in OpenTelemetry integration for tracing evaluations, debugging issues, and profiling performance.

### Installation

```bash
# Install with tracing support
pip install evaris[tracing]

# Or install OpenTelemetry separately
pip install evaris
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

### Quick Start - Tracing

```python
from evaris import evaluate
from evaris.tracing import configure_tracing, configure_debug_logging

# Configure tracing (console exporter)
configure_tracing(
    service_name="my-evaluation",
    exporter_type="console",  # or "otlp" for Jaeger/Zipkin
    enabled=True
)

# Enable debug logging for detailed output
configure_debug_logging(enabled=True)

# Run evaluation - tracing happens automatically
result = evaluate(
    name="traced-eval",
    task=my_agent,
    data=test_cases,
    metrics=["exact_match", "latency"]
)
```

### Trace Exporters

**Console Exporter** (development):
```python
configure_tracing(exporter_type="console")
```

**OTLP Exporter** (production - Jaeger, Zipkin, cloud providers):
```python
configure_tracing(
    exporter_type="otlp",
    otlp_endpoint="http://localhost:4317"  # Jaeger endpoint
)
```

**Running Jaeger** (for trace visualization):
```bash
docker run -d --name jaeger \
  -p 16686:16686 \  # UI
  -p 4317:4317 \    # OTLP gRPC
  jaegertracing/all-in-one:latest

# Open UI: http://localhost:16686
```

### Environment Variables

Configure tracing via environment variables instead of code:

```bash
export EVARIS_TRACING=true
export EVARIS_DEBUG=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

### Span Hierarchy

Evaris creates hierarchical spans for complete observability:

```
evaluation
  ├── dataset_normalization
  ├── generate_test_cases
  ├── metric_resolution
  └── test_execution
        ├── test_case_execution (per test)
        │     ├── agent_execution
        │     └── metric.{MetricName}
        │           └── llm_api_call (for LLM metrics)
```

### Debug Logging

Enable detailed logging for LLM prompts, responses, and intermediate values:

```python
configure_debug_logging(enabled=True)

# Now LLM metrics will log:
# - Prompts sent to LLMs
# - Responses received
# - Reasoning and scores
# - Token usage
# - Intermediate computation steps
```

### Per-Evaluation Configuration

Override global settings per evaluation:

```python
result = evaluate(
    name="special-eval",
    task=my_agent,
    data=test_cases,
    metrics=["llm_judge"],
    enable_tracing=True,   # Override global tracing
    enable_debug=True      # Override global debug logging
)
```

### Example: Full Observability Stack

See [examples/tracing_example.py](./examples/tracing_example.py) for a complete example.

```python
from evaris import evaluate
from evaris.tracing import configure_tracing, configure_debug_logging
from evaris.metrics.llm_judge import LLMJudgeMetric, LLMJudgeConfig

# 1. Configure tracing
configure_tracing(
    service_name="production-eval",
    exporter_type="otlp",
    otlp_endpoint="http://localhost:4317"
)

# 2. Enable debug logging
configure_debug_logging(enabled=True)

# 3. Run evaluation
llm_judge = LLMJudgeMetric(LLMJudgeConfig(provider="openai", model="gpt-4"))

result = evaluate(
    name="production-qa-eval",
    task=my_agent,
    data=test_cases,
    metrics=[llm_judge, "exact_match", "latency"]
)

# 4. View traces in Jaeger UI
print("View traces at: http://localhost:16686")
```

### Trace Attributes

Evaris automatically tracks:

**Evaluation-level:**
- `eval.name` - Evaluation name
- `eval.total` - Total test cases
- `eval.passed` - Tests passed
- `eval.accuracy` - Final accuracy
- `eval.avg_latency_ms` - Average latency

**Test-level:**
- `test.passed` - Test result
- `latency_ms` - Execution time

**Metric-level:**
- `metric.score` - Metric score
- `metric.passed` - Pass/fail status

**LLM-level** (for LLM metrics):
- `llm.provider` - Provider (openai, anthropic, qwen, gemini)
- `llm.model` - Model name
- `llm.prompt_tokens` - Prompt tokens
- `llm.completion_tokens` - Completion tokens
- `llm.total_tokens` - Total tokens

## Available Metrics

See **[Metrics Guide](./docs/METRICS.md)** for comprehensive documentation.

### Built-in Metrics

| Metric | Use Case | ABC Compliance |
|--------|----------|----------------|
| `exact_match` | Exact string comparison | - |
| `latency` | Performance measurement | - |

### Advanced Metrics

| Metric | Use Case | ABC Compliance |
|--------|----------|----------------|
| **SemanticSimilarity** | Natural language outputs | O.a.1, O.a.2 |
| **LLMJudge** | Complex semantic evaluation | O.c.1, O.c.2 |
| **AnswerMatch** | Structured output parsing | O.h.1, O.h.2 |
| **UnitTest** | Code generation | O.d.1, O.d.2 |
| **FuzzTest** | Code robustness | O.e.1, O.e.2, O.e.3 |
| **StateMatch** | Environment interaction | O.g.1, O.g.2, O.g.3 |

### Example: LLM-as-Judge

```python
from evaris.metrics.llm_judge import LLMJudgeMetric, LLMJudgeConfig

config = LLMJudgeConfig(
    provider="openai",
    model="gpt-4",
    enable_self_consistency=True
)
metric = LLMJudgeMetric(config)

result = evaluate(
    name="semantic-eval",
    task=my_agent,
    data=test_cases,
    metrics=[metric]
)
```

### Example: ABC-Compliant Evaluation

```python
from evaris import evaluate
from evaris.statistics import StatisticalAnalyzer
from evaris.baselines import BaselineManager
from evaris.contamination import ContaminationDetector
from evaris.metrics.semantic_similarity import SemanticSimilarityMetric

# 1. Check for contamination
detector = ContaminationDetector()
detector.register_dataset(test_cases, version="1.0", release_date="2025-01-01")

# 2. Evaluate with semantic similarity
metric = SemanticSimilarityMetric()
result = evaluate(name="my-eval", task=my_agent, data=test_cases, metrics=[metric])

# 3. Statistical analysis
analyzer = StatisticalAnalyzer()
stats = analyzer.analyze_evaluation(result)
print(f"Mean: {stats.mean:.2f} [{stats.ci.lower:.2f}, {stats.ci.upper:.2f}]")

# 4. Baseline comparison
baseline_manager = BaselineManager()
baseline_report = baseline_manager.compare_with_baselines(result, "MyAgent")
print(baseline_manager.format_comparison_report(baseline_report))
```

## Development

See [DEVELOPMENT.md](./docs/DEVELOPMENT.md) for detailed development instructions.

### Quick Start

```bash
# Setup
uv venv
uv pip install -e ".[dev]"

# Run all checks
pytest --cov=evaris --cov-fail-under=90 && \
black evaris tests && \
ruff check evaris tests && \
mypy evaris
```

### Testing

```bash
# Run tests
pytest -v

# With coverage
pytest --cov=evaris --cov-report=term-missing

# Specific test
pytest tests/test_evaluate.py::TestEvaluateBasic::test_evaluate_single_test_case_exact_match
```

## Roadmap

- Core evaluation API
- Dataset loading (JSONL, JSON, CSV)
- ABC-compliant metrics (all 48 checks)
- Statistical analysis with confidence intervals
- Baseline comparison utilities
- Contamination detection
- Oracle validation
- Comprehensive validation
- OpenTelemetry tracing integration
- Debug logging for LLM prompts/responses
- OTLP and console exporters

**Future:**
- [ ] Async agent support
- [ ] Progress bars for long evaluations
- [ ] Result export (JSON, Markdown, HTML)
- [ ] CLI tool (`evaris run config.yaml`)
- [ ] Trace sampling and context propagation
- [ ] Web-based trace visualization
- [ ] LangChain/CrewAI integrations
- [ ] Web dashboard for results
- [ ] Distributed evaluation
- [ ] Cost profiling and optimization

## Releasing

```bash
# 1. Verify package is ready
python verify_package.py

# 2. Update version in pyproject.toml
# 3. Update CHANGELOG.md
```
**Automated (Recommended):**
```bash
./publish.sh test   # Test on TestPyPI first
./publish.sh prod   # Publish to production PyPI
```

## Contributing

See [DEVELOPMENT.md](./docs/DEVELOPMENT.md) for development workflow and coding standards.

## License

Apache-2.0 - See [LICENSE](./LICENSE) for details

## Links

- [GitHub](https://github.com/swaingotnochill/evaris)
- [Documentation](https://docs.evaris.dev) (coming soon)
- [Issues](https://github.com/swaingotnochill/evaris/issues)

NOTE: The development of the project has been shifted to a monorepo in a separate organization.
https://github.com/evarisai/evaris

Any new changes will be made in that repository itself.

# Evaris

> AI Agent Evaluation, Tracing, Profiling & Observability Platform

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Evaris is a comprehensive platform for evaluating, tracing, and observing AI agents. Build reliable AI systems with confidence through systematic evaluation and deep observability.

## Project Components

###  Packages

#### [evaris-py](./packages/evaris-py) - Python SDK

Python SDK for evaluating AI agents with built-in metrics and type-safe evaluation framework.

**Status:**  v0.0.1-dev-001 ( Work in Progress)

**Features:**
- Simple evaluation API: `evaluate(name, task, data, metrics)`
- Type-safe Pydantic models
- Built-in metrics: `exact_match`, `latency`
- 100% test coverage
- Full async support (coming soon)

**Installation:**
```bash
pip install evaris
```

**Quick Start:**
```python
from evaris import evaluate

def my_agent(query: str) -> str:
    return f"Answer: {query}"

result = evaluate(
    name="my-first-eval",
    task=my_agent,
    data=[{"input": "What is 2+2?", "expected": "Answer: What is 2+2?"}],
    metrics=["exact_match", "latency"]
)

print(result)
# Evaluation: my-first-eval
# Total: 1 | Passed: 1 | Failed: 0
# Accuracy: 100.00% | Avg Latency: 0.15ms
```

**Documentation:**
- [Python SDK README](./packages/evaris-py/README.md)
- [API Reference](./packages/evaris-py/docs/API.md)
- [Development Guide](./packages/evaris-py/docs/DEVELOPMENT.md)

---

### Coming Soon

#### evaris-ts - TypeScript SDK
TypeScript/JavaScript SDK for evaluating AI agents in Node.js and browser environments.

**Planned Features:**
- Native TypeScript support with full type safety
- Framework integrations: LangChain.js, Vercel AI SDK
- Streaming evaluation support
- Web worker support for browser environments

#### evaris-trace - Tracing & Observability
Distributed tracing system for AI agents inspired by Langfuse and OpenTelemetry.

**Planned Features:**
- Hierarchical trace visualization (Trace -> Span -> Event)
- LLM call tracking with token usage and costs
- Session grouping and user attribution
- High-throughput trace ingestion API
- Real-time trace dashboard

#### evaris-cli - Command Line Interface
CLI tool for running evaluations locally and in CI/CD pipelines.

**Planned Features:**
- YAML-based evaluation configurations
- Local evaluation execution
- CI/CD integrations (GitHub Actions, GitLab CI)
- Result visualization and reporting

## Architecture

Evaris follows a modular architecture with language-specific SDKs and shared core services.

```
evaris/
 packages/
    evaris-py/        # Python SDK (Released)
    evaris-ts/        # TypeScript SDK (Planned)
    evaris-trace/     # Tracing system (Planned)
    evaris-cli/       # CLI tool (Planned)
 apps/
     platform/         # Web dashboard (Planned)
```

## Development Philosophy

Evaris is built following strict development principles:

- **Documentation First** - Every feature is documented before/during implementation
- **Atomic Features** - Small, isolated features that are tested and merged independently
- **Type Safety** - Strong typing with runtime validation across all SDKs
- **YAGNI** - Only build what's needed now, no speculative features
- **TDD** - Comprehensive test coverage (>90%) with unit, integration, and system tests

## Roadmap
- [x] Python SDK with evaluation API
- [x] Pydantic models for type safety
- [x] Built-in metrics: exact_match, latency
- [x] Comprehensive test suite (54 tests, 100% coverage)
- [x] Development documentation
- [ ] TypeScript SDK
- [ ] CLI tool with YAML configs
- [ ] Tracing system
- [ ] Web dashboard
- [ ] Framework integrations (LangChain, CrewAI, AutoGen)
- [ ] CI/CD integrations (GitHub Actions, GitLab CI)

## Contributing

Contributions are welcome! Please see individual package documentation for development guidelines.
Everytime you come across some code which can be cleaned or improved, please
feel free to drop a PR.

### Getting Started

1. Clone the repository:
```bash
git clone https://github.com/swaingotnochill/evaris.git
cd evaris
```

2. Work on a specific package:
```bash
cd packages/evaris-py
uv venv
uv pip install -e ".[dev]"
uv run pytest
```

See package-specific `DEVELOPMENT.md` files for detailed instructions.

## Testing

Each package maintains comprehensive test coverage:

- **Unit tests** - Individual functions and classes
- **Integration tests** - Component interactions
- **System tests** - End-to-end flows

All packages target >90% code coverage.

## License

Apache-2.0 - See [LICENSE](./LICENSE) for details.

Copyright 2025 Roshan Nrusing Swain

## Links

- [GitHub](https://github.com/swaingotnochill/evaris)
- [Documentation](https://docs.evaris.dev) (coming soon)
- [Issues](https://github.com/swaingotnochill/evaris/issues)
- [Python SDK on PyPI](https://pypi.org/project/evaris/) (coming soon)

## Support

- [Documentation](./packages/evaris-py/README.md)
- [Report Issues](https://github.com/swaingotnochill/evaris/issues)
- [Discussions](https://github.com/swaingotnochill/evaris/discussions)

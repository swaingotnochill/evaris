# Changelog

All notable changes to evaris-py will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- None

### Changed
- None

### Fixed
- None

## [0.0.1] - 2025-01-18

### Added
- Core evaluation API with `evaluate()` function
- Type-safe Pydantic models for test cases and results
- Dataset loading from JSONL, JSON, and CSV files
- Built-in metrics: `exact_match` and `latency`
- ABC-compliant advanced metrics:
  - Semantic similarity (embedding-based)
  - LLM-as-Judge (GPT-4, Claude, Qwen, Gemini)
  - Answer matching with structured output parsing
  - Unit testing for code generation
  - Fuzz testing for robustness
  - State matching for environment interactions
- Statistical analysis utilities:
  - Confidence intervals
  - Significance testing
  - Effect size calculations
- Baseline comparison tools (random, do-nothing, trivial)
- Contamination detection with dataset fingerprinting
- Oracle validation for benchmark solvability
- OpenTelemetry tracing integration:
  - Console and OTLP exporters
  - Hierarchical span tracking
  - Debug logging for LLM prompts/responses
- Conversation agent wrappers for multi-turn evaluation
- Comprehensive test suite (100% coverage)

[Unreleased]: https://github.com/swaingotnochill/evaris/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/swaingotnochill/evaris/releases/tag/v0.1.0

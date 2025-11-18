# Evaris Examples

This directory contains example projects demonstrating how to use the Evaris evaluation framework.

## Available Examples

### 1. Chatbot Evaluation (`chatbot-evaluation/`)

A comprehensive example showing how to evaluate an LLM-based chatbot using multiple metrics. Similar to the [DeepEval DataCamp tutorial](https://www.datacamp.com/tutorial/deepeval) but using Evaris.

**Features demonstrated:**
- Basic metrics (exact match, latency)
- Semantic similarity evaluation
- LLM-as-Judge with self-consistency
- Dataset loading from JSONL files
- Multiple LLM providers (QWEN/DashScope, Google Gemini)
- Custom metric configuration

**Quick Start:**
```bash
# Set up environment and install evaris
uv venv
source .venv/bin/activate  # On macOS/Linux
uv pip install -e packages/evaris-py

# Run example
cd examples/chatbot-evaluation
export DASHSCOPE_API_KEY="your-key"  # or GEMINI_API_KEY
python evaluate_chatbot.py
```

See [chatbot-evaluation/README.md](chatbot-evaluation/README.md) for full documentation.

## Prerequisites

All examples require:
- Python 3.9+
- uv (recommended): See installation instructions below
- Evaris installed: `uv pip install -e packages/evaris-py`

## Example Structure

```
examples/
├── README.md                  # This file
└── chatbot-evaluation/       # Chatbot evaluation example
    ├── README.md             # Full documentation
    ├── QUICKSTART.md         # 5-minute quick start
    ├── FILES.md              # File descriptions
    ├── chatbot.py            # Chatbot implementation
    ├── evaluate_chatbot.py   # Main evaluation script
    ├── test_basic.py         # Basic test (no API key)
    ├── test_data.jsonl       # Test dataset
    ├── requirements.txt      # Dependencies
    └── .env.example          # API key template
```

## Running Examples

### Step 1: Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Set Up Environment and Install Evaris

From the repository root:
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On macOS/Linux (.venv\Scripts\activate on Windows)

# Install Evaris
uv pip install -e packages/evaris-py
```

### Step 3: Install Example Dependencies

```bash
cd examples/chatbot-evaluation
uv pip install -r requirements.txt
```

### Step 4: Set API Keys

```bash
export DASHSCOPE_API_KEY="your-key"
# or
export GEMINI_API_KEY="your-key"
```

### Step 5: Run Example

```bash
# Basic test (no API key required)
python test_basic.py

# Full evaluation (requires API key)
python evaluate_chatbot.py
```

## Contributing Examples

To add a new example:

1. Create a new directory: `examples/your-example/`
2. Add your example code
3. Include a detailed README.md with:
   - Overview
   - Prerequisites
   - Setup instructions
   - Usage examples
   - Expected output
4. Add a `requirements.txt` for any optional dependencies
5. Update this README with a link to your example

## Example Ideas

Interested in contributing? Here are some example ideas:

- **RAG Evaluation**: Evaluating retrieval-augmented generation systems
- **Multi-Agent Systems**: Evaluating agent collaboration
- **Code Generation**: Evaluating code-generating agents with unit tests
- **Tool-Using Agents**: Evaluating agents that use external tools
- **Conversation Evaluation**: Multi-turn dialogue evaluation
- **Batch Processing**: Evaluating agents at scale
- **A/B Testing**: Comparing different agent versions

## Getting Help

- **Documentation**: See `packages/evaris-py/docs/` directory
- **API Reference**: `packages/evaris-py/docs/API.md`
- **Metrics Guide**: `packages/evaris-py/docs/METRICS.md`
- **ABC Compliance**: `packages/evaris-py/docs/ABC_COMPLIANCE.md`

## License

All examples are provided under the same license as Evaris (Apache 2.0).

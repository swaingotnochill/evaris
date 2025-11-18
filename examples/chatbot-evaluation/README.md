# Chatbot Evaluation Example

Demonstrates the async evaluation workflow:
1. Load goldens (input + expected)
2. Generate outputs by running the actual chatbot (QWEN)
3. Evaluate with multiple metrics (Exact Match, Semantic Similarity, LLM Judge)

## Files

- `chatbot.py` - Simple chatbot using QWEN model (qwen-turbo)
- `evaluate_chatbot.py` - Evaluation script
- `test_data.jsonl` - Test data (10 questions)
- `debug_outputs.py` - Debug script to analyze saved results

## Setup

Install dependencies:

```bash
cd packages/evaris-py
pip install -e .
pip install sentence-transformers openai google-generativeai
```

Set API key:

```bash
export DASHSCOPE_API_KEY="your-key"
```

## Usage

Run evaluation (with result saving enabled by default):

```bash
cd examples/chatbot-evaluation
python evaluate_chatbot.py
```

Results will be saved to the `outputs/` directory with timestamps:
- `generated_outputs_qwen_<timestamp>.jsonl` - All chatbot responses
- `exact_match_results_qwen_<timestamp>.json` - Exact match evaluation results
- `semantic_results_qwen_<timestamp>.json` - Semantic similarity results
- `llm_judge_results_qwen_<timestamp>.json` - LLM judge results
- `summary_qwen_<timestamp>.json` - Overall summary

To disable result saving, modify the script to call `main(save_results=False)`.

**Debug saved results:**
```bash
python debug_outputs.py
```

## Workflow

The evaluation follows this architecture:

```python
import asyncio
from evaris import EvaluationDataset, evaluate
from chatbot import chatbot

async def main():
    # Step 1: Load goldens (input + expected)
    dataset = EvaluationDataset.from_file("test_data.jsonl", as_goldens=True)

    # Step 2: Generate actual outputs by running the real chatbot
    async def async_agent(input_text: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, chatbot, input_text, "qwen")

    await dataset.generate_test_cases_async(async_agent, max_concurrency=3)

    # Step 3: Evaluate - TestCases already have actual_output from Step 2
    result = evaluate(
        name="eval",
        task=lambda x: "",  # Not used, TestCases have actual_output
        data=dataset.test_cases,
        metrics=["exact_match"]
    )
    print(f"Accuracy: {result.accuracy:.1%}")

asyncio.run(main())
```

**Key Points:**
- Goldens contain only `input` + `expected` (no actual_output)
- Step 2 runs the actual chatbot to generate `actual_output`
- Step 3 evaluates using TestCases that already have `actual_output`
- No dummy/mock data - everything uses real model responses

## Test Data Format

```jsonl
{"input": "What is 2+2?", "expected": "4", "metadata": {"category": "math"}}
{"input": "Capital of France?", "expected": "Paris", "metadata": {"category": "geography"}}
```

No `actual_output` field - this is generated when you run your agent.

## Expected Output

```
======================================================================
Chatbot Evaluation
======================================================================

Using QWEN

Step 1: Loading test data
Loaded 10 test cases

Step 2: Generating outputs
Progress: 10/10
Generated 10 outputs

Step 3: Evaluating with metrics

[1/3] Exact Match
Accuracy: 30.0%
Passed: 3/10

[2/3] Semantic Similarity
Accuracy: 80.0%
Passed: 8/10

[3/3] LLM Judge
Accuracy: 70.0%
Passed: 7/10

======================================================================
Summary
======================================================================
Provider: QWEN
Test cases: 10
Exact match: 30.0%
Semantic similarity: 80.0%
LLM Judge: 70.0%
```

**Note:** The exact scores will vary based on the model's responses and the LLM judge's evaluation.

## Metrics

The example demonstrates three evaluation metrics:

1. **Exact Match**: Checks if the actual output exactly matches the expected output
2. **Semantic Similarity**: Uses sentence embeddings to measure semantic similarity (requires `sentence-transformers`)
3. **LLM Judge**: Uses QWEN (qwen-plus) to judge if the response is semantically correct

## Troubleshooting

**No API key found**
- Set `DASHSCOPE_API_KEY` environment variable

**Missing dependencies**
```bash
pip install sentence-transformers openai
```

**Debugging low accuracy**
- Run `python debug_outputs.py` to see what the chatbot generated vs. expected
- Check `outputs/exact_match_results_qwen_*.json` for detailed per-question results
- LLM judge results include reasoning for each evaluation

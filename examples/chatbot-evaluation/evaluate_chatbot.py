"""Chatbot evaluation using Evaris framework.

Demonstrates the async evaluation workflow:
1. Load goldens from file
2. Generate outputs by running the chatbot
3. Evaluate with metrics
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from evaris import EvaluationDataset, evaluate
from evaris.metrics.semantic_similarity import SemanticSimilarityConfig, SemanticSimilarityMetric
from evaris.metrics.llm_judge import LLMJudgeConfig, LLMJudgeMetric
from chatbot import chatbot

import dotenv
dotenv.load_dotenv()


async def main(save_results: bool = True):
    print("=" * 70)
    print("Chatbot Evaluation")
    print("=" * 70)
    print()

    has_qwen = bool(os.getenv("DASHSCOPE_API_KEY"))

    if not has_qwen:
        print("Error: DASHSCOPE_API_KEY not found")
        print("Set DASHSCOPE_API_KEY environment variable")
        return

    provider = "qwen"
    print(f"Using {provider.upper()}")
    print()

    # Step 1: Load goldens
    print("Step 1: Loading test data")
    data_file = Path(__file__).parent / "test_data.jsonl"
    dataset = EvaluationDataset.from_file(str(data_file), as_goldens=True)
    print(f"Loaded {len(dataset.goldens)} test cases")
    print()

    # Step 2: Generate outputs
    print("Step 2: Generating outputs")

    async def async_agent(input_text: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, chatbot, input_text, provider)

    def progress(current: int, total: int):
        print(f"Progress: {current}/{total}", end='\r')

    await dataset.generate_test_cases_async(
        async_agent,
        max_concurrency=3,
        progress_callback=progress
    )
    print(f"\nGenerated {len(dataset.test_cases)} outputs")
    print()

    # Prepare for saving results if enabled
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = Path(__file__).parent / "outputs"

    if save_results:
        outputs_dir.mkdir(exist_ok=True)

        # Save generated outputs for debugging
        outputs_file = outputs_dir / \
            f"generated_outputs_{provider}_{timestamp}.jsonl"
        with open(outputs_file, "w") as f:
            for tc in dataset.test_cases:
                f.write(json.dumps({
                    "input": tc.input,
                    "expected": tc.expected,
                    "actual_output": tc.actual_output,
                    "metadata": tc.metadata
                }) + "\n")
        print(f"Saved generated outputs to: {outputs_file}")
        print()

    # Step 3: Evaluate with metrics
    print("Step 3: Evaluating with metrics")
    print()

    # Since we already generated actual_output in Step 2,
    # we can directly pass the TestCases to evaluate()
    # evaluate() will not re-run the agent since they already have actual_output

    print("[1/3] Exact Match")
    result_exact = evaluate(
        name="chatbot-exact-match",
        task=lambda x: "",  # Not used since TestCases already have actual_output
        data=dataset.test_cases,
        metrics=["exact_match"]
    )
    print(f"Accuracy: {result_exact.accuracy:.1%}")
    print(f"Passed: {result_exact.passed}/{result_exact.total}")

    if save_results:
        # Save exact match results
        exact_results_file = outputs_dir / \
            f"exact_match_results_{provider}_{timestamp}.json"
        with open(exact_results_file, "w") as f:
            results_data = {
                "metric": "exact_match",
                "accuracy": result_exact.accuracy,
                "passed": result_exact.passed,
                "failed": result_exact.failed,
                "total": result_exact.total,
                "results": [
                    {
                        "input": r.test_case.input,
                        "expected": r.test_case.expected,
                        "actual_output": r.output,
                        "passed": all(m.passed for m in r.metrics),
                        "error": r.error,
                        "metrics": [{"name": m.name, "score": m.score, "passed": m.passed, "metadata": m.metadata} for m in r.metrics]
                    }
                    for r in result_exact.results
                ]
            }
            json.dump(results_data, f, indent=2)
        print(f"Saved exact match results to: {exact_results_file}")
    print()

    try:
        print("[2/3] Semantic Similarity")
        config = SemanticSimilarityConfig(threshold=0.75)
        metric = SemanticSimilarityMetric(config=config)

        result_semantic = evaluate(
            name="chatbot-semantic",
            task=lambda x: "",  # Not used since TestCases already have actual_output
            data=dataset.test_cases,
            metrics=[metric]
        )
        print(f"Accuracy: {result_semantic.accuracy:.1%}")
        print(f"Passed: {result_semantic.passed}/{result_semantic.total}")

        if save_results:
            # Save semantic similarity results
            semantic_results_file = outputs_dir / \
                f"semantic_results_{provider}_{timestamp}.json"
            with open(semantic_results_file, "w") as f:
                results_data = {
                    "metric": "semantic_similarity",
                    "accuracy": result_semantic.accuracy,
                    "passed": result_semantic.passed,
                    "failed": result_semantic.failed,
                    "total": result_semantic.total,
                    "results": [
                        {
                            "input": r.test_case.input,
                            "expected": r.test_case.expected,
                            "actual_output": r.output,
                            "passed": all(m.passed for m in r.metrics),
                            "error": r.error,
                            "metrics": [{"name": m.name, "score": m.score, "passed": m.passed, "metadata": m.metadata} for m in r.metrics]
                        }
                        for r in result_semantic.results
                    ]
                }
                json.dump(results_data, f, indent=2)
            print(f"Saved semantic similarity results to: {
                  semantic_results_file}")
        print()
    except ImportError:
        print("Skipped (install sentence-transformers)")
        print()

    try:
        print("[3/3] LLM Judge")
        # Use QWEN for the judge
        judge_config = LLMJudgeConfig(
            provider="qwen",
            model="qwen-plus",
            enable_self_consistency=False,  # Disable for faster evaluation
            threshold=0.7
        )
        judge_metric = LLMJudgeMetric(judge_config)

        result_judge = evaluate(
            name="chatbot-llm-judge",
            task=lambda x: "",  # Not used since TestCases already have actual_output
            data=dataset.test_cases,
            metrics=[judge_metric]
        )
        print(f"Accuracy: {result_judge.accuracy:.1%}")
        print(f"Passed: {result_judge.passed}/{result_judge.total}")

        if save_results:
            # Save LLM judge results
            judge_results_file = outputs_dir / \
                f"llm_judge_results_qwen_{timestamp}.json"
            with open(judge_results_file, "w") as f:
                results_data = {
                    "metric": "llm_judge",
                    "judge_provider": "qwen",
                    "judge_model": "qwen-plus",
                    "accuracy": result_judge.accuracy,
                    "passed": result_judge.passed,
                    "failed": result_judge.failed,
                    "total": result_judge.total,
                    "results": [
                        {
                            "input": r.test_case.input,
                            "expected": r.test_case.expected,
                            "actual_output": r.output,
                            "passed": all(m.passed for m in r.metrics),
                            "error": r.error,
                            "metrics": [{"name": m.name, "score": m.score, "passed": m.passed, "metadata": m.metadata} for m in r.metrics]
                        }
                        for r in result_judge.results
                    ]
                }
                json.dump(results_data, f, indent=2)
            print(f"Saved LLM judge results to: {judge_results_file}")
        print()
    except Exception as e:
        print(f"Skipped: {e}")
        print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Provider: QWEN")
    print(f"Test cases: {len(dataset.test_cases)}")
    print(f"Exact match: {result_exact.accuracy:.1%}")
    if 'result_semantic' in locals():
        print(f"Semantic similarity: {result_semantic.accuracy:.1%}")
    if 'result_judge' in locals():
        print(f"LLM Judge: {result_judge.accuracy:.1%}")
    print()

    if save_results:
        # Save final summary
        summary_file = outputs_dir / f"summary_qwen_{timestamp}.json"
        summary_data = {
            "provider": "qwen",
            "timestamp": timestamp,
            "test_cases": len(dataset.test_cases),
            "metrics": {
                "exact_match": {
                    "accuracy": result_exact.accuracy,
                    "passed": result_exact.passed,
                    "failed": result_exact.failed
                }
            }
        }
        if 'result_semantic' in locals():
            summary_data["metrics"]["semantic_similarity"] = {
                "accuracy": result_semantic.accuracy,
                "passed": result_semantic.passed,
                "failed": result_semantic.failed
            }
        if 'result_judge' in locals():
            summary_data["metrics"]["llm_judge"] = {
                "accuracy": result_judge.accuracy,
                "passed": result_judge.passed,
                "failed": result_judge.failed
            }

        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved evaluation summary to: {summary_file}")
        print(f"All results saved to: {outputs_dir}")
        print()


if __name__ == "__main__":
    asyncio.run(main())

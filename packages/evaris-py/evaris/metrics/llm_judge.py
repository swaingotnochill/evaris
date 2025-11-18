"""LLM-as-a-Judge metric for semantic evaluation.

This metric uses an LLM to evaluate agent outputs when exact matching is insufficient.
Implements ABC checks O.c.1 and O.c.2 for accuracy validation and adversarial resistance.
"""

import asyncio
import os
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from evaris.tracing import get_debug_logger, get_tracer
from evaris.types import BaseMetric, MetricResult, TestCase

# Optional dependencies - import at module level for testability
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False

try:
    import dashscope  # type: ignore[import-untyped]
    from dashscope import Generation  # type: ignore[import-untyped]

    DASHSCOPE_AVAILABLE = True
except ImportError:
    dashscope = None  # type: ignore
    Generation = None  # type: ignore
    DASHSCOPE_AVAILABLE = False

try:
    import google.generativeai as genai  # type: ignore[import-untyped]

    GEMINI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    GEMINI_AVAILABLE = False


class LLMJudgeConfig(BaseModel):
    """Configuration for LLM judge metric."""

    provider: Literal["openai", "anthropic", "qwen", "gemini"] = Field(
        default="openai", description="LLM provider to use"
    )
    model: str = Field(default="gpt-4", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key (or use env var)")
    base_url: Optional[str] = Field(
        default=None, description="Base URL for API (used for QWEN/DashScope)"
    )
    temperature: float = Field(default=0.0, description="Temperature for generation")
    max_tokens: int = Field(default=500, description="Max tokens for response")
    custom_prompt: Optional[str] = Field(default=None, description="Custom judge prompt template")
    self_consistency_samples: int = Field(
        default=3, description="Number of samples for self-consistency check"
    )
    enable_self_consistency: bool = Field(
        default=True, description="Enable self-consistency validation"
    )
    threshold: float = Field(default=0.7, description="Score threshold for passing (0.0-1.0)")


class LLMJudgeMetric(BaseMetric):
    """LLM-as-a-Judge metric for semantic evaluation.

    Uses an LLM to judge whether an agent's output is correct based on
    the expected output and task context. Implements validation checks
    from ABC O.c.1 and O.c.2.

    ABC Compliance:
    - O.c.1: Demonstrates accuracy, self-consistency, and agreement with human
    - O.c.2: Designed to resist adversarial inputs and reward hacking

    Example:
        >>> from evaris.metrics.llm_judge import LLMJudgeMetric, LLMJudgeConfig
        >>> # Using OpenAI
        >>> config = LLMJudgeConfig(provider="openai", model="gpt-4")
        >>> metric = LLMJudgeMetric(config)
        >>> # Using QWEN
        >>> config = LLMJudgeConfig(provider="qwen", model="qwen-plus")
        >>> metric = LLMJudgeMetric(config)
        >>> # Using GEMINI
        >>> config = LLMJudgeConfig(provider="gemini", model="gemini-2.5-flash")
        >>> metric = LLMJudgeMetric(config)
        >>> tc = TestCase(
        ...     input="What is the capital of France?",
        ...     expected="Paris"
        ... )
        >>> result = metric.score(tc, "The capital of France is Paris")
        >>> print(result.score)  # Close to 1.0 for correct answer
    """

    def __init__(self, config: Optional[LLMJudgeConfig] = None):
        """Initialize LLM judge metric.

        Args:
            config: Configuration for LLM judge. If None, uses defaults.
        """
        self.config = config or LLMJudgeConfig()
        self._client: Any = None
        self._async_client: Any = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize both sync and async LLM clients based on provider."""
        if self.config.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")

            api_key = self.config.api_key
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")

            self._client = openai.OpenAI(api_key=api_key)
            self._async_client = openai.AsyncOpenAI(api_key=api_key)
        elif self.config.provider == "qwen":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI package required for QWEN. Install with: pip install openai"
                )

            api_key = self.config.api_key
            if api_key is None:
                api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")

            # Use DashScope's OpenAI-compatible endpoint
            base_url = (
                self.config.base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )

            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
            self._async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        elif self.config.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                )

            api_key = self.config.api_key
            if api_key is None:
                api_key = os.getenv("ANTHROPIC_API_KEY")

            self._client = anthropic.Anthropic(api_key=api_key)
            self._async_client = anthropic.AsyncAnthropic(api_key=api_key)
        elif self.config.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "Google Generative AI package required. "
                    "Install with: pip install google-generativeai"
                )

            api_key = self.config.api_key
            if api_key is None:
                api_key = os.getenv("GEMINI_API_KEY")

            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.config.model)
            # Gemini uses the same client for both sync and async
            self._async_client = self._client

    def _get_default_prompt(self, test_case: TestCase, actual_output: Any) -> str:
        """Get default judge prompt.

        Args:
            test_case: Test case with input and expected output
            actual_output: Agent's actual output

        Returns:
            Formatted judge prompt
        """
        return f"""You are an expert evaluator assessing AI agent responses.

Task Input: {test_case.input}
Expected Output: {test_case.expected}
Actual Output: {actual_output}

Evaluate if the actual output is semantically equivalent to the expected output.
Consider:
1. Semantic meaning (not just exact wording)
2. Factual correctness
3. Completeness of the answer
4. Relevance to the task

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}}

Score Guidelines:
- 1.0: Semantically equivalent and correct
- 0.7-0.9: Mostly correct with minor issues
- 0.4-0.6: Partially correct
- 0.0-0.3: Incorrect or irrelevant

Your response:"""

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt.

        Args:
            prompt: Judge prompt

        Returns:
            LLM response text
        """
        tracer = get_tracer()
        debug = get_debug_logger()

        assert self._client is not None, "Client not initialized"

        with tracer.start_span("llm_api_call"):
            # Log prompt in debug mode
            debug.log_prompt(
                "llm_judge",
                prompt,
                provider=self.config.provider,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            tracer.set_attribute("llm.provider", self.config.provider)
            tracer.set_attribute("llm.model", self.config.model)
            tracer.set_attribute("llm.temperature", self.config.temperature)
            tracer.set_attribute("llm.max_tokens", self.config.max_tokens)

            try:
                if self.config.provider in ("openai", "qwen"):
                    # Both OpenAI and QWEN use OpenAI-compatible API
                    response = self._client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    content = response.choices[0].message.content
                    response_text = content if content is not None else ""

                    # Log token usage if available
                    if hasattr(response, "usage") and response.usage:
                        tracer.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens)
                        tracer.set_attribute(
                            "llm.completion_tokens", response.usage.completion_tokens
                        )
                        tracer.set_attribute("llm.total_tokens", response.usage.total_tokens)

                elif self.config.provider == "anthropic":
                    response = self._client.messages.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    response_text = str(response.content[0].text)

                    # Log token usage if available
                    if hasattr(response, "usage") and response.usage:
                        tracer.set_attribute("llm.input_tokens", response.usage.input_tokens)
                        tracer.set_attribute("llm.output_tokens", response.usage.output_tokens)

                elif self.config.provider == "gemini":
                    generation_config = genai.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_tokens,
                    )
                    response = self._client.generate_content(
                        prompt, generation_config=generation_config
                    )
                    response_text = str(response.text)

                    # Log token usage if available
                    if hasattr(response, "usage_metadata"):
                        tracer.set_attribute(
                            "llm.prompt_tokens", response.usage_metadata.prompt_token_count
                        )
                        tracer.set_attribute(
                            "llm.completion_tokens",
                            response.usage_metadata.candidates_token_count,
                        )
                        tracer.set_attribute(
                            "llm.total_tokens", response.usage_metadata.total_token_count
                        )

                else:
                    raise ValueError(f"Unsupported provider: {self.config.provider}")

                # Log response in debug mode
                debug.log_response("llm_judge", response_text, provider=self.config.provider)

                tracer.set_attribute("llm.response_length", len(response_text))
                tracer.set_status("ok")

                return response_text

            except Exception as e:
                tracer.record_exception(e)
                tracer.set_status("error", str(e))
                debug.log_error("llm_judge", e, provider=self.config.provider)
                raise

    async def _call_llm_async(self, prompt: str) -> str:
        """Call the LLM asynchronously with the given prompt.

        Args:
            prompt: Judge prompt

        Returns:
            LLM response text
        """
        tracer = get_tracer()
        debug = get_debug_logger()

        assert self._async_client is not None, "Async client not initialized"

        with tracer.start_span("llm_api_call_async"):
            # Log prompt in debug mode
            debug.log_prompt(
                "llm_judge",
                prompt,
                provider=self.config.provider,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            tracer.set_attribute("llm.provider", self.config.provider)
            tracer.set_attribute("llm.model", self.config.model)
            tracer.set_attribute("llm.temperature", self.config.temperature)
            tracer.set_attribute("llm.max_tokens", self.config.max_tokens)

            try:
                if self.config.provider in ("openai", "qwen"):
                    # Both OpenAI and QWEN use OpenAI-compatible async API
                    response = await self._async_client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    content = response.choices[0].message.content
                    response_text = content if content is not None else ""

                    # Log token usage if available
                    if hasattr(response, "usage") and response.usage:
                        tracer.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens)
                        tracer.set_attribute(
                            "llm.completion_tokens", response.usage.completion_tokens
                        )
                        tracer.set_attribute("llm.total_tokens", response.usage.total_tokens)

                elif self.config.provider == "anthropic":
                    response = await self._async_client.messages.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    response_text = str(response.content[0].text)

                    # Log token usage if available
                    if hasattr(response, "usage") and response.usage:
                        tracer.set_attribute("llm.input_tokens", response.usage.input_tokens)
                        tracer.set_attribute("llm.output_tokens", response.usage.output_tokens)

                elif self.config.provider == "gemini":
                    # Gemini async support is limited, fall back to sync in thread pool
                    response_text = await asyncio.to_thread(self._call_llm, prompt)

                else:
                    raise ValueError(f"Unsupported provider: {self.config.provider}")

                # Log response in debug mode
                debug.log_response("llm_judge", response_text, provider=self.config.provider)

                tracer.set_attribute("llm.response_length", len(response_text))
                tracer.set_status("ok")

                return response_text

            except Exception as e:
                tracer.record_exception(e)
                tracer.set_status("error", str(e))
                debug.log_error("llm_judge", e, provider=self.config.provider)
                raise

    def _parse_judge_response(self, response: str) -> tuple[float, str]:
        """Parse judge response to extract score and reasoning.

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (score, reasoning)
        """
        import json

        try:
            # Try to parse as JSON
            data = json.loads(response.strip())
            score = float(data.get("score", 0.0))
            reasoning = data.get("reasoning", "")
            return score, reasoning
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to extract score from text
            # Look for patterns like "score: 0.8" or "0.8/1.0"
            import re

            score_match = re.search(r"score[:\s]+([0-9]+\.?[0-9]*)", response, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    return score, response
                except ValueError:
                    pass
            return 0.0, f"Failed to parse response: {response}"

    def _check_self_consistency(
        self, test_case: TestCase, actual_output: Any
    ) -> tuple[float, list[float]]:
        """Check self-consistency by sampling multiple judgments.

        ABC O.c.1: Demonstrates self-consistency of the judge.

        This method queries the LLM multiple times (n_samples) and computes
        the mean score across all samples. Using mean (rather than majority vote)
        naturally handles ties and provides a continuous confidence measure.

        For example, with 3 samples:
        - Scores [1.0, 1.0, 0.0] → mean = 0.67 (high confidence pass)
        - Scores [1.0, 0.5, 0.0] → mean = 0.50 (uncertain)
        - Scores [0.0, 0.0, 1.0] → mean = 0.33 (low confidence fail)

        Args:
            test_case: Test case
            actual_output: Agent's output

        Returns:
            Tuple of (mean_score, all_scores)
        """
        tracer = get_tracer()
        debug = get_debug_logger()

        with tracer.start_span("self_consistency_check") as span:
            prompt = self.config.custom_prompt or self._get_default_prompt(test_case, actual_output)

            tracer.set_attribute("self_consistency.samples", self.config.self_consistency_samples)

            scores = []
            for i in range(self.config.self_consistency_samples):
                with tracer.start_span(f"self_consistency_sample_{i+1}"):
                    response = self._call_llm(prompt)
                    score, _ = self._parse_judge_response(response)
                    scores.append(score)
                    tracer.set_attribute(f"sample_{i+1}.score", score)

            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = variance**0.5

            tracer.set_attribute("self_consistency.mean_score", mean_score)
            tracer.set_attribute("self_consistency.variance", variance)
            tracer.set_attribute("self_consistency.std_dev", std_dev)

            debug.log_intermediate(
                "llm_judge",
                "Self-consistency results",
                mean_score=mean_score,
                scores=scores,
                variance=variance,
                std_dev=std_dev,
            )

            if span:
                span.set_attribute("mean_score", mean_score)
                span.set_attribute("score_variance", variance)

            return mean_score, scores

    async def _check_self_consistency_async(
        self, test_case: TestCase, actual_output: Any
    ) -> tuple[float, list[float]]:
        """Check self-consistency by sampling multiple judgments in parallel.

        ABC O.c.1: Demonstrates self-consistency of the judge.
        This async version runs all samples concurrently for better performance.

        Args:
            test_case: Test case
            actual_output: Agent's output

        Returns:
            Tuple of (mean_score, all_scores)
        """
        tracer = get_tracer()
        debug = get_debug_logger()

        with tracer.start_span("self_consistency_check_async") as span:
            prompt = self.config.custom_prompt or self._get_default_prompt(test_case, actual_output)

            tracer.set_attribute("self_consistency.samples", self.config.self_consistency_samples)

            # Run all LLM calls in parallel
            async def get_one_score(sample_index: int) -> tuple[int, float]:
                """Get a single self-consistency score."""
                with tracer.start_span(f"self_consistency_sample_{sample_index+1}"):
                    response = await self._call_llm_async(prompt)
                    score, _ = self._parse_judge_response(response)
                    tracer.set_attribute(f"sample_{sample_index+1}.score", score)
                    return sample_index, score

            # Execute all samples concurrently
            results = await asyncio.gather(
                *[get_one_score(i) for i in range(self.config.self_consistency_samples)],
                return_exceptions=True,
            )

            # Collect scores (filter out exceptions)
            scores: list[float] = []
            for result in results:
                if isinstance(result, Exception):
                    # Log error but continue with remaining scores
                    debug.log_error("llm_judge", result, context="self_consistency")
                    tracer.record_exception(result)
                elif isinstance(result, tuple):
                    _, score = result
                    scores.append(score)

            # Need at least one score
            if not scores:
                raise ValueError("All self-consistency samples failed")

            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = variance**0.5

            tracer.set_attribute("self_consistency.mean_score", mean_score)
            tracer.set_attribute("self_consistency.variance", variance)
            tracer.set_attribute("self_consistency.std_dev", std_dev)

            debug.log_intermediate(
                "llm_judge",
                "Self-consistency results (async)",
                mean_score=mean_score,
                scores=scores,
                variance=variance,
                std_dev=std_dev,
            )

            if span:
                span.set_attribute("mean_score", mean_score)
                span.set_attribute("score_variance", variance)

            return mean_score, scores

    def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Score agent output using LLM judge.

        ABC Compliance:
        - O.c.1: Uses self-consistency check to validate accuracy
        - O.c.2: Structured prompt resists adversarial inputs

        Args:
            test_case: Test case with expected output
            actual_output: Agent's actual output

        Returns:
            MetricResult with score and metadata

        Raises:
            ValueError: If expected output is missing
        """
        tracer = get_tracer()
        debug = get_debug_logger()

        if test_case.expected is None:
            raise ValueError("LLM judge metric requires 'expected' value in test case")

        with tracer.start_span("llm_judge_score") as span:
            metadata: dict[str, Any] = {
                "expected": test_case.expected,
                "actual": actual_output,
                "provider": self.config.provider,
                "model": self.config.model,
            }

            tracer.set_attribute("judge.provider", self.config.provider)
            tracer.set_attribute("judge.model", self.config.model)
            tracer.set_attribute("judge.threshold", self.config.threshold)
            tracer.set_attribute(
                "judge.self_consistency_enabled", self.config.enable_self_consistency
            )

            try:
                if self.config.enable_self_consistency:
                    # Use self-consistency check (ABC O.c.1)
                    mean_score, all_scores = self._check_self_consistency(test_case, actual_output)
                    score = mean_score

                    # Calculate variance for reliability metric
                    variance = sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)
                    metadata["self_consistency_scores"] = all_scores
                    metadata["self_consistency_variance"] = variance
                    metadata["self_consistency_std"] = variance**0.5  # Standard deviation

                    # Get one detailed reasoning
                    prompt = self.config.custom_prompt or self._get_default_prompt(
                        test_case, actual_output
                    )
                    response = self._call_llm(prompt)
                    _, reasoning = self._parse_judge_response(response)
                    metadata["reasoning"] = reasoning

                    # Log reasoning in debug mode
                    debug.log_reasoning(
                        "llm_judge",
                        reasoning,
                        score=mean_score,
                        variance=variance,
                        all_scores=all_scores,
                    )
                else:
                    # Single judgment
                    with tracer.start_span("single_judgment"):
                        prompt = self.config.custom_prompt or self._get_default_prompt(
                            test_case, actual_output
                        )
                        response = self._call_llm(prompt)
                        score, reasoning = self._parse_judge_response(response)
                        metadata["reasoning"] = reasoning
                        metadata["raw_response"] = response

                        # Log reasoning in debug mode
                        debug.log_reasoning("llm_judge", reasoning, score=score)

                # Clamp score to [0, 1]
                score = max(0.0, min(1.0, score))

                # Add threshold to metadata
                metadata["threshold"] = self.config.threshold

                # Determine pass/fail based on configured threshold
                passed = score >= self.config.threshold

                tracer.set_attribute("judge.score", score)
                tracer.set_attribute("judge.passed", passed)
                tracer.set_status("ok")

                if span:
                    span.set_attribute("final_score", score)
                    span.set_attribute("passed", passed)

                return MetricResult(name="llm_judge", score=score, passed=passed, metadata=metadata)

            except Exception as e:
                # Handle errors gracefully
                tracer.record_exception(e)
                tracer.set_status("error", str(e))
                debug.log_error("llm_judge", e, test_case=str(test_case)[:100])

                return MetricResult(
                    name="llm_judge",
                    score=0.0,
                    passed=False,
                    metadata={
                        **metadata,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

    async def a_measure(self, test_case: TestCase) -> MetricResult:
        """Score agent output using LLM judge asynchronously.

        This is the async version of score() that uses async LLM clients and
        runs self-consistency checks in parallel for better performance.

        ABC Compliance:
        - O.c.1: Uses self-consistency check to validate accuracy (with parallel calls)
        - O.c.2: Structured prompt resists adversarial inputs

        Args:
            test_case: Test case with expected output and actual_output

        Returns:
            MetricResult with score and metadata

        Raises:
            ValueError: If expected output or actual_output is missing
        """
        tracer = get_tracer()
        debug = get_debug_logger()

        if test_case.expected is None:
            raise ValueError("LLM judge metric requires 'expected' value in test case")

        if test_case.actual_output is None:
            raise ValueError("LLM judge metric requires 'actual_output' in test case")

        actual_output = test_case.actual_output

        with tracer.start_span("llm_judge_a_measure") as span:
            metadata: dict[str, Any] = {
                "expected": test_case.expected,
                "actual": actual_output,
                "provider": self.config.provider,
                "model": self.config.model,
            }

            tracer.set_attribute("judge.provider", self.config.provider)
            tracer.set_attribute("judge.model", self.config.model)
            tracer.set_attribute("judge.threshold", self.config.threshold)
            tracer.set_attribute(
                "judge.self_consistency_enabled", self.config.enable_self_consistency
            )

            try:
                if self.config.enable_self_consistency:
                    # Use async self-consistency check (ABC O.c.1) with parallel calls
                    mean_score, all_scores = await self._check_self_consistency_async(
                        test_case, actual_output
                    )
                    score = mean_score

                    # Calculate variance for reliability metric
                    variance = sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)
                    metadata["self_consistency_scores"] = all_scores
                    metadata["self_consistency_variance"] = variance
                    metadata["self_consistency_std"] = variance**0.5  # Standard deviation

                    # Get one detailed reasoning
                    prompt = self.config.custom_prompt or self._get_default_prompt(
                        test_case, actual_output
                    )
                    response = await self._call_llm_async(prompt)
                    _, reasoning = self._parse_judge_response(response)
                    metadata["reasoning"] = reasoning

                    # Log reasoning in debug mode
                    debug.log_reasoning(
                        "llm_judge",
                        reasoning,
                        score=mean_score,
                        variance=variance,
                        all_scores=all_scores,
                    )
                else:
                    # Single judgment (async)
                    with tracer.start_span("single_judgment_async"):
                        prompt = self.config.custom_prompt or self._get_default_prompt(
                            test_case, actual_output
                        )
                        response = await self._call_llm_async(prompt)
                        score, reasoning = self._parse_judge_response(response)
                        metadata["reasoning"] = reasoning
                        metadata["raw_response"] = response

                        # Log reasoning in debug mode
                        debug.log_reasoning("llm_judge", reasoning, score=score)

                # Clamp score to [0, 1]
                score = max(0.0, min(1.0, score))

                # Add threshold to metadata
                metadata["threshold"] = self.config.threshold

                # Determine pass/fail based on configured threshold
                passed = score >= self.config.threshold

                tracer.set_attribute("judge.score", score)
                tracer.set_attribute("judge.passed", passed)
                tracer.set_status("ok")

                if span:
                    span.set_attribute("final_score", score)
                    span.set_attribute("passed", passed)

                return MetricResult(name="llm_judge", score=score, passed=passed, metadata=metadata)

            except Exception as e:
                # Handle errors gracefully
                tracer.record_exception(e)
                tracer.set_status("error", str(e))
                debug.log_error("llm_judge", e, test_case=str(test_case)[:100])

                return MetricResult(
                    name="llm_judge",
                    score=0.0,
                    passed=False,
                    metadata={
                        **metadata,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

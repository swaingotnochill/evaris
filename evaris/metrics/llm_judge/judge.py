"""Modular LLM Judge metric with three modes.

This module provides a flexible LLM Judge that supports:
1. Standalone Mode: Direct LLM evaluation (default)
2. Tool Mode: LLM can call tools to verify outputs
3. Meta Mode: LLM synthesizes judgment from other metrics' results
"""

import asyncio
import json
import re
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric, BaseTool
from evaris.core.registry import ToolRegistry
from evaris.core.types import MetricResult, TestCase
from evaris.providers.base import BaseLLMProvider
from evaris.providers.factory import get_provider
from evaris.tracing import get_debug_logger, get_tracer


class JudgeMode(str, Enum):
    """Mode of operation for the LLM Judge."""

    STANDALONE = "standalone"  # Direct LLM evaluation
    TOOLS = "tools"  # LLM can call tools to verify
    META = "meta"  # Synthesize from other metrics


class JudgeConfig(BaseModel):
    """Configuration for the modular LLM Judge.

    Attributes:
        mode: Operating mode (standalone, tools, meta)
        provider: LLM provider name (e.g., "openrouter")
        model: Model to use (e.g., "anthropic/claude-3.5-sonnet")
        api_key: API key (optional, uses env var if not provided)
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
        threshold: Score threshold for passing
        enable_self_consistency: Run multiple samples for reliability
        self_consistency_samples: Number of samples
        tools: List of tool names or instances (for tools mode)
        max_tool_iterations: Maximum tool call iterations
        custom_prompt: Custom evaluation prompt template
    """

    mode: JudgeMode = Field(
        default=JudgeMode.STANDALONE,
        description="Operating mode for the judge",
    )
    provider: str = Field(
        default="openrouter",
        description="LLM provider name",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use (uses provider default if not specified)",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (uses env var if not provided)",
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum response tokens",
    )
    threshold: float = Field(
        default=0.7,
        description="Score threshold for passing (0.0-1.0)",
    )
    enable_self_consistency: bool = Field(
        default=False,
        description="Enable self-consistency checking",
    )
    self_consistency_samples: int = Field(
        default=3,
        description="Number of samples for self-consistency",
    )
    tools: list[Union[str, BaseTool]] = Field(
        default_factory=list,
        description="Tools for tool mode (names or instances)",
    )
    max_tool_iterations: int = Field(
        default=5,
        description="Maximum tool call iterations",
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom evaluation prompt template",
    )

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True


class LLMJudge(BaseMetric):
    """Modular LLM-as-a-Judge metric.

    A flexible evaluation metric that uses an LLM to judge agent outputs.
    Supports three modes of operation:

    1. Standalone Mode (default):
       Direct LLM evaluation of expected vs actual output.

       >>> judge = LLMJudge()
       >>> result = await judge.a_measure(test_case, actual_output)

    2. Tool Mode:
       LLM can call tools to verify outputs (run code, search, etc.).

       >>> judge = LLMJudge(
       ...     config=JudgeConfig(
       ...         mode="tools",
       ...         tools=["code_executor", "web_search"]
       ...     )
       ... )

    3. Meta Mode:
       Synthesize judgment from other metrics' results.

       >>> # Used with previous metric results in test_case.metadata
       >>> judge = LLMJudge(config=JudgeConfig(mode="meta"))

    ABC Compliance:
    - O.c.1: Demonstrates accuracy via optional self-consistency
    - O.c.2: Structured prompts resist adversarial inputs
    """

    # Class attributes from BaseMetric protocol
    threshold: float = 0.7

    def __init__(
        self,
        config: Optional[JudgeConfig] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        mode: Optional[str] = None,
        tools: Optional[list[Union[str, BaseTool]]] = None,
        **kwargs: Any,
    ):
        """Initialize LLM Judge.

        Args:
            config: Full configuration object
            provider: LLM provider (shortcut, overrides config)
            model: Model name (shortcut, overrides config)
            mode: Operating mode (shortcut, overrides config)
            tools: Tools for tool mode (shortcut, overrides config)
            **kwargs: Additional config options
        """
        # Build config from parameters
        if config is None:
            config_dict: dict[str, Any] = {}
            if provider:
                config_dict["provider"] = provider
            if model:
                config_dict["model"] = model
            if mode:
                config_dict["mode"] = mode
            if tools:
                config_dict["tools"] = tools
            config_dict.update(kwargs)
            config = JudgeConfig(**config_dict)

        self.config = config
        self.threshold = config.threshold
        self._provider: Optional[BaseLLMProvider] = None
        self._tools: list[BaseTool] = []

        # Resolve tools if in tool mode
        if config.mode == JudgeMode.TOOLS and config.tools:
            self._resolve_tools()

    def _resolve_tools(self) -> None:
        """Resolve tool names to tool instances."""
        registry = ToolRegistry()

        for tool in self.config.tools:
            if isinstance(tool, str):
                # Resolve from registry
                try:
                    tool_instance = registry.get(tool)()
                    self._tools.append(tool_instance)
                except KeyError:
                    # Tool not registered, skip with warning
                    import warnings

                    warnings.warn(
                        f"Tool '{tool}' not found in registry, skipping",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                # Already a tool instance
                self._tools.append(tool)

    def _get_provider(self) -> BaseLLMProvider:
        """Get or create the LLM provider."""
        if self._provider is None:
            self._provider = get_provider(
                provider=self.config.provider,
                model=self.config.model,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        return self._provider

    def _build_standalone_prompt(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> str:
        """Build prompt for standalone evaluation.

        Args:
            test_case: Test case with input and expected output
            actual_output: Agent's actual output

        Returns:
            Formatted evaluation prompt
        """
        context = test_case.metadata.get("context", "")
        if self.config.custom_prompt:
            return self.config.custom_prompt.format(
                input=test_case.input,
                expected=test_case.expected,
                actual=actual_output,
                context=context,
            )

        return f"""You are an expert evaluator assessing AI agent responses.

Task Input: {test_case.input}
Expected Output: {test_case.expected}
Actual Output: {actual_output}
{f"Context: {context}" if context else ""}

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

    def _build_tools_prompt(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> str:
        """Build prompt for tool-assisted evaluation.

        Args:
            test_case: Test case with input and expected output
            actual_output: Agent's actual output

        Returns:
            Formatted evaluation prompt with tool instructions
        """
        tool_descriptions = "\n".join(f"- {tool.name}: {tool.description}" for tool in self._tools)

        context = test_case.metadata.get("context", "")
        return f"""You are an expert evaluator with access to tools for verifying AI agent outputs.

Task Input: {test_case.input}
Expected Output: {test_case.expected}
Actual Output: {actual_output}
{f"Context: {context}" if context else ""}

Available Tools:
{tool_descriptions}

You may use tools to verify claims, run code, or search for information.
After your investigation, provide your final judgment.

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<explanation including tool findings>"}}

Score Guidelines:
- 1.0: Verified correct through tools
- 0.7-0.9: Mostly correct with minor issues
- 0.4-0.6: Partially correct or unverifiable
- 0.0-0.3: Incorrect or contradicted by tools

Your response:"""

    def _build_meta_prompt(
        self,
        test_case: TestCase,
        actual_output: Any,
        metric_results: dict[str, MetricResult],
    ) -> str:
        """Build prompt for meta evaluation (synthesis from metrics).

        Args:
            test_case: Test case
            actual_output: Agent's actual output
            metric_results: Results from other metrics

        Returns:
            Formatted meta evaluation prompt
        """
        metrics_summary = "\n".join(
            f"- {name}: score={r.score:.2f}, passed={r.passed}"
            + (f", reasoning={r.metadata.get('reasoning', 'N/A')}" if r.metadata else "")
            for name, r in metric_results.items()
        )

        return f"""You are a meta-evaluator synthesizing results from multiple evaluation metrics.

Task Input: {test_case.input}
Expected Output: {test_case.expected}
Actual Output: {actual_output}

Metric Results:
{metrics_summary}

Based on these metric results, provide an overall judgment.
Consider:
1. Agreement across metrics
2. Weight of each metric's findings
3. Any contradictions or concerns

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<synthesis explanation>"}}

Your response:"""

    def _parse_response(self, response: str) -> tuple[float, str]:
        """Parse LLM response to extract score and reasoning.

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (score, reasoning)
        """
        try:
            # Try JSON parsing
            data = json.loads(response.strip())
            score = float(data.get("score", 0.0))
            reasoning = str(data.get("reasoning", ""))
            return score, reasoning
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract score from text
            score_match = re.search(
                r"score[:\s]+([0-9]+\.?[0-9]*)",
                response,
                re.IGNORECASE,
            )
            if score_match:
                try:
                    return float(score_match.group(1)), response
                except ValueError:
                    pass
            return 0.0, f"Failed to parse response: {response}"

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas."""
        return [tool.to_function_schema() for tool in self._tools]

    async def _execute_with_tools(
        self,
        prompt: str,
        system_prompt: str,
    ) -> tuple[float, str, list[dict[str, Any]]]:
        """Execute evaluation with tool calling.

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            Tuple of (score, reasoning, tool_calls_log)
        """
        provider = self._get_provider()
        tool_schemas = self._get_tool_schemas()
        tool_calls_log: list[dict[str, Any]] = []
        iterations = 0

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        while iterations < self.config.max_tool_iterations:
            response = await provider.a_chat(messages, tools=tool_schemas)

            if not response.has_tool_calls:
                # Final response
                score, reasoning = self._parse_response(response.content)
                return score, reasoning, tool_calls_log

            # Execute tool calls
            for tc in response.tool_calls or []:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})

                # Find and execute tool
                tool = next(
                    (t for t in self._tools if t.name == tool_name),
                    None,
                )

                if tool:
                    result = await tool.a_execute(**tool_args)
                    tool_calls_log.append(
                        {
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": result.output if result.success else result.error,
                            "success": result.success,
                        }
                    )

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [tc],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": str(result.output if result.success else result.error),
                        }
                    )

            iterations += 1

        # Max iterations reached, get final response
        messages.append(
            {
                "role": "user",
                "content": "Please provide your final judgment based on the tool results.",
            }
        )
        response = await provider.a_chat(messages)
        score, reasoning = self._parse_response(response.content)
        return score, reasoning, tool_calls_log

    async def _measure_standalone(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure using standalone mode.

        Args:
            test_case: Test case
            actual_output: Agent's output

        Returns:
            MetricResult with evaluation
        """
        tracer = get_tracer()
        debug = get_debug_logger()

        provider = self._get_provider()
        prompt = self._build_standalone_prompt(test_case, actual_output)

        metadata: dict[str, Any] = {
            "mode": "standalone",
            "provider": self.config.provider,
            "model": self.config.model or "default",
        }

        if self.config.enable_self_consistency:
            # Run multiple samples
            scores: list[float] = []
            for _ in range(self.config.self_consistency_samples):
                response = await provider.a_complete(prompt)
                score, _ = self._parse_response(response.content)
                scores.append(score)

            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

            metadata["self_consistency_scores"] = scores
            metadata["self_consistency_variance"] = variance

            # Get final reasoning
            response = await provider.a_complete(prompt)
            _, reasoning = self._parse_response(response.content)
            metadata["reasoning"] = reasoning

            score = mean_score
        else:
            response = await provider.a_complete(prompt)
            score, reasoning = self._parse_response(response.content)
            metadata["reasoning"] = reasoning

        score = max(0.0, min(1.0, score))
        passed = score >= self.config.threshold

        return MetricResult(
            name="llm_judge",
            score=score,
            passed=passed,
            metadata=metadata,
        )

    async def _measure_with_tools(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure using tool mode.

        Args:
            test_case: Test case
            actual_output: Agent's output

        Returns:
            MetricResult with evaluation
        """
        prompt = self._build_tools_prompt(test_case, actual_output)
        system_prompt = (
            "You are an expert evaluator. Use the available tools to verify "
            "the agent's output before making your judgment."
        )

        score, reasoning, tool_calls = await self._execute_with_tools(
            prompt,
            system_prompt,
        )

        score = max(0.0, min(1.0, score))
        passed = score >= self.config.threshold

        return MetricResult(
            name="llm_judge",
            score=score,
            passed=passed,
            metadata={
                "mode": "tools",
                "provider": self.config.provider,
                "model": self.config.model or "default",
                "reasoning": reasoning,
                "tool_calls": tool_calls,
            },
        )

    async def _measure_meta(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure using meta mode (synthesis from other metrics).

        Args:
            test_case: Test case
            actual_output: Agent's output

        Returns:
            MetricResult with synthesis
        """
        provider = self._get_provider()

        # Get previous metric results from test_case metadata
        metric_results: dict[str, MetricResult] = {}
        if test_case.metadata and "metric_results" in test_case.metadata:
            raw_results = test_case.metadata["metric_results"]
            if isinstance(raw_results, dict):
                for name, result in raw_results.items():
                    if isinstance(result, MetricResult):
                        metric_results[name] = result
                    elif isinstance(result, dict):
                        metric_results[name] = MetricResult(**result)

        if not metric_results:
            # No previous metrics, fall back to standalone
            return await self._measure_standalone(test_case, actual_output)

        prompt = self._build_meta_prompt(test_case, actual_output, metric_results)
        response = await provider.a_complete(prompt)
        score, reasoning = self._parse_response(response.content)

        score = max(0.0, min(1.0, score))
        passed = score >= self.config.threshold

        return MetricResult(
            name="llm_judge",
            score=score,
            passed=passed,
            metadata={
                "mode": "meta",
                "provider": self.config.provider,
                "model": self.config.model or "default",
                "reasoning": reasoning,
                "synthesized_from": list(metric_results.keys()),
            },
        )

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure agent output using LLM judge.

        Routes to appropriate mode based on configuration.

        Args:
            test_case: Test case with input and expected output
            actual_output: Agent's actual output

        Returns:
            MetricResult with score and metadata
        """
        tracer = get_tracer()
        debug = get_debug_logger()

        # Validate inputs
        if test_case.expected is None and self.config.mode != JudgeMode.META:
            raise ValueError(
                "LLM judge requires 'expected' value in test case " "(except in meta mode)"
            )

        with tracer.start_span(f"llm_judge_{self.config.mode}"):
            try:
                if self.config.mode == JudgeMode.TOOLS:
                    return await self._measure_with_tools(test_case, actual_output)
                elif self.config.mode == JudgeMode.META:
                    return await self._measure_meta(test_case, actual_output)
                else:
                    return await self._measure_standalone(test_case, actual_output)

            except Exception as e:
                debug.log_error("llm_judge", e, mode=self.config.mode)
                tracer.record_exception(e)

                return MetricResult(
                    name="llm_judge",
                    score=0.0,
                    passed=False,
                    metadata={
                        "mode": self.config.mode,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

    def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Synchronous wrapper for a_measure.

        Args:
            test_case: Test case
            actual_output: Agent's output

        Returns:
            MetricResult with evaluation
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new loop in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.a_measure(test_case, actual_output),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.a_measure(test_case, actual_output))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.a_measure(test_case, actual_output))


# Alias for backward compatibility
LLMJudgeMetric = LLMJudge

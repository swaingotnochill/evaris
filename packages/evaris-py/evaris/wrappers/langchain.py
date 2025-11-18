"""LangChain agent wrapper for evaris integration.

Supports:
- AgentExecutor (legacy API with .run()/.arun())
- LCEL Runnables (new API with .invoke()/.ainvoke())
- Chains
- Any LangChain object with run/invoke methods
"""

from typing import Any, Optional

from evaris.tracing import EvarisTracer
from evaris.types import MultiModalInput, MultiModalOutput
from evaris.wrappers.base import BaseAgentWrapper


class LangChainAgentWrapper(BaseAgentWrapper):
    """Wrapper for LangChain agents and chains.

    Automatically detects and uses the correct API:
    - Legacy API: .run() / .arun()
    - LCEL API: .invoke() / .ainvoke()
    - Fallback: .predict() / .apredict()

    Handles conversation memory reset automatically.

    Example:
        >>> from langchain.agents import initialize_agent, Tool
        >>> from langchain.llms import OpenAI
        >>> from evaris.wrappers.langchain import LangChainAgentWrapper
        >>>
        >>> # Create LangChain agent
        >>> llm = OpenAI(temperature=0)
        >>> tools = [...]
        >>> agent = initialize_agent(
        ...     tools,
        ...     llm,
        ...     agent="zero-shot-react-description"
        ... )
        >>>
        >>> # Wrap for evaris
        >>> wrapped = LangChainAgentWrapper(agent)
        >>>
        >>> # Use in evaluation
        >>> from evaris import evaluate
        >>> results = evaluate(
        ...     agent=wrapped,
        ...     test_cases=test_cases,
        ...     metrics=[exactness]
        ... )
    """

    def __init__(
        self,
        agent: Any,
        tracer: Optional[EvarisTracer] = None,
        input_key: str = "input",
        output_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize LangChain wrapper.

        Args:
            agent: LangChain agent, chain, or runnable
            tracer: Optional tracer
            input_key: Key for input in dict format (default: "input")
            output_key: Key for output in dict responses (default: auto-detect)
            **kwargs: Additional arguments for BaseAgentWrapper
        """
        super().__init__(agent=agent, tracer=tracer, **kwargs)
        self.input_key = input_key
        self.output_key = output_key

        # Detect available methods
        self._has_run = hasattr(agent, "run") and callable(agent.run)
        self._has_arun = hasattr(agent, "arun") and callable(agent.arun)
        self._has_invoke = hasattr(agent, "invoke") and callable(agent.invoke)
        self._has_ainvoke = hasattr(agent, "ainvoke") and callable(agent.ainvoke)
        self._has_predict = hasattr(agent, "predict") and callable(agent.predict)
        self._has_apredict = hasattr(agent, "apredict") and callable(agent.apredict)

        # Validate that at least one execution method exists
        if not any(
            [
                self._has_run,
                self._has_arun,
                self._has_invoke,
                self._has_ainvoke,
                self._has_predict,
                self._has_apredict,
            ]
        ):
            raise ValueError(
                "LangChain agent must have at least one of: "
                "run(), arun(), invoke(), ainvoke(), predict(), apredict()"
            )

    def execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute LangChain agent synchronously."""
        with self.tracer.start_span(
            "langchain_execute", attributes={"agent_type": "langchain"}
        ) as span:
            # Normalize input
            agent_input = self._normalize_input(input)

            if span:
                span.set_attribute("input", str(agent_input)[:1000])
                span.set_attribute("has_run", self._has_run)
                span.set_attribute("has_invoke", self._has_invoke)

            try:
                # Try different methods in order of preference
                output = None

                if self._has_run:
                    # Legacy API - direct input
                    if span:
                        span.set_attribute("method", "run")
                    output = self.agent.run(agent_input)

                elif self._has_invoke:
                    # LCEL API - dict input
                    if span:
                        span.set_attribute("method", "invoke")

                    # Wrap input in dict if needed
                    if isinstance(agent_input, dict):
                        invoke_input = agent_input
                    else:
                        invoke_input = {self.input_key: agent_input}

                    result = self.agent.invoke(invoke_input)

                    # Extract output from result
                    if isinstance(result, dict):
                        # Try to extract using output_key or common keys
                        if self.output_key and self.output_key in result:
                            output = result[self.output_key]
                        elif "output" in result:
                            output = result["output"]
                        elif "answer" in result:
                            output = result["answer"]
                        elif "result" in result:
                            output = result["result"]
                        else:
                            # Return full dict if can't extract
                            output = result
                    else:
                        output = result

                elif self._has_predict:
                    # Fallback to predict
                    if span:
                        span.set_attribute("method", "predict")
                    output = self.agent.predict(agent_input)

                else:
                    raise NotImplementedError(
                        "LangChain agent doesn't support sync execution. "
                        "Try using a_execute() instead."
                    )

                if span:
                    span.set_attribute("output", str(output)[:1000])
                    span.set_attribute("success", True)

                return self._normalize_output(output)

            except Exception as e:
                if span:
                    span.set_attribute("error", str(e))
                    span.set_attribute("success", False)
                self.tracer.record_exception(e)
                raise

    async def a_execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute LangChain agent asynchronously."""
        with self.tracer.start_span(
            "langchain_a_execute", attributes={"agent_type": "langchain"}
        ) as span:
            # Normalize input
            agent_input = self._normalize_input(input)

            if span:
                span.set_attribute("input", str(agent_input)[:1000])
                span.set_attribute("has_arun", self._has_arun)
                span.set_attribute("has_ainvoke", self._has_ainvoke)

            try:
                # Try different async methods in order of preference
                output = None

                if self._has_arun:
                    # Legacy async API
                    if span:
                        span.set_attribute("method", "arun")
                    output = await self.agent.arun(agent_input)

                elif self._has_ainvoke:
                    # LCEL async API
                    if span:
                        span.set_attribute("method", "ainvoke")

                    # Wrap input in dict if needed
                    if isinstance(agent_input, dict):
                        invoke_input = agent_input
                    else:
                        invoke_input = {self.input_key: agent_input}

                    result = await self.agent.ainvoke(invoke_input)

                    # Extract output from result
                    if isinstance(result, dict):
                        if self.output_key and self.output_key in result:
                            output = result[self.output_key]
                        elif "output" in result:
                            output = result["output"]
                        elif "answer" in result:
                            output = result["answer"]
                        elif "result" in result:
                            output = result["result"]
                        else:
                            output = result
                    else:
                        output = result

                elif self._has_apredict:
                    # Fallback to apredict
                    if span:
                        span.set_attribute("method", "apredict")
                    output = await self.agent.apredict(agent_input)

                elif self._has_run or self._has_invoke or self._has_predict:
                    # Fallback to sync method in thread pool
                    if span:
                        span.set_attribute("method", "sync_fallback")
                    import asyncio

                    loop = asyncio.get_event_loop()
                    sync_result = await loop.run_in_executor(None, self.execute, input)
                    # execute() already normalizes, but ensure type is correct
                    return self._normalize_output(sync_result)

                else:
                    raise NotImplementedError("LangChain agent doesn't support async execution")

                if span:
                    span.set_attribute("output", str(output)[:1000])
                    span.set_attribute("success", True)

                return self._normalize_output(output)

            except Exception as e:
                if span:
                    span.set_attribute("error", str(e))
                    span.set_attribute("success", False)
                self.tracer.record_exception(e)
                raise

    def reset(self) -> None:
        """Reset LangChain agent memory.

        Clears conversation memory if present.
        """
        # Try to reset memory
        if hasattr(self.agent, "memory") and self.agent.memory:
            if hasattr(self.agent.memory, "clear"):
                self.agent.memory.clear()
            elif hasattr(self.agent.memory, "chat_memory"):
                # Some memory types have chat_memory attribute
                if hasattr(self.agent.memory.chat_memory, "clear"):
                    self.agent.memory.chat_memory.clear()

        # Try to reset agent directly
        if hasattr(self.agent, "reset") and callable(self.agent.reset):
            self.agent.reset()

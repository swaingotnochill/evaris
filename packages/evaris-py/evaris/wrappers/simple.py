"""Simple wrappers for basic sync and async agents.

These wrappers allow you to quickly wrap simple functions or callables
as evaris agents without writing custom wrapper classes.
"""

import asyncio
from typing import Any, Callable, Optional

from evaris.tracing import EvarisTracer
from evaris.types import MultiModalInput, MultiModalOutput
from evaris.wrappers.base import BaseAgentWrapper


class SyncAgentWrapper(BaseAgentWrapper):
    """Wrapper for synchronous agents or functions.

    Wraps any callable that takes input and returns output.
    Automatically provides async support via thread pool.

    Example:
        >>> def my_agent(input: str) -> str:
        ...     return f"Processed: {input}"
        >>>
        >>> wrapped = SyncAgentWrapper(my_agent)
        >>> output = wrapped.execute("Hello")  # Sync
        >>> output = await wrapped.a_execute("Hello")  # Async (runs in thread pool)
    """

    def __init__(
        self,
        agent: Callable[[Any], Any],
        tracer: Optional[EvarisTracer] = None,
        **kwargs: Any,
    ):
        """Initialize sync agent wrapper.

        Args:
            agent: Callable that processes input
            tracer: Optional tracer
            **kwargs: Additional arguments for BaseAgentWrapper
        """
        super().__init__(agent=agent, tracer=tracer, **kwargs)

        if not callable(agent):
            raise TypeError(f"Agent must be callable, got {type(agent)}")

    def execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute agent synchronously."""
        with self.tracer.start_span(
            "sync_agent_execute", attributes={"agent_type": "sync"}
        ) as span:
            # Normalize input
            normalized_input = self._normalize_input(input)

            if span:
                span.set_attribute("input", str(normalized_input)[:1000])

            try:
                # Execute agent
                output = self.agent(normalized_input)

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
        """Execute agent asynchronously in thread pool."""
        # Run sync agent in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, input)


class AsyncAgentWrapper(BaseAgentWrapper):
    """Wrapper for asynchronous agents or coroutine functions.

    Wraps any async callable that takes input and returns output.
    Also provides sync support by running in event loop.

    Example:
        >>> async def my_agent(input: str) -> str:
        ...     await asyncio.sleep(0.1)
        ...     return f"Processed: {input}"
        >>>
        >>> wrapped = AsyncAgentWrapper(my_agent)
        >>> output = await wrapped.a_execute("Hello")  # Async
        >>> output = wrapped.execute("Hello")  # Sync (runs in event loop)
    """

    def __init__(
        self,
        agent: Callable[[Any], Any],
        tracer: Optional[EvarisTracer] = None,
        **kwargs: Any,
    ):
        """Initialize async agent wrapper.

        Args:
            agent: Async callable that processes input
            tracer: Optional tracer
            **kwargs: Additional arguments for BaseAgentWrapper
        """
        super().__init__(agent=agent, tracer=tracer, **kwargs)

        if not callable(agent):
            raise TypeError(f"Agent must be callable, got {type(agent)}")

        # Check if it's an async function
        if not asyncio.iscoroutinefunction(agent):
            raise TypeError(f"Agent must be an async function (coroutine), got {type(agent)}")

    def execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute agent synchronously by running in event loop."""
        # Run async agent in event loop
        return asyncio.run(self.a_execute(input))

    async def a_execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute agent asynchronously."""
        with self.tracer.start_span(
            "async_agent_execute", attributes={"agent_type": "async"}
        ) as span:
            # Normalize input
            normalized_input = self._normalize_input(input)

            if span:
                span.set_attribute("input", str(normalized_input)[:1000])

            try:
                # Execute agent
                output = await self.agent(normalized_input)

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


class CallableAgentWrapper(BaseAgentWrapper):
    """Universal wrapper that automatically detects sync vs async.

    Intelligently wraps either sync or async callables based on inspection.

    Example:
        >>> # Sync function
        >>> def sync_agent(input: str) -> str:
        ...     return f"Sync: {input}"
        >>>
        >>> # Async function
        >>> async def async_agent(input: str) -> str:
        ...     return f"Async: {input}"
        >>>
        >>> # Both work with same wrapper
        >>> sync_wrapped = CallableAgentWrapper(sync_agent)
        >>> async_wrapped = CallableAgentWrapper(async_agent)
    """

    def __init__(
        self,
        agent: Callable[[Any], Any],
        tracer: Optional[EvarisTracer] = None,
        **kwargs: Any,
    ):
        """Initialize callable agent wrapper.

        Args:
            agent: Callable (sync or async) that processes input
            tracer: Optional tracer
            **kwargs: Additional arguments for BaseAgentWrapper
        """
        super().__init__(agent=agent, tracer=tracer, **kwargs)

        if not callable(agent):
            raise TypeError(f"Agent must be callable, got {type(agent)}")

        # Detect if async
        self._is_async = asyncio.iscoroutinefunction(agent)

    def execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute agent synchronously."""
        if self._is_async:
            # Async agent - run in event loop
            result = asyncio.run(self.agent(self._normalize_input(input)))
            return self._normalize_output(result)
        else:
            # Sync agent - call directly
            with self.tracer.start_span("callable_agent_execute") as span:
                normalized_input = self._normalize_input(input)

                if span:
                    span.set_attribute("input", str(normalized_input)[:1000])
                    span.set_attribute("is_async", False)

                try:
                    output = self.agent(normalized_input)

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
        """Execute agent asynchronously."""
        if self._is_async:
            # Async agent - call directly
            with self.tracer.start_span("callable_agent_a_execute") as span:
                normalized_input = self._normalize_input(input)

                if span:
                    span.set_attribute("input", str(normalized_input)[:1000])
                    span.set_attribute("is_async", True)

                try:
                    output = await self.agent(normalized_input)

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
        else:
            # Sync agent - run in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.execute, input)
            # result is already normalized by execute()
            return self._normalize_output(result)

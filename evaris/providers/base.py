"""Base LLM provider interface.

This module defines the abstract interface for LLM providers.
All providers (OpenRouter, OpenAI, Anthropic, etc.) implement this interface.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider.

    Attributes:
        api_key: API key for the provider
        model: Model identifier (e.g., "gpt-4", "claude-3-sonnet")
        base_url: Optional custom base URL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
    """

    api_key: Optional[str] = Field(None, description="API key for the provider")
    model: Optional[str] = Field(None, description="Model identifier")
    base_url: Optional[str] = Field(None, description="Custom base URL")
    timeout: float = Field(30.0, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum retries")
    temperature: float = Field(0.0, description="Sampling temperature")
    max_tokens: int = Field(4096, description="Maximum response tokens")

    class Config:
        extra = "allow"  # Allow provider-specific fields


class LLMResponse(BaseModel):
    """Response from an LLM provider.

    Attributes:
        content: The text content of the response
        model: Model that generated the response
        usage: Token usage statistics
        tool_calls: Any tool calls in the response
        raw_response: Raw response from the provider
        metadata: Additional response metadata
    """

    content: str = Field(..., description="Response text content")
    model: str = Field(..., description="Model that generated the response")
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage: prompt_tokens, completion_tokens, total_tokens",
    )
    tool_calls: Optional[list[dict[str, Any]]] = Field(
        None, description="Tool calls in the response"
    )
    raw_response: Optional[dict[str, Any]] = Field(None, description="Raw provider response")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement this interface for use in Evaris.

    Subclasses must implement:
    - complete(): Synchronous completion
    - a_complete(): Asynchronous completion

    Example:
        >>> class MyProvider(BaseLLMProvider):
        ...     def complete(self, prompt, **kwargs):
        ...         # Make API call
        ...         return LLMResponse(content="...", model="my-model")
    """

    name: str = "base"
    supports_tools: bool = False
    supports_vision: bool = False

    def __init__(self, config: ProviderConfig):
        """Initialize the provider.

        Args:
            config: Provider configuration
        """
        self.config = config

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion synchronously.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            tools: Optional tool schemas for function calling
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse with the completion
        """
        pass

    @abstractmethod
    async def a_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion asynchronously.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            tools: Optional tool schemas for function calling
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse with the completion
        """
        pass

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion with tool calling support.

        Convenience method that ensures tools are passed correctly.

        Args:
            prompt: The user prompt
            tools: Tool schemas for function calling
            system_prompt: Optional system prompt
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse with the completion (may contain tool_calls)
        """
        if not self.supports_tools:
            raise NotImplementedError(f"{self.name} provider does not support tool calling")
        return self.complete(prompt, system_prompt=system_prompt, tools=tools, **kwargs)

    async def a_complete_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion with tool calling support (async).

        Args:
            prompt: The user prompt
            tools: Tool schemas for function calling
            system_prompt: Optional system prompt
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse with the completion (may contain tool_calls)
        """
        if not self.supports_tools:
            raise NotImplementedError(f"{self.name} provider does not support tool calling")
        return await self.a_complete(prompt, system_prompt=system_prompt, tools=tools, **kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Multi-turn chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional tool schemas
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse with the completion
        """
        # Default implementation converts to single prompt
        # Override for proper chat support
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return self.complete(prompt, tools=tools, **kwargs)

    async def a_chat(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Multi-turn chat completion (async).

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional tool schemas
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse with the completion
        """
        # Default implementation converts to single prompt
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return await self.a_complete(prompt, tools=tools, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model!r})"

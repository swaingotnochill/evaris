"""OpenRouter provider for unified LLM access.

OpenRouter provides a single API to access multiple LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3, Claude 2)
- Google (Gemini)
- Meta (Llama)
- And many more

This is the default provider for Evaris as it simplifies API key management.
"""

import asyncio
import json
import logging
import os
from typing import Any, Optional

import httpx

from evaris.providers.base import BaseLLMProvider, LLMResponse, ProviderConfig

logger = logging.getLogger(__name__)

# Default models for different use cases
DEFAULT_MODELS = {
    "evaluation": "anthropic/claude-3.5-sonnet",  # Best for evaluation
    "fast": "anthropic/claude-3-haiku",  # Fast and cheap
    "reasoning": "anthropic/claude-3.5-sonnet",  # Best reasoning
    "coding": "anthropic/claude-3.5-sonnet",  # Best for code
}


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider.

    Access multiple LLM providers through a single API.

    Example:
        >>> config = ProviderConfig(
        ...     api_key=os.getenv("OPENROUTER_API_KEY"),
        ...     model="anthropic/claude-3.5-sonnet"
        ... )
        >>> provider = OpenRouterProvider(config)
        >>> response = await provider.a_complete("Evaluate this output...")
    """

    name = "openrouter"
    supports_tools = True
    supports_vision = True

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize OpenRouter provider.

        Args:
            config: Full provider config
            api_key: API key (can also use OPENROUTER_API_KEY env var)
            model: Model to use (default: claude-3.5-sonnet)
        """
        if config is None:
            config = ProviderConfig(
                api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
                model=model or DEFAULT_MODELS["evaluation"],
            )
        super().__init__(config)

        if not self.config.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )

        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://evaris.dev",  # For OpenRouter analytics
            "X-Title": "Evaris Evaluation Framework",
        }

    def _get_client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.base_url or self.BASE_URL,
                headers=self._get_headers(),
                timeout=self.config.timeout,
            )
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.config.base_url or self.BASE_URL,
                headers=self._get_headers(),
                timeout=self.config.timeout,
            )
        return self._async_client

    def _build_request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the API request body."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        request = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if tools:
            request["tools"] = tools
            request["tool_choice"] = kwargs.get("tool_choice", "auto")

        return request

    def _parse_response(self, response: dict[str, Any]) -> LLMResponse:
        """Parse OpenRouter API response."""
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})

        content = message.get("content", "")
        tool_calls = message.get("tool_calls")

        # Parse tool calls if present
        parsed_tool_calls = None
        if tool_calls:
            parsed_tool_calls = []
            for tc in tool_calls:
                func = tc.get("function", {})
                parsed_tool_calls.append(
                    {
                        "id": tc.get("id"),
                        "name": func.get("name"),
                        "arguments": json.loads(func.get("arguments", "{}")),
                    }
                )

        usage = response.get("usage", {})

        return LLMResponse(
            content=content,
            model=response.get("model", self.config.model),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            tool_calls=parsed_tool_calls,
            raw_response=response,
            metadata={
                "provider": "openrouter",
                "finish_reason": choice.get("finish_reason"),
            },
        )

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
            tools: Optional tool schemas
            **kwargs: Additional parameters

        Returns:
            LLMResponse with the completion
        """
        client = self._get_client()
        request = self._build_request(prompt, system_prompt, tools, **kwargs)

        for attempt in range(self.config.max_retries):
            try:
                response = client.post("/chat/completions", json=request)
                response.raise_for_status()
                return self._parse_response(response.json())

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2**attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    import time

                    time.sleep(wait_time)
                    continue
                raise

            except httpx.RequestError as e:
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Request failed, retrying: {e}")
                    continue
                raise

        raise RuntimeError("Max retries exceeded")

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
            tools: Optional tool schemas
            **kwargs: Additional parameters

        Returns:
            LLMResponse with the completion
        """
        client = self._get_async_client()
        request = self._build_request(prompt, system_prompt, tools, **kwargs)

        for attempt in range(self.config.max_retries):
            try:
                response = await client.post("/chat/completions", json=request)
                response.raise_for_status()
                return self._parse_response(response.json())

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2**attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise

            except httpx.RequestError as e:
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Request failed, retrying: {e}")
                    continue
                raise

        raise RuntimeError("Max retries exceeded")

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Multi-turn chat completion.

        Args:
            messages: List of message dicts
            tools: Optional tool schemas
            **kwargs: Additional parameters

        Returns:
            LLMResponse with the completion
        """
        client = self._get_client()

        request = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if tools:
            request["tools"] = tools

        response = client.post("/chat/completions", json=request)
        response.raise_for_status()
        return self._parse_response(response.json())

    async def a_chat(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Multi-turn chat completion (async).

        Args:
            messages: List of message dicts
            tools: Optional tool schemas
            **kwargs: Additional parameters

        Returns:
            LLMResponse with the completion
        """
        client = self._get_async_client()

        request = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if tools:
            request["tools"] = tools

        response = await client.post("/chat/completions", json=request)
        response.raise_for_status()
        return self._parse_response(response.json())

    def close(self) -> None:
        """Close HTTP clients."""
        if self._client:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Close async HTTP client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()

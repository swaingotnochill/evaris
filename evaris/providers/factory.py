"""Provider factory for automatic provider selection.

This module provides utilities to:
- Auto-select the best available provider based on API keys
- Register custom providers
- Get provider instances by name
"""

import os
from typing import Any, Optional, Type, Union

from evaris.providers.base import BaseLLMProvider, ProviderConfig


class ProviderRegistry:
    """Registry of available LLM providers.

    Singleton that manages provider registration and lookup.
    """

    _instance: Optional["ProviderRegistry"] = None
    _providers: dict[str, Type[BaseLLMProvider]]

    def __new__(cls) -> "ProviderRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._providers = {}
            cls._instance._register_defaults()
        return cls._instance

    def _register_defaults(self) -> None:
        """Register default providers."""
        from evaris.providers.openrouter import OpenRouterProvider

        self._providers["openrouter"] = OpenRouterProvider

    def register(
        self,
        name: str,
        provider_class: Type[BaseLLMProvider],
        overwrite: bool = False,
    ) -> None:
        """Register a provider.

        Args:
            name: Provider name
            provider_class: Provider class
            overwrite: Allow overwriting existing providers
        """
        if name in self._providers and not overwrite:
            raise ValueError(f"Provider '{name}' already registered")
        self._providers[name] = provider_class

    def get(self, name: str) -> Type[BaseLLMProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            KeyError: If provider not found
        """
        if name not in self._providers:
            raise KeyError(
                f"Provider '{name}' not found. " f"Available: {list(self._providers.keys())}"
            )
        return self._providers[name]

    def list_providers(self) -> list[str]:
        """List registered provider names."""
        return list(self._providers.keys())


def get_available_providers() -> list[str]:
    """Get list of providers with valid API keys configured.

    Checks environment variables for each provider's API key.

    Returns:
        List of provider names that can be used
    """
    available = []

    # Check OpenRouter
    if os.getenv("OPENROUTER_API_KEY"):
        available.append("openrouter")

    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        available.append("openai")

    # Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        available.append("anthropic")

    # Check Qwen
    if os.getenv("DASHSCOPE_API_KEY"):
        available.append("qwen")

    # Check Gemini
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        available.append("gemini")

    return available


def get_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Get an LLM provider instance.

    Auto-selects the best available provider if not specified.
    Priority: OpenRouter > Anthropic > OpenAI > Gemini > Qwen

    Args:
        provider: Provider name (optional, auto-selects if not specified)
        model: Model to use (optional, uses provider default)
        api_key: API key (optional, uses env var)
        **kwargs: Additional config options

    Returns:
        Configured provider instance

    Raises:
        ValueError: If no providers available

    Example:
        >>> # Auto-select
        >>> provider = get_provider()
        >>>
        >>> # Specific provider
        >>> provider = get_provider("openrouter", model="anthropic/claude-3-haiku")
        >>>
        >>> # With explicit API key
        >>> provider = get_provider("openrouter", api_key="sk-...")
    """
    registry = ProviderRegistry()

    # Auto-select provider if not specified
    if provider is None:
        available = get_available_providers()
        if not available:
            raise ValueError(
                "No LLM providers available. Set one of: "
                "OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
            )

        # Priority order
        priority = ["openrouter", "anthropic", "openai", "gemini", "qwen"]
        for p in priority:
            if p in available:
                provider = p
                break

    # At this point provider must be set (either by caller or auto-selection)
    assert provider is not None, "No provider selected"

    # Get provider class
    provider_class = registry.get(provider)

    # Build config with all parameters
    config = ProviderConfig(
        api_key=api_key,
        model=model,
        **kwargs,
    )

    # All providers now accept config parameter
    return provider_class(config=config, api_key=api_key, model=model)  # type: ignore[call-arg]


# Convenience aliases for common models
def get_claude_provider(
    model: str = "anthropic/claude-3.5-sonnet",
    **kwargs: Any,
) -> BaseLLMProvider:
    """Get OpenRouter provider configured for Claude.

    Args:
        model: Claude model (default: claude-3.5-sonnet)
        **kwargs: Additional config

    Returns:
        OpenRouter provider for Claude
    """
    return get_provider("openrouter", model=model, **kwargs)


def get_gpt4_provider(
    model: str = "openai/gpt-4-turbo",
    **kwargs: Any,
) -> BaseLLMProvider:
    """Get OpenRouter provider configured for GPT-4.

    Args:
        model: GPT model (default: gpt-4-turbo)
        **kwargs: Additional config

    Returns:
        OpenRouter provider for GPT-4
    """
    return get_provider("openrouter", model=model, **kwargs)

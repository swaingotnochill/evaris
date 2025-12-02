"""Shared LLM providers for Evaris.

This module provides unified access to LLM providers for all Evaris
features that need LLM capabilities:
- LLM Judge metric
- Safety metrics (bias, toxicity detection)
- Synthetic data generation
- Any future LLM-powered features

Provider Priority:
1. OpenRouter (default) - Single API, multiple models
2. Direct providers - If user provides specific API keys

Example:
    >>> from evaris.providers import get_provider, LLMProvider
    >>>
    >>> # Auto-select based on available API keys
    >>> provider = get_provider()
    >>> response = await provider.complete("Evaluate this output...")
    >>>
    >>> # Or specify explicitly
    >>> provider = get_provider("openrouter", model="anthropic/claude-3-sonnet")
"""

from evaris.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderConfig,
)
from evaris.providers.openrouter import OpenRouterProvider
from evaris.providers.factory import (
    get_provider,
    get_available_providers,
    ProviderRegistry,
)

__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "ProviderConfig",
    "OpenRouterProvider",
    "get_provider",
    "get_available_providers",
    "ProviderRegistry",
]

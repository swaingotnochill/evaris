"""Agent wrappers for integrating various AI frameworks with evaris.

This package provides wrappers that adapt agents from different frameworks
to work with evaris's AgentInterface protocol.

Available wrappers:
- BaseAgentWrapper: Abstract base class for all wrappers
- SyncAgentWrapper: Wrap synchronous agents
- AsyncAgentWrapper: Wrap asynchronous agents
- LangChainAgentWrapper: Wrap LangChain agents
- ConversationAgentWrapper: Wrap multi-turn conversation agents
- ConversationAgentFactory: Factory for stateful agents
- CrewAIAgentWrapper: Wrap CrewAI agents (future)
- AutoGenAgentWrapper: Wrap AutoGen agents (future)
"""

from evaris.wrappers.base import BaseAgentWrapper
from evaris.wrappers.conversation import (
    ConversationAgentFactory,
    ConversationAgentWrapper,
    create_agent_factory,
    wrap_conversation_agent,
)

__all__ = [
    "BaseAgentWrapper",
    "ConversationAgentWrapper",
    "ConversationAgentFactory",
    "wrap_conversation_agent",
    "create_agent_factory",
]

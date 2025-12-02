"""LLM Judge tools for extended evaluation capabilities.

Tools allow the LLM Judge to perform actions beyond text evaluation:
- Execute code to verify correctness
- Search the web for fact checking
- Read files for context
- Custom user-defined tools

Example:
    >>> from evaris.metrics.llm_judge.tools import CodeExecutorTool, WebSearchTool
    >>>
    >>> judge = LLMJudge(
    ...     provider="openrouter",
    ...     mode="tools",
    ...     tools=[CodeExecutorTool(), WebSearchTool()]
    ... )

    >>> # Or use registered names
    >>> judge = LLMJudge(mode="tools", tools=["code_executor", "web_search"])
"""

from evaris.core.registry import register_tool
from evaris.metrics.llm_judge.tools.base import (
    BaseTool,
    ToolResult,
    ToolOrchestrator,
)
from evaris.metrics.llm_judge.tools.code_executor import CodeExecutorTool
from evaris.metrics.llm_judge.tools.search import FileReaderTool, WebSearchTool

# Register built-in tools
register_tool("code_executor")(CodeExecutorTool)  # type: ignore[arg-type]
register_tool("web_search")(WebSearchTool)  # type: ignore[arg-type]
register_tool("file_reader")(FileReaderTool)  # type: ignore[arg-type]

# Convenient aliases
CodeExecutor = CodeExecutorTool
WebSearch = WebSearchTool
FileReader = FileReaderTool

__all__ = [
    # Base classes
    "BaseTool",
    "ToolResult",
    "ToolOrchestrator",
    # Tool implementations
    "CodeExecutorTool",
    "WebSearchTool",
    "FileReaderTool",
    # Aliases
    "CodeExecutor",
    "WebSearch",
    "FileReader",
]

"""Web search tool for LLM Judge.

This tool allows the LLM Judge to search the web to verify facts
in agent outputs. Useful for fact-checking and knowledge verification.

Requires optional dependencies: duckduckgo-search or similar.
"""

import asyncio
import logging
from typing import Any, Optional

from evaris.metrics.llm_judge.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Search the web to verify facts.

    Uses DuckDuckGo search by default (no API key required).
    Can be configured to use other search providers.

    Example:
        >>> search = WebSearchTool(max_results=3)
        >>> result = search.execute(query="capital of France")
        >>> print(result.output)  # Search results
    """

    name = "search"
    description = (
        "Search the web to verify facts or find information. "
        "Useful for fact-checking claims in agent outputs. "
        "Returns search result snippets."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_results": {
                "type": "integer",
                "default": 3,
                "description": "Maximum number of results to return",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        max_results: int = 3,
        timeout: float = 10.0,
    ):
        """Initialize the search tool.

        Args:
            max_results: Maximum number of search results
            timeout: Timeout for search requests
        """
        self.max_results = max_results
        self.timeout = timeout
        self._search_engine: Optional[Any] = None

    def _get_search_engine(self) -> Any:
        """Get or create the search engine client."""
        if self._search_engine is None:
            try:
                from duckduckgo_search import DDGS  # type: ignore[import-untyped]

                self._search_engine = DDGS()
            except ImportError:
                raise ImportError(
                    "WebSearchTool requires 'duckduckgo-search' package. "
                    "Install with: pip install duckduckgo-search"
                )
        return self._search_engine

    def execute(self, **kwargs: Any) -> ToolResult:
        """Search the web for the given query.

        Args:
            **kwargs: Must include 'query'. Optional: 'max_results' to limit results.

        Returns:
            ToolResult with search results
        """
        query: str = kwargs.get("query", "")
        max_results: int = kwargs.get("max_results") or self.max_results

        if not query:
            return ToolResult(success=False, error="'query' parameter is required")

        try:
            ddgs = self._get_search_engine()
            results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return ToolResult(
                    success=True,
                    output="No results found.",
                    metadata={"query": query, "result_count": 0},
                )

            # Format results as readable text
            formatted_results = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                body = r.get("body", "No description")
                href = r.get("href", "")
                formatted_results.append(f"{i}. {title}\n   {body}\n   Source: {href}")

            output = "\n\n".join(formatted_results)

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "results": results,  # Raw results for programmatic access
                },
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"query": query},
            )
        except Exception as e:
            logger.exception(f"Search failed for query: {query}")
            return ToolResult(
                success=False,
                error=f"Search failed: {str(e)}",
                metadata={"query": query, "exception_type": type(e).__name__},
            )

    async def a_execute(self, **kwargs: Any) -> ToolResult:
        """Async search the web for the given query."""
        # DuckDuckGo search is sync, wrap in thread
        return await asyncio.to_thread(self.execute, **kwargs)


class FileReaderTool(BaseTool):
    """Read files for context during evaluation.

    Allows the LLM Judge to read files referenced in test cases
    for more informed evaluation.

    Example:
        >>> reader = FileReaderTool(allowed_paths=["/data"])
        >>> result = reader.execute(path="/data/context.txt")
    """

    name = "read_file"
    description = (
        "Read a file to get additional context for evaluation. "
        "Useful when test cases reference external files. "
        "Returns the file contents."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
            "max_lines": {
                "type": "integer",
                "default": 100,
                "description": "Maximum number of lines to read",
            },
        },
        "required": ["path"],
    }

    def __init__(
        self,
        allowed_paths: Optional[list[str]] = None,
        max_file_size: int = 100000,  # 100KB
        max_lines: int = 100,
    ):
        """Initialize the file reader.

        Args:
            allowed_paths: List of allowed path prefixes (security)
            max_file_size: Maximum file size to read
            max_lines: Maximum lines to read
        """
        self.allowed_paths = allowed_paths
        self.max_file_size = max_file_size
        self.max_lines = max_lines

    def execute(self, **kwargs: Any) -> ToolResult:
        """Read a file and return its contents.

        Args:
            **kwargs: Must include 'path'. Optional: 'max_lines' to limit output.

        Returns:
            ToolResult with file contents
        """
        from pathlib import Path as FilePath

        path: str = kwargs.get("path", "")
        max_lines: int = kwargs.get("max_lines") or self.max_lines

        if not path:
            return ToolResult(success=False, error="'path' parameter is required")

        file_path = FilePath(path).resolve()

        # Security check: allowed paths
        if self.allowed_paths:
            allowed = any(str(file_path).startswith(allowed) for allowed in self.allowed_paths)
            if not allowed:
                return ToolResult(
                    success=False,
                    error=f"Path not allowed: {path}",
                    metadata={"path": path, "allowed_paths": self.allowed_paths},
                )

        try:
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}",
                    metadata={"path": path},
                )

            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    error=f"Not a file: {path}",
                    metadata={"path": path},
                )

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                return ToolResult(
                    success=False,
                    error=f"File too large: {file_size} bytes (max: {self.max_file_size})",
                    metadata={"path": path, "file_size": file_size},
                )

            # Read file
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... (truncated after {max_lines} lines)")
                        break
                    lines.append(line)

            content = "".join(lines)

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "path": path,
                    "file_size": file_size,
                    "lines_read": len(lines),
                },
            )

        except Exception as e:
            logger.exception(f"Failed to read file: {path}")
            return ToolResult(
                success=False,
                error=f"Read error: {str(e)}",
                metadata={"path": path, "exception_type": type(e).__name__},
            )

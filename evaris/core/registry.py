"""Registry for metrics and tools in Evaris.

This module provides global registries for:
- Metrics: Evaluation metrics that can be referenced by name
- Tools: LLM Judge tools that can be referenced by name

The registry pattern allows:
- Easy registration of custom metrics/tools via decorators
- Discovery of available metrics/tools at runtime
- String-based metric/tool references in evaluate() calls

Example:
    >>> from evaris.core import register_metric, BaseMetric
    >>>
    >>> @register_metric("my_metric")
    ... class MyMetric(BaseMetric):
    ...     async def a_measure(self, test_case, actual_output):
    ...         ...
    >>>
    >>> # Now usable by name
    >>> evaluate(data=data, metrics=["my_metric"])
"""

from typing import Any, Callable, Optional, Type, Union, overload

from evaris.core.protocols import BaseMetric, BaseTool


class MetricRegistry:
    """Global registry for evaluation metrics.

    Metrics can be registered by name and later retrieved for use
    in evaluate() calls.

    Attributes:
        _metrics: Dictionary mapping metric names to metric classes

    Example:
        >>> registry = MetricRegistry()
        >>> registry.register("exact_match", ExactMatchMetric)
        >>> metric = registry.get("exact_match")
    """

    _instance: Optional["MetricRegistry"] = None
    _metrics: dict[str, Type[BaseMetric]]

    def __new__(cls) -> "MetricRegistry":
        """Singleton pattern - only one registry exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics = {}
        return cls._instance

    def register(
        self,
        name: str,
        metric_class: Type[BaseMetric],
        overwrite: bool = False,
    ) -> None:
        """Register a metric class.

        Args:
            name: Unique name for the metric
            metric_class: The metric class to register
            overwrite: If True, overwrite existing registration

        Raises:
            ValueError: If name already exists and overwrite=False
        """
        if name in self._metrics and not overwrite:
            raise ValueError(
                f"Metric '{name}' already registered. " "Use overwrite=True to replace."
            )
        self._metrics[name] = metric_class

    def get(self, name: str) -> Type[BaseMetric]:
        """Get a metric class by name.

        Args:
            name: The metric name

        Returns:
            The metric class

        Raises:
            KeyError: If metric not found
        """
        if name not in self._metrics:
            available = ", ".join(sorted(self._metrics.keys()))
            raise KeyError(f"Metric '{name}' not found. Available metrics: {available}")
        return self._metrics[name]

    def get_instance(self, name: str, **kwargs: Any) -> BaseMetric:
        """Get an instantiated metric by name.

        Args:
            name: The metric name
            **kwargs: Arguments to pass to metric constructor

        Returns:
            An instance of the metric
        """
        metric_class = self.get(name)
        return metric_class(**kwargs)

    def list_metrics(self) -> list[str]:
        """List all registered metric names.

        Returns:
            Sorted list of metric names
        """
        return sorted(self._metrics.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a metric is registered.

        Args:
            name: The metric name

        Returns:
            True if registered
        """
        return name in self._metrics

    def unregister(self, name: str) -> None:
        """Unregister a metric.

        Args:
            name: The metric name

        Raises:
            KeyError: If metric not found
        """
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found")
        del self._metrics[name]

    def clear(self) -> None:
        """Clear all registered metrics."""
        self._metrics.clear()


class ToolRegistry:
    """Global registry for LLM Judge tools.

    Tools can be registered by name and later retrieved for use
    in LLMJudge configurations.

    Attributes:
        _tools: Dictionary mapping tool names to tool classes

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register("run_code", CodeExecutorTool)
        >>> tool = registry.get("run_code")
    """

    _instance: Optional["ToolRegistry"] = None
    _tools: dict[str, Type[BaseTool]]

    def __new__(cls) -> "ToolRegistry":
        """Singleton pattern - only one registry exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(
        self,
        name: str,
        tool_class: Type[BaseTool],
        overwrite: bool = False,
    ) -> None:
        """Register a tool class.

        Args:
            name: Unique name for the tool
            tool_class: The tool class to register
            overwrite: If True, overwrite existing registration

        Raises:
            ValueError: If name already exists and overwrite=False
        """
        if name in self._tools and not overwrite:
            raise ValueError(f"Tool '{name}' already registered. " "Use overwrite=True to replace.")
        self._tools[name] = tool_class

    def get(self, name: str) -> Type[BaseTool]:
        """Get a tool class by name.

        Args:
            name: The tool name

        Returns:
            The tool class

        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            available = ", ".join(sorted(self._tools.keys()))
            raise KeyError(f"Tool '{name}' not found. Available tools: {available}")
        return self._tools[name]

    def get_instance(self, name: str, **kwargs: Any) -> BaseTool:
        """Get an instantiated tool by name.

        Args:
            name: The tool name
            **kwargs: Arguments to pass to tool constructor

        Returns:
            An instance of the tool
        """
        tool_class = self.get(name)
        return tool_class(**kwargs)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            Sorted list of tool names
        """
        return sorted(self._tools.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: The tool name

        Returns:
            True if registered
        """
        return name in self._tools

    def unregister(self, name: str) -> None:
        """Unregister a tool.

        Args:
            name: The tool name

        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        del self._tools[name]

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()


# Singleton accessors
def get_metric_registry() -> MetricRegistry:
    """Get the global metric registry.

    Returns:
        The singleton MetricRegistry instance
    """
    return MetricRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry.

    Returns:
        The singleton ToolRegistry instance
    """
    return ToolRegistry()


# Decorator for registering metrics
@overload
def register_metric(
    name: Type[BaseMetric],
    overwrite: bool = False,
) -> Type[BaseMetric]: ...


@overload
def register_metric(
    name: Optional[str] = None,
    overwrite: bool = False,
) -> Callable[[Type[BaseMetric]], Type[BaseMetric]]: ...


def register_metric(
    name: Union[Optional[str], Type[BaseMetric]] = None,
    overwrite: bool = False,
) -> Union[Type[BaseMetric], Callable[[Type[BaseMetric]], Type[BaseMetric]]]:
    """Decorator to register a metric class.

    Can be used with or without arguments:

        @register_metric
        class MyMetric(BaseMetric):
            ...

        @register_metric("custom_name")
        class AnotherMetric(BaseMetric):
            ...

    Args:
        name: Optional custom name (defaults to class name in snake_case)
        overwrite: If True, overwrite existing registration

    Returns:
        Decorator function or decorated class
    """

    def _to_snake_case(class_name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re

        # Remove 'Metric' suffix if present
        if class_name.endswith("Metric"):
            class_name = class_name[:-6]
        # Convert CamelCase to snake_case
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def decorator(cls: Type[BaseMetric]) -> Type[BaseMetric]:
        metric_name = name if isinstance(name, str) else _to_snake_case(cls.__name__)
        get_metric_registry().register(metric_name, cls, overwrite=overwrite)
        return cls

    # Handle @register_metric without parentheses
    if isinstance(name, type) and issubclass(name, BaseMetric):
        cls = name
        return decorator(cls)

    return decorator


# Decorator for registering tools
@overload
def register_tool(
    name: Type[BaseTool],
    overwrite: bool = False,
) -> Type[BaseTool]: ...


@overload
def register_tool(
    name: Optional[str] = None,
    overwrite: bool = False,
) -> Callable[[Type[BaseTool]], Type[BaseTool]]: ...


def register_tool(
    name: Union[Optional[str], Type[BaseTool]] = None,
    overwrite: bool = False,
) -> Union[Type[BaseTool], Callable[[Type[BaseTool]], Type[BaseTool]]]:
    """Decorator to register a tool class.

    Can be used with or without arguments:

        @register_tool
        class MyTool(BaseTool):
            ...

        @register_tool("custom_name")
        class AnotherTool(BaseTool):
            ...

    Args:
        name: Optional custom name (defaults to tool's name attribute)
        overwrite: If True, overwrite existing registration

    Returns:
        Decorator function or decorated class
    """

    def decorator(cls: Type[BaseTool]) -> Type[BaseTool]:
        tool_name = name if isinstance(name, str) else getattr(cls, "name", cls.__name__)
        get_tool_registry().register(tool_name, cls, overwrite=overwrite)
        return cls

    # Handle @register_tool without parentheses
    if isinstance(name, type) and issubclass(name, BaseTool):
        cls = name
        return decorator(cls)

    return decorator


# Resolve metric from string or instance
def resolve_metric(metric: Union[str, BaseMetric, Type[BaseMetric]]) -> BaseMetric:
    """Resolve a metric from string name, class, or instance.

    Args:
        metric: Metric name (str), class, or instance

    Returns:
        Metric instance

    Raises:
        KeyError: If string metric not found in registry
        TypeError: If metric is not a valid type
    """
    if isinstance(metric, str):
        return get_metric_registry().get_instance(metric)
    elif isinstance(metric, type) and issubclass(metric, BaseMetric):
        return metric()
    elif isinstance(metric, BaseMetric):
        return metric
    else:
        raise TypeError(f"Expected metric name (str), class, or instance, got {type(metric)}")


# Resolve tool from string or instance
def resolve_tool(tool: Union[str, BaseTool, Type[BaseTool]]) -> BaseTool:
    """Resolve a tool from string name, class, or instance.

    Args:
        tool: Tool name (str), class, or instance

    Returns:
        Tool instance

    Raises:
        KeyError: If string tool not found in registry
        TypeError: If tool is not a valid type
    """
    if isinstance(tool, str):
        return get_tool_registry().get_instance(tool)
    elif isinstance(tool, type) and issubclass(tool, BaseTool):
        return tool()
    elif isinstance(tool, BaseTool):
        return tool
    else:
        raise TypeError(f"Expected tool name (str), class, or instance, got {type(tool)}")

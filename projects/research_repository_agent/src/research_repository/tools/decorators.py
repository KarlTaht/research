"""Decorator for registering agent tools."""

from functools import wraps
from typing import Any, Callable

from research_repository.tools.registry import (
    ToolCategory,
    ToolSpec,
    get_global_registry,
)


def tool(
    name: str,
    description: str,
    category: ToolCategory,
    parameters: dict[str, Any],
    requires_confirmation: bool = False,
    read_only: bool = True,
    examples: list[str] = None,
    auto_register: bool = True,
):
    """Decorator to register a function as an agent tool.

    Args:
        name: Tool name (used for invocation).
        description: Human-readable description of what the tool does.
        category: Tool category for organization.
        parameters: JSON Schema for tool parameters.
        requires_confirmation: If True, requires user confirmation before execution.
        read_only: If True, tool doesn't modify state (safe to auto-approve).
        examples: Example usage strings for documentation.
        auto_register: If True, automatically register with global registry.

    Returns:
        Decorator function.

    Example:
        @tool(
            name="explore_repo",
            description="Explore the repository structure",
            category=ToolCategory.CODEBASE,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to explore"},
                    "depth": {"type": "integer", "description": "Max depth"}
                },
                "required": []
            }
        )
        async def explore_repo(path: str = ".", depth: int = 2) -> dict:
            ...
    """
    if examples is None:
        examples = []

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        # Create and attach tool spec
        spec = ToolSpec(
            name=name,
            description=description,
            category=category,
            handler=func,
            parameters=parameters,
            requires_confirmation=requires_confirmation,
            read_only=read_only,
            examples=examples,
        )
        wrapper._tool_spec = spec

        # Auto-register if requested
        if auto_register:
            registry = get_global_registry()
            if name not in registry:
                registry.register(spec)

        return wrapper

    return decorator


def get_tool_spec(func: Callable) -> ToolSpec | None:
    """Get the tool spec attached to a decorated function.

    Args:
        func: Decorated function.

    Returns:
        ToolSpec if the function is a registered tool, None otherwise.
    """
    return getattr(func, "_tool_spec", None)

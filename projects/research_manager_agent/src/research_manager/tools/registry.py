"""Central registry for agent tools."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ToolCategory(Enum):
    """Categories of agent tools."""

    CODEBASE = "codebase"  # explore_repo, find_script, read_config, list_projects
    EXPERIMENTS = "experiments"  # query_experiments, analyze_logs, compare_runs
    ASSISTANT = "assistant"  # run_command, suggest_cleanup, explain_error


@dataclass
class ToolSpec:
    """Specification for a registered tool."""

    name: str
    description: str
    category: ToolCategory
    handler: Callable
    parameters: dict[str, Any]
    requires_confirmation: bool = False
    read_only: bool = True
    examples: list[str] = field(default_factory=list)

    def to_mcp_schema(self) -> dict:
        """Convert to MCP tool schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for inspection."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters,
            "requires_confirmation": self.requires_confirmation,
            "read_only": self.read_only,
            "examples": self.examples,
        }


class ToolRegistry:
    """Central registry for all agent tools.

    Handles tool registration, discovery, and schema export.
    """

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self._by_category: dict[ToolCategory, list[str]] = {cat: [] for cat in ToolCategory}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool.

        Args:
            spec: Tool specification to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered")

        self._tools[spec.name] = spec
        self._by_category[spec.category].append(spec.name)

    def unregister(self, name: str) -> None:
        """Unregister a tool.

        Args:
            name: Name of the tool to unregister.
        """
        if name in self._tools:
            spec = self._tools.pop(name)
            self._by_category[spec.category].remove(name)

    def get(self, name: str) -> Optional[ToolSpec]:
        """Get tool by name.

        Args:
            name: Tool name.

        Returns:
            ToolSpec if found, None otherwise.
        """
        return self._tools.get(name)

    def list_all(self) -> list[ToolSpec]:
        """List all registered tools.

        Returns:
            List of all tool specifications.
        """
        return list(self._tools.values())

    def list_by_category(self, category: ToolCategory) -> list[ToolSpec]:
        """List tools in a category.

        Args:
            category: Category to filter by.

        Returns:
            List of tool specifications in the category.
        """
        return [self._tools[name] for name in self._by_category[category]]

    def list_names(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def get_mcp_schema(self) -> list[dict]:
        """Export all tools as MCP schema.

        Returns:
            List of tool schemas for MCP registration.
        """
        return [spec.to_mcp_schema() for spec in self._tools.values()]

    def get_read_only_tools(self) -> list[ToolSpec]:
        """Get tools that are read-only (safe to auto-approve).

        Returns:
            List of read-only tool specifications.
        """
        return [spec for spec in self._tools.values() if spec.read_only]

    def get_confirmation_required(self) -> list[ToolSpec]:
        """Get tools that require user confirmation.

        Returns:
            List of tool specifications requiring confirmation.
        """
        return [spec for spec in self._tools.values() if spec.requires_confirmation]

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry.

    Creates one if it doesn't exist.

    Returns:
        The global ToolRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _global_registry
    _global_registry = None

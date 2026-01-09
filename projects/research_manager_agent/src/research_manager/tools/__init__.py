"""Tool management for the research manager agent."""

from research_manager.tools.registry import (
    ToolRegistry,
    ToolSpec,
    ToolCategory,
    get_global_registry,
    reset_global_registry,
)
from research_manager.tools.decorators import tool

# Import tool modules to trigger registration
from research_manager.tools import explore  # noqa: F401
from research_manager.tools import projects  # noqa: F401
from research_manager.tools import scripts  # noqa: F401
from research_manager.tools import configs  # noqa: F401
from research_manager.tools import experiments  # noqa: F401
from research_manager.tools import assistant  # noqa: F401

__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "ToolCategory",
    "tool",
    "get_global_registry",
    "reset_global_registry",
]

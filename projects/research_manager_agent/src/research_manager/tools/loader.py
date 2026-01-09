"""Dynamic tool discovery and loading."""

import importlib
import pkgutil
from pathlib import Path
from typing import Optional

from research_manager.tools.registry import ToolRegistry, get_global_registry
from research_manager.tools.decorators import get_tool_spec


def discover_tools(package_path: str = "research_manager.tools") -> list[str]:
    """Discover tool modules in the tools package.

    Args:
        package_path: Python package path to search for tool modules.

    Returns:
        List of module names containing tools.
    """
    tool_modules = []

    try:
        package = importlib.import_module(package_path)
        package_dir = Path(package.__file__).parent

        # Skip these modules as they're infrastructure, not tools
        skip_modules = {"__init__", "registry", "decorators", "loader"}

        for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
            if module_name not in skip_modules:
                tool_modules.append(f"{package_path}.{module_name}")

    except ImportError as e:
        print(f"Warning: Could not import tools package: {e}")

    return tool_modules


def load_tools(
    module_names: Optional[list[str]] = None,
    registry: Optional[ToolRegistry] = None,
) -> int:
    """Load tools from specified modules.

    Args:
        module_names: List of module paths to load. If None, auto-discovers.
        registry: Registry to register tools into. Uses global if None.

    Returns:
        Number of tools loaded.
    """
    if registry is None:
        registry = get_global_registry()

    if module_names is None:
        module_names = discover_tools()

    loaded_count = 0

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)

            # Find all decorated functions in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                spec = get_tool_spec(attr)

                if spec is not None and spec.name not in registry:
                    registry.register(spec)
                    loaded_count += 1

        except ImportError as e:
            print(f"Warning: Could not import tool module {module_name}: {e}")
        except Exception as e:
            print(f"Warning: Error loading tools from {module_name}: {e}")

    return loaded_count


def reload_tools(
    module_names: Optional[list[str]] = None,
    registry: Optional[ToolRegistry] = None,
) -> int:
    """Reload tools (useful for development).

    Args:
        module_names: List of module paths to reload. If None, auto-discovers.
        registry: Registry to use. Uses global if None.

    Returns:
        Number of tools loaded after reload.
    """
    if registry is None:
        registry = get_global_registry()

    if module_names is None:
        module_names = discover_tools()

    # Unregister existing tools from these modules
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                spec = get_tool_spec(attr)

                if spec is not None and spec.name in registry:
                    registry.unregister(spec.name)

            # Force reload
            importlib.reload(module)

        except ImportError:
            pass

    # Re-load tools
    return load_tools(module_names, registry)

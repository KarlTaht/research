"""Tests for tool registry and decorators."""

import pytest

from research_repository.tools import ToolRegistry, ToolSpec, ToolCategory, tool
from research_repository.tools.registry import get_global_registry, reset_global_registry
from research_repository.tools.decorators import get_tool_spec


class TestToolCategory:
    """Tests for ToolCategory enum."""

    def test_categories_exist(self):
        """Test that all expected categories exist."""
        assert ToolCategory.CODEBASE.value == "codebase"
        assert ToolCategory.EXPERIMENTS.value == "experiments"
        assert ToolCategory.ASSISTANT.value == "assistant"


class TestToolSpec:
    """Tests for ToolSpec."""

    def test_create_spec(self):
        """Test creating a tool spec."""
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.CODEBASE,
            handler=lambda: None,
            parameters={"type": "object", "properties": {}},
        )
        assert spec.name == "test_tool"
        assert spec.read_only is True
        assert spec.requires_confirmation is False

    def test_to_mcp_schema(self):
        """Test converting to MCP schema."""
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.CODEBASE,
            handler=lambda: None,
            parameters={"type": "object"},
        )
        schema = spec.to_mcp_schema()
        assert schema["name"] == "test_tool"
        assert schema["description"] == "A test tool"
        assert "parameters" in schema

    def test_to_dict(self):
        """Test converting to dict."""
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.EXPERIMENTS,
            handler=lambda: None,
            parameters={},
            examples=["example 1", "example 2"],
        )
        d = spec.to_dict()
        assert d["category"] == "experiments"
        assert len(d["examples"]) == 2


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self, tool_registry: ToolRegistry):
        """Test registering a tool."""
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.CODEBASE,
            handler=lambda: None,
            parameters={},
        )
        tool_registry.register(spec)

        assert "test_tool" in tool_registry
        assert len(tool_registry) == 1

    def test_register_duplicate_fails(self, tool_registry: ToolRegistry):
        """Test that registering duplicate name fails."""
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.CODEBASE,
            handler=lambda: None,
            parameters={},
        )
        tool_registry.register(spec)

        with pytest.raises(ValueError):
            tool_registry.register(spec)

    def test_unregister_tool(self, tool_registry: ToolRegistry):
        """Test unregistering a tool."""
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.CODEBASE,
            handler=lambda: None,
            parameters={},
        )
        tool_registry.register(spec)
        tool_registry.unregister("test_tool")

        assert "test_tool" not in tool_registry

    def test_get_tool(self, tool_registry: ToolRegistry):
        """Test getting a tool by name."""
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.CODEBASE,
            handler=lambda: None,
            parameters={},
        )
        tool_registry.register(spec)

        retrieved = tool_registry.get("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"

        assert tool_registry.get("nonexistent") is None

    def test_list_by_category(self, tool_registry: ToolRegistry):
        """Test listing tools by category."""
        spec1 = ToolSpec(
            name="codebase_tool",
            description="A codebase tool",
            category=ToolCategory.CODEBASE,
            handler=lambda: None,
            parameters={},
        )
        spec2 = ToolSpec(
            name="experiment_tool",
            description="An experiment tool",
            category=ToolCategory.EXPERIMENTS,
            handler=lambda: None,
            parameters={},
        )
        tool_registry.register(spec1)
        tool_registry.register(spec2)

        codebase_tools = tool_registry.list_by_category(ToolCategory.CODEBASE)
        assert len(codebase_tools) == 1
        assert codebase_tools[0].name == "codebase_tool"

    def test_get_mcp_schema(self, tool_registry: ToolRegistry):
        """Test exporting MCP schema."""
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.CODEBASE,
            handler=lambda: None,
            parameters={"type": "object"},
        )
        tool_registry.register(spec)

        schemas = tool_registry.get_mcp_schema()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "test_tool"

    def test_get_read_only_tools(self, tool_registry: ToolRegistry):
        """Test getting read-only tools."""
        spec1 = ToolSpec(
            name="read_tool",
            description="Read-only tool",
            category=ToolCategory.CODEBASE,
            handler=lambda: None,
            parameters={},
            read_only=True,
        )
        spec2 = ToolSpec(
            name="write_tool",
            description="Write tool",
            category=ToolCategory.ASSISTANT,
            handler=lambda: None,
            parameters={},
            read_only=False,
        )
        tool_registry.register(spec1)
        tool_registry.register(spec2)

        read_only = tool_registry.get_read_only_tools()
        assert len(read_only) == 1
        assert read_only[0].name == "read_tool"


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator(self, tool_registry: ToolRegistry):
        """Test basic tool decorator."""
        # Reset global registry for this test
        reset_global_registry()

        @tool(
            name="decorated_tool",
            description="A decorated tool",
            category=ToolCategory.CODEBASE,
            parameters={"type": "object"},
            auto_register=True,
        )
        async def decorated_tool():
            return {"result": "success"}

        # Should have tool spec attached
        spec = get_tool_spec(decorated_tool)
        assert spec is not None
        assert spec.name == "decorated_tool"

        # Should be registered in global registry
        global_reg = get_global_registry()
        assert "decorated_tool" in global_reg

    def test_tool_decorator_no_auto_register(self):
        """Test decorator without auto-registration."""
        reset_global_registry()

        @tool(
            name="manual_tool",
            description="A manual tool",
            category=ToolCategory.CODEBASE,
            parameters={},
            auto_register=False,
        )
        async def manual_tool():
            return {}

        spec = get_tool_spec(manual_tool)
        assert spec is not None

        global_reg = get_global_registry()
        assert "manual_tool" not in global_reg

    def test_tool_decorator_with_options(self):
        """Test decorator with all options."""
        reset_global_registry()

        @tool(
            name="full_tool",
            description="A full tool",
            category=ToolCategory.ASSISTANT,
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
            requires_confirmation=True,
            read_only=False,
            examples=["example usage"],
            auto_register=True,
        )
        async def full_tool(arg: str):
            return {"arg": arg}

        spec = get_tool_spec(full_tool)
        assert spec.requires_confirmation is True
        assert spec.read_only is False
        assert len(spec.examples) == 1

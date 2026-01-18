"""Claude Agent SDK integration for the research manager.

This module provides the core agent loop that integrates with Claude's API
to process natural language queries and execute tools.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from research_repository.indexer import RepoIndexer
from research_repository.memory import MemoryBackend, SessionMemory
from research_repository.safety import SafetyHooks, AuditLog
from research_repository.safety.hooks import SafetyDecision
from research_repository.tools import ToolRegistry, get_global_registry
from research_repository.tools.loader import load_tools


@dataclass
class ToolCall:
    """A tool call from the agent."""

    id: str
    name: str
    input: dict
    output: Optional[dict] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class AgentTurn:
    """A single turn in the agent conversation."""

    user_message: str
    assistant_response: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentConfig:
    """Configuration for the agent."""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.0
    system_prompt: Optional[str] = None
    max_tool_calls_per_turn: int = 10
    timeout_seconds: float = 60.0


class ToolExecutor:
    """Executes tools with safety checks and auditing."""

    def __init__(
        self,
        registry: ToolRegistry,
        safety: SafetyHooks,
        audit: AuditLog,
        indexer: RepoIndexer,
        repo_root: Path,
    ):
        self.registry = registry
        self.safety = safety
        self.audit = audit
        self.indexer = indexer
        self.repo_root = repo_root

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        user_confirmed: bool = False,
    ) -> tuple[dict, Optional[str]]:
        """Execute a tool with safety checks.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Input parameters for the tool.
            user_confirmed: Whether user confirmed risky operation.

        Returns:
            Tuple of (result dict, error string or None).
        """
        start_time = datetime.now()

        # Check safety
        safety_result = self.safety.pre_tool_use(tool_name, tool_input)

        if safety_result.decision == SafetyDecision.DENY:
            error = f"Blocked: {safety_result.reason}"
            self.audit.log_action(tool_name, tool_input, safety_result.decision, error=error)
            return {"error": error}, error

        if safety_result.decision == SafetyDecision.CONFIRM and not user_confirmed:
            error = f"Requires confirmation: {safety_result.reason}"
            return {"requires_confirmation": True, "reason": safety_result.reason}, None

        # Get the tool
        tool_spec = self.registry.get(tool_name)
        if not tool_spec:
            error = f"Unknown tool: {tool_name}"
            return {"error": error}, error

        # Inject common parameters
        enriched_input = self._enrich_input(tool_name, tool_input)

        # Execute
        try:
            result = await tool_spec.handler(**enriched_input)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.audit.log_action(
                tool_name,
                tool_input,
                safety_result.decision,
                tool_output=result,
                user_confirmed=user_confirmed,
                execution_time_ms=execution_time,
            )

            return result, None

        except Exception as e:
            error = str(e)
            self.audit.log_action(
                tool_name,
                tool_input,
                safety_result.decision,
                error=error,
            )
            return {"error": error}, error

    def _enrich_input(self, tool_name: str, tool_input: dict) -> dict:
        """Add common parameters to tool input."""
        enriched = dict(tool_input)

        # Add repo_root for tools that need it
        if "repo_root" not in enriched:
            enriched["repo_root"] = self.repo_root

        # Add indexer for tools that need it
        if "indexer" not in enriched:
            enriched["indexer"] = self.indexer

        return enriched


class ResearchAgent:
    """Research Manager Agent with Claude SDK integration.

    This agent uses Claude to process natural language queries about
    the research repository and execute appropriate tools.
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        config: Optional[AgentConfig] = None,
        memory: Optional[MemoryBackend] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the research agent.

        Args:
            repo_root: Root of the monorepo.
            config: Agent configuration.
            memory: Memory backend to use.
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
        """
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.config = config or AgentConfig()
        self.api_key = api_key

        # Initialize components
        self.indexer = RepoIndexer(self.repo_root)
        self.memory = memory or SessionMemory()
        self.safety = SafetyHooks()
        self.audit = AuditLog()
        self.registry = get_global_registry()

        # Tool executor
        self.executor = ToolExecutor(
            registry=self.registry,
            safety=self.safety,
            audit=self.audit,
            indexer=self.indexer,
            repo_root=self.repo_root,
        )

        # Conversation history
        self.history: list[AgentTurn] = []

        # Anthropic client (lazy loaded)
        self._client = None
        self._initialized = False

    @property
    def client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. " "Install with: pip install anthropic"
                )
        return self._client

    async def initialize(self) -> None:
        """Initialize the agent (index repo, load tools)."""
        if self._initialized:
            return

        self.indexer.refresh()
        load_tools(registry=self.registry)
        self._initialized = True

    def _get_system_prompt(self) -> str:
        """Build the system prompt with context."""
        stats = self.indexer.get_stats()

        base_prompt = self.config.system_prompt or ""

        context_prompt = f"""
You are a Research Manager Agent helping navigate an ML research monorepo.

Repository: {self.repo_root}
Projects: {stats['projects']}
Scripts: {stats['scripts']}
Configs: {stats['configs']}

You have access to tools for:
1. Exploring the repository structure
2. Finding and understanding projects, scripts, and configs
3. Querying experiment results
4. Analyzing logs and checkpoints
5. Generating training commands
6. Explaining errors

Always use tools to answer questions about the codebase. Be concise and helpful.
"""

        return base_prompt + context_prompt

    def _get_tools_schema(self) -> list[dict]:
        """Get tool schemas for Claude API."""
        return self.registry.get_mcp_schema()

    async def chat(
        self,
        message: str,
        confirm_callback: Optional[Callable[[str], bool]] = None,
    ) -> str:
        """Send a message and get a response.

        Args:
            message: User message.
            confirm_callback: Function to call for user confirmation.
                              Takes reason string, returns bool.

        Returns:
            Assistant response string.
        """
        await self.initialize()

        # Build messages
        messages = self._build_messages(message)

        # Get tools schema
        tools = self._get_tools_schema()

        # Create turn
        turn = AgentTurn(user_message=message, assistant_response="")

        # Call Claude
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self._get_system_prompt(),
            messages=messages,
            tools=tools if tools else None,
        )

        # Process response
        tool_calls_made = 0
        while (
            response.stop_reason == "tool_use"
            and tool_calls_made < self.config.max_tool_calls_per_turn
        ):
            # Extract tool uses
            tool_uses = [block for block in response.content if block.type == "tool_use"]

            # Execute tools
            tool_results = []
            for tool_use in tool_uses:
                tool_call = ToolCall(
                    id=tool_use.id,
                    name=tool_use.name,
                    input=tool_use.input,
                )

                # Execute with confirmation if needed
                result, error = await self.executor.execute(
                    tool_use.name,
                    tool_use.input,
                    user_confirmed=False,
                )

                # Handle confirmation requirement
                if result.get("requires_confirmation"):
                    if confirm_callback and confirm_callback(result.get("reason", "")):
                        result, error = await self.executor.execute(
                            tool_use.name,
                            tool_use.input,
                            user_confirmed=True,
                        )
                    else:
                        result = {"error": "User declined to confirm operation"}
                        error = result["error"]

                tool_call.output = result
                tool_call.error = error
                turn.tool_calls.append(tool_call)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(result),
                    }
                )

            tool_calls_made += len(tool_uses)

            # Continue conversation with tool results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self._get_system_prompt(),
                messages=messages,
                tools=tools if tools else None,
            )

        # Extract final text response
        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        turn.assistant_response = "\n".join(text_blocks)

        # Store turn
        self.history.append(turn)

        # Update memory
        await self.memory.store(
            key=f"turn_{len(self.history)}",
            value={"message": message, "response": turn.assistant_response},
            category="conversation",
        )

        return turn.assistant_response

    def _build_messages(self, new_message: str) -> list[dict]:
        """Build messages list from history."""
        messages = []

        # Add history (limited to recent turns)
        for turn in self.history[-10:]:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_response})

        # Add new message
        messages.append({"role": "user", "content": new_message})

        return messages

    def get_session_summary(self) -> dict:
        """Get summary of current session."""
        return {
            "turns": len(self.history),
            "total_tool_calls": sum(len(t.tool_calls) for t in self.history),
            "audit": self.audit.get_session_summary(),
        }

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []


class MockAgent(ResearchAgent):
    """Mock agent for testing without API calls.

    Uses predefined responses for testing tool execution
    and workflows without requiring API access.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_responses: list[dict] = []
        self._response_index = 0

    def add_mock_response(
        self,
        text: str,
        tool_calls: Optional[list[dict]] = None,
    ) -> None:
        """Add a mock response.

        Args:
            text: Text response.
            tool_calls: Optional list of tool calls to make.
        """
        self.mock_responses.append(
            {
                "text": text,
                "tool_calls": tool_calls or [],
            }
        )

    async def chat(
        self,
        message: str,
        confirm_callback: Optional[Callable[[str], bool]] = None,
    ) -> str:
        """Mock chat that returns predefined responses."""
        await self.initialize()

        turn = AgentTurn(user_message=message, assistant_response="")

        if self._response_index < len(self.mock_responses):
            mock = self.mock_responses[self._response_index]
            self._response_index += 1

            # Execute any tool calls
            for tc in mock.get("tool_calls", []):
                tool_call = ToolCall(
                    id=f"mock_{len(turn.tool_calls)}",
                    name=tc["name"],
                    input=tc.get("input", {}),
                )

                result, error = await self.executor.execute(
                    tc["name"],
                    tc.get("input", {}),
                    user_confirmed=tc.get("confirmed", False),
                )

                tool_call.output = result
                tool_call.error = error
                turn.tool_calls.append(tool_call)

            turn.assistant_response = mock["text"]
        else:
            turn.assistant_response = "No mock response configured."

        self.history.append(turn)
        return turn.assistant_response

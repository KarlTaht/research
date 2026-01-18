"""Safety hooks for validating tool calls."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SafetyDecision(Enum):
    """Decision for tool execution."""

    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


# Commands that should never be executed
BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+[/~]",  # rm -rf / or rm -rf ~
    r"rm\s+-rf\s+\*",  # rm -rf *
    r"git\s+push.*--force",  # git push --force
    r"git\s+push.*-f\s",  # git push -f
    r"sudo\s+",  # Any sudo command
    r">\s*/dev/",  # Redirecting to /dev/
    r"chmod\s+777",  # chmod 777
    r"mkfs\.",  # Format filesystem
    r"dd\s+if=",  # dd command
    r":(){.*};:",  # Fork bomb
]

# Commands that require confirmation
CONFIRM_PATTERNS = [
    r"python.*train\.py",  # Training runs
    r"python.*download",  # Downloads
    r"git\s+reset",  # Git reset
    r"git\s+checkout\s+--",  # Git checkout with --
    r"pip\s+install",  # Package installation
    r"uv\s+pip\s+install",  # UV package installation
    r"rm\s+",  # Any remove command
    r"mv\s+",  # Any move command (could overwrite)
]


@dataclass
class HookResult:
    """Result of a safety hook evaluation."""

    decision: SafetyDecision
    reason: Optional[str] = None
    modified_input: Optional[dict] = None


class SafetyHooks:
    """Safety hooks for validating tool calls before execution."""

    def __init__(
        self,
        blocked_patterns: Optional[list[str]] = None,
        confirm_patterns: Optional[list[str]] = None,
    ):
        """Initialize safety hooks.

        Args:
            blocked_patterns: Regex patterns for commands to block.
            confirm_patterns: Regex patterns for commands requiring confirmation.
        """
        self.blocked_patterns = blocked_patterns or BLOCKED_PATTERNS
        self.confirm_patterns = confirm_patterns or CONFIRM_PATTERNS

        # Compile patterns for efficiency
        self._blocked_compiled = [re.compile(p, re.IGNORECASE) for p in self.blocked_patterns]
        self._confirm_compiled = [re.compile(p, re.IGNORECASE) for p in self.confirm_patterns]

    def pre_tool_use(self, tool_name: str, tool_input: dict) -> HookResult:
        """Validate a tool call before execution.

        Args:
            tool_name: Name of the tool being called.
            tool_input: Input parameters for the tool.

        Returns:
            HookResult with decision (allow/deny/confirm) and reason.
        """
        # Check for command execution
        if tool_name == "run_command":
            command = tool_input.get("command", "")
            execute = tool_input.get("execute", False)

            # If not executing, just generating command - always allow
            if not execute:
                return HookResult(SafetyDecision.ALLOW)

            # Check blocked patterns
            for pattern in self._blocked_compiled:
                if pattern.search(command):
                    return HookResult(
                        SafetyDecision.DENY,
                        reason="Blocked: Command matches dangerous pattern",
                    )

            # Check confirmation patterns
            for pattern in self._confirm_compiled:
                if pattern.search(command):
                    return HookResult(
                        SafetyDecision.CONFIRM,
                        reason=f"Requires confirmation: {command}",
                    )

        # For other tools, check if they're read-only
        # This will be enhanced when we have the registry available
        write_tools = ["run_command", "write_file", "delete_file"]
        if tool_name in write_tools:
            return HookResult(
                SafetyDecision.CONFIRM,
                reason=f"Tool '{tool_name}' can modify state",
            )

        return HookResult(SafetyDecision.ALLOW)

    def post_tool_use(self, tool_name: str, tool_input: dict, result: dict) -> dict:
        """Process tool result after execution.

        Can be used for logging, result sanitization, etc.

        Args:
            tool_name: Name of the tool that was called.
            tool_input: Input parameters that were used.
            result: Result from the tool execution.

        Returns:
            Potentially modified result.
        """
        # For now, just pass through
        # Could add result sanitization, logging, etc.
        return result

    def is_safe_command(self, command: str) -> bool:
        """Quick check if a command is safe (not blocked).

        Args:
            command: Command to check.

        Returns:
            True if the command is not blocked.
        """
        for pattern in self._blocked_compiled:
            if pattern.search(command):
                return False
        return True

    def requires_confirmation(self, command: str) -> bool:
        """Check if a command requires confirmation.

        Args:
            command: Command to check.

        Returns:
            True if the command requires user confirmation.
        """
        for pattern in self._confirm_compiled:
            if pattern.search(command):
                return True
        return False

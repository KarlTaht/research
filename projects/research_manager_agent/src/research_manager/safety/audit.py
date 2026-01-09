"""Audit logging for tracking agent actions."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from research_manager.safety.hooks import SafetyDecision


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: datetime
    tool_name: str
    tool_input: dict
    tool_output: Optional[dict] = None
    decision: SafetyDecision = SafetyDecision.ALLOW
    user_confirmed: bool = False
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "decision": self.decision.value,
            "user_confirmed": self.user_confirmed,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AuditEntry":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tool_name=data["tool_name"],
            tool_input=data["tool_input"],
            tool_output=data.get("tool_output"),
            decision=SafetyDecision(data.get("decision", "allow")),
            user_confirmed=data.get("user_confirmed", False),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            error=data.get("error"),
            session_id=data.get("session_id"),
        )


class AuditLog:
    """Track all agent actions for review and debugging."""

    def __init__(self, log_path: Optional[Path] = None, session_id: Optional[str] = None):
        """Initialize audit log.

        Args:
            log_path: Path to persist audit log. If None, only keeps in memory.
            session_id: Identifier for this session.
        """
        self.log_path = log_path
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entries: list[AuditEntry] = []
        self._load()

    def _load(self) -> None:
        """Load existing entries from file."""
        if self.log_path and self.log_path.exists():
            try:
                with open(self.log_path) as f:
                    data = json.load(f)
                    self.entries = [AuditEntry.from_dict(e) for e in data.get("entries", [])]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load audit log: {e}")

    def _save(self) -> None:
        """Persist entries to file."""
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w") as f:
                json.dump(
                    {
                        "session_id": self.session_id,
                        "entries": [e.to_dict() for e in self.entries],
                        "saved_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

    def log_action(
        self,
        tool_name: str,
        tool_input: dict,
        decision: SafetyDecision = SafetyDecision.ALLOW,
        tool_output: Optional[dict] = None,
        user_confirmed: bool = False,
        execution_time_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> AuditEntry:
        """Log a tool action.

        Args:
            tool_name: Name of the tool.
            tool_input: Input parameters.
            decision: Safety decision for this action.
            tool_output: Output from tool execution.
            user_confirmed: Whether user confirmed the action.
            execution_time_ms: Execution time in milliseconds.
            error: Error message if execution failed.

        Returns:
            The created audit entry.
        """
        entry = AuditEntry(
            timestamp=datetime.now(),
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            decision=decision,
            user_confirmed=user_confirmed,
            execution_time_ms=execution_time_ms,
            error=error,
            session_id=self.session_id,
        )
        self.entries.append(entry)
        self._save()
        return entry

    def get_session_summary(self) -> dict:
        """Get summary of this session's actions."""
        session_entries = [e for e in self.entries if e.session_id == self.session_id]

        tool_counts: dict[str, int] = {}
        total_time_ms = 0.0
        errors = 0
        blocked = 0
        confirmed = 0

        for entry in session_entries:
            tool_counts[entry.tool_name] = tool_counts.get(entry.tool_name, 0) + 1
            total_time_ms += entry.execution_time_ms

            if entry.error:
                errors += 1
            if entry.decision == SafetyDecision.DENY:
                blocked += 1
            if entry.user_confirmed:
                confirmed += 1

        return {
            "session_id": self.session_id,
            "total_actions": len(session_entries),
            "tool_counts": tool_counts,
            "total_time_ms": total_time_ms,
            "errors": errors,
            "blocked": blocked,
            "confirmed": confirmed,
        }

    def get_recent_entries(self, limit: int = 10) -> list[AuditEntry]:
        """Get most recent entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of recent audit entries.
        """
        return self.entries[-limit:]

    def get_entries_by_tool(self, tool_name: str) -> list[AuditEntry]:
        """Get all entries for a specific tool.

        Args:
            tool_name: Tool name to filter by.

        Returns:
            List of entries for the tool.
        """
        return [e for e in self.entries if e.tool_name == tool_name]

    def get_errors(self) -> list[AuditEntry]:
        """Get all entries that had errors.

        Returns:
            List of entries with errors.
        """
        return [e for e in self.entries if e.error is not None]

    def clear_session(self) -> None:
        """Clear entries from current session."""
        self.entries = [e for e in self.entries if e.session_id != self.session_id]
        self._save()

    def clear_all(self) -> None:
        """Clear all entries."""
        self.entries.clear()
        if self.log_path and self.log_path.exists():
            self.log_path.unlink()

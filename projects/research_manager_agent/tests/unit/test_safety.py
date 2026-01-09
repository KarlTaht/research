"""Tests for safety hooks and audit logging."""


from research_manager.safety import SafetyHooks, SafetyDecision, AuditLog, AuditEntry


class TestSafetyHooks:
    """Tests for SafetyHooks."""

    def test_block_dangerous_commands(
        self, safety_hooks: SafetyHooks, dangerous_commands: list[str]
    ):
        """Test that dangerous commands are blocked."""
        for command in dangerous_commands:
            result = safety_hooks.pre_tool_use("run_command", {"command": command, "execute": True})
            assert result.decision == SafetyDecision.DENY, f"Should block: {command}"

    def test_allow_safe_commands(self, safety_hooks: SafetyHooks, safe_commands: list[str]):
        """Test that safe commands are allowed."""
        for command in safe_commands:
            result = safety_hooks.pre_tool_use("run_command", {"command": command, "execute": True})
            assert result.decision != SafetyDecision.DENY, f"Should allow: {command}"

    def test_confirm_risky_commands(
        self, safety_hooks: SafetyHooks, confirmation_commands: list[str]
    ):
        """Test that risky commands require confirmation."""
        for command in confirmation_commands:
            result = safety_hooks.pre_tool_use("run_command", {"command": command, "execute": True})
            assert result.decision == SafetyDecision.CONFIRM, f"Should require confirm: {command}"

    def test_allow_non_execute_commands(
        self, safety_hooks: SafetyHooks, dangerous_commands: list[str]
    ):
        """Test that non-executed commands (just generating) are allowed."""
        for command in dangerous_commands:
            result = safety_hooks.pre_tool_use(
                "run_command", {"command": command, "execute": False}  # Not executing, just showing
            )
            assert result.decision == SafetyDecision.ALLOW

    def test_is_safe_command(self, safety_hooks: SafetyHooks):
        """Test is_safe_command helper."""
        assert safety_hooks.is_safe_command("ls -la") is True
        assert safety_hooks.is_safe_command("rm -rf /") is False
        assert safety_hooks.is_safe_command("git push --force") is False

    def test_requires_confirmation(self, safety_hooks: SafetyHooks):
        """Test requires_confirmation helper."""
        assert safety_hooks.requires_confirmation("python train.py --config x.yaml") is True
        assert safety_hooks.requires_confirmation("ls -la") is False
        assert safety_hooks.requires_confirmation("pip install numpy") is True

    def test_custom_patterns(self):
        """Test custom blocked/confirm patterns."""
        hooks = SafetyHooks(
            blocked_patterns=[r"custom_dangerous"],
            confirm_patterns=[r"custom_risky"],
        )

        result = hooks.pre_tool_use(
            "run_command", {"command": "custom_dangerous command", "execute": True}
        )
        assert result.decision == SafetyDecision.DENY

        result = hooks.pre_tool_use(
            "run_command", {"command": "custom_risky command", "execute": True}
        )
        assert result.decision == SafetyDecision.CONFIRM


class TestAuditEntry:
    """Tests for AuditEntry."""

    def test_create_entry(self):
        """Test creating an audit entry."""
        from datetime import datetime

        entry = AuditEntry(
            timestamp=datetime.now(),
            tool_name="test_tool",
            tool_input={"arg": "value"},
            decision=SafetyDecision.ALLOW,
        )
        assert entry.tool_name == "test_tool"
        assert entry.user_confirmed is False

    def test_entry_to_dict(self):
        """Test converting entry to dict."""
        from datetime import datetime

        entry = AuditEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            tool_name="test_tool",
            tool_input={"arg": "value"},
            tool_output={"result": "success"},
            decision=SafetyDecision.CONFIRM,
            user_confirmed=True,
            execution_time_ms=123.45,
        )
        d = entry.to_dict()
        assert d["tool_name"] == "test_tool"
        assert d["decision"] == "confirm"
        assert d["user_confirmed"] is True
        assert d["execution_time_ms"] == 123.45

    def test_entry_from_dict(self):
        """Test creating entry from dict."""
        data = {
            "timestamp": "2024-01-01T12:00:00",
            "tool_name": "test_tool",
            "tool_input": {"arg": "value"},
            "decision": "deny",
            "error": "Permission denied",
        }
        entry = AuditEntry.from_dict(data)
        assert entry.tool_name == "test_tool"
        assert entry.decision == SafetyDecision.DENY
        assert entry.error == "Permission denied"


class TestAuditLog:
    """Tests for AuditLog."""

    def test_log_action(self, audit_log: AuditLog):
        """Test logging an action."""
        entry = audit_log.log_action(
            tool_name="test_tool",
            tool_input={"arg": "value"},
            decision=SafetyDecision.ALLOW,
            tool_output={"result": "success"},
            execution_time_ms=100.0,
        )
        assert entry.tool_name == "test_tool"
        assert len(audit_log.entries) == 1

    def test_get_session_summary(self, audit_log: AuditLog):
        """Test getting session summary."""
        audit_log.log_action("tool1", {"a": 1})
        audit_log.log_action("tool1", {"a": 2})
        audit_log.log_action("tool2", {"b": 1}, decision=SafetyDecision.DENY)
        audit_log.log_action("tool3", {"c": 1}, error="Test error")

        summary = audit_log.get_session_summary()
        assert summary["total_actions"] == 4
        assert summary["tool_counts"]["tool1"] == 2
        assert summary["blocked"] == 1
        assert summary["errors"] == 1

    def test_get_recent_entries(self, audit_log: AuditLog):
        """Test getting recent entries."""
        for i in range(20):
            audit_log.log_action(f"tool_{i}", {})

        recent = audit_log.get_recent_entries(limit=5)
        assert len(recent) == 5

    def test_get_entries_by_tool(self, audit_log: AuditLog):
        """Test getting entries by tool name."""
        audit_log.log_action("target_tool", {"a": 1})
        audit_log.log_action("other_tool", {"b": 1})
        audit_log.log_action("target_tool", {"a": 2})

        entries = audit_log.get_entries_by_tool("target_tool")
        assert len(entries) == 2

    def test_get_errors(self, audit_log: AuditLog):
        """Test getting error entries."""
        audit_log.log_action("good_tool", {})
        audit_log.log_action("bad_tool", {}, error="Error 1")
        audit_log.log_action("bad_tool", {}, error="Error 2")

        errors = audit_log.get_errors()
        assert len(errors) == 2

    def test_persistence(self, temp_dir):
        """Test audit log persistence."""

        log_path = temp_dir / "audit.json"

        # First log
        log1 = AuditLog(log_path)
        log1.log_action("test_tool", {"arg": "value"})

        # Second log - should load from file
        log2 = AuditLog(log_path)
        assert len(log2.entries) == 1
        assert log2.entries[0].tool_name == "test_tool"

    def test_clear_session(self, audit_log: AuditLog):
        """Test clearing session entries."""
        audit_log.log_action("tool1", {})
        audit_log.log_action("tool2", {})

        audit_log.clear_session()
        assert len(audit_log.entries) == 0

    def test_clear_all(self, temp_dir):
        """Test clearing all entries."""

        log_path = temp_dir / "audit.json"
        log = AuditLog(log_path)
        log.log_action("tool", {})

        assert log_path.exists()

        log.clear_all()
        assert not log_path.exists()
        assert len(log.entries) == 0

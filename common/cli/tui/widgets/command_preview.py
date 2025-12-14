"""Command preview widget that shows the generated command."""

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Static


class CommandPreview(Container):
    """Widget that shows a preview of the command that will be executed."""

    command: reactive[str] = reactive("")

    DEFAULT_CSS = """
    CommandPreview {
        height: auto;
        border: solid #555c65;
        padding: 1;
        margin-top: 1;
        background: #141a26;
    }

    CommandPreview .preview-label {
        color: #555c65;
    }

    CommandPreview .preview-command {
        color: #78DCE8;
        text-style: bold;
    }
    """

    def __init__(
        self,
        module: str = "",
        initial_args: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.module = module
        self._args = initial_args or []
        self._update_command()

    def compose(self) -> ComposeResult:
        yield Static("Command:", classes="preview-label")
        yield Static(self.command, id="preview-text", classes="preview-command")

    def _update_command(self) -> None:
        """Update the command string."""
        if not self.module:
            self.command = "[dim]No command configured[/dim]"
        else:
            cmd = f"python -m {self.module}"
            if self._args:
                cmd += " " + " ".join(self._args)
            self.command = cmd

    def update_args(self, args: list[str]) -> None:
        """Update the command arguments."""
        self._args = args
        self._update_command()
        try:
            preview_text = self.query_one("#preview-text", Static)
            preview_text.update(self.command)
        except Exception:
            pass  # Widget not mounted yet

    def watch_command(self, command: str) -> None:
        """React to command changes."""
        try:
            preview_text = self.query_one("#preview-text", Static)
            preview_text.update(command)
        except Exception:
            pass

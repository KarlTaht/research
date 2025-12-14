"""Output screen for displaying command execution results."""

import asyncio
import io
import sys
from contextlib import redirect_stdout

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, RichLog, Static

from ..executors import EXECUTORS


class RichLogWriter(io.TextIOBase):
    """A file-like object that writes to a RichLog widget."""

    def __init__(self, rich_log: RichLog) -> None:
        self.rich_log = rich_log
        self._buffer = ""

    def write(self, text: str) -> int:
        """Write text to the RichLog, handling newlines properly."""
        if not text:
            return 0

        self._buffer += text

        # Process complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.rich_log.write(line)

        return len(text)

    def flush(self) -> None:
        """Flush any remaining buffered text."""
        if self._buffer:
            self.rich_log.write(self._buffer)
            self._buffer = ""


class OutputScreen(Screen):
    """Screen that shows live command output."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "back", "Back"),
        Binding("r", "rerun", "Re-run"),
    ]

    def __init__(
        self,
        executor_name: str,
        params: dict,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.executor_name = executor_name
        self.params = params
        self._running = False
        self._cancelled = False

    def compose(self) -> ComposeResult:
        """Compose the output screen."""
        yield Header()
        with Container(id="output-container"):
            # Show a friendly description of the operation
            params_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items() if v)
            yield Static(
                f"[bold]Running:[/bold] [cyan]{self.executor_name}({params_str})[/cyan]",
                id="command-display",
            )
            yield RichLog(id="output-log", highlight=True, markup=True)
            yield Static("", id="status-display")
            with Horizontal(classes="button-bar"):
                yield Button("Back", variant="default", id="back-btn")
                yield Button("Re-run", variant="primary", id="rerun-btn")
        yield Footer()

    def on_mount(self) -> None:
        """Start command execution when screen is mounted."""
        self.run_command()

    def run_command(self) -> None:
        """Start the command execution."""
        self._running = True
        self._cancelled = False
        self.update_status("running")
        self.run_worker(self._execute_command(), exclusive=True)

    async def _execute_command(self) -> None:
        """Execute the command directly via the executor function."""
        log = self.query_one("#output-log", RichLog)
        log.clear()

        executor = EXECUTORS.get(self.executor_name)
        if not executor:
            log.write(f"[red]Unknown executor: {self.executor_name}[/red]")
            self.update_status("failed")
            self._running = False
            return

        log.write(f"[dim]Executing {self.executor_name}...[/dim]\n")

        try:
            # Create a writer that sends output to RichLog
            log_writer = RichLogWriter(log)

            # Run the executor in a thread to avoid blocking
            # Redirect stdout to our RichLog
            def run_with_redirect():
                old_stdout = sys.stdout
                try:
                    sys.stdout = log_writer
                    executor(**self.params)
                finally:
                    log_writer.flush()
                    sys.stdout = old_stdout

            await asyncio.to_thread(run_with_redirect)

            if not self._cancelled:
                self.update_status("success")
                log.write("\n[green]Completed successfully.[/green]")

        except Exception as e:
            self.update_status("failed")
            log.write(f"\n[red]Error: {e}[/red]")

        finally:
            self._running = False

    def update_status(self, status: str) -> None:
        """Update the status display."""
        status_widget = self.query_one("#status-display", Static)
        if status == "running":
            status_widget.update("[yellow]Status: Running...[/yellow]")
        elif status == "success":
            status_widget.update("[green]Status: Completed[/green]")
        elif status == "failed":
            status_widget.update("[red]Status: Failed[/red]")

    def action_back(self) -> None:
        """Go back to previous screen."""
        if self._running:
            self._cancelled = True
        self.app.pop_screen()

    def action_rerun(self) -> None:
        """Re-run the command."""
        if not self._running:
            self.run_command()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-btn":
            self.action_back()
        elif event.button.id == "rerun-btn":
            self.action_rerun()

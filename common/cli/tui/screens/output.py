"""Output screen for displaying command execution results."""

import asyncio
import sys

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, RichLog, Static


class OutputScreen(Screen):
    """Screen that shows live command output."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "back", "Back"),
        Binding("r", "rerun", "Re-run"),
    ]

    def __init__(self, module: str, args: list[str], name: str | None = None) -> None:
        super().__init__(name=name)
        self.module = module
        self.args = args
        self._process: asyncio.subprocess.Process | None = None
        self._running = False

    def compose(self) -> ComposeResult:
        """Compose the output screen."""
        yield Header()
        with Container(id="output-container"):
            cmd_str = f"python -m {self.module} {' '.join(self.args)}"
            yield Static(f"[bold]Running:[/bold] [cyan]{cmd_str}[/cyan]", id="command-display")
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
        self.update_status("running")
        self.run_worker(self._execute_command(), exclusive=True)

    async def _execute_command(self) -> None:
        """Execute the command and stream output."""
        log = self.query_one("#output-log", RichLog)
        log.clear()

        cmd = [sys.executable, "-m", self.module] + self.args
        log.write(f"[dim]$ {' '.join(cmd)}[/dim]\n")

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            if self._process.stdout:
                async for line in self._process.stdout:
                    text = line.decode("utf-8", errors="replace")
                    log.write(text.rstrip("\n"))

            await self._process.wait()
            return_code = self._process.returncode

            if return_code == 0:
                self.update_status("success")
                log.write("\n[green]Command completed successfully.[/green]")
            else:
                self.update_status("failed")
                log.write(f"\n[red]Command failed with exit code {return_code}.[/red]")

        except Exception as e:
            self.update_status("failed")
            log.write(f"\n[red]Error: {e}[/red]")

        finally:
            self._running = False
            self._process = None

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
        if self._running and self._process:
            self._process.terminate()
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

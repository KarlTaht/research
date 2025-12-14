"""Main Textual TUI application for the research CLI.

Launch with:
    python -m common.cli.tui
    # or after installing:
    research
"""

from pathlib import Path

from textual.app import App
from textual.binding import Binding
from textual.command import Hit, Hits, Provider

from .screens import HomeScreen, OutputScreen


# Command definitions for the command palette (power-user shortcut)
COMMANDS = [
    # Data commands
    (
        "data download dataset",
        "Download a HuggingFace dataset",
        "common.cli.data",
        ["download", "dataset", "--help"],
    ),
    (
        "data download model",
        "Download a HuggingFace model",
        "common.cli.data",
        ["download", "model", "--help"],
    ),
    ("data analyze", "Analyze token distributions", "common.cli.data", ["analyze", "--help"]),
    (
        "data pretokenize",
        "Pre-tokenize dataset for training",
        "common.cli.data",
        ["pretokenize", "--help"],
    ),
    (
        "data fineweb sample",
        "Download FineWeb sample",
        "common.cli.data",
        ["fineweb", "sample", "--help"],
    ),
    (
        "data fineweb index",
        "Build FineWeb domain index",
        "common.cli.data",
        ["fineweb", "index", "--help"],
    ),
    (
        "data fineweb query",
        "Query FineWeb domain index",
        "common.cli.data",
        ["fineweb", "query", "--help"],
    ),
    (
        "data fineweb extract",
        "Extract domain corpus from FineWeb",
        "common.cli.data",
        ["fineweb", "extract", "--help"],
    ),
    # Experiment commands
    ("experiments list", "List all experiments", "common.cli.experiments", ["list"]),
    (
        "experiments summary",
        "Show experiment summary stats",
        "common.cli.experiments",
        ["summary"],
    ),
    (
        "experiments best",
        "Get best experiments by metric",
        "common.cli.experiments",
        ["best", "--help"],
    ),
    (
        "experiments query",
        "Run SQL query on experiments",
        "common.cli.experiments",
        ["query", "--help"],
    ),
    (
        "experiments compare",
        "Compare specific experiments",
        "common.cli.experiments",
        ["compare", "--help"],
    ),
    # Infrastructure commands
    ("infra env", "Test environment setup", "common.cli.infra", ["env"]),
    ("infra lambda", "Check Lambda Labs availability", "common.cli.infra", ["lambda"]),
]


class ResearchCommandProvider(Provider):
    """Command palette provider for research commands."""

    def _make_command(self, module: str, args: list[str]):
        """Create an async command callback."""

        async def run_command() -> None:
            self.app.push_screen(OutputScreen(module, args))

        return run_command

    async def search(self, query: str) -> Hits:
        """Search commands matching the query."""
        matcher = self.matcher(query)

        for cmd_name, description, module, args in COMMANDS:
            score = matcher.match(cmd_name)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(cmd_name),
                    self._make_command(module, args),
                    help=description,
                )

    async def discover(self) -> Hits:
        """Show suggestions when input is empty."""
        for cmd_name, description, module, args in COMMANDS[:8]:
            yield Hit(
                1.0,
                cmd_name,
                self._make_command(module, args),
                help=description,
            )


class ResearchApp(App):
    """Research CLI TUI application."""

    TITLE = "Research CLI"
    SUB_TITLE = "ML Research Tools"

    # Load CSS from external file
    CSS_PATH = Path(__file__).parent / "styles.tcss"

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    COMMANDS = App.COMMANDS | {ResearchCommandProvider}

    def on_mount(self) -> None:
        """Push the home screen on mount."""
        self.push_screen(HomeScreen())


def main():
    """Main entry point for the TUI."""
    app = ResearchApp()
    app.run()


if __name__ == "__main__":
    main()

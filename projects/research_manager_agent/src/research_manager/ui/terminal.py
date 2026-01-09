"""Rich-based terminal UI for the research manager agent."""

from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm
from rich.markdown import Markdown


class TerminalUI:
    """Rich-based terminal interface for agent interactions."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the terminal UI.

        Args:
            console: Rich console to use. Creates new one if not provided.
        """
        self.console = console or Console()

    def show_thinking(self, thought: str) -> None:
        """Display agent's reasoning.

        Args:
            thought: The agent's thought/reasoning to display.
        """
        self.console.print(Panel(thought, title="Thinking", border_style="blue", expand=False))

    def show_tool_call(self, tool_name: str, args: dict) -> None:
        """Display a tool invocation.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments being passed to the tool.
        """
        # Format arguments nicely
        args_str = "\n".join(f"  {k}: {v}" for k, v in args.items())
        content = f"[bold]{tool_name}[/bold]\n{args_str}" if args_str else tool_name

        self.console.print(Panel(content, title="Tool Call", border_style="cyan", expand=False))

    def show_result(self, result: Any, title: str = "Result") -> None:
        """Display a tool result.

        Args:
            result: Result to display.
            title: Title for the result panel.
        """
        if isinstance(result, dict):
            self._show_dict_result(result, title)
        elif isinstance(result, list):
            self._show_list_result(result, title)
        elif isinstance(result, str):
            self.console.print(Panel(result, title=title, border_style="green"))
        else:
            self.console.print(Panel(str(result), title=title, border_style="green"))

    def _show_dict_result(self, result: dict, title: str) -> None:
        """Display a dictionary result as a table."""
        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("Key", style="cyan")
        table.add_column("Value")

        for key, value in result.items():
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            table.add_row(str(key), value_str)

        self.console.print(table)

    def _show_list_result(self, result: list, title: str) -> None:
        """Display a list result."""
        if not result:
            self.console.print(Panel("(empty)", title=title, border_style="green"))
            return

        # If list of dicts, show as table
        if isinstance(result[0], dict):
            table = Table(title=title, show_header=True, header_style="bold")

            # Use keys from first item as columns
            keys = list(result[0].keys())
            for key in keys:
                table.add_column(str(key))

            for item in result[:20]:  # Limit to 20 rows
                row = [str(item.get(k, ""))[:50] for k in keys]
                table.add_row(*row)

            if len(result) > 20:
                self.console.print(table)
                self.console.print(f"[dim]... and {len(result) - 20} more[/dim]")
            else:
                self.console.print(table)
        else:
            # Simple list
            content = "\n".join(f"- {item}" for item in result[:20])
            if len(result) > 20:
                content += f"\n... and {len(result) - 20} more"
            self.console.print(Panel(content, title=title, border_style="green"))

    def show_error(self, error: str) -> None:
        """Display an error message.

        Args:
            error: Error message to display.
        """
        self.console.print(Panel(error, title="Error", border_style="red", expand=False))

    def show_warning(self, warning: str) -> None:
        """Display a warning message.

        Args:
            warning: Warning message to display.
        """
        self.console.print(Panel(warning, title="Warning", border_style="yellow", expand=False))

    def show_info(self, info: str) -> None:
        """Display an info message.

        Args:
            info: Info message to display.
        """
        self.console.print(Panel(info, border_style="dim"))

    def show_markdown(self, content: str) -> None:
        """Display markdown content.

        Args:
            content: Markdown content to render.
        """
        self.console.print(Markdown(content))

    def confirm_action(self, action: str, command: Optional[str] = None) -> bool:
        """Prompt user to confirm a risky action.

        Args:
            action: Description of the action.
            command: Optional command being executed.

        Returns:
            True if user confirms, False otherwise.
        """
        if command:
            self.console.print(
                Panel(
                    f"[bold]{action}[/bold]\n\nCommand: [cyan]{command}[/cyan]",
                    title="Confirmation Required",
                    border_style="yellow",
                )
            )
        else:
            self.console.print(Panel(action, title="Confirmation Required", border_style="yellow"))

        return Confirm.ask("Proceed?", default=False)

    def show_cost(self, tokens: int, cost_usd: float) -> None:
        """Display token usage and cost.

        Args:
            tokens: Number of tokens used.
            cost_usd: Cost in USD.
        """
        self.console.print(f"[dim]Tokens: {tokens:,} | Cost: ${cost_usd:.4f}[/dim]")

    def show_projects_table(self, projects: list[dict]) -> None:
        """Display a table of projects.

        Args:
            projects: List of project dictionaries.
        """
        table = Table(title="Projects", show_header=True, header_style="bold")
        table.add_column("Name", style="cyan")
        table.add_column("Type")
        table.add_column("Description")
        table.add_column("Scripts")

        for project in projects:
            scripts = []
            if project.get("has_train_script"):
                scripts.append("train")
            if project.get("has_eval_script"):
                scripts.append("eval")

            table.add_row(
                project["name"],
                project.get("type", ""),
                (project.get("description") or "")[:50],
                ", ".join(scripts) if scripts else "-",
            )

        self.console.print(table)

    def show_experiments_table(self, experiments: list[dict]) -> None:
        """Display a table of experiments.

        Args:
            experiments: List of experiment dictionaries.
        """
        table = Table(title="Experiments", show_header=True, header_style="bold")
        table.add_column("Name", style="cyan")
        table.add_column("Project")
        table.add_column("Perplexity", justify="right")
        table.add_column("Date")

        for exp in experiments:
            metrics = exp.get("metrics", {})
            perplexity = metrics.get("perplexity", metrics.get("val_perplexity", "-"))
            if isinstance(perplexity, float):
                perplexity = f"{perplexity:.2f}"

            table.add_row(
                exp["name"],
                exp.get("project", "-"),
                str(perplexity),
                exp.get("timestamp", "-")[:10] if exp.get("timestamp") else "-",
            )

        self.console.print(table)

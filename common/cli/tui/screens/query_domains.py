"""Query domains wizard screen."""

from textual.app import ComposeResult
from textual.widgets import Input

from ..widgets import CommandPreview, LabeledInput
from .wizard_base import WizardScreen


class QueryDomainsScreen(WizardScreen):
    """Wizard for querying FineWeb domain index."""

    TITLE = "Query Domain Index"
    COMMAND_MODULE = "common.cli.data"

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        yield LabeledInput(
            label="Top domains",
            placeholder="e.g., 50",
            hint="Show top N domains by document count",
            input_id="top-domains",
        )

        yield LabeledInput(
            label="TLD filter",
            placeholder="e.g., .edu, .gov",
            hint="Filter by top-level domain",
            input_id="tld",
        )

        yield LabeledInput(
            label="Domain contains",
            placeholder="e.g., wiki, news",
            hint="Filter domains containing pattern",
            input_id="domain-contains",
        )

        yield LabeledInput(
            label="Custom SQL",
            placeholder="e.g., SELECT * FROM idx WHERE ...",
            hint="Run custom SQL query (table: idx)",
            input_id="sql",
        )

    def compose_command_preview(self) -> ComposeResult:
        """Compose the command preview."""
        yield CommandPreview(
            module=self.COMMAND_MODULE,
            initial_args=self.get_command_args(),
            id="preview",
        )

    def _get_execute_label(self) -> str:
        return "Query"

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update command preview when inputs change."""
        self._update_preview()

    def _update_preview(self) -> None:
        """Update the command preview widget."""
        try:
            preview = self.query_one("#preview", CommandPreview)
            preview.update_args(self.get_command_args())
        except Exception:
            pass

    def get_command_args(self) -> list[str]:
        """Build CLI arguments from form state."""
        args = ["fineweb", "query"]

        try:
            # Top domains
            top_input = self.query_one("#top-domains", Input)
            top = top_input.value.strip()
            if top:
                args.extend(["--top-domains", top])

            # TLD filter
            tld_input = self.query_one("#tld", Input)
            tld = tld_input.value.strip()
            if tld:
                args.extend(["--tld", tld])

            # Domain contains
            contains_input = self.query_one("#domain-contains", Input)
            contains = contains_input.value.strip()
            if contains:
                args.extend(["--domain-contains", contains])

            # Custom SQL
            sql_input = self.query_one("#sql", Input)
            sql = sql_input.value.strip()
            if sql:
                args.extend(["--sql", sql])

        except Exception:
            pass

        return args

    def validate(self) -> tuple[bool, str]:
        """Validate the form before execution."""
        try:
            # Validate top-domains is a positive integer if provided
            top_input = self.query_one("#top-domains", Input)
            top = top_input.value.strip()
            if top:
                try:
                    val = int(top)
                    if val <= 0:
                        return False, "Top domains must be positive"
                except ValueError:
                    return False, "Top domains must be a valid integer"

        except Exception:
            return False, "Form not ready"

        return True, ""

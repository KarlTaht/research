"""Download FineWeb sample wizard screen."""

from textual.app import ComposeResult
from textual.widgets import Input

from ..widgets import LabeledInput
from .wizard_base import WizardScreen


class DownloadFinewebScreen(WizardScreen):
    """Wizard for downloading FineWeb sample data."""

    TITLE = "Download FineWeb Sample"
    EXECUTOR_NAME = "download_fineweb"

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        yield LabeledInput(
            label="Target tokens",
            placeholder="10000000",
            hint="Number of tokens to download (default: 10M)",
            input_id="tokens",
        )

    def _get_execute_label(self) -> str:
        return "Download"

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update command preview when inputs change."""
        self.update_command_preview()

    def get_params(self) -> dict:
        """Build parameters for the executor."""
        params = {}

        try:
            tokens_input = self.query_one("#tokens", Input)
            tokens = tokens_input.value.strip()
            if tokens:
                params["tokens"] = int(tokens)
        except Exception:
            pass

        return params

    def validate(self) -> tuple[bool, str]:
        """Validate the form before execution."""
        try:
            tokens_input = self.query_one("#tokens", Input)
            tokens = tokens_input.value.strip()
            if tokens:
                # Validate it's a positive integer
                try:
                    val = int(tokens)
                    if val <= 0:
                        return False, "Tokens must be a positive number"
                except ValueError:
                    return False, "Tokens must be a valid integer"
        except Exception:
            return False, "Form not ready"

        return True, ""

"""Download FineWeb sample wizard screen."""

from textual.app import ComposeResult
from textual.widgets import Input

from ..widgets import CommandPreview, LabeledInput
from .wizard_base import WizardScreen


class DownloadFinewebScreen(WizardScreen):
    """Wizard for downloading FineWeb sample data."""

    TITLE = "Download FineWeb Sample"
    COMMAND_MODULE = "common.cli.data"

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        yield LabeledInput(
            label="Target tokens",
            placeholder="10000000",
            hint="Number of tokens to download (default: 10M)",
            input_id="tokens",
        )

    def compose_command_preview(self) -> ComposeResult:
        """Compose the command preview."""
        yield CommandPreview(
            module=self.COMMAND_MODULE,
            initial_args=self.get_command_args(),
            id="preview",
        )

    def _get_execute_label(self) -> str:
        return "Download"

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
        args = ["fineweb", "sample"]

        try:
            tokens_input = self.query_one("#tokens", Input)
            tokens = tokens_input.value.strip()
            if tokens:
                args.extend(["--tokens", tokens])
        except Exception:
            pass

        return args

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

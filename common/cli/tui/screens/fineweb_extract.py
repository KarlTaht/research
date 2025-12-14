"""FineWeb extract wizard screen."""

from textual.app import ComposeResult
from textual.widgets import Checkbox, Input, Select, Static

from ..widgets import CommandPreview, LabeledInput
from .wizard_base import WizardScreen


class FinewebExtractScreen(WizardScreen):
    """Wizard for extracting domain corpus from FineWeb."""

    TITLE = "Extract FineWeb Corpus"
    COMMAND_MODULE = "common.cli.data"

    CORPUS_OPTIONS = [
        ("Both corpora", "both"),
        ("Automotive", "automotive"),
        ("Food", "food"),
    ]

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        yield Static("Corpus:", classes="form-label")
        yield Select(
            [(label, value) for label, value in self.CORPUS_OPTIONS],
            value="both",
            id="corpus",
        )

        yield LabeledInput(
            label="Target tokens",
            placeholder="125000000",
            hint="Target tokens per corpus (~500MB at ~4 bytes/token)",
            input_id="target-tokens",
        )

        yield Checkbox("Preview only (show stats without extracting)", id="preview")

    def compose_command_preview(self) -> ComposeResult:
        """Compose the command preview."""
        yield CommandPreview(
            module=self.COMMAND_MODULE,
            initial_args=self.get_command_args(),
            id="preview-widget",
        )

    def _get_execute_label(self) -> str:
        return "Extract"

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update command preview when inputs change."""
        self._update_preview()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update command preview when select changes."""
        self._update_preview()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Update command preview when checkbox changes."""
        self._update_preview()

    def _update_preview(self) -> None:
        """Update the command preview widget."""
        try:
            preview = self.query_one("#preview-widget", CommandPreview)
            preview.update_args(self.get_command_args())
        except Exception:
            pass

    def get_command_args(self) -> list[str]:
        """Build CLI arguments from form state."""
        args = ["fineweb", "extract"]

        try:
            # Corpus selection
            corpus_select = self.query_one("#corpus", Select)
            if corpus_select.value and corpus_select.value != "both":
                args.extend(["--corpus", str(corpus_select.value)])

            # Target tokens
            tokens_input = self.query_one("#target-tokens", Input)
            tokens = tokens_input.value.strip()
            if tokens:
                args.extend(["--target-tokens", tokens])

            # Preview flag
            preview_checkbox = self.query_one("#preview", Checkbox)
            if preview_checkbox.value:
                args.append("--preview")

        except Exception:
            pass

        return args

    def validate(self) -> tuple[bool, str]:
        """Validate the form before execution."""
        try:
            # Validate target-tokens is a positive integer if provided
            tokens_input = self.query_one("#target-tokens", Input)
            tokens = tokens_input.value.strip()
            if tokens:
                try:
                    val = int(tokens)
                    if val <= 0:
                        return False, "Target tokens must be positive"
                except ValueError:
                    return False, "Target tokens must be a valid integer"

        except Exception:
            return False, "Form not ready"

        return True, ""

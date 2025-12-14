"""Download dataset wizard screen."""

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Input, Select, Static

from ..widgets import CommandPreview, LabeledInput
from .wizard_base import WizardScreen


class DownloadDatasetScreen(WizardScreen):
    """Wizard for downloading HuggingFace datasets."""

    TITLE = "Download Dataset"
    COMMAND_MODULE = "common.cli.data"

    # Common splits for HuggingFace datasets
    SPLIT_OPTIONS = [
        ("all", "All splits"),
        ("train", "Train only"),
        ("test", "Test only"),
        ("validation", "Validation only"),
    ]

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        yield LabeledInput(
            label="Dataset name",
            placeholder="e.g., squad, wmt14, HuggingFaceFW/fineweb",
            hint="HuggingFace Hub dataset name or path",
            required=True,
            input_id="dataset-name",
        )

        yield LabeledInput(
            label="Config/subset",
            placeholder="e.g., de-en, default",
            hint="Optional: dataset configuration or subset name",
            input_id="config",
        )

        yield Static("Split:", classes="form-label")
        yield Select(
            [(label, value) for value, label in self.SPLIT_OPTIONS],
            value="all",
            id="split",
        )

        yield LabeledInput(
            label="Output directory",
            placeholder="Leave empty for default (assets/datasets/)",
            hint="Optional: custom output directory",
            input_id="output",
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

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update command preview when select changes."""
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
        args = ["download", "dataset"]

        try:
            # Dataset name (required)
            name_input = self.query_one("#dataset-name", Input)
            name = name_input.value.strip()
            if name:
                args.extend(["--name", name])

            # Config (optional)
            config_input = self.query_one("#config", Input)
            config = config_input.value.strip()
            if config:
                args.extend(["--config", config])

            # Split (optional, skip if "all")
            split_select = self.query_one("#split", Select)
            if split_select.value and split_select.value != "all":
                args.extend(["--split", str(split_select.value)])

            # Output directory (optional)
            output_input = self.query_one("#output", Input)
            output = output_input.value.strip()
            if output:
                args.extend(["--output", output])

        except Exception:
            pass  # Widgets not mounted yet

        return args

    def validate(self) -> tuple[bool, str]:
        """Validate the form before execution."""
        try:
            name_input = self.query_one("#dataset-name", Input)
            name = name_input.value.strip()
            if not name:
                return False, "Dataset name is required"
        except Exception:
            return False, "Form not ready"

        return True, ""

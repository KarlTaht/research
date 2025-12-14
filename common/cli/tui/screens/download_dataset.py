"""Download dataset wizard screen."""

from textual.app import ComposeResult
from textual.widgets import Input, Select, Static

from ..widgets import DatasetSelect, LabeledInput
from .wizard_base import WizardScreen


class DownloadDatasetScreen(WizardScreen):
    """Wizard for downloading HuggingFace datasets."""

    TITLE = "Download Dataset"
    EXECUTOR_NAME = "download_dataset"

    # Common splits for HuggingFace datasets
    SPLIT_OPTIONS = [
        ("all", "All splits"),
        ("train", "Train only"),
        ("test", "Test only"),
        ("validation", "Validation only"),
    ]

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        yield DatasetSelect(
            label="Dataset name",
            placeholder="Type to search local datasets or enter HF path...",
            hint="Select from local datasets or enter HuggingFace Hub path",
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

    def _get_execute_label(self) -> str:
        return "Download"

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update command preview when inputs change."""
        self.update_command_preview()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update command preview when select changes."""
        self.update_command_preview()

    def on_dataset_select_changed(self, event: DatasetSelect.Changed) -> None:
        """Update command preview when dataset selection changes."""
        self.update_command_preview()

    def get_params(self) -> dict:
        """Build parameters for the executor."""
        params = {}

        try:
            # Dataset name (required)
            name_input = self.query_one("#dataset-name", Input)
            name = name_input.value.strip()
            if name:
                params["name"] = name

            # Config (optional)
            config_input = self.query_one("#config", Input)
            config = config_input.value.strip()
            if config:
                params["config"] = config

            # Split (optional, skip if "all")
            split_select = self.query_one("#split", Select)
            if split_select.value and split_select.value != "all":
                params["split"] = str(split_select.value)

            # Output directory (optional)
            output_input = self.query_one("#output", Input)
            output = output_input.value.strip()
            if output:
                params["output"] = output

        except Exception:
            pass  # Widgets not mounted yet

        return params

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

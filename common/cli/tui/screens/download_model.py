"""Download model wizard screen."""

from textual.app import ComposeResult
from textual.widgets import Input

from ..widgets import LabeledInput
from .wizard_base import WizardScreen


class DownloadModelScreen(WizardScreen):
    """Wizard for downloading HuggingFace models."""

    TITLE = "Download Model"
    EXECUTOR_NAME = "download_model"

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        yield LabeledInput(
            label="Repo ID",
            placeholder="e.g., gpt2, bert-base-uncased, openai/clip-vit-base-patch32",
            hint="HuggingFace model repository ID",
            required=True,
            input_id="repo-id",
        )

        yield LabeledInput(
            label="Filename",
            placeholder="e.g., pytorch_model.bin",
            hint="Optional: specific file to download (downloads all if empty)",
            input_id="filename",
        )

        yield LabeledInput(
            label="Revision",
            placeholder="main",
            hint="Git revision (branch, tag, or commit hash)",
            input_id="revision",
        )

        yield LabeledInput(
            label="Output directory",
            placeholder="Leave empty for default (assets/models/)",
            hint="Optional: custom output directory",
            input_id="output",
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
            # Repo ID (required)
            repo_input = self.query_one("#repo-id", Input)
            repo = repo_input.value.strip()
            if repo:
                params["repo_id"] = repo

            # Filename (optional)
            filename_input = self.query_one("#filename", Input)
            filename = filename_input.value.strip()
            if filename:
                params["filename"] = filename

            # Revision (optional)
            revision_input = self.query_one("#revision", Input)
            revision = revision_input.value.strip()
            if revision:
                params["revision"] = revision

            # Output directory (optional)
            output_input = self.query_one("#output", Input)
            output = output_input.value.strip()
            if output:
                params["output"] = output

        except Exception:
            pass

        return params

    def validate(self) -> tuple[bool, str]:
        """Validate the form before execution."""
        try:
            repo_input = self.query_one("#repo-id", Input)
            repo = repo_input.value.strip()
            if not repo:
                return False, "Repo ID is required"
        except Exception:
            return False, "Form not ready"

        return True, ""

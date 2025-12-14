"""Analyze tokens wizard screen."""

from textual.app import ComposeResult
from textual.widgets import Checkbox, Input

from ..widgets import DataSourceInput, LabeledInput, SourceType, TokenizerSelect
from .wizard_base import WizardScreen


class AnalyzeTokensScreen(WizardScreen):
    """Wizard for analyzing token distributions."""

    TITLE = "Analyze Tokens"
    EXECUTOR_NAME = "analyze"

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        yield DataSourceInput(
            label="Data source",
            placeholder="Dataset name or JSONL file path...",
            hint="Enter a local dataset name or path to a JSONL file",
            required=True,
            input_id="source",
        )

        yield TokenizerSelect(
            label="Tokenizer",
            placeholder="gpt2",
            hint="HuggingFace tokenizer or custom trained",
            input_id="tokenizer",
        )

        yield LabeledInput(
            label="Subset size",
            placeholder="Leave empty for all",
            hint="Optional: analyze only first N examples",
            input_id="subset",
        )

        yield Checkbox("Show recommendations", id="recommendations")

    def _get_execute_label(self) -> str:
        return "Analyze"

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update command preview when inputs change."""
        self.update_command_preview()

    def on_data_source_input_changed(self, event: DataSourceInput.Changed) -> None:
        """Update command preview when source changes."""
        self.update_command_preview()

    def on_tokenizer_select_changed(self, event: TokenizerSelect.Changed) -> None:
        """Update command preview when tokenizer changes."""
        self.update_command_preview()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Update command preview when checkbox changes."""
        self.update_command_preview()

    def get_params(self) -> dict:
        """Build parameters for the executor."""
        params = {}

        try:
            source_input = self.query_one(DataSourceInput)
            source_value = source_input.get_value().strip()
            source_type = source_input.get_source_type()

            if source_type == SourceType.FILE:
                if source_value:
                    params["jsonl"] = source_value
            else:  # DATASET or UNKNOWN
                if source_value:
                    params["dataset"] = source_value

            # Tokenizer
            tokenizer_select = self.query_one(TokenizerSelect)
            tokenizer = tokenizer_select.get_value().strip()
            if tokenizer:
                params["tokenizer"] = tokenizer

            # Subset
            subset_input = self.query_one("#subset", Input)
            subset = subset_input.value.strip()
            if subset:
                params["subset"] = int(subset)

            # Recommendations
            rec_checkbox = self.query_one("#recommendations", Checkbox)
            if rec_checkbox.value:
                params["recommendations"] = True

        except Exception:
            pass

        return params

    def validate(self) -> tuple[bool, str]:
        """Validate the form before execution."""
        try:
            source_input = self.query_one(DataSourceInput)
            source_value = source_input.get_value().strip()

            if not source_value:
                return False, "Data source is required"

            # Validate subset is a positive integer if provided
            subset_input = self.query_one("#subset", Input)
            subset = subset_input.value.strip()
            if subset:
                try:
                    val = int(subset)
                    if val <= 0:
                        return False, "Subset size must be positive"
                except ValueError:
                    return False, "Subset size must be a valid integer"

        except Exception:
            return False, "Form not ready"

        return True, ""

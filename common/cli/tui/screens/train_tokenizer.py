"""Train tokenizer wizard screen."""

from textual.app import ComposeResult
from textual.widgets import Input, Select, Static

from common.data import discover_local_datasets
from ..widgets import DataSourceInput, LabeledInput, SourceType
from .wizard_base import WizardScreen


class TrainTokenizerScreen(WizardScreen):
    """Wizard for training custom BPE tokenizer."""

    TITLE = "Train Tokenizer"
    EXECUTOR_NAME = "train_tokenizer"

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        # Static Select for quick dataset selection (fallback)
        datasets = discover_local_datasets()
        if datasets:
            yield Static("Quick select dataset:", classes="form-label")
            yield Select(
                [(ds.split("/")[-1], ds) for ds in datasets],
                prompt="Choose a local dataset...",
                id="quick-select",
            )
            yield Static("[dim]Or type below for custom input:[/dim]")

        yield DataSourceInput(
            label="Training data source",
            placeholder="Dataset name or JSONL file path...",
            hint="Enter a local dataset name or path to a JSONL file",
            required=True,
            input_id="source",
        )

        yield LabeledInput(
            label="Vocabulary size",
            placeholder="32768",
            hint="Target vocabulary size for BPE tokenizer",
            required=True,
            input_id="vocab-size",
        )

        yield LabeledInput(
            label="Training subset",
            placeholder="Leave empty for all",
            hint="Optional: use only first N examples for training",
            input_id="train-subset",
        )

        yield LabeledInput(
            label="Output directory",
            placeholder="assets/models/tokenizers/",
            hint="Output directory for trained tokenizer",
            input_id="output",
        )

    def _get_execute_label(self) -> str:
        return "Train"

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update command preview when inputs change."""
        self.update_command_preview()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle quick select dropdown change."""
        if event.value and event.select.id == "quick-select":
            try:
                # Update the DataSourceInput with selected dataset
                source_input = self.query_one(DataSourceInput)
                source_input.set_value(str(event.value))
            except Exception:
                pass
        self.update_command_preview()

    def on_data_source_input_changed(self, event: DataSourceInput.Changed) -> None:
        """Update command preview when source changes."""
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

            # Vocab size (required for training)
            vocab_input = self.query_one("#vocab-size", Input)
            vocab = vocab_input.value.strip()
            if vocab:
                params["vocab_size"] = int(vocab)

            # Training subset
            subset_input = self.query_one("#train-subset", Input)
            subset = subset_input.value.strip()
            if subset:
                params["subset"] = int(subset)

            # Output directory
            output_input = self.query_one("#output", Input)
            output = output_input.value.strip()
            if output:
                params["output_dir"] = output

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

            # Validate vocab size is a positive integer
            vocab_input = self.query_one("#vocab-size", Input)
            vocab = vocab_input.value.strip()
            if not vocab:
                return False, "Vocabulary size is required"
            try:
                val = int(vocab)
                if val <= 0:
                    return False, "Vocabulary size must be positive"
            except ValueError:
                return False, "Vocabulary size must be a valid integer"

            # Validate train-subset is a positive integer if provided
            subset_input = self.query_one("#train-subset", Input)
            subset = subset_input.value.strip()
            if subset:
                try:
                    val = int(subset)
                    if val <= 0:
                        return False, "Training subset must be positive"
                except ValueError:
                    return False, "Training subset must be a valid integer"

        except Exception:
            return False, "Form not ready"

        return True, ""

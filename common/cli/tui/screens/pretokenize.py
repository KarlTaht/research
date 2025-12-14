"""Pretokenize wizard screen with step-by-step flow."""

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Input, Static

from ..widgets import DataSourceInput, FileSelect, LabeledInput, SourceType, TokenizerSelect
from .step_wizard_base import StepWizardScreen


class PretokenizeScreen(StepWizardScreen):
    """Step-by-step wizard for pre-tokenizing datasets."""

    TITLE = "Pre-tokenize Dataset"
    EXECUTOR_NAME = "pretokenize"

    # Step configuration for guided flow
    STEPS = [
        {"id": "source", "label": "Data source", "type": "select", "required": True},
        {"id": "tokenizer", "label": "Tokenizer", "type": "select", "required": False},
        {"id": "max-length", "label": "Max sequence length", "type": "input", "required": False},
        {"id": "max-tokens", "label": "Max total tokens", "type": "input", "required": False},
    ]

    def compose_form(self) -> ComposeResult:
        """Compose the step-based form fields."""
        # Step 1: Data source
        with Container(id="step-source", classes="step-field current"):
            yield Static("1. Data source [red]*[/red]", classes="step-label")
            yield DataSourceInput(
                label="",  # Label is in step-label
                placeholder="Dataset name or JSONL file path...",
                hint="Enter a local dataset name or path to a JSONL file",
                required=True,
                input_id="source",
            )

        # Extra JSONL input for validation file (shown when source type is file)
        with Container(id="jsonl-extras", classes="hidden"):
            yield FileSelect(
                label="Validation JSONL",
                placeholder="e.g., data/corpus/val.jsonl",
                hint="Optional: validation file",
                extensions=[".jsonl", ".json"],
                input_id="val-jsonl",
            )

        # Step 2: Tokenizer
        with Container(id="step-tokenizer", classes="step-field pending"):
            yield Static("2. Tokenizer", classes="step-label")
            yield TokenizerSelect(
                label="",  # Label is in step-label
                placeholder="gpt2",
                hint="HuggingFace tokenizer or custom trained",
                input_id="tokenizer",
            )

        # Step 3: Max sequence length
        with Container(id="step-max-length", classes="step-field pending"):
            yield Static("3. Max sequence length", classes="step-label")
            yield LabeledInput(
                label="",  # Label is in step-label
                placeholder="1024",
                hint="Maximum tokens per sequence (default: 1024)",
                input_id="max-length",
            )

        # Step 4: Max total tokens
        with Container(id="step-max-tokens", classes="step-field pending"):
            yield Static("4. Max total tokens", classes="step-label")
            yield LabeledInput(
                label="",  # Label is in step-label
                placeholder="Leave empty for all",
                hint="Optional: limit total tokens (e.g., 10000000 for 10M)",
                input_id="max-tokens",
            )

    def _get_execute_label(self) -> str:
        return "Pretokenize"

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update command preview when inputs change."""
        self.update_command_preview()

    def on_data_source_input_changed(self, event: DataSourceInput.Changed) -> None:
        """Update visibility and command preview when source changes."""
        self._update_source_visibility(event.source_type)
        self.update_command_preview()

    def on_data_source_input_selected(self, event) -> None:
        """Auto-advance when data source is selected from dropdown."""
        self.advance_step()

    def on_tokenizer_select_changed(self, event: TokenizerSelect.Changed) -> None:
        """Update command preview when tokenizer changes."""
        self.update_command_preview()

    def on_tokenizer_select_selected(self, event) -> None:
        """Auto-advance when tokenizer is selected from dropdown."""
        self.advance_step()

    def on_file_select_changed(self, event: FileSelect.Changed) -> None:
        """Update command preview when file selection changes."""
        self.update_command_preview()

    def _update_source_visibility(self, source_type: SourceType) -> None:
        """Show/hide extra inputs based on source type."""
        try:
            jsonl_extras = self.query_one("#jsonl-extras", Container)

            if source_type == SourceType.FILE:
                jsonl_extras.remove_class("hidden")
            else:
                jsonl_extras.add_class("hidden")
        except Exception:
            pass

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

                val_jsonl_select = self.query_one(FileSelect)
                val_jsonl = val_jsonl_select.get_value().strip()
                if val_jsonl:
                    params["val_jsonl"] = val_jsonl
            else:  # DATASET or UNKNOWN
                if source_value:
                    params["dataset"] = source_value

            # Tokenizer
            tokenizer_select = self.query_one(TokenizerSelect)
            tokenizer = tokenizer_select.get_value().strip()
            if tokenizer:
                params["tokenizer"] = tokenizer

            # Max length
            max_length_input = self.query_one("#max-length", Input)
            max_length = max_length_input.value.strip()
            if max_length:
                params["max_length"] = int(max_length)

            # Max tokens (new field)
            max_tokens_input = self.query_one("#max-tokens", Input)
            max_tokens = max_tokens_input.value.strip()
            if max_tokens:
                params["max_tokens"] = int(max_tokens)

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

            # Validate max-length is a positive integer if provided
            max_length_input = self.query_one("#max-length", Input)
            max_length = max_length_input.value.strip()
            if max_length:
                try:
                    val = int(max_length)
                    if val <= 0:
                        return False, "Max length must be positive"
                except ValueError:
                    return False, "Max length must be a valid integer"

            # Validate max-tokens is a positive integer if provided
            max_tokens_input = self.query_one("#max-tokens", Input)
            max_tokens = max_tokens_input.value.strip()
            if max_tokens:
                try:
                    val = int(max_tokens)
                    if val <= 0:
                        return False, "Max tokens must be positive"
                except ValueError:
                    return False, "Max tokens must be a valid integer"

        except Exception:
            return False, "Form not ready"

        return True, ""

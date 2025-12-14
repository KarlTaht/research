"""Pretokenize wizard screen."""

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Input, Select, Static

from ..widgets import CommandPreview, LabeledInput
from .wizard_base import WizardScreen


class PretokenizeScreen(WizardScreen):
    """Wizard for pre-tokenizing datasets."""

    TITLE = "Pre-tokenize Dataset"
    COMMAND_MODULE = "common.cli.data"

    SOURCE_OPTIONS = [
        ("Dataset (from registry)", "dataset"),
        ("JSONL file", "jsonl"),
    ]

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        yield Static("Source type:", classes="form-label")
        yield Select(
            [(label, value) for label, value in self.SOURCE_OPTIONS],
            value="dataset",
            id="source-type",
        )

        # Dataset input (shown when source-type is "dataset")
        with Container(id="dataset-container"):
            yield LabeledInput(
                label="Dataset name",
                placeholder="e.g., tinystories, squad",
                hint="Dataset name from registry",
                required=True,
                input_id="dataset",
            )

        # JSONL inputs (shown when source-type is "jsonl")
        with Container(id="jsonl-container", classes="hidden"):
            yield LabeledInput(
                label="JSONL file",
                placeholder="e.g., data/corpus/train.jsonl",
                hint="Path to JSONL file with {\"text\": \"...\"} format",
                required=True,
                input_id="jsonl",
            )
            yield LabeledInput(
                label="Validation JSONL",
                placeholder="e.g., data/corpus/val.jsonl",
                hint="Optional: validation file",
                input_id="val-jsonl",
            )

        yield LabeledInput(
            label="Tokenizer",
            placeholder="gpt2",
            hint="Tokenizer name or path (default: gpt2)",
            input_id="tokenizer",
        )

        yield LabeledInput(
            label="Max sequence length",
            placeholder="1024",
            hint="Maximum sequence length (default: 1024)",
            input_id="max-length",
        )

    def compose_command_preview(self) -> ComposeResult:
        """Compose the command preview."""
        yield CommandPreview(
            module=self.COMMAND_MODULE,
            initial_args=self.get_command_args(),
            id="preview",
        )

    def _get_execute_label(self) -> str:
        return "Pretokenize"

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update command preview when inputs change."""
        self._update_preview()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update visibility and command preview when source type changes."""
        self._update_source_visibility()
        self._update_preview()

    def _update_source_visibility(self) -> None:
        """Show/hide inputs based on source type selection."""
        try:
            source_select = self.query_one("#source-type", Select)
            dataset_container = self.query_one("#dataset-container", Container)
            jsonl_container = self.query_one("#jsonl-container", Container)

            if source_select.value == "dataset":
                dataset_container.remove_class("hidden")
                jsonl_container.add_class("hidden")
            else:
                dataset_container.add_class("hidden")
                jsonl_container.remove_class("hidden")
        except Exception:
            pass

    def _update_preview(self) -> None:
        """Update the command preview widget."""
        try:
            preview = self.query_one("#preview", CommandPreview)
            preview.update_args(self.get_command_args())
        except Exception:
            pass

    def get_command_args(self) -> list[str]:
        """Build CLI arguments from form state."""
        args = ["pretokenize"]

        try:
            source_select = self.query_one("#source-type", Select)

            if source_select.value == "dataset":
                dataset_input = self.query_one("#dataset", Input)
                dataset = dataset_input.value.strip()
                if dataset:
                    args.extend(["--dataset", dataset])
            else:
                jsonl_input = self.query_one("#jsonl", Input)
                jsonl = jsonl_input.value.strip()
                if jsonl:
                    args.extend(["--jsonl", jsonl])

                val_jsonl_input = self.query_one("#val-jsonl", Input)
                val_jsonl = val_jsonl_input.value.strip()
                if val_jsonl:
                    args.extend(["--val-jsonl", val_jsonl])

            # Tokenizer
            tokenizer_input = self.query_one("#tokenizer", Input)
            tokenizer = tokenizer_input.value.strip()
            if tokenizer:
                args.extend(["--tokenizer", tokenizer])

            # Max length
            max_length_input = self.query_one("#max-length", Input)
            max_length = max_length_input.value.strip()
            if max_length:
                args.extend(["--max-length", max_length])

        except Exception:
            pass

        return args

    def validate(self) -> tuple[bool, str]:
        """Validate the form before execution."""
        try:
            source_select = self.query_one("#source-type", Select)

            if source_select.value == "dataset":
                dataset_input = self.query_one("#dataset", Input)
                if not dataset_input.value.strip():
                    return False, "Dataset name is required"
            else:
                jsonl_input = self.query_one("#jsonl", Input)
                if not jsonl_input.value.strip():
                    return False, "JSONL file path is required"

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

        except Exception:
            return False, "Form not ready"

        return True, ""

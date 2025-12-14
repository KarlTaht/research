"""Train tokenizer wizard screen."""

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Input, Select, Static

from ..widgets import CommandPreview, LabeledInput
from .wizard_base import WizardScreen


class TrainTokenizerScreen(WizardScreen):
    """Wizard for training custom BPE tokenizer."""

    TITLE = "Train Tokenizer"
    COMMAND_MODULE = "common.cli.data"

    SOURCE_OPTIONS = [
        ("JSONL file", "jsonl"),
        ("Dataset (from registry)", "dataset"),
    ]

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        yield Static("Source type:", classes="form-label")
        yield Select(
            [(label, value) for label, value in self.SOURCE_OPTIONS],
            value="jsonl",
            id="source-type",
        )

        # JSONL input (shown when source-type is "jsonl")
        with Container(id="jsonl-container"):
            yield LabeledInput(
                label="JSONL file",
                placeholder="e.g., data/corpus/train.jsonl",
                hint="Path to JSONL file with {\"text\": \"...\"} format",
                required=True,
                input_id="jsonl",
            )

        # Dataset input (shown when source-type is "dataset")
        with Container(id="dataset-container", classes="hidden"):
            yield LabeledInput(
                label="Dataset name",
                placeholder="e.g., tinystories, squad",
                hint="Dataset name from registry",
                required=True,
                input_id="dataset",
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

    def compose_command_preview(self) -> ComposeResult:
        """Compose the command preview."""
        yield CommandPreview(
            module=self.COMMAND_MODULE,
            initial_args=self.get_command_args(),
            id="preview",
        )

    def _get_execute_label(self) -> str:
        return "Train"

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

            if source_select.value == "jsonl":
                jsonl_container.remove_class("hidden")
                dataset_container.add_class("hidden")
            else:
                jsonl_container.add_class("hidden")
                dataset_container.remove_class("hidden")
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
        args = ["analyze"]

        try:
            source_select = self.query_one("#source-type", Select)

            if source_select.value == "jsonl":
                jsonl_input = self.query_one("#jsonl", Input)
                jsonl = jsonl_input.value.strip()
                if jsonl:
                    args.extend(["--jsonl", jsonl])
            else:
                dataset_input = self.query_one("#dataset", Input)
                dataset = dataset_input.value.strip()
                if dataset:
                    args.extend(["--dataset", dataset])

            # Vocab size (required for training)
            vocab_input = self.query_one("#vocab-size", Input)
            vocab = vocab_input.value.strip()
            if vocab:
                args.extend(["--train-tokenizer", vocab])

            # Training subset
            subset_input = self.query_one("#train-subset", Input)
            subset = subset_input.value.strip()
            if subset:
                args.extend(["--train-subset", subset])

            # Output directory
            output_input = self.query_one("#output", Input)
            output = output_input.value.strip()
            if output:
                args.extend(["--tokenizer-output", output])

        except Exception:
            pass

        return args

    def validate(self) -> tuple[bool, str]:
        """Validate the form before execution."""
        try:
            source_select = self.query_one("#source-type", Select)

            if source_select.value == "jsonl":
                jsonl_input = self.query_one("#jsonl", Input)
                if not jsonl_input.value.strip():
                    return False, "JSONL file path is required"
            else:
                dataset_input = self.query_one("#dataset", Input)
                if not dataset_input.value.strip():
                    return False, "Dataset name is required"

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

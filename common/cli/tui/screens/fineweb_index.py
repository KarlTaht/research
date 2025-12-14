"""FineWeb index wizard screen."""

from textual.app import ComposeResult
from textual.widgets import Checkbox, Input

from ..widgets import CommandPreview, LabeledInput
from .wizard_base import WizardScreen


class FinewebIndexScreen(WizardScreen):
    """Wizard for building FineWeb domain index."""

    TITLE = "Build FineWeb Index"
    COMMAND_MODULE = "common.cli.data"

    def compose_form(self) -> ComposeResult:
        """Compose the form fields."""
        yield LabeledInput(
            label="Start shard",
            placeholder="0",
            hint="Starting shard index (for incremental runs)",
            input_id="start-shard",
        )

        yield LabeledInput(
            label="Number of shards",
            placeholder="Leave empty for all remaining",
            hint="Number of shards to process",
            input_id="num-shards",
        )

        yield Checkbox("Status only (show progress without building)", id="status")

    def compose_command_preview(self) -> ComposeResult:
        """Compose the command preview."""
        yield CommandPreview(
            module=self.COMMAND_MODULE,
            initial_args=self.get_command_args(),
            id="preview",
        )

    def _get_execute_label(self) -> str:
        return "Build Index"

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update command preview when inputs change."""
        self._update_preview()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Update command preview when checkbox changes."""
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
        args = ["fineweb", "index"]

        try:
            # Status flag
            status_checkbox = self.query_one("#status", Checkbox)
            if status_checkbox.value:
                args.append("--status")
            else:
                # Only include shard args if not status-only
                start_input = self.query_one("#start-shard", Input)
                start = start_input.value.strip()
                if start:
                    args.extend(["--start-shard", start])

                num_input = self.query_one("#num-shards", Input)
                num = num_input.value.strip()
                if num:
                    args.extend(["--num-shards", num])

        except Exception:
            pass

        return args

    def validate(self) -> tuple[bool, str]:
        """Validate the form before execution."""
        try:
            # Validate start-shard is a non-negative integer if provided
            start_input = self.query_one("#start-shard", Input)
            start = start_input.value.strip()
            if start:
                try:
                    val = int(start)
                    if val < 0:
                        return False, "Start shard must be non-negative"
                except ValueError:
                    return False, "Start shard must be a valid integer"

            # Validate num-shards is a positive integer if provided
            num_input = self.query_one("#num-shards", Input)
            num = num_input.value.strip()
            if num:
                try:
                    val = int(num)
                    if val <= 0:
                        return False, "Number of shards must be positive"
                except ValueError:
                    return False, "Number of shards must be a valid integer"

        except Exception:
            return False, "Form not ready"

        return True, ""

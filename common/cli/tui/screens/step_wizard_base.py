"""Base class for step-based wizard screens."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Input, Static

from .wizard_base import WizardScreen


class StepWizardScreen(WizardScreen):
    """Base class for step-by-step wizard screens.

    Shows all fields but highlights/enables current step.
    Progression happens automatically for selections, via Enter for inputs.

    Subclasses should:
    1. Set TITLE, EXECUTOR_NAME, and STEPS class attributes
    2. Override compose_form() to add form fields wrapped in step containers
    3. Override get_params() to return parameters for the executor
    4. Optionally override validate() for form validation

    Example STEPS configuration:
        STEPS = [
            {"id": "source", "label": "Data source", "type": "select", "required": True},
            {"id": "tokenizer", "label": "Tokenizer", "type": "select", "required": False},
            {"id": "max-length", "label": "Max sequence length", "type": "input", "required": False},
        ]

    Step types:
        - "select": Auto-advances on selection (e.g., from DataSourceInput, TokenizerSelect)
        - "input": Advances on Enter key (e.g., from LabeledInput)
    """

    # Step configuration - subclasses must define this
    STEPS: list[dict] = []

    # Current step index (0-based)
    current_step: reactive[int] = reactive(0)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._step_count = len(self.STEPS)

    def on_mount(self) -> None:
        """Initialize step highlighting on mount."""
        super().on_mount()
        self._update_step_highlighting()

    def _get_step_container(self, step_index: int) -> Container | None:
        """Get the container for a specific step by index."""
        if step_index < 0 or step_index >= self._step_count:
            return None
        step_id = self.STEPS[step_index]["id"]
        try:
            return self.query_one(f"#step-{step_id}", Container)
        except Exception:
            return None

    def _update_step_highlighting(self) -> None:
        """Update visual highlighting for all steps based on current_step."""
        for i, step in enumerate(self.STEPS):
            container = self._get_step_container(i)
            if container is None:
                continue

            # Remove all state classes
            container.remove_class("current", "completed", "pending")

            if i < self.current_step:
                container.add_class("completed")
            elif i == self.current_step:
                container.add_class("current")
            else:
                container.add_class("pending")

        # Update step indicator if present
        try:
            indicator = self.query_one("#step-indicator", Static)
            indicator.update(f"Step {self.current_step + 1} of {self._step_count}")
        except Exception:
            pass

    def watch_current_step(self, new_step: int) -> None:
        """React to step changes."""
        self._update_step_highlighting()
        self._focus_current_step()

    def _focus_current_step(self) -> None:
        """Focus the first focusable widget in the current step."""
        container = self._get_step_container(self.current_step)
        if container is None:
            return

        # Find focusable widgets in this step
        for widget in container.query("Input, Select, Checkbox").results():
            if not widget.has_class("hidden") and widget.display:
                widget.focus()
                return

    def advance_step(self) -> None:
        """Move to the next step."""
        if self.current_step < self._step_count - 1:
            self.current_step += 1
        else:
            # Last step completed - focus execute button
            try:
                self.query_one("#execute-btn", Button).focus()
            except Exception:
                pass

    def go_to_step(self, step_index: int) -> None:
        """Go to a specific step (for clicking on previous steps)."""
        if 0 <= step_index < self._step_count:
            self.current_step = step_index

    def is_step_complete(self, step_index: int) -> bool:
        """Check if a step has been completed (has a value)."""
        if step_index < 0 or step_index >= self._step_count:
            return False

        step = self.STEPS[step_index]
        step_id = step["id"]

        try:
            # Try to find the input widget for this step
            input_widget = self.query_one(f"#{step_id}", Input)
            value = input_widget.value.strip()
            return bool(value) or not step.get("required", False)
        except Exception:
            pass

        # For custom widgets, check by querying the step container
        container = self._get_step_container(step_index)
        if container:
            for input_widget in container.query(Input).results():
                if input_widget.value.strip():
                    return True
        return not step.get("required", False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input fields - advance to next step."""
        # Find which step this input belongs to
        for i, step in enumerate(self.STEPS):
            if step["type"] == "input":
                try:
                    container = self._get_step_container(i)
                    if container and event.input in container.query(Input).results():
                        if i == self.current_step:
                            self.advance_step()
                        return
                except Exception:
                    pass

    def compose(self) -> ComposeResult:
        """Compose the wizard screen layout with step indicator."""
        yield Header()
        with Container(classes="wizard-container"):
            yield Static(f"[bold]{self.TITLE}[/bold]", classes="wizard-title")
            yield Static(
                f"Step 1 of {self._step_count}",
                id="step-indicator",
                classes="step-indicator"
            )
            with Vertical(classes="wizard-form"):
                yield from self.compose_form()
            yield from self.compose_command_preview()
            yield from self.compose_buttons()
        yield Footer()

    def _get_execute_label(self) -> str:
        """Get the label for the execute button."""
        return "Execute"

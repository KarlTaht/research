"""Base class for wizard screens."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Header, Input, Select, Static


class WizardScreen(Screen):
    """Base class for all wizard screens.

    Subclasses should:
    1. Set TITLE and EXECUTOR_NAME class attributes
    2. Override compose_form() to add form fields
    3. Override get_params() to return parameters for the executor
    4. Optionally override validate() for form validation
    """

    TITLE: str = "Wizard"
    EXECUTOR_NAME: str = ""  # e.g., "download_dataset"

    # Define layers for dropdown overlays (higher index = renders on top)
    LAYERS = ("below", "default", "above")

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+enter", "execute", "Run", show=True),
        Binding("down", "focus_next_field", "Next", show=False),
        Binding("up", "focus_prev_field", "Prev", show=False),
        Binding("left", "focus_prev_button", show=False),
        Binding("right", "focus_next_button", show=False),
    ]

    def on_mount(self) -> None:
        """Focus the first input on mount."""
        focusables = self._get_focusable_widgets()
        if focusables:
            focusables[0].focus()

    def _get_focusable_widgets(self) -> list:
        """Get all focusable widgets in order (inputs, selects, checkboxes, buttons)."""
        focusables = []
        # Form elements first
        for widget in self.query("Input, Select, Checkbox").results():
            # Skip hidden widgets
            if not widget.has_class("hidden") and widget.display:
                parent = widget.parent
                while parent:
                    if hasattr(parent, "has_class") and parent.has_class("hidden"):
                        break
                    if hasattr(parent, "display") and not parent.display:
                        break
                    parent = parent.parent
                else:
                    focusables.append(widget)
        # Then buttons
        focusables.extend(self.query(Button).results())
        return focusables

    def action_focus_next_field(self) -> None:
        """Focus the next focusable widget."""
        focusables = self._get_focusable_widgets()
        if not focusables:
            return
        focused = self.focused
        if focused in focusables:
            idx = focusables.index(focused)
            next_idx = (idx + 1) % len(focusables)
            focusables[next_idx].focus()
        elif focusables:
            focusables[0].focus()

    def action_focus_prev_field(self) -> None:
        """Focus the previous focusable widget."""
        focusables = self._get_focusable_widgets()
        if not focusables:
            return
        focused = self.focused
        if focused in focusables:
            idx = focusables.index(focused)
            prev_idx = (idx - 1) % len(focusables)
            focusables[prev_idx].focus()
        elif focusables:
            focusables[-1].focus()

    def action_focus_next_button(self) -> None:
        """Focus the next button (for left/right navigation in button bar)."""
        buttons = list(self.query(Button).results())
        if not buttons:
            return
        focused = self.focused
        if focused in buttons:
            idx = buttons.index(focused)
            next_idx = (idx + 1) % len(buttons)
            buttons[next_idx].focus()

    def action_focus_prev_button(self) -> None:
        """Focus the previous button (for left/right navigation in button bar)."""
        buttons = list(self.query(Button).results())
        if not buttons:
            return
        focused = self.focused
        if focused in buttons:
            idx = buttons.index(focused)
            prev_idx = (idx - 1) % len(buttons)
            buttons[prev_idx].focus()

    def compose(self) -> ComposeResult:
        """Compose the wizard screen layout."""
        yield Header()
        with Container(classes="wizard-container"):
            yield Static(f"[bold]{self.TITLE}[/bold]", classes="wizard-title")
            with Vertical(classes="wizard-form"):
                yield from self.compose_form()
            yield from self.compose_command_preview()
            yield from self.compose_buttons()
        yield Footer()

    def compose_form(self) -> ComposeResult:
        """Override to compose form fields.

        Example:
            yield Static("Dataset name:", classes="form-label")
            yield Input(placeholder="e.g., squad", id="dataset-name")
        """
        yield Static("[dim]No form fields defined[/dim]")

    def compose_command_preview(self) -> ComposeResult:
        """Compose the command preview section."""
        with Container(classes="command-preview"):
            yield Static("Command:", classes="command-preview-label")
            yield Static(
                self._build_command_string(),
                id="command-text",
                classes="command-preview-text",
            )

    def compose_buttons(self) -> ComposeResult:
        """Compose the action buttons."""
        with Container(classes="button-bar"):
            yield Button("Cancel", variant="default", id="cancel-btn")
            yield Button(self._get_execute_label(), variant="primary", id="execute-btn")

    def _get_execute_label(self) -> str:
        """Get the label for the execute button. Override for custom labels."""
        return "Execute"

    def _build_command_string(self) -> str:
        """Build the command string for preview."""
        params = self.get_params()
        if not self.EXECUTOR_NAME:
            return "[dim]No executor configured[/dim]"
        # Show a preview of the operation
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items() if v)
        return f"[cyan]{self.EXECUTOR_NAME}[/cyan]({params_str})"

    def update_command_preview(self) -> None:
        """Update the command preview text."""
        try:
            preview = self.query_one("#command-text", Static)
            preview.update(self._build_command_string())
        except Exception:
            pass  # Widget not mounted yet

    def get_params(self) -> dict:
        """Return parameters for the executor function.

        Override in subclasses to return the appropriate parameters.

        Returns:
            Dictionary of parameters to pass to the executor.
        """
        return {}

    def validate(self) -> tuple[bool, str]:
        """Validate form before execution.

        Override in subclasses for custom validation.

        Returns:
            Tuple of (is_valid, error_message).
        """
        return True, ""

    def action_cancel(self) -> None:
        """Handle cancel action."""
        self.app.pop_screen()

    def action_execute(self) -> None:
        """Handle execute action."""
        self.execute()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "execute-btn":
            self.execute()

    def execute(self) -> None:
        """Validate and execute the command."""
        is_valid, error_msg = self.validate()
        if not is_valid:
            self.notify(error_msg, severity="error")
            return

        params = self.get_params()
        # Import here to avoid circular imports
        from .output import OutputScreen

        self.app.push_screen(OutputScreen(self.EXECUTOR_NAME, params))

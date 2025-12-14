"""Labeled input widget with optional validation message."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, Static


class LabeledInput(Vertical):
    """An input field with a label and optional hint/error message."""

    DEFAULT_CSS = """
    LabeledInput {
        height: auto;
        margin-bottom: 1;
    }

    LabeledInput .input-label {
        margin-bottom: 0;
    }

    LabeledInput .input-hint {
        color: #555c65;
        margin-top: 0;
    }

    LabeledInput .input-error {
        color: #F25E86;
        margin-top: 0;
    }
    """

    value: reactive[str] = reactive("")
    error: reactive[str] = reactive("")

    class Changed(Message):
        """Message sent when input value changes."""

        def __init__(self, labeled_input: "LabeledInput", value: str) -> None:
            super().__init__()
            self.labeled_input = labeled_input
            self.value = value

    def __init__(
        self,
        label: str,
        placeholder: str = "",
        hint: str = "",
        required: bool = False,
        input_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.placeholder = placeholder
        self.hint = hint
        self.required = required
        self.input_id = input_id or f"input-{id(self)}"

    def compose(self) -> ComposeResult:
        label_text = self.label
        if self.required:
            label_text += " [red]*[/red]"
        yield Static(label_text, classes="input-label")
        yield Input(placeholder=self.placeholder, id=self.input_id)
        if self.hint:
            yield Static(f"[dim]{self.hint}[/dim]", id="hint", classes="input-hint")
        yield Static("", id="error", classes="input-error")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        self.value = event.value
        self.post_message(self.Changed(self, event.value))

    def watch_error(self, error: str) -> None:
        """Update error message display."""
        try:
            error_widget = self.query_one("#error", Static)
            error_widget.update(error)
        except Exception:
            pass

    def set_error(self, message: str) -> None:
        """Set an error message."""
        self.error = message
        try:
            input_widget = self.query_one(Input)
            input_widget.add_class("-invalid")
        except Exception:
            pass

    def clear_error(self) -> None:
        """Clear the error message."""
        self.error = ""
        try:
            input_widget = self.query_one(Input)
            input_widget.remove_class("-invalid")
        except Exception:
            pass

    def get_value(self) -> str:
        """Get the current input value."""
        try:
            return self.query_one(Input).value
        except Exception:
            return self.value

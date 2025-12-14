"""Tokenizer selection widget with autocomplete for common and custom tokenizers."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click, Focus
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from common.data import get_default_assets_dir
from .fuzzy import fuzzy_filter


# Common tokenizers that are frequently used
COMMON_TOKENIZERS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "bert-base-uncased",
    "bert-base-cased",
    "bert-large-uncased",
    "roberta-base",
    "roberta-large",
    "t5-small",
    "t5-base",
    "facebook/opt-125m",
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/pythia-70m",
    "meta-llama/Llama-2-7b-hf",
]


def discover_local_tokenizers(assets_dir: Path | None = None) -> list[str]:
    """Discover custom trained tokenizers from assets/models/tokenizers/.

    Returns:
        List of tokenizer paths relative to assets directory.
    """
    if assets_dir is None:
        assets_dir = get_default_assets_dir()

    tokenizers_dir = assets_dir / "models" / "tokenizers"
    if not tokenizers_dir.exists():
        return []

    tokenizers = []
    # Look for directories containing tokenizer files
    for item in tokenizers_dir.iterdir():
        if item.is_dir():
            # Check if it looks like a tokenizer directory
            if (item / "tokenizer.json").exists() or (item / "vocab.json").exists():
                tokenizers.append(str(item))
    return sorted(tokenizers)


class TokenizerSelect(Vertical):
    """An autocomplete widget for selecting tokenizers.

    Shows common tokenizers and discovers custom trained tokenizers.
    """

    DEFAULT_CSS = """
    TokenizerSelect {
        height: auto;
        margin-bottom: 1;
    }

    TokenizerSelect .tokenizer-label {
        margin-bottom: 0;
    }

    TokenizerSelect .tokenizer-hint {
        color: #555c65;
        margin-top: 0;
    }

    TokenizerSelect .tokenizer-error {
        color: #F25E86;
        margin-top: 0;
    }

    TokenizerSelect OptionList {
        max-height: 8;
        background: #141a26;
        layer: above;
        display: none;
    }

    TokenizerSelect OptionList.visible {
        display: block;
        height: auto;
        border: solid #555c65;
    }

    TokenizerSelect OptionList:focus {
        border: solid #78DCE8;
    }
    """

    value: reactive[str] = reactive("")
    error: reactive[str] = reactive("")

    class Changed(Message):
        """Message sent when tokenizer selection changes."""

        def __init__(self, widget: "TokenizerSelect", value: str) -> None:
            super().__init__()
            self.tokenizer_select = widget
            self.value = value

    class Selected(Message):
        """Message sent when a tokenizer is explicitly selected from the list."""

        def __init__(self, widget: "TokenizerSelect", value: str) -> None:
            super().__init__()
            self.tokenizer_select = widget
            self.value = value

    def __init__(
        self,
        label: str = "Tokenizer",
        placeholder: str = "gpt2",
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
        self.input_id = input_id or f"tokenizer-input-{id(self)}"
        self._all_tokenizers: list[str] = []
        self._selecting = False

    def compose(self) -> ComposeResult:
        label_text = self.label
        if self.required:
            label_text += " [red]*[/red]"
        yield Static(label_text, classes="tokenizer-label")
        yield Input(placeholder=self.placeholder, id=self.input_id)
        yield OptionList(id="tokenizer-options")
        if self.hint:
            yield Static(f"[dim]{self.hint}[/dim]", id="hint", classes="tokenizer-hint")
        yield Static("", id="error", classes="tokenizer-error")

    def on_mount(self) -> None:
        """Load available tokenizers on mount."""
        self._refresh_tokenizers()

    def _refresh_tokenizers(self) -> None:
        """Refresh the list of available tokenizers."""
        # Start with common tokenizers
        self._all_tokenizers = list(COMMON_TOKENIZERS)

        # Add custom trained tokenizers
        custom = discover_local_tokenizers()
        for t in custom:
            if t not in self._all_tokenizers:
                self._all_tokenizers.append(t)

        self._update_options("")

    def _update_options(self, filter_text: str, show_all: bool = False) -> None:
        """Update the options list based on filter text (fuzzy search)."""
        try:
            option_list = self.query_one("#tokenizer-options", OptionList)
            option_list.clear_options()

            if filter_text:
                # Fuzzy filter by typed text
                matching = fuzzy_filter(filter_text, self._all_tokenizers, limit=10)
            elif show_all:
                matching = self._all_tokenizers[:10]
            else:
                matching = []

            if matching:
                for t in matching:
                    # Add label for custom tokenizers
                    if t.startswith("/") or t.startswith("assets"):
                        label = f"{Path(t).name} [dim](custom)[/dim]"
                    else:
                        label = t
                    option_list.add_option(Option(label, id=t))
                option_list.add_class("visible")
            else:
                option_list.remove_class("visible")
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if self._selecting:
            return

        self.value = event.value
        self._update_options(event.value)
        self.post_message(self.Changed(self, event.value))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key."""
        try:
            option_list = self.query_one("#tokenizer-options", OptionList)
            if option_list.option_count > 0:
                first_option = option_list.get_option_at_index(0)
                self._select_tokenizer(first_option.id)
            else:
                option_list.remove_class("visible")
                option_list.highlighted = None  # Reset cursor
                self.post_message(self.Selected(self, self.value))
        except Exception:
            pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection from the list."""
        event.stop()  # Prevent event bubbling
        self._select_tokenizer(event.option_id)

    def _select_tokenizer(self, tokenizer_id: str) -> None:
        """Select a tokenizer and update the input."""
        self._selecting = True
        try:
            input_widget = self.query_one(f"#{self.input_id}", Input)
            input_widget.value = tokenizer_id
            self.value = tokenizer_id

            option_list = self.query_one("#tokenizer-options", OptionList)
            option_list.remove_class("visible")
            option_list.highlighted = None  # Reset cursor

            self.post_message(self.Selected(self, tokenizer_id))
            self.post_message(self.Changed(self, tokenizer_id))
        except Exception:
            pass
        finally:
            self._selecting = False

    def on_focus(self, event: Focus) -> None:
        """When widget gets focus, show options."""
        try:
            input_widget = self.query_one(f"#{self.input_id}", Input)
            if event.widget == self:
                input_widget.focus()
            if not input_widget.value.strip():
                self._update_options("", show_all=True)
        except Exception:
            pass

    def on_click(self, event: Click) -> None:
        """Handle clicks on the widget to show options."""
        try:
            input_widget = self.query_one(f"#{self.input_id}", Input)
            input_widget.focus()
            # Always show options on click
            current_value = input_widget.value.strip()
            if current_value:
                self._update_options(current_value)
            else:
                self._update_options("", show_all=True)
        except Exception:
            pass

    def on_blur(self) -> None:
        """Hide options when focus leaves the widget."""
        self.set_timer(0.2, self._maybe_hide_options)

    def _maybe_hide_options(self) -> None:
        """Hide options if neither input nor option list has focus."""
        # Don't hide if we're in the middle of selecting
        if self._selecting:
            return
        try:
            input_focused = self.query_one(f"#{self.input_id}", Input).has_focus
            options = self.query_one("#tokenizer-options", OptionList)
            options_focused = options.has_focus

            if not input_focused and not options_focused:
                options.remove_class("visible")
                options.highlighted = None  # Reset cursor
        except Exception:
            pass

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
            input_widget = self.query_one(f"#{self.input_id}", Input)
            input_widget.add_class("-invalid")
        except Exception:
            pass

    def clear_error(self) -> None:
        """Clear the error message."""
        self.error = ""
        try:
            input_widget = self.query_one(f"#{self.input_id}", Input)
            input_widget.remove_class("-invalid")
        except Exception:
            pass

    def get_value(self) -> str:
        """Get the current input value."""
        try:
            return self.query_one(f"#{self.input_id}", Input).value
        except Exception:
            return self.value

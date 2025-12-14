"""Dataset selection widget with autocomplete from discovered local datasets."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click, Focus
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from common.data import discover_local_datasets
from .autocomplete_base import AutocompleteMixin
from .fuzzy import fuzzy_filter


class DatasetSelect(AutocompleteMixin, Vertical):
    """An autocomplete widget for selecting datasets from local storage or entering custom paths.

    Uses AutocompleteMixin for reliable dropdown behavior.
    """

    DEFAULT_CSS = """
    DatasetSelect {
        height: auto;
        margin-bottom: 1;
    }

    DatasetSelect .dataset-label {
        margin-bottom: 0;
    }

    DatasetSelect .dataset-hint {
        color: #555c65;
        margin-top: 0;
    }

    DatasetSelect .dataset-error {
        color: #F25E86;
        margin-top: 0;
    }

    DatasetSelect OptionList {
        max-height: 8;
        background: #141a26;
        layer: above;
        display: none;
    }

    DatasetSelect OptionList.visible {
        display: block;
        height: auto;
        border: solid #555c65;
    }

    DatasetSelect OptionList:focus {
        border: solid #78DCE8;
    }
    """

    value: reactive[str] = reactive("")
    error: reactive[str] = reactive("")

    class Changed(Message):
        """Message sent when dataset selection changes."""

        def __init__(self, widget: "DatasetSelect", value: str) -> None:
            super().__init__()
            self.dataset_select = widget
            self.value = value

    class Selected(Message):
        """Message sent when a dataset is explicitly selected from the list."""

        def __init__(self, widget: "DatasetSelect", value: str) -> None:
            super().__init__()
            self.dataset_select = widget
            self.value = value

    def __init__(
        self,
        label: str = "Dataset",
        placeholder: str = "Start typing or select from list...",
        hint: str = "",
        required: bool = False,
        input_id: str | None = None,
        allow_custom: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._init_autocomplete()  # Initialize mixin state
        self.label = label
        self.placeholder = placeholder
        self.hint = hint
        self.required = required
        self.input_id = input_id or f"dataset-input-{id(self)}"
        self.allow_custom = allow_custom
        self._all_datasets: list[str] = []

    def compose(self) -> ComposeResult:
        label_text = self.label
        if self.required:
            label_text += " [red]*[/red]"
        yield Static(label_text, classes="dataset-label")
        yield Input(placeholder=self.placeholder, id=self.input_id)
        yield OptionList(id="dataset-options")
        if self.hint:
            yield Static(f"[dim]{self.hint}[/dim]", id="hint", classes="dataset-hint")
        yield Static("", id="error", classes="dataset-error")

    def on_mount(self) -> None:
        """Load available datasets on mount."""
        self._refresh_datasets()

    def _get_input_widget(self) -> Input:
        """Return the Input widget for the mixin."""
        return self.query_one(f"#{self.input_id}", Input)

    def _get_option_list(self) -> OptionList:
        """Return the OptionList widget for the mixin."""
        return self.query_one("#dataset-options", OptionList)

    def _refresh_datasets(self) -> None:
        """Refresh the list of available datasets."""
        self._all_datasets = discover_local_datasets()
        self._update_options("")

    def _update_options(self, filter_text: str, show_all: bool = False) -> None:
        """Update the options list based on filter text (fuzzy search).

        Args:
            filter_text: Text to filter datasets by
            show_all: If True, show all datasets when filter is empty
        """
        try:
            option_list = self._get_option_list()
            option_list.clear_options()

            if filter_text:
                # Fuzzy filter by typed text
                matching = fuzzy_filter(filter_text, self._all_datasets, limit=10)
            elif show_all:
                # Show all datasets when focused with empty input
                matching = self._all_datasets[:10]
            else:
                matching = []

            # Show options if there are matches
            if matching:
                for ds in matching:
                    option_list.add_option(Option(ds, id=ds))
                self._show_dropdown()
            else:
                self._hide_dropdown()
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes - filter options and emit change message."""
        if self._selecting:
            return

        self.value = event.value
        self._update_options(event.value)
        self.post_message(self.Changed(self, event.value))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key - select first matching option or use current value."""
        try:
            option_list = self._get_option_list()
            if option_list.option_count > 0:
                # Select the first option
                first_option = option_list.get_option_at_index(0)
                self._select_dataset(first_option.id)
            else:
                # Use current value as-is (custom dataset)
                self._hide_dropdown()
                self.post_message(self.Selected(self, self.value))
        except Exception:
            pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection from the list."""
        event.stop()  # Prevent event bubbling
        self._select_dataset(event.option_id)

    def _select_dataset(self, dataset_id: str) -> None:
        """Select a dataset and update the input."""

        def on_selected(value: str) -> None:
            """Callback after selection is complete."""
            self.value = value
            self.post_message(self.Selected(self, value))
            self.post_message(self.Changed(self, value))

        # Use mixin's selection handling for reliable dropdown behavior
        self._select_value(dataset_id, on_selected)

    def on_focus(self, event: Focus) -> None:
        """When widget or its children get focus, show options."""
        # Don't show dropdown if we just selected (cooldown period)
        if not self._should_show_on_focus():
            return

        try:
            input_widget = self._get_input_widget()
            # Focus input if widget itself is focused
            if event.widget == self:
                input_widget.focus()
            # Show all options when input is empty
            if not input_widget.value.strip():
                self._update_options("", show_all=True)
        except Exception:
            pass

    def on_click(self, event: Click) -> None:
        """Handle clicks on the widget to show options."""
        # Don't show dropdown if we just selected (cooldown period)
        if not self._should_show_on_focus():
            return

        try:
            input_widget = self._get_input_widget()
            # Focus the input
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
        # Use mixin's delayed hide with cooldown check
        self._maybe_hide_after_blur(200)

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
            self._get_input_widget().add_class("-invalid")
        except Exception:
            pass

    def clear_error(self) -> None:
        """Clear the error message."""
        self.error = ""
        try:
            self._get_input_widget().remove_class("-invalid")
        except Exception:
            pass

    def get_value(self) -> str:
        """Get the current input value."""
        try:
            return self._get_input_widget().value
        except Exception:
            return self.value

    def is_local_dataset(self) -> bool:
        """Check if current value matches a local dataset."""
        return self.value in self._all_datasets

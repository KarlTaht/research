"""Smart data source input with auto-detection and optional source type toggle."""

from enum import Enum
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Click, Focus
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, OptionList, RadioButton, RadioSet, Static
from textual.widgets.option_list import Option

from common.data import discover_local_datasets
from .fuzzy import fuzzy_filter


class SourceType(Enum):
    """Enumeration of supported data source types."""
    DATASET = "dataset"
    FILE = "file"
    UNKNOWN = "unknown"


class DataSourceInput(Vertical):
    """Smart data source input with auto-detection of source type.

    Features:
    - Auto-detects whether input is a dataset name or file path
    - Shows a toggle for source type (grayed out when only one option)
    - Displays detected type badge next to input
    - Pre-selects source type based on input pattern
    """

    DEFAULT_CSS = """
    DataSourceInput {
        height: auto;
        margin-bottom: 1;
    }

    DataSourceInput .source-label {
        margin-bottom: 0;
    }

    DataSourceInput .source-hint {
        color: #555c65;
        margin-top: 0;
    }

    DataSourceInput .source-error {
        color: #F25E86;
        margin-top: 0;
    }

    DataSourceInput .source-row {
        height: auto;
        width: 100%;
    }

    DataSourceInput .source-input-container {
        width: 1fr;
    }

    DataSourceInput .source-badge {
        width: auto;
        min-width: 10;
        padding: 0 1;
        margin-left: 1;
        color: #78DCE8;
        text-style: bold;
    }

    DataSourceInput .source-badge.file {
        color: #B2D977;
    }

    DataSourceInput .source-badge.unknown {
        color: #555c65;
    }

    DataSourceInput RadioSet {
        height: auto;
        width: auto;
        layout: horizontal;
        margin-top: 0;
        padding: 0;
    }

    DataSourceInput RadioSet.single-option {
        opacity: 0.4;
    }

    DataSourceInput RadioButton {
        margin: 0 1 0 0;
        padding: 0;
    }

    DataSourceInput OptionList {
        max-height: 8;
        background: #141a26;
        layer: above;
        display: none;
    }

    DataSourceInput OptionList.visible {
        display: block;
        height: auto;
        border: solid #555c65;
    }

    DataSourceInput OptionList:focus {
        border: solid #78DCE8;
    }
    """

    value: reactive[str] = reactive("")
    source_type: reactive[SourceType] = reactive(SourceType.UNKNOWN)
    error: reactive[str] = reactive("")

    class Changed(Message):
        """Message sent when input value or source type changes."""

        def __init__(
            self,
            widget: "DataSourceInput",
            value: str,
            source_type: SourceType,
        ) -> None:
            super().__init__()
            self.data_source_input = widget
            self.value = value
            self.source_type = source_type

    def __init__(
        self,
        label: str = "Data Source",
        placeholder: str = "Dataset name or file path...",
        hint: str = "",
        required: bool = False,
        input_id: str | None = None,
        supported_sources: list[SourceType] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.placeholder = placeholder
        self.hint = hint
        self.required = required
        self.input_id = input_id or f"source-input-{id(self)}"
        self.supported_sources = supported_sources or [SourceType.DATASET, SourceType.FILE]
        self._all_datasets: list[str] = []
        self._updating = False
        self._selecting = False  # Prevent recursion when selecting

    def compose(self) -> ComposeResult:
        label_text = self.label
        if self.required:
            label_text += " [red]*[/red]"
        yield Static(label_text, classes="source-label")

        # Input row with badge
        with Horizontal(classes="source-row"):
            with Vertical(classes="source-input-container"):
                yield Input(placeholder=self.placeholder, id=self.input_id)
            yield Static("[dim]?[/dim]", id="type-badge", classes="source-badge unknown")

        # Autocomplete options list
        yield OptionList(id="source-options")

        # Source type toggle (only if multiple sources supported)
        if len(self.supported_sources) > 1:
            with RadioSet(id="source-toggle"):
                if SourceType.DATASET in self.supported_sources:
                    yield RadioButton("Dataset", id="radio-dataset", value=True)
                if SourceType.FILE in self.supported_sources:
                    yield RadioButton("File", id="radio-file")
        else:
            # Single option - show grayed out label
            single_type = self.supported_sources[0]
            with RadioSet(id="source-toggle", classes="single-option"):
                yield RadioButton(
                    single_type.value.title(),
                    id=f"radio-{single_type.value}",
                    value=True,
                    disabled=True,
                )

        if self.hint:
            yield Static(f"[dim]{self.hint}[/dim]", id="hint", classes="source-hint")
        yield Static("", id="error", classes="source-error")

    def on_mount(self) -> None:
        """Load available datasets on mount."""
        self._all_datasets = discover_local_datasets()
        self._update_options("")

    def _update_options(self, filter_text: str, show_all: bool = False) -> None:
        """Update the options list based on filter text (fuzzy search)."""
        try:
            option_list = self.query_one("#source-options", OptionList)
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
                option_list.add_class("visible")
            else:
                option_list.remove_class("visible")
        except Exception:
            pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection from the list."""
        event.stop()  # Prevent event bubbling
        self._select_source(event.option_id)

    def _select_source(self, source_id: str) -> None:
        """Select a source and update the input."""
        self._selecting = True
        try:
            input_widget = self.query_one(f"#{self.input_id}", Input)
            input_widget.value = source_id
            self.value = source_id

            option_list = self.query_one("#source-options", OptionList)
            option_list.remove_class("visible")
            option_list.highlighted = None  # Reset cursor

            # Detect source type
            detected = self._detect_source_type(source_id)
            if detected != SourceType.UNKNOWN:
                self.source_type = detected
                self._update_badge(detected)
                self._update_radio_selection(detected)

            self.post_message(self.Changed(self, source_id, self.source_type))
        except Exception:
            pass
        finally:
            self._selecting = False

    def on_focus(self, event: Focus) -> None:
        """When widget or its children get focus, show options."""
        try:
            input_widget = self.query_one(f"#{self.input_id}", Input)
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
        try:
            input_widget = self.query_one(f"#{self.input_id}", Input)
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
        # Delay hiding to allow click on options
        self.set_timer(0.2, self._maybe_hide_options)

    def _maybe_hide_options(self) -> None:
        """Hide options if neither input nor option list has focus."""
        # Don't hide if we're in the middle of selecting
        if self._selecting:
            return
        try:
            input_focused = self.query_one(f"#{self.input_id}", Input).has_focus
            options = self.query_one("#source-options", OptionList)
            options_focused = options.has_focus

            if not input_focused and not options_focused:
                options.remove_class("visible")
                options.highlighted = None  # Reset cursor
        except Exception:
            pass

    def _detect_source_type(self, value: str) -> SourceType:
        """Detect the source type from the input value."""
        if not value:
            return SourceType.UNKNOWN

        # Check if it's a known local dataset
        if value in self._all_datasets:
            return SourceType.DATASET

        # Check if it looks like a file path
        if value.endswith('.jsonl') or value.endswith('.json'):
            return SourceType.FILE

        # Check if it's an absolute or relative path
        if value.startswith('/') or value.startswith('./') or value.startswith('~'):
            return SourceType.FILE

        # Check if path exists (could be relative path)
        if Path(value).exists():
            return SourceType.FILE

        # Check if it looks like a HuggingFace dataset (org/name pattern)
        if '/' in value and not value.startswith('/'):
            # Could be either HF dataset or relative path
            # Prefer dataset if no file extension
            if not any(value.endswith(ext) for ext in ['.jsonl', '.json', '.txt', '.csv']):
                return SourceType.DATASET

        return SourceType.UNKNOWN

    def _update_badge(self, source_type: SourceType) -> None:
        """Update the type badge display."""
        try:
            badge = self.query_one("#type-badge", Static)
            badge.remove_class("file", "unknown")

            if source_type == SourceType.DATASET:
                badge.update("[Dataset]")
            elif source_type == SourceType.FILE:
                badge.update("[File]")
                badge.add_class("file")
            else:
                badge.update("[?]")
                badge.add_class("unknown")
        except Exception:
            pass

    def _update_radio_selection(self, source_type: SourceType) -> None:
        """Update the radio button selection to match detected type."""
        if self._updating or len(self.supported_sources) <= 1:
            return

        self._updating = True
        try:
            if source_type == SourceType.DATASET:
                radio = self.query_one("#radio-dataset", RadioButton)
                radio.value = True
            elif source_type == SourceType.FILE:
                radio = self.query_one("#radio-file", RadioButton)
                radio.value = True
        except Exception:
            pass
        finally:
            self._updating = False

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes - detect source type and emit change message."""
        if self._selecting:
            return

        self.value = event.value
        detected = self._detect_source_type(event.value)

        # Only auto-update if we have a clear detection
        if detected != SourceType.UNKNOWN:
            self.source_type = detected
            self._update_badge(detected)
            self._update_radio_selection(detected)
        else:
            self._update_badge(SourceType.UNKNOWN)

        # Update autocomplete options
        self._update_options(event.value)

        self.post_message(self.Changed(self, event.value, self.source_type))

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle manual source type toggle."""
        if self._updating:
            return

        if event.pressed.id == "radio-dataset":
            self.source_type = SourceType.DATASET
        elif event.pressed.id == "radio-file":
            self.source_type = SourceType.FILE

        self._update_badge(self.source_type)
        self.post_message(self.Changed(self, self.value, self.source_type))

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

    def set_value(self, value: str) -> None:
        """Set the input value programmatically without showing options."""
        self._selecting = True
        try:
            input_widget = self.query_one(f"#{self.input_id}", Input)
            input_widget.value = value
            self.value = value

            # Hide any visible options
            option_list = self.query_one("#source-options", OptionList)
            option_list.remove_class("visible")
            option_list.highlighted = None  # Reset cursor

            # Detect and update source type
            detected = self._detect_source_type(value)
            if detected != SourceType.UNKNOWN:
                self.source_type = detected
                self._update_badge(detected)
                self._update_radio_selection(detected)

            self.post_message(self.Changed(self, value, self.source_type))
        except Exception:
            pass
        finally:
            self._selecting = False

    def get_source_type(self) -> SourceType:
        """Get the current source type."""
        return self.source_type

    def is_dataset(self) -> bool:
        """Check if current source type is dataset."""
        return self.source_type == SourceType.DATASET

    def is_file(self) -> bool:
        """Check if current source type is file."""
        return self.source_type == SourceType.FILE

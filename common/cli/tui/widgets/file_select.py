"""File path selection widget with autocomplete for directories and files."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click, Focus
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option


def get_path_completions(
    partial_path: str,
    base_dir: Path | None = None,
    extensions: list[str] | None = None,
) -> list[str]:
    """Get path completions for a partial path.

    Args:
        partial_path: Partial path to complete.
        base_dir: Base directory to search from. Defaults to current directory.
        extensions: List of file extensions to filter by (e.g., ['.jsonl', '.json']).

    Returns:
        List of matching paths.
    """
    if base_dir is None:
        base_dir = Path.cwd()

    # Handle empty input - show items in base_dir
    if not partial_path:
        search_dir = base_dir
        prefix = ""
    else:
        path = Path(partial_path)

        # If it's an absolute path
        if path.is_absolute():
            if path.is_dir():
                search_dir = path
                prefix = str(path) + "/"
            else:
                search_dir = path.parent
                prefix = str(search_dir) + "/" if search_dir != Path("/") else "/"
        else:
            # Relative path
            full_path = base_dir / path
            if full_path.is_dir():
                search_dir = full_path
                prefix = partial_path + ("" if partial_path.endswith("/") else "/")
            else:
                search_dir = full_path.parent
                prefix = str(path.parent) + "/" if str(path.parent) != "." else ""

    if not search_dir.exists():
        return []

    completions = []
    try:
        for item in sorted(search_dir.iterdir()):
            # Skip hidden files
            if item.name.startswith("."):
                continue

            if item.is_dir():
                completions.append(prefix + item.name + "/")
            elif extensions is None or item.suffix.lower() in extensions:
                completions.append(prefix + item.name)
    except PermissionError:
        pass

    # Filter by partial filename if provided
    if partial_path and not partial_path.endswith("/"):
        filename_part = Path(partial_path).name.lower()
        completions = [
            c for c in completions
            if Path(c).name.lower().startswith(filename_part)
        ]

    return completions[:20]  # Limit results


class FileSelect(Vertical):
    """An autocomplete widget for selecting file paths.

    Shows path completions as user types, with optional extension filtering.
    """

    DEFAULT_CSS = """
    FileSelect {
        height: auto;
        margin-bottom: 1;
    }

    FileSelect .file-label {
        margin-bottom: 0;
    }

    FileSelect .file-hint {
        color: #555c65;
        margin-top: 0;
    }

    FileSelect .file-error {
        color: #F25E86;
        margin-top: 0;
    }

    FileSelect OptionList {
        max-height: 8;
        background: #141a26;
        layer: above;
        display: none;
    }

    FileSelect OptionList.visible {
        display: block;
        height: auto;
        border: solid #555c65;
    }

    FileSelect OptionList:focus {
        border: solid #78DCE8;
    }
    """

    value: reactive[str] = reactive("")
    error: reactive[str] = reactive("")

    class Changed(Message):
        """Message sent when file selection changes."""

        def __init__(self, widget: "FileSelect", value: str) -> None:
            super().__init__()
            self.file_select = widget
            self.value = value

    class Selected(Message):
        """Message sent when a file is explicitly selected from the list."""

        def __init__(self, widget: "FileSelect", value: str) -> None:
            super().__init__()
            self.file_select = widget
            self.value = value

    def __init__(
        self,
        label: str = "File",
        placeholder: str = "Enter file path...",
        hint: str = "",
        required: bool = False,
        extensions: list[str] | None = None,
        base_dir: Path | None = None,
        input_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.placeholder = placeholder
        self.hint = hint
        self.required = required
        self.extensions = extensions  # e.g., ['.jsonl', '.json']
        self.base_dir = base_dir
        self.input_id = input_id or f"file-input-{id(self)}"
        self._selecting = False

    def compose(self) -> ComposeResult:
        label_text = self.label
        if self.required:
            label_text += " [red]*[/red]"
        yield Static(label_text, classes="file-label")
        yield Input(placeholder=self.placeholder, id=self.input_id)
        yield OptionList(id="file-options")
        if self.hint:
            yield Static(f"[dim]{self.hint}[/dim]", id="hint", classes="file-hint")
        yield Static("", id="error", classes="file-error")

    def _update_options(self, path_text: str, show_all: bool = False) -> None:
        """Update the options list based on current path."""
        try:
            option_list = self.query_one("#file-options", OptionList)
            option_list.clear_options()

            if path_text or show_all:
                completions = get_path_completions(
                    path_text,
                    base_dir=self.base_dir,
                    extensions=self.extensions,
                )

                if completions:
                    for path in completions:
                        # Show shorter label for display
                        display_path = path
                        if path.endswith("/"):
                            label = f"{Path(path.rstrip('/')).name}/ [dim](dir)[/dim]"
                        else:
                            label = Path(path).name
                        option_list.add_option(Option(label, id=display_path))
                    option_list.add_class("visible")
                else:
                    option_list.remove_class("visible")
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
        """Handle Enter key - select first option or use current value."""
        try:
            option_list = self.query_one("#file-options", OptionList)
            if option_list.option_count > 0:
                first_option = option_list.get_option_at_index(0)
                self._select_path(first_option.id)
            else:
                option_list.remove_class("visible")
                option_list.highlighted = None  # Reset cursor
                self.post_message(self.Selected(self, self.value))
        except Exception:
            pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection from the list."""
        event.stop()  # Prevent event bubbling
        self._select_path(event.option_id)

    def _select_path(self, path: str) -> None:
        """Select a path and update the input."""
        self._selecting = True
        try:
            input_widget = self.query_one(f"#{self.input_id}", Input)
            input_widget.value = path
            self.value = path

            option_list = self.query_one("#file-options", OptionList)

            # If it's a directory, show contents
            if path.endswith("/"):
                option_list.highlighted = None  # Reset cursor before repopulating
                self._update_options(path)
            else:
                option_list.remove_class("visible")
                option_list.highlighted = None  # Reset cursor
                self.post_message(self.Selected(self, path))
                self.post_message(self.Changed(self, path))
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
            current_value = input_widget.value.strip()
            if not current_value:
                self._update_options("", show_all=True)
            else:
                self._update_options(current_value)
        except Exception:
            pass

    def on_click(self, event: Click) -> None:
        """Handle clicks on the widget to show options."""
        try:
            input_widget = self.query_one(f"#{self.input_id}", Input)
            input_widget.focus()
            current_value = input_widget.value.strip()
            if not current_value:
                self._update_options("", show_all=True)
            else:
                self._update_options(current_value)
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
            options = self.query_one("#file-options", OptionList)
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

"""Base mixin for autocomplete widgets with reliable dropdown behavior."""

import time
from typing import Callable

from textual.widgets import Input, OptionList


class AutocompleteMixin:
    """Mixin providing reliable dropdown behavior for autocomplete widgets.

    Solves the race condition where dropdown re-shows after selection due to
    focus events firing immediately after selection.

    Usage:
        class MyWidget(AutocompleteMixin, Vertical):
            def __init__(self, ...):
                super().__init__(...)
                self._init_autocomplete()

            def _get_input_widget(self) -> Input:
                return self.query_one("#my-input", Input)

            def _get_option_list(self) -> OptionList:
                return self.query_one("#my-options", OptionList)
    """

    # Timestamp of last selection - prevents re-show for SELECTION_COOLDOWN_MS
    _selection_timestamp: float = 0
    SELECTION_COOLDOWN_MS: int = 300

    # Flag to prevent input change events during programmatic updates
    _selecting: bool = False

    def _init_autocomplete(self) -> None:
        """Initialize autocomplete state. Call from __init__."""
        self._selection_timestamp = 0
        self._selecting = False

    def _get_input_widget(self) -> Input:
        """Override to return the Input widget. Required."""
        raise NotImplementedError("Subclass must implement _get_input_widget()")

    def _get_option_list(self) -> OptionList:
        """Override to return the OptionList widget. Required."""
        raise NotImplementedError("Subclass must implement _get_option_list()")

    def _is_in_selection_cooldown(self) -> bool:
        """Check if we're within the cooldown period after a selection."""
        if self._selection_timestamp == 0:
            return False
        elapsed_ms = (time.time() - self._selection_timestamp) * 1000
        return elapsed_ms < self.SELECTION_COOLDOWN_MS

    def _show_dropdown(self) -> None:
        """Show the dropdown options list, unless in cooldown."""
        if self._is_in_selection_cooldown():
            return

        try:
            option_list = self._get_option_list()
            option_list.add_class("visible")
        except Exception:
            pass

    def _hide_dropdown(self) -> None:
        """Hide the dropdown options list."""
        try:
            option_list = self._get_option_list()
            option_list.remove_class("visible")
            option_list.highlighted = None
        except Exception:
            pass

    def _is_dropdown_visible(self) -> bool:
        """Check if the dropdown is currently visible."""
        try:
            option_list = self._get_option_list()
            return option_list.has_class("visible")
        except Exception:
            return False

    def _select_value(
        self,
        value: str,
        on_selected: Callable[[str], None] | None = None,
    ) -> None:
        """Handle selection of a value from the dropdown.

        Args:
            value: The selected value to set in the input
            on_selected: Optional callback to run after selection is complete
        """
        self._selecting = True
        self._selection_timestamp = time.time()

        try:
            # Update input value
            input_widget = self._get_input_widget()
            input_widget.value = value

            # Hide dropdown immediately
            self._hide_dropdown()

            # Move focus back to input (not option list)
            input_widget.focus()

            # Call the optional callback
            if on_selected:
                on_selected(value)

        except Exception:
            pass
        finally:
            self._selecting = False

    def _should_show_on_focus(self) -> bool:
        """Determine if dropdown should show on focus event.

        Returns False during selection cooldown to prevent re-showing.
        """
        return not self._is_in_selection_cooldown()

    def _maybe_hide_after_blur(self, delay_ms: int = 200) -> None:
        """Schedule hiding dropdown after blur, unless focus moved to option list.

        Call this from on_blur() handler. Uses set_timer for delayed check.
        """
        # Don't schedule hide if we just selected
        if self._is_in_selection_cooldown():
            return

        def check_and_hide() -> None:
            # Don't hide if in cooldown (selection happened during delay)
            if self._is_in_selection_cooldown():
                return

            try:
                input_widget = self._get_input_widget()
                option_list = self._get_option_list()

                input_focused = input_widget.has_focus
                options_focused = option_list.has_focus

                if not input_focused and not options_focused:
                    self._hide_dropdown()
            except Exception:
                pass

        # Schedule the check
        if hasattr(self, "set_timer"):
            self.set_timer(delay_ms / 1000.0, check_and_hide)

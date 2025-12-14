"""Monokai Pro theme colors for Omarchy/Ghostty compatibility."""


class Theme:
    """Color constants for the Monokai Omarchy theme."""

    # Base colors
    BACKGROUND = "#0a1220"
    SURFACE = "#141a26"
    BOOST = "#1a2233"
    FOREGROUND = "#f0f2f5"
    MUTED = "#555c65"

    # Accent colors
    PINK = "#F25E86"
    GREEN = "#B2D977"
    YELLOW = "#F2C063"
    CYAN = "#78DCE8"
    PURPLE = "#ABA0F2"
    ORANGE = "#F28B66"

    # Semantic aliases
    PRIMARY = CYAN
    SECONDARY = PURPLE
    SUCCESS = GREEN
    WARNING = YELLOW
    ERROR = PINK
    INFO = CYAN

    @classmethod
    def rgba(cls, hex_color: str, alpha: float) -> str:
        """Convert hex to rgba string for CSS."""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"


# Status icons with colors
STATUS_ICONS = {
    "running": ("◐", Theme.YELLOW),
    "success": ("✓", Theme.GREEN),
    "failed": ("✗", Theme.PINK),
    "queued": ("○", Theme.MUTED),
    "warning": ("⚠", Theme.YELLOW),
    "info": ("ℹ", Theme.CYAN),
}

"""Home screen with category selection."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static


class CategoryCard(Static):
    """A clickable card showing a command category."""

    can_focus = True

    def __init__(
        self,
        title: str,
        commands: list[tuple[str, str]],
        category_id: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.title = title
        self.commands = commands
        self.category_id = category_id

    def compose(self) -> ComposeResult:
        yield Static(f"[bold]{self.title}[/bold]", classes="card-title")
        for cmd, desc in self.commands:
            yield Static(f"  [cyan]{cmd}[/cyan] - {desc}", classes="card-command")

    def on_click(self) -> None:
        """Handle click on card."""
        self.focus()
        self._select_category()

    def on_key(self, event) -> None:
        """Handle key press on focused card."""
        if event.key == "enter":
            event.stop()
            self._select_category()

    def _select_category(self) -> None:
        """Push the category selection screen."""
        from .category import CategoryScreen

        self.app.push_screen(CategoryScreen(self.category_id))


# Category definitions
DOWNLOAD_COMMANDS = [
    ("dataset", "Download HF dataset"),
    ("model", "Download HF model"),
    ("fineweb", "Download FineWeb sample"),
]

PREPROCESS_COMMANDS = [
    ("tokenize", "Pre-tokenize dataset"),
    ("train-tok", "Train tokenizer"),
    ("index", "Build FineWeb index"),
    ("extract", "Extract domain corpus"),
]

ANALYZE_COMMANDS = [
    ("tokens", "Token distribution"),
    ("domains", "Query domain index"),
]


class HomeScreen(Screen):
    """Home screen with category cards."""

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("1", "select_download", "Download", show=False),
        Binding("2", "select_preprocess", "Preprocess", show=False),
        Binding("3", "select_analyze", "Analyze", show=False),
        Binding("left", "focus_prev", "Previous", show=False),
        Binding("right", "focus_next", "Next", show=False),
        Binding("enter", "select_focused", "Select", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            with Vertical(id="welcome"):
                yield Static("[bold]Research CLI[/bold]", id="welcome-title", classes="center")
                yield Static(
                    "Select a category or press Ctrl+P for command palette",
                    id="welcome-subtitle",
                    classes="center text-muted",
                )

            with Horizontal(id="cards-container"):
                yield CategoryCard(
                    "Download",
                    DOWNLOAD_COMMANDS,
                    "download",
                    classes="category-card",
                )
                yield CategoryCard(
                    "Preprocess",
                    PREPROCESS_COMMANDS,
                    "preprocess",
                    classes="category-card",
                )
                yield CategoryCard(
                    "Analyze",
                    ANALYZE_COMMANDS,
                    "analyze",
                    classes="category-card",
                )

            yield Static(
                "[dim]Press [/dim][cyan]←/→[/cyan][dim] to navigate  •  "
                "[/dim][cyan]Enter[/cyan][dim] to select  •  "
                "[/dim][cyan]Ctrl+P[/cyan][dim] for commands  •  "
                "[/dim][cyan]q[/cyan][dim] to quit[/dim]",
                id="footer-hint",
                classes="center",
            )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the first card on mount."""
        cards = self.query(CategoryCard)
        if cards:
            cards.first().focus()

    def _get_cards(self) -> list[CategoryCard]:
        """Get all category cards in order."""
        return list(self.query(CategoryCard))

    def action_focus_prev(self) -> None:
        """Focus the previous category card."""
        cards = self._get_cards()
        if not cards:
            return
        focused = self.focused
        if focused in cards:
            idx = cards.index(focused)
            # Wrap around to last card if at first
            new_idx = (idx - 1) % len(cards)
            cards[new_idx].focus()
        else:
            cards[-1].focus()

    def action_focus_next(self) -> None:
        """Focus the next category card."""
        cards = self._get_cards()
        if not cards:
            return
        focused = self.focused
        if focused in cards:
            idx = cards.index(focused)
            # Wrap around to first card if at last
            new_idx = (idx + 1) % len(cards)
            cards[new_idx].focus()
        else:
            cards[0].focus()

    def action_select_focused(self) -> None:
        """Select the currently focused category card."""
        focused = self.focused
        if isinstance(focused, CategoryCard):
            focused._select_category()

    def action_select_download(self) -> None:
        """Select download category."""
        from .category import CategoryScreen

        self.app.push_screen(CategoryScreen("download"))

    def action_select_preprocess(self) -> None:
        """Select preprocess category."""
        from .category import CategoryScreen

        self.app.push_screen(CategoryScreen("preprocess"))

    def action_select_analyze(self) -> None:
        """Select analyze category."""
        from .category import CategoryScreen

        self.app.push_screen(CategoryScreen("analyze"))

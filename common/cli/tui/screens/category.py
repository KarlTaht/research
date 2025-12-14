"""Category selection screen for choosing sub-commands."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, ListView, ListItem, Static


# Map category IDs to their sub-commands and wizard screens
CATEGORIES = {
    "download": {
        "title": "Download",
        "description": "Fetch data from external sources",
        "commands": [
            ("dataset", "Download HuggingFace dataset", "download_dataset"),
            ("model", "Download HuggingFace model", "download_model"),
            ("fineweb", "Download FineWeb sample", "download_fineweb"),
        ],
    },
    "preprocess": {
        "title": "Preprocess",
        "description": "Transform raw data for training",
        "commands": [
            ("tokenize", "Pre-tokenize dataset for training", "pretokenize"),
            ("train-tokenizer", "Train custom BPE tokenizer", "train_tokenizer"),
            ("index", "Build FineWeb domain index", "fineweb_index"),
            ("extract", "Extract domain corpus from FineWeb", "fineweb_extract"),
        ],
    },
    "analyze": {
        "title": "Analyze",
        "description": "Inspect and understand data",
        "commands": [
            ("tokens", "Analyze token distributions", "analyze_tokens"),
            ("domains", "Query FineWeb domain index", "query_domains"),
        ],
    },
}


class CommandListItem(ListItem):
    """List item for a command."""

    def __init__(self, name: str, description: str, wizard_name: str) -> None:
        super().__init__()
        self.command_name = name
        self.description = description
        self.wizard_name = wizard_name

    def compose(self) -> ComposeResult:
        yield Static(f"[cyan]{self.command_name}[/cyan]  [dim]{self.description}[/dim]")


class CategoryScreen(Screen):
    """Screen for selecting a sub-command within a category."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("enter", "select", "Select", show=False),
    ]

    def __init__(self, category_id: str, name: str | None = None) -> None:
        super().__init__(name=name)
        self.category_id = category_id
        self.category = CATEGORIES.get(category_id, {})

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="category-container"):
            title = self.category.get("title", "Unknown")
            desc = self.category.get("description", "")
            yield Static(f"[bold]{title}[/bold]", classes="wizard-title center")
            yield Static(f"[dim]{desc}[/dim]", classes="center text-muted")

            with Vertical(id="command-list-container"):
                # Create list items from commands
                items = [
                    CommandListItem(name, description, wizard)
                    for name, description, wizard in self.category.get("commands", [])
                ]
                yield ListView(*items, id="command-list")

            with Container(classes="button-bar"):
                yield Button("Back", variant="default", id="back-btn")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the list view on mount."""
        self.query_one("#command-list", ListView).focus()

    def action_back(self) -> None:
        """Go back to home screen."""
        self.app.pop_screen()

    def action_select(self) -> None:
        """Select the highlighted command."""
        list_view = self.query_one("#command-list", ListView)
        if list_view.highlighted_child is not None:
            self._open_wizard(list_view.highlighted_child)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection."""
        self._open_wizard(event.item)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "back-btn":
            self.action_back()

    def _open_wizard(self, item: ListItem) -> None:
        """Open the wizard screen for the selected command."""
        if not isinstance(item, CommandListItem):
            return

        wizard_name = item.wizard_name
        wizard_class = self._get_wizard_class(wizard_name)
        if wizard_class:
            self.app.push_screen(wizard_class())
        else:
            self.notify(f"Wizard '{wizard_name}' not implemented yet", severity="warning")

    def _get_wizard_class(self, wizard_name: str):
        """Get the wizard screen class by name."""
        # Import wizards lazily to avoid circular imports
        try:
            if wizard_name == "download_dataset":
                from .download_dataset import DownloadDatasetScreen

                return DownloadDatasetScreen
            elif wizard_name == "download_model":
                from .download_model import DownloadModelScreen

                return DownloadModelScreen
            elif wizard_name == "download_fineweb":
                from .download_fineweb import DownloadFinewebScreen

                return DownloadFinewebScreen
            elif wizard_name == "pretokenize":
                from .pretokenize import PretokenizeScreen

                return PretokenizeScreen
            elif wizard_name == "fineweb_index":
                from .fineweb_index import FinewebIndexScreen

                return FinewebIndexScreen
            elif wizard_name == "fineweb_extract":
                from .fineweb_extract import FinewebExtractScreen

                return FinewebExtractScreen
            elif wizard_name == "analyze_tokens":
                from .analyze_tokens import AnalyzeTokensScreen

                return AnalyzeTokensScreen
            elif wizard_name == "query_domains":
                from .query_domains import QueryDomainsScreen

                return QueryDomainsScreen
            elif wizard_name == "train_tokenizer":
                from .train_tokenizer import TrainTokenizerScreen

                return TrainTokenizerScreen
        except ImportError:
            pass
        return None

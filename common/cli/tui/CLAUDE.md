# TUI Development Guide

## Status: All Wizards Complete

All 9 wizard screens are implemented and working. 28 tests passing.

### Recent Updates (Dec 2024)
- Added `DatasetSelect` widget with autocomplete for local datasets
- Added `DataSourceInput` widget with auto-detection + autocomplete
- Added `TokenizerSelect` widget with common tokenizers + custom discovery
- Added `FileSelect` widget with path completion + extension filtering
- Moved Train Tokenizer to Preprocess category
- Added inverted selection colors for better visibility
- Fixed click-to-show-options behavior

---

# Completed Work

## Infrastructure

| Task | Status | File |
|------|--------|------|
| Theme colors | Done | `theme.py` |
| Global CSS styles | Done | `styles.tcss` |
| Screen base classes | Done | `screens/__init__.py` |
| WizardScreen base | Done | `screens/wizard_base.py` |
| OutputScreen | Done | `screens/output.py` |
| HomeScreen | Done | `screens/home.py` |
| CategoryScreen | Done | `screens/category.py` |
| CommandPreview widget | Done | `widgets/command_preview.py` |
| LabeledInput widget | Done | `widgets/labeled_input.py` |
| App refactored | Done | `app.py` |
| Arrow key navigation | Done | `screens/home.py` |
| Tests | Done | `tests/test_tui.py` (28 passing) |

## All Wizards Implemented

### Download (3)
| Wizard | File |
|--------|------|
| Download Dataset | `screens/download_dataset.py` |
| Download Model | `screens/download_model.py` |
| Download FineWeb | `screens/download_fineweb.py` |

### Preprocess (4)
| Wizard | File |
|--------|------|
| Pretokenize | `screens/pretokenize.py` |
| Train Tokenizer | `screens/train_tokenizer.py` |
| FineWeb Index | `screens/fineweb_index.py` |
| FineWeb Extract | `screens/fineweb_extract.py` |

### Analyze (2)
| Wizard | File |
|--------|------|
| Analyze Tokens | `screens/analyze_tokens.py` |
| Query Domains | `screens/query_domains.py` |

---

# File Structure

```
common/cli/tui/
├── __init__.py              # Exports ResearchApp, main
├── __main__.py              # python -m common.cli.tui
├── app.py                   # Main app, command palette provider
├── theme.py                 # Monokai Pro color constants
├── styles.tcss              # Global Textual CSS
├── design_principles.md     # Full design system documentation
├── CLAUDE.md                # This file
├── screens/
│   ├── __init__.py          # Exports all screens
│   ├── home.py              # Home screen with 3 category cards
│   ├── category.py          # Sub-command selection
│   ├── wizard_base.py       # Base class for all wizards
│   ├── output.py            # Live command output with RichLog
│   └── [wizard screens]     # One per operation
└── widgets/
    ├── __init__.py          # Exports widgets
    ├── command_preview.py   # Shows generated command live
    ├── labeled_input.py     # Input with label + validation
    ├── dataset_select.py    # Autocomplete for local datasets
    ├── data_source_input.py # Smart source type selector
    ├── tokenizer_select.py  # Autocomplete for tokenizers
    └── file_select.py       # File path completion
```

---

# User Flow

```
research (launch TUI)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ HOME SCREEN                                      │
│  ┌──────────┐  ┌────────────┐  ┌─────────┐     │
│  │ Download │  │ Preprocess │  │ Analyze │     │
│  └──────────┘  └────────────┘  └─────────┘     │
│  Press 1-3 or click | Ctrl+P for palette        │
└─────────────────────────────────────────────────┘
    │ (press 1)
    ▼
┌─────────────────────────────────────────────────┐
│ CATEGORY: Download                               │
│  > dataset   Download HuggingFace dataset       │
│    model     Download HuggingFace model         │
│    fineweb   Download FineWeb sample            │
│  [Back]                                         │
└─────────────────────────────────────────────────┘
    │ (select dataset)
    ▼
┌─────────────────────────────────────────────────┐
│ WIZARD: Download Dataset                         │
│  Dataset name: [squad________________]          │
│  Config:       [default______________]          │
│  Split:        [all ▼]                          │
│  Command: python -m common.cli.data ...         │
│  [Cancel]                    [Download]         │
└─────────────────────────────────────────────────┘
    │ (click Download)
    ▼
┌─────────────────────────────────────────────────┐
│ OUTPUT                                           │
│  Running: python -m common.cli.data ...         │
│  ┌─────────────────────────────────────────┐   │
│  │ Downloading squad...                     │   │
│  │ Command completed successfully.          │   │
│  └─────────────────────────────────────────┘   │
│  Status: Completed                              │
│  [Back]                          [Re-run]       │
└─────────────────────────────────────────────────┘
```

---

# How to Add a New Wizard

## 1. Create the wizard file

`common/cli/tui/screens/<wizard_name>.py`:

```python
from textual.app import ComposeResult
from textual.widgets import Input, Select, Static
from ..widgets import CommandPreview, LabeledInput
from .wizard_base import WizardScreen

class MyWizardScreen(WizardScreen):
    TITLE = "My Wizard"
    COMMAND_MODULE = "common.cli.data"

    def compose_form(self) -> ComposeResult:
        yield LabeledInput(
            label="Field Name",
            placeholder="Enter value",
            required=True,
            input_id="field-name",
        )

    def get_command_args(self) -> list[str]:
        args = ["subcommand"]
        try:
            value = self.query_one("#field-name", Input).value.strip()
            if value:
                args.extend(["--flag", value])
        except Exception:
            pass
        return args

    def validate(self) -> tuple[bool, str]:
        try:
            value = self.query_one("#field-name", Input).value.strip()
            if not value:
                return False, "Field is required"
        except Exception:
            return False, "Form not ready"
        return True, ""
```

## 2. Register in category.py

In `screens/category.py`, add to `_get_wizard_class()`:

```python
elif wizard_name == "my_wizard":
    from .my_wizard import MyWizardScreen
    return MyWizardScreen
```

## 3. Export from __init__.py

In `screens/__init__.py`:

```python
from .my_wizard import MyWizardScreen
__all__ = [..., "MyWizardScreen"]
```

---

# Key Classes

## WizardScreen (Base Class)

```python
class WizardScreen(Screen):
    TITLE: str           # Screen title
    COMMAND_MODULE: str  # e.g., "common.cli.data"

    def compose_form(self) -> ComposeResult:
        """Override to add form fields."""
        pass

    def get_command_args(self) -> list[str]:
        """Return CLI arguments from form state."""
        return []

    def validate(self) -> tuple[bool, str]:
        """Validate form. Return (is_valid, error_message)."""
        return True, ""
```

## OutputScreen

- Runs subprocess with `asyncio.create_subprocess_exec`
- Streams stdout/stderr to `RichLog` widget
- Shows success/error status on completion
- Supports re-run

## DatasetSelect Widget

Autocomplete widget for selecting datasets from local storage (`widgets/dataset_select.py`).

```python
from ..widgets import DatasetSelect

yield DatasetSelect(
    label="Dataset name",
    placeholder="Type to search local datasets...",
    required=True,
    input_id="dataset-name",
)

def on_dataset_select_changed(self, event: DatasetSelect.Changed) -> None:
    dataset_name = event.value  # e.g., "roneneldan/TinyStories"
```

Features:
- Discovers datasets from `assets/datasets/{org}/{name}/`
- Fuzzy-match filtering as user types
- Shows dropdown with matching datasets
- Accepts arbitrary HuggingFace paths

## DataSourceInput Widget

Smart source type selector with auto-detection and autocomplete (`widgets/data_source_input.py`).

```python
from ..widgets import DataSourceInput, SourceType

yield DataSourceInput(
    label="Data source",
    placeholder="Dataset name or JSONL path...",
    required=True,
    input_id="source",
)

def on_data_source_input_changed(self, event: DataSourceInput.Changed) -> None:
    value = event.value
    source_type = event.source_type  # SourceType.DATASET, FILE, or UNKNOWN
```

Features:
- Auto-detects source type as user types
- **Autocomplete dropdown** for local datasets (click to show all)
- Shows type badge (`[Dataset]` or `[File]`)
- Toggle grayed out when only one type applies
- Supports both dataset names and file paths

## TokenizerSelect Widget

Autocomplete widget for selecting tokenizers (`widgets/tokenizer_select.py`).

```python
from ..widgets import TokenizerSelect

yield TokenizerSelect(
    label="Tokenizer",
    placeholder="gpt2",
    hint="HuggingFace tokenizer or custom trained",
    input_id="tokenizer",
)

def on_tokenizer_select_changed(self, event: TokenizerSelect.Changed) -> None:
    tokenizer = event.value  # e.g., "bert-base-uncased"
```

Features:
- 14 common tokenizers (gpt2, bert variants, roberta, t5, llama, etc.)
- Discovers custom trained tokenizers from `assets/models/tokenizers/`
- Fuzzy-match filtering as user types
- Shows all options when clicked with empty input

## FileSelect Widget

File path completion widget (`widgets/file_select.py`).

```python
from ..widgets import FileSelect

yield FileSelect(
    label="Validation JSONL",
    placeholder="path/to/file.jsonl",
    extensions=[".jsonl", ".json"],  # Filter by extension
    input_id="val-jsonl",
)

def on_file_select_changed(self, event: FileSelect.Changed) -> None:
    path = event.value  # e.g., "data/corpus/val.jsonl"
```

Features:
- Path completion for directories and files
- Extension filtering (e.g., only show `.jsonl` files)
- Navigates into directories when selected
- Shows current directory contents when clicked with empty input

---

# Theme Colors (Monokai Pro)

```python
BACKGROUND = "#0a1220"  # Deep blue-black
SURFACE = "#141a26"     # Panel backgrounds
CYAN = "#78DCE8"        # Primary/focus
GREEN = "#B2D977"       # Success
YELLOW = "#F2C063"      # Warning/selection
PINK = "#F25E86"        # Error
PURPLE = "#ABA0F2"      # Headers
```

---

# Testing

```bash
pytest tests/test_tui.py -v
```

Current: 28 tests passing
- 9 widget tests: DatasetSelect (3), TokenizerSelect (3), FileSelect (3)

Test pattern:
```python
@pytest.mark.asyncio(loop_scope="function")
async def test_something(self):
    app = ResearchApp()
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.1)
        # assertions
```

---

# Next Steps

## Completed
1. **Arrow key navigation** - ✅ DONE
2. **All 9 wizards** - ✅ DONE

## Future Enhancements
- More tests for wizard flows
- Recent commands history
- Model autocomplete for download_model.py
- Corpus name autocomplete for fineweb_extract.py

---

## Future Direction: Beautification Roadmap

The TUI has extensive design principles documented in `design_principles.md` that represent the target state. Key areas for future beautification work:

### New Screens to Build
- **Datasets Screen**: Browse/tokenize/analyze tabs with dataset table
- **Experiments Screen**: Split-pane layout with experiment list + detail view
- **Experiment Detail**: Metric progress bars, live logs, export functionality
- **Tools/Environment Screen**: Status checklist, quick actions

### Layout Patterns to Implement
- Multi-pane split views (master list + detail panel)
- Tabbed interfaces for multi-view screens
- Modal dialogs for confirmations and details

### New Widgets to Create
- `widgets/sparkline.py`: Sparkline graphs for metrics (loss trends)
- `widgets/status.py`: Status badges (checkmark/x/half/empty)
- `widgets/progress.py`: Determinate/indeterminate progress indicators

### Visual Polish
- Consistent focus rings on all interactive elements
- Contextual footer keybindings per screen
- Search/filter functionality (`/` key)
- Help overlay (`?` key)
- Semantic color formatting (orange for numbers, cyan for keybindings)

See `design_principles.md` for full specifications.

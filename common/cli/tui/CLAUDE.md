# TUI Development Guide

## Status: All Wizards Complete

All 9 wizard screens are implemented and working. 19 tests passing.

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
| Tests | Done | `tests/test_tui.py` (19 passing) |

## All Wizards Implemented

### Download (3)
| Wizard | File |
|--------|------|
| Download Dataset | `screens/download_dataset.py` |
| Download Model | `screens/download_model.py` |
| Download FineWeb | `screens/download_fineweb.py` |

### Preprocess (3)
| Wizard | File |
|--------|------|
| Pretokenize | `screens/pretokenize.py` |
| FineWeb Index | `screens/fineweb_index.py` |
| FineWeb Extract | `screens/fineweb_extract.py` |

### Analyze (3)
| Wizard | File |
|--------|------|
| Analyze Tokens | `screens/analyze_tokens.py` |
| Query Domains | `screens/query_domains.py` |
| Train Tokenizer | `screens/train_tokenizer.py` |

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
│   └── download_dataset.py  # First wizard implementation
└── widgets/
    ├── __init__.py          # Exports widgets
    ├── command_preview.py   # Shows generated command live
    └── labeled_input.py     # Input with label + validation
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

Current: 19 tests passing

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
- Visual polish (focus indicators, hover states)
- File browser for path inputs
- Autocomplete for dataset/model names
- Recent commands history
- Keyboard shortcut help overlay

# TUI Design Guide: Research Experiment Manager

A comprehensive design system for building a terminal user interface to manage ML research experiments, datasets, and analysis workflows. Built for Textual (Python) with Ghostty/Omarchy compatibility.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Color System](#color-system)
3. [Typography & Spacing](#typography--spacing)
4. [Layout Patterns](#layout-patterns)
5. [Component Catalog](#component-catalog)
6. [Interaction Patterns](#interaction-patterns)
7. [Domain-Specific Guidelines](#domain-specific-guidelines)
8. [Implementation Reference](#implementation-reference)

---

## Design Philosophy

### Core Principles

**1. Clarity over decoration**
Research workflows are complex enough. The interface should reduce cognitive load, not add to it. Every visual element must earn its place.

**2. Information density with hierarchy**
Show what matters at a glance. Use progressive disclosure—summary views expand to detail views. Never overwhelm with data dumps.

**3. Keyboard-first, mouse-friendly**
Power users live on the keyboard. Every action should be one or two keypresses away. Mouse support is a convenience, not a crutch. Arrow keys are the default, not Vim-style.

**4. Immediate feedback**
Every action produces visible acknowledgment. Long operations show progress. Errors are surfaced clearly, not buried in logs.

**5. Consistent visual language**
Same patterns, same meanings. A cyan border always means focused. Yellow always means selected. Users build muscle memory.

### Reference Implementations

This design draws from three exemplary TUIs:

| Project | What to emulate |
|---------|-----------------|
| **impala** | Minimal chrome, clean single-line borders, contextual footer keybindings |
| **lazydocker** | Multi-pane layouts, mouse support, graceful resizing, help discoverability |
| **btop** | Sparkline graphs for metrics (use sparingly), semantic color gradients |

---

## Color System

### Palette: Monokai Pro (Omarchy Variant)

Designed for true-color terminals (Ghostty, Kitty, Alacritty). Falls back gracefully to 256-color.

```
┌─────────────────────────────────────────────────────────────┐
│  BASE COLORS                                                │
├─────────────────────────────────────────────────────────────┤
│  Background     #0a1220    Deep blue-black                  │
│  Surface        #141a26    Panel backgrounds                │
│  Boost          #1a2233    Hover/focus backgrounds          │
│  Foreground     #f0f2f5    Primary text                     │
│  Muted          #555c65    Secondary text, borders          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ACCENT COLORS                                              │
├─────────────────────────────────────────────────────────────┤
│  Pink           #F25E86    Errors, destructive actions      │
│  Green          #B2D977    Success, completion, strings     │
│  Yellow         #F2C063    Warnings, selection, active      │
│  Cyan           #78DCE8    Info, focus, links, primary      │
│  Purple         #ABA0F2    Headers, special elements        │
│  Orange         #F28B66    Numbers, secondary accents       │
└─────────────────────────────────────────────────────────────┘
```

### Semantic Color Mapping

| Context | Color | Hex | Usage |
|---------|-------|-----|-------|
| **Primary action** | Cyan | `#78DCE8` | Focused elements, interactive hints, links |
| **Selection** | Yellow | `#F2C063` | Currently selected item, active state |
| **Success** | Green | `#B2D977` | Completed operations, passing tests, valid states |
| **Warning** | Yellow | `#F2C063` | Caution states, pending operations |
| **Error** | Pink | `#F25E86` | Failed operations, invalid input, destructive actions |
| **Info** | Cyan | `#78DCE8` | Informational messages, help text |
| **Disabled** | Muted | `#555c65` | Inactive elements, placeholder text |
| **Header/Title** | Purple | `#ABA0F2` | Section headers, panel titles |
| **Metrics/Numbers** | Orange | `#F28B66` | Numerical data, statistics, counts |

### Color Usage Rules

1. **Background**: Always `#0a1220`. Panels may use `#141a26` for subtle elevation.
2. **Text**: Default to `#f0f2f5`. Use `#555c65` for secondary/helper text only.
3. **Borders**: Default `#555c65`. Focused elements use `#78DCE8`.
4. **Never use pure white** (`#ffffff`) or pure black (`#000000`).
5. **Accent colors at 20% opacity** for backgrounds (selection highlighting).

---

## Typography & Spacing

### Text Hierarchy

```
PANEL TITLE          Purple (#ABA0F2), bold
Section Header       Foreground (#f0f2f5), bold  
Body Text            Foreground (#f0f2f5), regular
Secondary Text       Muted (#555c65), regular
Keybinding Labels    Cyan (#78DCE8), bold
Values/Numbers       Orange (#F28B66), regular
```

### Spacing Units

All spacing is measured in terminal character cells.

| Unit | Characters | Usage |
|------|------------|-------|
| `xs` | 0 | Tight grouping |
| `sm` | 1 | Inside components, between related items |
| `md` | 2 | Between components |
| `lg` | 3 | Between sections |
| `xl` | 4+ | Major visual breaks |

### Padding Guidelines

- **Inside panels**: 1 character horizontal, 0 vertical
- **List items**: 1 character horizontal padding
- **Buttons**: 1 character horizontal padding
- **Between panels**: 1 character gap (handled by layout)

---

## Layout Patterns

### Primary Layout: Split View

The default layout for most screens. Master list on left, detail view on right.

```
┌─ Research Manager ──────────────────────────────────────────┐
│                                                              │
│  ┌─ Experiments ───────┐  ┌─ Details ─────────────────────┐ │
│  │                     │  │                               │ │
│  │    exp_001          │  │  Name: gpt2-finetune-v3       │ │
│  │  ▸ exp_002 ◀────────│──│  Status: Running              │ │
│  │    exp_003          │  │  Started: 2025-01-12 14:32    │ │
│  │    exp_004          │  │  Steps: 1,240 / 10,000        │ │
│  │                     │  │                               │ │
│  │                     │  │  ┌─ Loss ──────────────────┐  │ │
│  │                     │  │  │ ▂▃▄▅▆▇▆▅▄▃▂▁▁▁▁▁▁▁▁▁▁▁  │  │ │
│  │                     │  │  └─────────────────────────┘  │ │
│  │                     │  │                               │ │
│  └─────────────────────┘  └───────────────────────────────┘ │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ [q] Quit  [↑↓] Navigate  [Enter] View  [d] Delete  [?] Help  │
└──────────────────────────────────────────────────────────────┘
```

**Proportions**: Left panel 30-40%, right panel 60-70%. Flexible based on content.

### Secondary Layout: Tabbed Sections

For screens with multiple distinct views (e.g., Datasets: Browse / Tokenize / Analyze).

```
┌─ Datasets ──────────────────────────────────────────────────┐
│                                                              │
│  ┌─────────┬───────────┬──────────┐                         │
│  │ Browse  │ Tokenize  │ Analyze  │                         │
│  └─────────┴───────────┴──────────┘                         │
│  ═══════════                                                 │
│                                                              │
│  ┌─ Available Datasets ────────────────────────────────────┐│
│  │                                                          ││
│  │  Name              Size        Samples    Tokenized      ││
│  │  ─────────────────────────────────────────────────────   ││
│  │  openwebtext       52.4 GB     8,013,769  ✓              ││
│  │▸ pile-subset       12.1 GB     2,100,000  ✗              ││
│  │  custom-corpus     3.2 GB      450,000    ✓              ││
│  │                                                          ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ [1-3] Switch Tab  [↑↓] Navigate  [t] Tokenize  [?] Help      │
└──────────────────────────────────────────────────────────────┘
```

**Tab indicator**: Active tab has underline (`═══`). Use numbers `[1]` `[2]` `[3]` for switching.

### Tertiary Layout: Modal Dialogs

For confirmations, input forms, and focused tasks.

```
┌─ Experiments ───────────────────────────────────────────────┐
│                                                              │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ │
│  │░░░░░┌─ Delete Experiment ─────────────────────┐░░░░░░░│ │
│  │░░░░░│                                         │░░░░░░░│ │
│  │░░░░░│  Are you sure you want to delete        │░░░░░░░│ │
│  │░░░░░│  experiment "gpt2-finetune-v3"?         │░░░░░░░│ │
│  │░░░░░│                                         │░░░░░░░│ │
│  │░░░░░│  This action cannot be undone.          │░░░░░░░│ │
│  │░░░░░│                                         │░░░░░░░│ │
│  │░░░░░│        [Cancel]        [Delete]         │░░░░░░░│ │
│  │░░░░░│                                         │░░░░░░░│ │
│  │░░░░░└─────────────────────────────────────────┘░░░░░░░│ │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ [Esc] Cancel  [Tab] Switch Button  [Enter] Confirm           │
└──────────────────────────────────────────────────────────────┘
```

**Modal rules**:
- Dim background (represented by `░`)
- Center modal vertically and horizontally
- Destructive button on right, styled with pink
- Escape always dismisses

---

## Component Catalog

### Panels

```
Default (unfocused):              Focused:
┌─ Panel Title ─────────┐        ┌─ Panel Title ─────────┐
│                       │        │                       │
│  Content here         │        │  Content here         │
│                       │        │                       │
└───────────────────────┘        └───────────────────────┘
 Border: #555c65                  Border: #78DCE8
 Title: #ABA0F2                   Title: #78DCE8
```

### Lists

```
Standard list item:
   item_name                      Color: #f0f2f5

Highlighted (hover/cursor):
 ▸ item_name                      Color: #f0f2f5, BG: #1a2233

Selected:
 ▸ item_name ✓                    Color: #F2C063, BG: #F2C063 @ 20%

Disabled:
   item_name                      Color: #555c65
```

### Tables

```
┌─ Results ─────────────────────────────────────────────────┐
│                                                           │
│  Experiment       Loss     Acc      Steps     Status      │
│  ──────────────────────────────────────────────────────   │
│  gpt2-base        0.342    89.2%    10,000    ✓ Done      │
│▸ gpt2-large       0.128    94.1%    8,240     ◐ Running   │
│  gpt2-xl          —        —        0         ○ Queued    │
│                                                           │
└───────────────────────────────────────────────────────────┘

Header row: Bold, #f0f2f5
Separator: ─ in #555c65
Numbers: #F28B66
Status icons: ✓ Green, ◐ Yellow, ○ Muted, ✗ Pink
```

### Progress Indicators

```
Determinate (known total):
  Training ████████████░░░░░░░░ 62%     Green fill, muted empty

Indeterminate (unknown total):
  Loading ◐                             Yellow, animated

Sparkline (mini graph):
  Loss ▂▃▅▇▆▄▃▂▁▁▁▁                     Cyan
```

### Buttons

```
Default:        [ Action ]              Border: #555c65, Text: #f0f2f5
Focused:        [ Action ]              Border: #78DCE8, Text: #78DCE8
Primary:        [ Confirm ]             BG: #78DCE8 @ 20%, Text: #78DCE8
Destructive:    [ Delete ]              BG: #F25E86 @ 20%, Text: #F25E86
Disabled:       [ Action ]              Border: #555c65, Text: #555c65
```

### Input Fields

```
Default:
  Label
  ┌────────────────────────────┐
  │ placeholder text           │     Border: #555c65, Text: #555c65
  └────────────────────────────┘

Focused:
  Label
  ┌────────────────────────────┐
  │ user input█                │     Border: #78DCE8, Text: #f0f2f5
  └────────────────────────────┘

Error:
  Label
  ┌────────────────────────────┐
  │ invalid input              │     Border: #F25E86, Text: #f0f2f5
  └────────────────────────────┘
  Invalid format                      Error text: #F25E86
```

### Status Badges

```
Running     ◐ Running        Yellow (#F2C063)
Success     ✓ Complete       Green (#B2D977)
Failed      ✗ Failed         Pink (#F25E86)
Queued      ○ Queued         Muted (#555c65)
Warning     ⚠ Warning        Yellow (#F2C063)
Info        ℹ Info           Cyan (#78DCE8)
```

### Footer / Keybinding Bar

```
┌──────────────────────────────────────────────────────────────┐
│ [q] Quit  [↑↓] Navigate  [Enter] Select  [/] Search  [?] Help│
└──────────────────────────────────────────────────────────────┘

Background: #141a26
Keys: #78DCE8, bold, inside brackets
Actions: #555c65
Separator: two spaces between items
```

---

## Interaction Patterns

### Navigation

| Key | Action | Scope |
|-----|--------|-------|
| `↑` `k` | Move up | Lists, tables |
| `↓` `j` | Move down | Lists, tables |
| `←` `h` | Previous panel / collapse | Multi-panel, trees |
| `→` `l` | Next panel / expand | Multi-panel, trees |
| `Tab` | Next focusable element | Global |
| `Shift+Tab` | Previous focusable element | Global |
| `Enter` | Select / confirm | Lists, buttons |
| `Space` | Toggle selection | Multi-select lists |
| `Esc` | Cancel / back / dismiss | Modals, search |

### Global Commands

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `?` | Show help |
| `/` | Open search |
| `1-9` | Switch tabs (when applicable) |
| `:` | Command palette (optional) |

### Mouse Support

- **Click** panel to focus
- **Click** list item to select
- **Scroll** in any scrollable region
- **Click** buttons to activate
- **Double-click** to expand/open (optional)

### Feedback Patterns

**Immediate actions**: Visual change happens instantly (selection highlight, focus ring).

**Short operations (<1s)**: Show brief flash or inline indicator.

**Long operations (>1s)**: 
- Show progress bar if determinate
- Show spinner if indeterminate
- Allow cancellation with `Esc` or `Ctrl+C`
- Show notification on completion

**Errors**:
- Inline validation for forms
- Toast notifications for operation failures
- Pink color + clear message
- Suggest remediation when possible

---

## Domain-Specific Guidelines

### Datasets Screen

```
┌─ Datasets ──────────────────────────────────────────────────┐
│                                                              │
│  ┌─────────┬───────────┬──────────┬─────────┐               │
│  │ Browse  │ Tokenize  │ Analyze  │ Create  │               │
│  └─────────┴───────────┴──────────┴─────────┘               │
│                                                              │
│  [Browse Tab]                                                │
│  ┌─ Datasets ─────────────────────────────────────────────┐ │
│  │ Name              Samples      Size      Tokenizer      │ │
│  │ ────────────────────────────────────────────────────── │ │
│  │ openwebtext       8,013,769    52.4 GB   gpt2    ✓     │ │
│  │▸pile-subset       2,100,000    12.1 GB   —       ○     │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  [Tokenize Tab]                                              │
│  - Dataset selector                                          │
│  - Tokenizer selector (dropdown: gpt2, llama, custom)        │
│  - Sequence length input                                     │
│  - Train/val/test split configuration                        │
│  - Progress display during tokenization                      │
│                                                              │
│  [Analyze Tab]                                               │
│  - Token distribution histogram                              │
│  - Sequence length statistics                                │
│  - Vocabulary coverage                                       │
│  - Sample preview                                            │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ [1-4] Tab  [t] Tokenize  [a] Analyze  [d] Delete  [?] Help   │
└──────────────────────────────────────────────────────────────┘
```

**Key features**:
- Tokenization status visible at a glance (✓ / ○ / ◐)
- Quick actions: `t` to tokenize selected, `a` to analyze
- Size shown in human-readable format (GB, MB)
- Sample counts with thousands separators

### Experiments Screen

```
┌─ Experiments ───────────────────────────────────────────────┐
│                                                              │
│  ┌─ Runs ─────────────────┐  ┌─ Details ─────────────────┐  │
│  │                        │  │                            │  │
│  │   exp_001   ✓ Done     │  │  gpt2-finetune-v3          │  │
│  │ ▸ exp_002   ◐ Running  │  │  ══════════════════════    │  │
│  │   exp_003   ○ Queued   │  │                            │  │
│  │   exp_004   ✗ Failed   │  │  Status    ◐ Running       │  │
│  │                        │  │  Started   2025-01-12      │  │
│  │                        │  │  Runtime   2h 34m          │  │
│  │                        │  │  Steps     1,240 / 10,000  │  │
│  │                        │  │                            │  │
│  │                        │  │  Config                    │  │
│  │                        │  │  ─────────────────────     │  │
│  │                        │  │  Model     gpt2-medium     │  │
│  │                        │  │  LR        3e-4            │  │
│  │                        │  │  Batch     32              │  │
│  │                        │  │                            │  │
│  │                        │  │  Loss ▂▃▅▇▆▄▃▂▁▁▁▁        │  │
│  │                        │  │                            │  │
│  └────────────────────────┘  └────────────────────────────┘  │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ [n] New  [Enter] View Logs  [s] Stop  [r] Restart  [?] Help  │
└──────────────────────────────────────────────────────────────┘
```

**Key features**:
- Status at a glance with color-coded icons
- Sparkline for loss trend (fits in small space)
- Key config values visible without drilling down
- Quick actions: stop, restart, view logs

### Experiment Detail / Logs View

```
┌─ exp_002: gpt2-finetune-v3 ─────────────────────────────────┐
│                                                              │
│  ┌─ Metrics ──────────────────────────────────────────────┐ │
│  │                                                         │ │
│  │  Loss        ████████████████░░░░  0.128               │ │
│  │  Accuracy    ██████████████████░░  94.1%               │ │
│  │  LR          ███░░░░░░░░░░░░░░░░░  3e-5 (decayed)      │ │
│  │                                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─ Log Output ───────────────────────────────────────────┐ │
│  │                                                         │ │
│  │  [14:32:01] Starting training...                        │ │
│  │  [14:32:05] Loaded checkpoint from step 1000            │ │
│  │  [14:35:12] Step 1100 | Loss: 0.142 | LR: 3.2e-5        │ │
│  │  [14:38:45] Step 1200 | Loss: 0.131 | LR: 3.1e-5        │ │
│  │  [14:42:18] Step 1300 | Loss: 0.128 | LR: 3.0e-5        │ │
│  │  █                                                      │ │
│  │                                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ [Esc] Back  [f] Follow Logs  [/] Search Logs  [e] Export     │
└──────────────────────────────────────────────────────────────┘
```

**Key features**:
- Metric bars with current values
- Live log output with auto-scroll option
- Searchable logs
- Export functionality

### Environment / Tools Screen

```
┌─ Tools ─────────────────────────────────────────────────────┐
│                                                              │
│  ┌─ Environment ──────────────────────────────────────────┐ │
│  │                                                         │ │
│  │  Python        3.11.5        ✓                         │ │
│  │  PyTorch       2.1.0+cu121   ✓                         │ │
│  │  CUDA          12.1          ✓                         │ │
│  │  GPU           RTX 4090      24GB free / 24GB          │ │
│  │  Disk          /data         142GB free / 1TB          │ │
│  │                                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─ Quick Actions ────────────────────────────────────────┐ │
│  │                                                         │ │
│  │  [ Clear Cache ]    [ Sync Checkpoints ]    [ Backup ] │ │
│  │                                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ [r] Refresh  [c] Clear Cache  [?] Help                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Implementation Reference

### Textual CSS (styles.tcss)

```css
/* ============================================
   BASE STYLES
   ============================================ */

Screen {
    background: #0a1220;
    color: #f0f2f5;
}

/* ============================================
   PANELS
   ============================================ */

.panel {
    border: solid #555c65;
    border-title-color: #ABA0F2;
    border-title-style: bold;
    padding: 0 1;
}

.panel:focus-within {
    border: solid #78DCE8;
    border-title-color: #78DCE8;
}

/* ============================================
   LISTS
   ============================================ */

ListView {
    background: transparent;
}

ListView > ListItem {
    padding: 0 1;
    background: transparent;
}

ListView > ListItem:hover {
    background: #1a2233;
}

ListView > ListItem.-selected {
    background: rgba(242, 192, 99, 0.2);
    color: #F2C063;
}

ListView:focus > ListItem.-selected {
    background: rgba(242, 192, 99, 0.25);
}

/* ============================================
   TABLES
   ============================================ */

DataTable {
    background: transparent;
}

DataTable > .datatable--header {
    color: #f0f2f5;
    text-style: bold;
    background: transparent;
}

DataTable > .datatable--cursor {
    background: #1a2233;
}

DataTable:focus > .datatable--cursor {
    background: rgba(242, 192, 99, 0.2);
    color: #F2C063;
}

/* ============================================
   BUTTONS
   ============================================ */

Button {
    border: tall #555c65;
    background: transparent;
    color: #f0f2f5;
    padding: 0 2;
    margin: 0 1;
}

Button:hover {
    border: tall #78DCE8;
    color: #78DCE8;
}

Button:focus {
    border: tall #78DCE8;
    color: #78DCE8;
    text-style: bold;
}

Button.-primary {
    border: tall #78DCE8;
    background: rgba(120, 220, 232, 0.15);
    color: #78DCE8;
}

Button.-destructive {
    border: tall #F25E86;
    background: rgba(242, 94, 134, 0.15);
    color: #F25E86;
}

Button.-success {
    border: tall #B2D977;
    background: rgba(178, 217, 119, 0.15);
    color: #B2D977;
}

/* ============================================
   INPUT FIELDS
   ============================================ */

Input {
    border: tall #555c65;
    background: transparent;
    padding: 0 1;
}

Input:focus {
    border: tall #78DCE8;
}

Input.-invalid {
    border: tall #F25E86;
}

/* ============================================
   TABS
   ============================================ */

Tabs {
    background: transparent;
}

Tab {
    color: #555c65;
    padding: 0 2;
}

Tab:hover {
    color: #f0f2f5;
}

Tab.-active {
    color: #78DCE8;
    text-style: bold underline;
}

/* ============================================
   PROGRESS BARS
   ============================================ */

ProgressBar > .bar--bar {
    color: #B2D977;
    background: #555c65;
}

ProgressBar > .bar--complete {
    color: #B2D977;
}

/* ============================================
   FOOTER
   ============================================ */

Footer {
    background: #141a26;
    color: #555c65;
    height: 1;
}

Footer > .footer--key {
    color: #78DCE8;
    text-style: bold;
    background: transparent;
}

Footer > .footer--description {
    color: #555c65;
}

/* ============================================
   MODALS
   ============================================ */

ModalScreen {
    align: center middle;
}

#modal-container {
    width: 60;
    height: auto;
    border: solid #555c65;
    background: #0a1220;
    padding: 1 2;
}

#modal-container:focus-within {
    border: solid #78DCE8;
}

/* ============================================
   STATUS INDICATORS
   ============================================ */

.status-running {
    color: #F2C063;
}

.status-success {
    color: #B2D977;
}

.status-failed {
    color: #F25E86;
}

.status-queued {
    color: #555c65;
}

/* ============================================
   UTILITY CLASSES
   ============================================ */

.text-muted {
    color: #555c65;
}

.text-primary {
    color: #78DCE8;
}

.text-success {
    color: #B2D977;
}

.text-warning {
    color: #F2C063;
}

.text-error {
    color: #F25E86;
}

.text-number {
    color: #F28B66;
}

.bold {
    text-style: bold;
}
```

### Python Theme Definition

```python
# theme.py
"""Monokai Pro theme colors for Omarchy/Ghostty compatibility."""

class MonokaiOmarchy:
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


# Status icons
STATUS_ICONS = {
    "running": ("◐", MonokaiOmarchy.YELLOW),
    "success": ("✓", MonokaiOmarchy.GREEN),
    "failed": ("✗", MonokaiOmarchy.PINK),
    "queued": ("○", MonokaiOmarchy.MUTED),
    "warning": ("⚠", MonokaiOmarchy.YELLOW),
    "info": ("ℹ", MonokaiOmarchy.CYAN),
}
```

### Recommended Project Structure

```
research_tui/
├── __init__.py
├── app.py                 # Main Textual application
├── theme.py               # Color definitions (above)
├── styles.tcss            # Global styles (above)
├── screens/
│   ├── __init__.py
│   ├── dashboard.py       # Home/overview screen
│   ├── datasets.py        # Dataset management
│   ├── experiments.py     # Experiment browser
│   ├── experiment_detail.py
│   └── tools.py           # Environment/utilities
├── components/
│   ├── __init__.py
│   ├── panels.py          # Reusable panel widgets
│   ├── lists.py           # Custom list implementations
│   ├── tables.py          # Data table variants
│   ├── sparkline.py       # Mini graph widget
│   ├── status.py          # Status badges/indicators
│   └── footer.py          # Keybinding footer
└── utils/
    ├── __init__.py
    ├── formatting.py      # Number formatting, dates
    └── keybindings.py     # Keybinding definitions
```

---

## Appendix: Character Reference

### Box Drawing (for custom borders)

```
Single line:  ─ │ ┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼
Double line:  ═ ║ ╔ ╗ ╚ ╝ ╠ ╣ ╦ ╩ ╬
Rounded:      ╭ ╮ ╰ ╯
```

### Progress/Indicators

```
Blocks:       ░ ▒ ▓ █
Bars:         ▏▎▍▌▋▊▉█
Spinners:     ◐ ◓ ◑ ◒  or  ⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏
Sparkline:    ▁ ▂ ▃ ▄ ▅ ▆ ▇ █
```

### Status Icons

```
Success:      ✓ ✔ ●
Error:        ✗ ✘ ○
Warning:      ⚠ ▲
Info:         ℹ ●
Running:      ◐ ◑ ◒ ◓
Arrow:        ▸ ▹ ► ▶
```

---

*Last updated: 2025-01-13*
*Compatible with: Textual 0.40+, Ghostty, Kitty, Alacritty*
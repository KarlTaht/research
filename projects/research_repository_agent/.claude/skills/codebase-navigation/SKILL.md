---
name: codebase-navigation
description: Navigate and understand the ML research monorepo structure. Use when user asks about projects, scripts, configs, or where things are located in the codebase.
---

# Codebase Navigation

Help users explore and understand the ML research monorepo structure.

## When to Use

- "What projects exist?"
- "Where is the training script?"
- "Show me the config for X"
- "What's in the custom_transformer project?"
- "Find scripts for training"

## Repository Structure

```
~/research/
├── common/           # Shared Python package
│   ├── models/       # Model architectures (BaseLanguageModel)
│   ├── training/     # Training loops, evaluators
│   ├── data/         # Dataset loaders, HuggingFace utils
│   └── utils/        # Logging, checkpointing, experiment storage
│
├── projects/         # Research projects
│   └── <project>/
│       ├── train.py
│       ├── evaluate.py
│       └── configs/
│
├── assets/           # Data and outputs (gitignored)
│   ├── datasets/
│   ├── models/
│   └── outputs/
│
└── tools/            # CLI utilities
```

## Available Tools

| Tool | Purpose |
|------|---------|
| `explore_repo` | Show directory structure |
| `list_projects` | List all projects with status |
| `get_project` | Detailed info about one project |
| `find_script` | Find scripts by task |
| `list_scripts` | List all scripts |
| `read_config` | Parse and explain YAML configs |
| `compare_configs` | Diff two configs |
| `list_configs` | List all configs |

## Common Patterns

### Finding a Training Script

```
User: "How do I train the custom transformer?"

1. Use list_projects() to find the project
2. Use get_project(name='custom_transformer') for details
3. Use read_config() to show the config options
4. Provide the exact command: python train.py --config configs/default.yaml
```

### Exploring a New Project

```
User: "What's in the embedded_attention project?"

1. Use explore_repo(path='projects/embedded_attention', depth=2)
2. Summarize the structure and key files
3. Read CLAUDE.md or README.md for description
```

### Comparing Configurations

```
User: "What's different between small and large configs?"

1. Use list_configs(project='...') to find config paths
2. Use compare_configs(config1='...', config2='...')
3. Highlight the important differences (model size, batch size, etc.)
```

## Conventions

See [CONVENTIONS.md](CONVENTIONS.md) for repo-specific naming patterns and conventions.

"""CLI utilities for the research framework.

This module provides command-line interfaces for:
- Data operations (download, analyze, pretokenize)
- Experiment tracking (query, compare, list)
- Infrastructure (environment check, cloud availability)

Usage:
    # Run individual CLIs
    python -m common.cli.data download dataset --name squad
    python -m common.cli.experiments list
    python -m common.cli.infra env

    # Launch TUI
    python -m common.cli.tui
"""

__all__ = []

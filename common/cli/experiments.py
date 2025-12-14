#!/usr/bin/env python3
"""Unified CLI for experiment tracking.

Usage:
    python -m common.cli.experiments <command> [options]

Commands:
    list       List all experiments
    query      Run SQL query
    compare    Compare experiments
    best       Get best by metric
    load       Load single experiment
    summary    Show summary stats

Examples:
    python -m common.cli.experiments list
    python -m common.cli.experiments summary
    python -m common.cli.experiments best perplexity --minimize --top 5
    python -m common.cli.experiments query "SELECT * FROM experiments WHERE epoch > 5"
    python -m common.cli.experiments compare exp_001 exp_002
    python -m common.cli.experiments load exp_001
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    # Map commands to internal CLI args
    if command == "list":
        sys.argv = [sys.argv[0], "--list"] + sys.argv[2:]
    elif command == "summary":
        sys.argv = [sys.argv[0], "--summary"] + sys.argv[2:]
    elif command == "best":
        if len(sys.argv) < 3:
            print("Usage: python -m common.cli.experiments best <metric> [--minimize|--no-minimize] [--top N]")
            sys.exit(1)
        metric = sys.argv[2]
        sys.argv = [sys.argv[0], "--best", metric] + sys.argv[3:]
    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python -m common.cli.experiments query <SQL>")
            sys.exit(1)
        sql = sys.argv[2]
        sys.argv = [sys.argv[0], "--sql", sql] + sys.argv[3:]
    elif command == "compare":
        if len(sys.argv) < 3:
            print("Usage: python -m common.cli.experiments compare <exp1> <exp2> ...")
            sys.exit(1)
        sys.argv = [sys.argv[0], "--compare"] + sys.argv[2:]
    elif command == "load":
        if len(sys.argv) < 3:
            print("Usage: python -m common.cli.experiments load <exp_name>")
            sys.exit(1)
        exp_name = sys.argv[2]
        sys.argv = [sys.argv[0], "--load", exp_name] + sys.argv[3:]
    elif command in ("-h", "--help", "help"):
        print(__doc__)
        sys.exit(0)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

    from common.cli._internal.query_experiments import main as query_main
    query_main()


if __name__ == "__main__":
    main()

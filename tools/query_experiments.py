#!/usr/bin/env python3
"""CLI tool for querying experiment results stored in Parquet format."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import common
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import (
    query_experiments,
    list_experiments,
    get_experiment_summary,
    get_best_experiments,
    compare_experiments,
    load_experiment,
)


def main():
    parser = argparse.ArgumentParser(
        description="Query experiment results using SQL or pre-defined filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python tools/query_experiments.py --list

  # Show summary of all experiments
  python tools/query_experiments.py --summary

  # Get top 5 experiments by lowest perplexity
  python tools/query_experiments.py --best perplexity --minimize --top 5

  # Get top 3 experiments by highest BLEU score
  python tools/query_experiments.py --best bleu_score --no-minimize --top 3

  # Custom SQL query
  python tools/query_experiments.py --sql "SELECT * FROM experiments WHERE epoch > 5"

  # Compare specific experiments
  python tools/query_experiments.py --compare exp_001 exp_002 exp_003

  # Load single experiment
  python tools/query_experiments.py --load exp_001
        """,
    )

    # Action group (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)

    action_group.add_argument(
        "--list", action="store_true", help="List all available experiments"
    )

    action_group.add_argument(
        "--summary", action="store_true", help="Show summary of all experiments"
    )

    action_group.add_argument(
        "--best",
        type=str,
        metavar="METRIC",
        help="Get top experiments by metric (e.g., 'perplexity', 'bleu_score')",
    )

    action_group.add_argument(
        "--sql", type=str, metavar="QUERY", help="Execute custom SQL query"
    )

    action_group.add_argument(
        "--compare",
        nargs="+",
        metavar="EXP_NAME",
        help="Compare specific experiments",
    )

    action_group.add_argument(
        "--load", type=str, metavar="EXP_NAME", help="Load a single experiment"
    )

    # Optional arguments for --best
    parser.add_argument(
        "--minimize",
        action="store_true",
        default=True,
        help="For --best: minimize metric (lower is better). Default: True",
    )

    parser.add_argument(
        "--no-minimize",
        action="store_false",
        dest="minimize",
        help="For --best: maximize metric (higher is better)",
    )

    parser.add_argument(
        "--top", type=int, default=10, help="For --best: number of top results (default: 10)"
    )

    # Optional arguments for --compare
    parser.add_argument(
        "--metrics",
        nargs="+",
        metavar="METRIC",
        help="For --compare: specific metrics to include",
    )

    # Output format
    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table)",
    )

    parser.add_argument(
        "--output", type=str, help="Save results to file (auto-detect format from extension)"
    )

    args = parser.parse_args()

    try:
        # Execute requested action
        if args.list:
            experiments = list_experiments()
            print(f"Found {len(experiments)} experiments:")
            for exp in sorted(experiments):
                print(f"  - {exp}")
            return

        elif args.summary:
            df = get_experiment_summary()
            print_dataframe(df, args.format, args.output)

        elif args.best:
            df = get_best_experiments(
                metric=args.best, minimize=args.minimize, top_n=args.top
            )
            print(f"\nTop {args.top} experiments by {args.best}:")
            print(f"({'minimizing' if args.minimize else 'maximizing'})\n")
            print_dataframe(df, args.format, args.output)

        elif args.sql:
            df = query_experiments(args.sql)
            print_dataframe(df, args.format, args.output)

        elif args.compare:
            df = compare_experiments(args.compare, metrics=args.metrics)
            print(f"\nComparing experiments: {', '.join(args.compare)}\n")
            print_dataframe(df, args.format, args.output)

        elif args.load:
            df = load_experiment(args.load)
            print(f"\nExperiment: {args.load}\n")
            print_dataframe(df, args.format, args.output)

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


def print_dataframe(df, format_type="table", output_file=None):
    """Print DataFrame in specified format."""
    if output_file:
        # Auto-detect format from file extension
        ext = Path(output_file).suffix.lower()
        if ext == ".csv":
            df.to_csv(output_file, index=False)
            print(f"✓ Saved to: {output_file}")
        elif ext == ".json":
            df.to_json(output_file, orient="records", indent=2)
            print(f"✓ Saved to: {output_file}")
        elif ext == ".parquet":
            df.to_parquet(output_file, index=False)
            print(f"✓ Saved to: {output_file}")
        else:
            print(f"✗ Unsupported file extension: {ext}", file=sys.stderr)
            sys.exit(1)
    else:
        # Print to stdout
        if format_type == "table":
            print(df.to_string())
        elif format_type == "csv":
            print(df.to_csv(index=False))
        elif format_type == "json":
            print(df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()

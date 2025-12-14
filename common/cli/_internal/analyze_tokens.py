#!/usr/bin/env python
"""CLI tool for analyzing token distributions in datasets.

This tool helps understand vocabulary usage patterns and recommends
optimal tokenizer sizes for small-scale experiments.

Examples:
    # Analyze TinyStories dataset (from registry)
    python -m common.cli._internal.analyze_tokens --dataset tinystories

    # Analyze a JSONL file directly
    python -m common.cli._internal.analyze_tokens --jsonl projects/continual_learning/data/corpus_automotive/train.jsonl

    # Analyze with a subset for faster results
    python -m common.cli._internal.analyze_tokens --dataset tinystories --subset 10000

    # Compare multiple datasets
    python -m common.cli._internal.analyze_tokens --dataset tinystories tiny-textbooks --compare

    # Export CDF data for plotting
    python -m common.cli._internal.analyze_tokens --dataset tinystories --export-cdf output.csv

    # Show essential tokens for 99% coverage
    python -m common.cli._internal.analyze_tokens --dataset tinystories --show-tokens 0.99

    # Train a custom tokenizer from a registry dataset
    python -m common.cli._internal.analyze_tokens --dataset tinystories --train-tokenizer 4096

    # Train a custom tokenizer from a JSONL file
    python -m common.cli._internal.analyze_tokens --jsonl data/corpus_automotive/train.jsonl --train-tokenizer 16384

    # Train a tokenizer from multiple JSONL files (combined corpora)
    python -m common.cli._internal.analyze_tokens --jsonl data/corpus_automotive/train.jsonl data/corpus_food/train.jsonl --train-tokenizer 32768

    # Get recommendations for a JSONL corpus
    python -m common.cli._internal.analyze_tokens --jsonl data/corpus_automotive/train.jsonl --recommendations
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import GPT2TokenizerFast

from common.data.token_analyzer import (
    TokenAnalyzer,
    train_custom_tokenizer,
    train_custom_tokenizer_from_file,
)
from common.data.dataset_registry import list_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token distributions in datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        nargs="+",
        help="Dataset name(s) to analyze (from registry)",
    )
    parser.add_argument(
        "--jsonl",
        nargs="+",
        metavar="PATH",
        help="Analyze JSONL file(s) directly (expects {\"text\": \"...\"} format)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets in the registry",
    )
    parser.add_argument(
        "--subset",
        "-n",
        type=int,
        default=None,
        help="Only analyze first N examples (for faster results)",
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare multiple datasets side by side",
    )
    parser.add_argument(
        "--export-cdf",
        type=str,
        metavar="PATH",
        help="Export CDF data to CSV file",
    )
    parser.add_argument(
        "--show-tokens",
        type=float,
        metavar="COVERAGE",
        help="Show essential tokens for given coverage (e.g., 0.99)",
    )
    parser.add_argument(
        "--show-edge-cases",
        action="store_true",
        help="Show detailed edge case analysis",
    )
    parser.add_argument(
        "--recommendations",
        "-r",
        action="store_true",
        help="Show tokenizer size recommendations",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=10,
        help="Threshold for 'rare' token classification (default: 10)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--train-tokenizer",
        type=int,
        metavar="VOCAB_SIZE",
        help="Train a custom BPE tokenizer with specified vocab size",
    )
    parser.add_argument(
        "--tokenizer-output",
        type=str,
        metavar="PATH",
        help="Output directory for trained tokenizer (default: assets/models/tokenizers/)",
    )
    parser.add_argument(
        "--train-subset",
        type=int,
        default=None,
        help="Number of examples to use for tokenizer training (default: all)",
    )

    args = parser.parse_args()

    # List datasets
    if args.list_datasets:
        print("Available datasets:")
        for name, desc in list_datasets().items():
            print(f"  {name}: {desc}")
        return 0

    # Validate arguments
    if not args.dataset and not args.jsonl:
        parser.error("--dataset or --jsonl is required (or use --list-datasets)")

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    # Create analyzer
    analyzer = TokenAnalyzer(tokenizer, rare_threshold=args.rare_threshold)

    # Analyze datasets
    reports = []

    # Handle JSONL file analysis
    if args.jsonl:
        for jsonl_file in args.jsonl:
            jsonl_path = Path(jsonl_file)
            print(f"\n{'='*60}")
            print(f"Analyzing JSONL: {jsonl_path}")
            print(f"{'='*60}")

            try:
                # Load texts from JSONL
                texts = []
                with open(jsonl_path) as f:
                    for line in f:
                        texts.append(json.loads(line)["text"])

                print(f"Loaded {len(texts):,} texts")

                # Apply subset if requested
                if args.subset and args.subset < len(texts):
                    texts = texts[:args.subset]

                # Get name from path
                name = jsonl_path.stem
                if name in ("train", "val", "test"):
                    name = jsonl_path.parent.name

                report = analyzer.analyze_texts(
                    texts,
                    name=name,
                    show_progress=not args.quiet,
                )
                reports.append(report)

                # Print summary
                print("\n" + report.summary())

            except Exception as e:
                print(f"Error analyzing {jsonl_path}: {e}")
                import traceback
                traceback.print_exc()

    # Handle registry dataset analysis
    if args.dataset:
        for dataset_name in args.dataset:
            print(f"\n{'='*60}")
            print(f"Analyzing: {dataset_name}")
            print(f"{'='*60}")

            try:
                report = analyzer.analyze_dataset(
                    dataset_name,
                    subset_size=args.subset,
                    show_progress=not args.quiet,
                )
                reports.append(report)

                # Print summary
                print("\n" + report.summary())

            except FileNotFoundError as e:
                print(f"Error: {e}")
                print(f"Try downloading with: python -m common.cli.data download dataset --name <hf-path>")
                continue
            except Exception as e:
                print(f"Error analyzing {dataset_name}: {e}")
                continue

    if not reports:
        print("No datasets were successfully analyzed.")
        return 1

    # Comparison mode
    if args.compare and len(reports) > 1:
        print("\n" + analyzer.compare_datasets(reports))

    # Export CDF
    if args.export_cdf and reports:
        report = reports[0]  # Export first dataset
        output_path = Path(args.export_cdf)
        analyzer.export_cdf_data(report, output_path)
        print(f"\nCDF data exported to: {output_path}")

    # Show recommendations
    if args.recommendations:
        for report in reports:
            print(f"\n{'='*60}")
            print(f"Recommendations for: {report.dataset_name}")
            print(f"{'='*60}")

            recommendations = analyzer.recommend_vocab_size(report)
            for rec in recommendations:
                print(f"\n{rec.summary()}")

    # Show essential tokens
    if args.show_tokens:
        for report in reports:
            print(f"\n{'='*60}")
            print(f"Essential tokens for {args.show_tokens:.1%} coverage: {report.dataset_name}")
            print(f"{'='*60}")

            tokens = analyzer.get_essential_tokens(report, args.show_tokens)
            print(f"\nShowing top 50 of {len(tokens)} tokens:")
            print(f"{'ID':>8} {'Token':<20} {'Count':>12} {'Pct':>8}")
            print("-" * 52)

            for i, (token_id, token_str, count) in enumerate(tokens[:50]):
                # Escape special characters for display
                display_str = repr(token_str)[1:-1]  # Remove quotes
                if len(display_str) > 18:
                    display_str = display_str[:15] + "..."
                pct = count / report.total_tokens * 100
                print(f"{token_id:>8} {display_str:<20} {count:>12,} {pct:>7.2f}%")

    # Show edge cases
    if args.show_edge_cases:
        for report in reports:
            print(f"\n{'='*60}")
            print(f"Edge Case Details: {report.dataset_name}")
            print(f"{'='*60}")

            edge = report.edge_cases

            # Show some examples from each category
            categories = [
                ("Single Character", edge.single_char_tokens),
                ("Punctuation", edge.punctuation_tokens),
                ("Numbers", edge.number_tokens),
                ("Hyphenated", edge.hyphenated_tokens),
            ]

            for cat_name, cat_tokens in categories:
                if not cat_tokens:
                    continue

                print(f"\n{cat_name} tokens (top 10):")
                sorted_tokens = sorted(cat_tokens.items(), key=lambda x: -x[1])[:10]
                for token_id, count in sorted_tokens:
                    token_str = analyzer._decode_token(token_id)
                    display_str = repr(token_str)[1:-1]
                    print(f"  {token_id:>6}: {display_str:<15} ({count:,} occurrences)")

    # Train custom tokenizer
    if args.train_tokenizer:
        output_dir = Path(args.tokenizer_output) if args.tokenizer_output else None

        # Train from JSONL file(s)
        if args.jsonl:
            jsonl_paths = [Path(p) for p in args.jsonl]
            print(f"\n{'='*60}")
            if len(jsonl_paths) == 1:
                print(f"Training tokenizer from: {jsonl_paths[0]}")
            else:
                print(f"Training tokenizer from {len(jsonl_paths)} files:")
                for p in jsonl_paths:
                    print(f"  - {p}")
            print(f"{'='*60}")

            try:
                tokenizer_path = train_custom_tokenizer_from_file(
                    jsonl_paths,
                    vocab_size=args.train_tokenizer,
                    output_dir=output_dir,
                    subset_size=args.train_subset,
                    show_progress=not args.quiet,
                )
                print(f"\nTokenizer saved to: {tokenizer_path}")
                print(f"\nTo use the tokenizer:")
                print(f"  from transformers import PreTrainedTokenizerFast")
                print(f"  tokenizer = PreTrainedTokenizerFast.from_pretrained('{tokenizer_path}')")

            except Exception as e:
                print(f"Error training tokenizer: {e}")
                import traceback
                traceback.print_exc()

        # Train from registry datasets
        if args.dataset:
            for dataset_name in args.dataset:
                print(f"\n{'='*60}")
                print(f"Training tokenizer for: {dataset_name}")
                print(f"{'='*60}")

                try:
                    tokenizer_path = train_custom_tokenizer(
                        dataset_name,
                        vocab_size=args.train_tokenizer,
                        output_dir=output_dir,
                        subset_size=args.train_subset,
                        show_progress=not args.quiet,
                    )
                    print(f"\nTokenizer saved to: {tokenizer_path}")
                    print(f"\nTo use the tokenizer:")
                    print(f"  from transformers import PreTrainedTokenizerFast")
                    print(f"  tokenizer = PreTrainedTokenizerFast.from_pretrained('{tokenizer_path}')")

                except Exception as e:
                    print(f"Error training tokenizer: {e}")
                    import traceback
                    traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())

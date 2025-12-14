#!/usr/bin/env python3
"""CLI script for downloading datasets from HuggingFace Hub."""

import argparse
from pathlib import Path
import sys

from common.data.hf_utils import download_dataset, get_datasets_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location (assets/datasets/)
  python -m common.cli._internal.download_dataset --name wmt14

  # Download specific split
  python -m common.cli._internal.download_dataset --name squad --split train

  # Download to custom location
  python -m common.cli._internal.download_dataset --name imagenet-1k --output /path/to/data

  # Download with config
  python -m common.cli._internal.download_dataset --name wmt14 --config de-en

  # Download FineWeb sample
  python -m common.cli._internal.download_dataset --name HuggingFaceFW/fineweb --config sample-100BT
        """,
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name on HuggingFace Hub (e.g., 'wmt14', 'squad')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ~/research/assets/datasets/<name>)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Specific split to download (e.g., 'train', 'test'). If not specified, downloads all.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dataset configuration/subset name",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory (default: ~/.cache/huggingface/datasets)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        output_dir = get_datasets_dir() / args.name
        # Append config name to output directory if specified
        if args.config:
            output_dir = output_dir / args.config

    print(f"Downloading dataset: {args.name}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    try:
        download_dataset(
            name=args.name,
            config=args.config,
            output_dir=output_dir,
            split=args.split,
            cache_dir=args.cache_dir,
        )

        print("-" * 50)
        print(f"Success! Dataset saved to: {output_dir}")

    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

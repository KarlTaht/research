#!/usr/bin/env python3
"""CLI script for downloading datasets from HuggingFace Hub."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path to import common
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.data.hf_utils import download_dataset, get_datasets_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location (assets/datasets/)
  python tools/download_hf_dataset.py --name wmt14

  # Download specific split
  python tools/download_hf_dataset.py --name squad --split train

  # Download to custom location
  python tools/download_hf_dataset.py --name imagenet-1k --output /path/to/data

  # Download with config
  python tools/download_hf_dataset.py --name wmt14 --config de-en
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

    print(f"Downloading dataset: {args.name}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    try:
        # Build kwargs for load_dataset
        kwargs = {}
        if args.config:
            kwargs["name"] = args.config

        download_dataset(
            name=args.name,
            output_dir=output_dir,
            split=args.split,
            cache_dir=args.cache_dir,
            **kwargs,
        )

        print("-" * 50)
        print(f"✓ Success! Dataset saved to: {output_dir}")

    except Exception as e:
        print(f"✗ Error downloading dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

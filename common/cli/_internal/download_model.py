#!/usr/bin/env python3
"""CLI script for downloading models from HuggingFace Hub."""

import argparse
from pathlib import Path
import sys

from common.data.hf_utils import download_model, get_models_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download entire model to default location (assets/models/)
  python -m common.cli._internal.download_model --repo-id bert-base-uncased

  # Download specific file
  python -m common.cli._internal.download_model --repo-id bert-base-uncased --filename pytorch_model.bin

  # Download to custom location
  python -m common.cli._internal.download_model --repo-id openai/clip-vit-base-patch32 --output /path/to/models

  # Download specific revision (branch/tag/commit)
  python -m common.cli._internal.download_model --repo-id facebook/opt-1.3b --revision main
        """,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Model repository ID (e.g., 'bert-base-uncased', 'openai/clip-vit-base-patch32')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ~/research/assets/models/<repo-name>)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Specific file to download. If not specified, downloads entire repository.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Git revision - branch, tag, or commit hash (default: main)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory (default: ~/.cache/huggingface/hub)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        # Use last part of repo_id as directory name
        repo_name = args.repo_id.split("/")[-1]
        output_dir = get_models_dir() / repo_name

    print(f"Downloading model: {args.repo_id}")
    if args.filename:
        print(f"File: {args.filename}")
    else:
        print("Mode: Full repository")
    print(f"Revision: {args.revision}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    try:
        result_path = download_model(
            repo_id=args.repo_id,
            output_dir=output_dir,
            filename=args.filename,
            cache_dir=args.cache_dir,
            revision=args.revision,
        )

        print("-" * 50)
        print(f"Success! Model saved to: {result_path}")

    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download a small sample from FineWeb dataset for testing."""

import sys
from pathlib import Path
from datasets import load_dataset

# Add parent directory to path to import common
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.data import get_datasets_dir


def download_small_sample(target_tokens: int = 10_000_000):
    """
    Download a small sample from FineWeb by streaming sample-10BT.

    Args:
        target_tokens: Number of tokens to download (default: 10M)
    """
    output_dir = get_datasets_dir() / "fineweb" / "test-sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ~{target_tokens:,} tokens from FineWeb sample-10BT...")
    print(f"Output directory: {output_dir}")
    print("-" * 70)

    # Stream the dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    # Collect samples until we reach target tokens
    samples = []
    total_tokens = 0

    print("Streaming and collecting samples...")
    for i, sample in enumerate(dataset):
        token_count = sample.get('token_count', 0)
        total_tokens += token_count
        samples.append(sample)

        if i % 100 == 0:
            print(f"  Collected {i+1} samples, {total_tokens:,} tokens so far...")

        if total_tokens >= target_tokens:
            print(f"\n✓ Reached target: {total_tokens:,} tokens in {len(samples)} samples")
            break

    # Convert to dataset and save
    print(f"\nSaving to {output_dir}...")
    from datasets import Dataset
    ds = Dataset.from_list(samples)
    ds.save_to_disk(str(output_dir))

    print("-" * 70)
    print(f"✓ Success!")
    print(f"  Samples: {len(samples):,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Location: {output_dir}")

    # Show a sample
    print("\nSample entry:")
    print(f"  URL: {samples[0]['url']}")
    print(f"  Tokens: {samples[0]['token_count']}")
    print(f"  Text preview: {samples[0]['text'][:200]}...")

    return ds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download a small test sample from FineWeb dataset"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=10_000_000,
        help="Target number of tokens to download (default: 10M)"
    )

    args = parser.parse_args()

    try:
        download_small_sample(target_tokens=args.tokens)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""
Build a domain index for FineWeb sample-100BT dataset.

Creates a lightweight parquet index with:
- shard_file: Arrow file name
- row_idx: Row index within shard
- domain: Full domain (e.g., forum.openvz.org)
- etld_plus_one: eTLD+1 (e.g., openvz.org)
- tld: Top-level domain (e.g., .org)
- url_path: URL path after domain (e.g., /wiki/BMW_3_Series)
- token_count: Token count from dataset

Usage:
    # Full run (all shards)
    python build_domain_index.py

    # Incremental runs
    python build_domain_index.py --start-shard 0 --num-shards 100
    python build_domain_index.py --start-shard 100 --num-shards 100

    # Check progress
    python build_domain_index.py --status
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import tldextract
from tqdm import tqdm

# Paths
FINEWEB_DIR = Path("/home/ktaht/research/assets/datasets/HuggingFaceFW/fineweb/sample-100BT/train")
OUTPUT_PATH = Path("/home/ktaht/research/assets/datasets/HuggingFaceFW/fineweb/domain_index.parquet")
PROGRESS_PATH = OUTPUT_PATH.with_suffix(".progress")


def get_shard_files() -> list[Path]:
    """Get sorted list of Arrow shard files."""
    files = sorted(FINEWEB_DIR.glob("data-*.arrow"))
    return files


def extract_domain_info(url: str) -> tuple[str, str, str, str]:
    """Extract domain, eTLD+1, TLD, and URL path from URL.

    Returns:
        (domain, etld_plus_one, tld, url_path)
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc or parsed.path.split("/")[0]

        # Remove www. prefix for consistency
        if hostname.startswith("www."):
            hostname = hostname[4:]

        # Use tldextract for proper eTLD+1 parsing
        extracted = tldextract.extract(url)

        # Build eTLD+1 (e.g., "example.co.uk")
        if extracted.suffix:
            etld_plus_one = f"{extracted.domain}.{extracted.suffix}"
            tld = f".{extracted.suffix}"
        else:
            etld_plus_one = extracted.domain
            tld = ""

        # Extract URL path (e.g., "/wiki/BMW_3_Series")
        url_path = parsed.path or ""

        return hostname, etld_plus_one, tld, url_path
    except Exception:
        return "", "", "", ""


def process_shard(shard_path: Path) -> list[dict]:
    """Process a single Arrow shard file and extract domain info."""
    rows = []
    shard_name = shard_path.name

    # Read Arrow file using streaming reader
    with pa.memory_map(str(shard_path), "r") as source:
        reader = pa.ipc.open_stream(source)

        row_idx = 0
        for batch in reader:
            urls = batch.column("url").to_pylist()
            token_counts = batch.column("token_count").to_pylist()

            for url, token_count in zip(urls, token_counts):
                domain, etld_plus_one, tld, url_path = extract_domain_info(url)

                rows.append({
                    "shard_file": shard_name,
                    "row_idx": row_idx,
                    "domain": domain,
                    "etld_plus_one": etld_plus_one,
                    "tld": tld,
                    "url_path": url_path,
                    "token_count": token_count,
                })
                row_idx += 1

    return rows


def save_progress(completed_shards: list[str]):
    """Save list of completed shards to progress file."""
    PROGRESS_PATH.write_text("\n".join(completed_shards))


def load_progress() -> set[str]:
    """Load set of completed shards from progress file."""
    if PROGRESS_PATH.exists():
        return set(PROGRESS_PATH.read_text().strip().split("\n"))
    return set()


def show_status():
    """Show current indexing progress."""
    shard_files = get_shard_files()
    total_shards = len(shard_files)
    completed = load_progress()

    print(f"Total shards: {total_shards}")
    print(f"Completed: {len(completed)}")
    print(f"Remaining: {total_shards - len(completed)}")
    print(f"Progress: {len(completed) / total_shards * 100:.1f}%")

    if OUTPUT_PATH.exists():
        # Read parquet metadata
        pf = pq.ParquetFile(OUTPUT_PATH)
        print(f"\nIndex file: {OUTPUT_PATH}")
        print(f"  Rows: {pf.metadata.num_rows:,}")
        print(f"  Size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")


def build_index(start_shard: int = 0, num_shards: int | None = None):
    """Build the domain index for specified shard range."""
    shard_files = get_shard_files()
    total_shards = len(shard_files)

    # Determine range
    end_shard = total_shards if num_shards is None else min(start_shard + num_shards, total_shards)
    shards_to_process = shard_files[start_shard:end_shard]

    # Load existing progress
    completed = load_progress()

    # Filter out already completed shards
    shards_to_process = [s for s in shards_to_process if s.name not in completed]

    if not shards_to_process:
        print("All shards in range already processed!")
        return

    print(f"Processing shards {start_shard} to {end_shard - 1} ({len(shards_to_process)} remaining)")

    # Schema for output
    schema = pa.schema([
        ("shard_file", pa.string()),
        ("row_idx", pa.int64()),
        ("domain", pa.string()),
        ("etld_plus_one", pa.string()),
        ("tld", pa.string()),
        ("url_path", pa.string()),
        ("token_count", pa.int64()),
    ])

    # Process shards with progress bar
    all_rows = []
    batch_size = 500_000  # Write every 500k rows to limit memory

    for shard_path in tqdm(shards_to_process, desc="Processing shards"):
        rows = process_shard(shard_path)
        all_rows.extend(rows)

        # Mark as completed
        completed.add(shard_path.name)
        save_progress(sorted(completed))

        # Write batch if large enough
        if len(all_rows) >= batch_size:
            write_to_parquet(all_rows, schema)
            all_rows = []

    # Write remaining rows
    if all_rows:
        write_to_parquet(all_rows, schema)

    print(f"\nDone! Index saved to: {OUTPUT_PATH}")
    show_status()


def write_to_parquet(rows: list[dict], schema: pa.Schema):
    """Append rows to parquet file."""
    table = pa.Table.from_pylist(rows, schema=schema)

    if OUTPUT_PATH.exists():
        # Append to existing file
        existing = pq.read_table(OUTPUT_PATH)
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, OUTPUT_PATH, compression="snappy")
    else:
        # Create new file
        pq.write_table(table, OUTPUT_PATH, compression="snappy")


def main():
    parser = argparse.ArgumentParser(description="Build FineWeb domain index")
    parser.add_argument("--start-shard", type=int, default=0, help="Starting shard index")
    parser.add_argument("--num-shards", type=int, default=None, help="Number of shards to process")
    parser.add_argument("--status", action="store_true", help="Show indexing progress")

    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        build_index(args.start_shard, args.num_shards)


if __name__ == "__main__":
    main()

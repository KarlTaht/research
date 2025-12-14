#!/usr/bin/env python3
"""
Extract domain-specific corpora from FineWeb for continual learning experiments.

Uses the domain index to identify matching documents, then extracts actual text
from the Arrow shard files.

Usage:
    # Extract both corpora (automotive and food)
    python extract_corpus.py

    # Extract specific corpus
    python extract_corpus.py --corpus automotive
    python extract_corpus.py --corpus food

    # Custom target size (in tokens, default 125M = ~500MB)
    python extract_corpus.py --target-tokens 50000000

    # Preview mode (show stats without extracting)
    python extract_corpus.py --preview
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
from tqdm import tqdm

# Paths
INDEX_PATH = Path("/home/ktaht/research/assets/datasets/HuggingFaceFW/fineweb/domain_index.parquet")
FINEWEB_DIR = Path("/home/ktaht/research/assets/datasets/HuggingFaceFW/fineweb/sample-100BT/train")
OUTPUT_DIR = Path("/home/ktaht/research/projects/continual_learning/data")

# Target tokens per corpus (~500MB at ~4 bytes/token)
DEFAULT_TARGET_TOKENS = 125_000_000

# Domain filters
AUTOMOTIVE_FILTER = """
    (etld_plus_one LIKE '%auto%'
        OR etld_plus_one LIKE '%car%'
        OR etld_plus_one LIKE '%motor%'
        OR etld_plus_one LIKE '%vehicle%')
    AND etld_plus_one NOT LIKE '%care%'
    AND etld_plus_one NOT LIKE '%career%'
    AND etld_plus_one NOT LIKE '%card%'
    AND etld_plus_one NOT LIKE '%carton%'
    AND etld_plus_one NOT LIKE '%cartoon%'
    AND etld_plus_one NOT LIKE '%scar%'
    AND etld_plus_one NOT LIKE '%carleton%'
    AND etld_plus_one NOT LIKE '%carry%'
    AND etld_plus_one NOT LIKE '%straddle%'
    AND etld_plus_one NOT LIKE '%carnegie%'
    AND etld_plus_one NOT LIKE '%cartel%'
    AND etld_plus_one NOT LIKE '%carnage%'
    AND etld_plus_one NOT LIKE '%caring%'
    AND etld_plus_one NOT LIKE '%ucar%'
    AND etld_plus_one NOT LIKE '%automaticearth%'
    AND etld_plus_one NOT LIKE '%carrot%'
    AND etld_plus_one NOT LIKE '%carpet%'
    AND etld_plus_one NOT LIKE '%carousel%'
"""

FOOD_FILTER = """
    etld_plus_one LIKE '%recipe%'
    OR etld_plus_one LIKE '%cooking%'
    OR etld_plus_one LIKE '%food%'
"""

CORPUS_CONFIGS = {
    "automotive": {
        "filter": AUTOMOTIVE_FILTER,
        "description": "Automotive articles (cars, motors, vehicles)",
    },
    "food": {
        "filter": FOOD_FILTER,
        "description": "Food, recipes, and cooking content",
    },
}


def get_connection():
    """Get DuckDB connection with index loaded."""
    if not INDEX_PATH.exists():
        print(f"Index not found at {INDEX_PATH}")
        print("Run build_domain_index.py first to create the index.")
        exit(1)

    con = duckdb.connect(":memory:")
    con.execute(f"CREATE VIEW idx AS SELECT * FROM read_parquet('{INDEX_PATH}')")
    return con


def get_corpus_stats(con, corpus_name: str) -> dict:
    """Get statistics for a corpus filter."""
    config = CORPUS_CONFIGS[corpus_name]
    result = con.execute(f"""
        SELECT
            COUNT(*) as docs,
            SUM(token_count) as tokens,
            COUNT(DISTINCT etld_plus_one) as domains
        FROM idx
        WHERE {config['filter']}
    """).fetchone()

    return {
        "docs": result[0],
        "tokens": result[1],
        "domains": result[2],
        "est_mb": result[1] * 4 / 1_000_000,  # ~4 bytes per token
    }


def get_matching_documents(con, corpus_name: str, target_tokens: int) -> list[dict]:
    """Get list of documents matching the corpus filter, sampled to target size."""
    config = CORPUS_CONFIGS[corpus_name]

    print(f"Querying index for {corpus_name} documents...")

    # Get all matching documents with their locations
    result = con.execute(f"""
        SELECT
            shard_file,
            row_idx,
            token_count,
            etld_plus_one
        FROM idx
        WHERE {config['filter']}
    """).fetchall()

    docs = [
        {"shard_file": r[0], "row_idx": r[1], "token_count": r[2], "domain": r[3]}
        for r in result
    ]

    total_tokens = sum(d["token_count"] for d in docs)
    print(f"  Found {len(docs):,} documents with {total_tokens:,} tokens")

    # Sample if we have more than target
    if total_tokens > target_tokens:
        print(f"  Sampling down to ~{target_tokens:,} tokens...")
        random.shuffle(docs)

        sampled = []
        current_tokens = 0
        for doc in docs:
            if current_tokens >= target_tokens:
                break
            sampled.append(doc)
            current_tokens += doc["token_count"]

        docs = sampled
        print(f"  Sampled {len(docs):,} documents with {current_tokens:,} tokens")

    return docs


def load_shard_texts(shard_path: Path, row_indices: set[int]) -> dict[int, str]:
    """Load text for specific rows from an Arrow shard file."""
    texts = {}

    with pa.memory_map(str(shard_path), "r") as source:
        reader = pa.ipc.open_stream(source)

        row_idx = 0
        for batch in reader:
            batch_texts = batch.column("text").to_pylist()

            for i, text in enumerate(batch_texts):
                if row_idx in row_indices:
                    texts[row_idx] = text
                row_idx += 1

    return texts


def extract_corpus(corpus_name: str, target_tokens: int, val_split: float = 0.1):
    """Extract a corpus and save to JSONL files."""
    con = get_connection()

    # Get matching documents
    docs = get_matching_documents(con, corpus_name, target_tokens)

    if not docs:
        print(f"No documents found for {corpus_name}!")
        return

    # Group documents by shard file for efficient reading
    docs_by_shard = defaultdict(list)
    for doc in docs:
        docs_by_shard[doc["shard_file"]].append(doc)

    print(f"\nExtracting text from {len(docs_by_shard)} shard files...")

    # Create output directory
    corpus_dir = OUTPUT_DIR / f"corpus_{corpus_name}"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Extract texts
    all_texts = []
    for shard_file, shard_docs in tqdm(docs_by_shard.items(), desc="Reading shards"):
        shard_path = FINEWEB_DIR / shard_file

        if not shard_path.exists():
            print(f"  Warning: Shard not found: {shard_path}")
            continue

        row_indices = {d["row_idx"] for d in shard_docs}
        texts = load_shard_texts(shard_path, row_indices)

        for doc in shard_docs:
            if doc["row_idx"] in texts:
                all_texts.append({
                    "text": texts[doc["row_idx"]],
                    "domain": doc["domain"],
                    "token_count": doc["token_count"],
                })

    print(f"\nExtracted {len(all_texts):,} documents")

    # Shuffle and split into train/val
    random.shuffle(all_texts)
    val_size = int(len(all_texts) * val_split)
    val_texts = all_texts[:val_size]
    train_texts = all_texts[val_size:]

    # Calculate token counts
    train_tokens = sum(t["token_count"] for t in train_texts)
    val_tokens = sum(t["token_count"] for t in val_texts)

    print(f"  Train: {len(train_texts):,} docs, {train_tokens:,} tokens")
    print(f"  Val: {len(val_texts):,} docs, {val_tokens:,} tokens")

    # Save to JSONL
    train_path = corpus_dir / "train.jsonl"
    val_path = corpus_dir / "val.jsonl"

    print(f"\nSaving to {corpus_dir}/...")

    with open(train_path, "w") as f:
        for item in train_texts:
            f.write(json.dumps({"text": item["text"]}) + "\n")

    with open(val_path, "w") as f:
        for item in val_texts:
            f.write(json.dumps({"text": item["text"]}) + "\n")

    # Save metadata
    metadata = {
        "corpus": corpus_name,
        "description": CORPUS_CONFIGS[corpus_name]["description"],
        "train_docs": len(train_texts),
        "train_tokens": train_tokens,
        "val_docs": len(val_texts),
        "val_tokens": val_tokens,
        "total_tokens": train_tokens + val_tokens,
        "est_size_mb": (train_tokens + val_tokens) * 4 / 1_000_000,
    }

    with open(corpus_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Files saved:")
    print(f"  {train_path} ({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {val_path} ({val_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {corpus_dir / 'metadata.json'}")


def preview_corpora(target_tokens: int):
    """Show preview of corpus extraction without actually extracting."""
    con = get_connection()

    print("=" * 60)
    print("Corpus Extraction Preview")
    print("=" * 60)
    print(f"Target tokens per corpus: {target_tokens:,} (~{target_tokens * 4 / 1e6:.0f} MB)")
    print()

    for corpus_name, config in CORPUS_CONFIGS.items():
        stats = get_corpus_stats(con, corpus_name)
        print(f"{corpus_name.upper()}: {config['description']}")
        print(f"  Available: {stats['docs']:,} docs, {stats['tokens']:,} tokens (~{stats['est_mb']:.0f} MB)")
        print(f"  Domains: {stats['domains']:,} unique")

        if stats["tokens"] >= target_tokens:
            print(f"  Status: OK (will sample to {target_tokens:,} tokens)")
        else:
            print(f"  Status: WARNING - only {stats['tokens'] / target_tokens * 100:.1f}% of target")
        print()


def main():
    parser = argparse.ArgumentParser(description="Extract domain-specific corpora from FineWeb")
    parser.add_argument(
        "--corpus",
        type=str,
        choices=list(CORPUS_CONFIGS.keys()),
        help="Extract specific corpus (default: both)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help=f"Target tokens per corpus (default: {DEFAULT_TARGET_TOKENS:,})",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview extraction stats without extracting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    if args.preview:
        preview_corpora(args.target_tokens)
        return

    corpora_to_extract = [args.corpus] if args.corpus else list(CORPUS_CONFIGS.keys())

    for corpus_name in corpora_to_extract:
        print("\n" + "=" * 60)
        print(f"Extracting corpus: {corpus_name}")
        print("=" * 60)
        extract_corpus(corpus_name, args.target_tokens)


if __name__ == "__main__":
    main()

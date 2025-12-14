#!/usr/bin/env python3
"""Pre-tokenize a dataset for faster training startup.

This script tokenizes a dataset once and caches it to disk.
Subsequent training runs will load from cache instantly.

Usage:
    # Pre-tokenize TinyStories with custom 4096 BPE tokenizer
    python -m common.cli._internal.pretokenize --dataset tinystories --tokenizer tinystories_bpe_4096 --max-length 1024

    # Pre-tokenize with GPT-2 tokenizer
    python -m common.cli._internal.pretokenize --dataset tinystories --tokenizer gpt2 --max-length 512

    # Pre-tokenize a JSONL file (e.g., custom corpus)
    python -m common.cli._internal.pretokenize --jsonl data/corpus_automotive/train.jsonl --tokenizer combined_bpe_32768 --max-length 1024

    # Pre-tokenize validation split too
    python -m common.cli._internal.pretokenize --jsonl data/corpus_automotive/train.jsonl --val-jsonl data/corpus_automotive/val.jsonl --tokenizer combined_bpe_32768
"""

import argparse
import json
import sys
import time
from pathlib import Path

from datasets import Dataset
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast

from common.data import get_models_dir
from common.data.dataset_registry import (
    load_dataset_from_registry,
    get_tokenized_cache_path,
    get_tokenizer_cache_key,
    create_lm_dataloader,
)


def load_tokenizer(tokenizer_name: str):
    """Load tokenizer by name."""
    if tokenizer_name == 'gpt2':
        print(f"Loading GPT-2 tokenizer (vocab_size=50257)")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # Try assets/models/tokenizers/ path
        tokenizer_path = get_models_dir() / 'tokenizers' / tokenizer_name
        if not tokenizer_path.exists():
            tokenizer_path = get_models_dir() / tokenizer_name
        if not tokenizer_path.exists():
            tokenizer_path = Path(tokenizer_name)

        print(f"Loading custom tokenizer from: {tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '<pad>'})

    print(f"Vocab size: {len(tokenizer)}")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description='Pre-tokenize a dataset for caching')
    parser.add_argument('--dataset', type=str,
                        help='Dataset name from registry')
    parser.add_argument('--jsonl', type=str, metavar='PATH',
                        help='JSONL file to pre-tokenize (expects {"text": "..."} format)')
    parser.add_argument('--val-jsonl', type=str, metavar='PATH',
                        help='Validation JSONL file (optional, use with --jsonl)')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='Tokenizer name or path (default: gpt2)')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='Max sequence length (default: 1024)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for DataLoader (default: 32)')
    args = parser.parse_args()

    # Validate arguments
    if not args.dataset and not args.jsonl:
        parser.error('Either --dataset or --jsonl is required')

    # Load tokenizer first (needed for both paths)
    tokenizer = load_tokenizer(args.tokenizer)

    # Handle JSONL files
    if args.jsonl:
        jsonl_path = Path(args.jsonl)

        print(f"\n{'='*60}")
        print(f"Pre-tokenizing JSONL: {jsonl_path}")
        print(f"Tokenizer: {args.tokenizer}")
        print(f"Max length: {args.max_length}")
        print(f"{'='*60}\n")

        # Load texts from JSONL
        print(f"Loading texts from: {jsonl_path}")
        texts = []
        with open(jsonl_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        print(f"  Loaded {len(texts):,} texts")

        # Convert to HuggingFace Dataset
        train_dataset = Dataset.from_dict({"text": texts})

        # Build cache path alongside the JSONL file
        cache_key = get_tokenizer_cache_key(tokenizer, args.max_length)
        cache_path = jsonl_path.parent / f"{jsonl_path.stem}_tokenized" / cache_key
        print(f"\nCache will be saved to: {cache_path}")

        # Tokenize train split
        print(f"\nTokenizing train split...")
        start_time = time.time()
        train_loader = create_lm_dataloader(
            train_dataset,
            tokenizer,
            "text",
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=False,
            cache_path=cache_path,
            split_name="train",
        )
        train_time = time.time() - start_time
        print(f"  Train tokenization completed in {train_time:.1f}s")

        # Handle validation JSONL if provided
        if args.val_jsonl:
            val_path = Path(args.val_jsonl)
            print(f"\nLoading validation texts from: {val_path}")
            val_texts = []
            with open(val_path) as f:
                for line in f:
                    val_texts.append(json.loads(line)["text"])
            print(f"  Loaded {len(val_texts):,} texts")

            val_dataset = Dataset.from_dict({"text": val_texts})

            print(f"\nTokenizing validation split...")
            start_time = time.time()
            val_loader = create_lm_dataloader(
                val_dataset,
                tokenizer,
                "text",
                max_length=args.max_length,
                batch_size=args.batch_size,
                shuffle=False,
                cache_path=cache_path,
                split_name="validation",
            )
            val_time = time.time() - start_time
            print(f"  Validation tokenization completed in {val_time:.1f}s")

    # Handle registry datasets
    else:
        print(f"\n{'='*60}")
        print(f"Pre-tokenizing dataset: {args.dataset}")
        print(f"Tokenizer: {args.tokenizer}")
        print(f"Max length: {args.max_length}")
        print(f"{'='*60}\n")

        # Load raw dataset
        print(f"\nLoading dataset: {args.dataset}")
        train_dataset, val_dataset, text_column, dataset_path = load_dataset_from_registry(
            args.dataset
        )
        print(f"  Train examples: {len(train_dataset):,}")
        if val_dataset:
            print(f"  Val examples: {len(val_dataset):,}")

        # Get cache path
        cache_path = get_tokenized_cache_path(dataset_path, tokenizer, args.max_length)
        print(f"\nCache will be saved to: {cache_path}")

        # Tokenize train split
        print(f"\nTokenizing train split...")
        start_time = time.time()
        train_loader = create_lm_dataloader(
            train_dataset,
            tokenizer,
            text_column,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=False,
            cache_path=cache_path,
            split_name="train",
        )
        train_time = time.time() - start_time
        print(f"  Train tokenization completed in {train_time:.1f}s")

        # Tokenize validation split
        if val_dataset:
            print(f"\nTokenizing validation split...")
            start_time = time.time()
            val_loader = create_lm_dataloader(
                val_dataset,
                tokenizer,
                text_column,
                max_length=args.max_length,
                batch_size=args.batch_size,
                shuffle=False,
                cache_path=cache_path,
                split_name="validation",
            )
            val_time = time.time() - start_time
            print(f"  Validation tokenization completed in {val_time:.1f}s")

    print(f"\n{'='*60}")
    print(f"Pre-tokenization complete!")
    print(f"Cache location: {cache_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

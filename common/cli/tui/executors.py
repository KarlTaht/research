"""Executor functions for TUI wizard actions.

These functions are called directly by the TUI instead of spawning subprocesses.
Each function contains the logic from the corresponding CLI module.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

from datasets import Dataset


def execute_download_dataset(
    name: str,
    output: Optional[str] = None,
    split: Optional[str] = None,
    config: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> None:
    """Download a dataset from HuggingFace Hub."""
    from common.data.hf_utils import download_dataset, get_datasets_dir

    # Determine output directory
    if output:
        output_dir = Path(output).expanduser().resolve()
    else:
        output_dir = get_datasets_dir() / name
        if config:
            output_dir = output_dir / config

    print(f"Downloading dataset: {name}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    download_dataset(
        name=name,
        config=config,
        output_dir=output_dir,
        split=split,
        cache_dir=cache_dir,
    )

    print("-" * 50)
    print(f"Success! Dataset saved to: {output_dir}")


def execute_download_model(
    repo_id: str,
    output: Optional[str] = None,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> None:
    """Download a model from HuggingFace Hub."""
    from common.data.hf_utils import download_model, get_models_dir

    if output:
        output_dir = Path(output).expanduser().resolve()
    else:
        output_dir = get_models_dir() / repo_id.replace("/", "_")

    print(f"Downloading model: {repo_id}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    download_model(
        repo_id=repo_id,
        output_dir=output_dir,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
    )

    print("-" * 50)
    print(f"Success! Model saved to: {output_dir}")


def execute_analyze(
    dataset: Optional[str] = None,
    jsonl: Optional[str] = None,
    tokenizer: str = "gpt2",
    subset: Optional[int] = None,
    recommendations: bool = False,
) -> None:
    """Analyze token distributions in a dataset."""
    from common.data.token_analyzer import TokenAnalyzer
    from common.data.dataset_registry import load_dataset_from_registry

    print(f"Analyzing tokens...")
    print(f"Tokenizer: {tokenizer}")
    print("-" * 50)

    # Create analyzer
    analyzer = TokenAnalyzer(tokenizer)

    if jsonl:
        # Load from JSONL file
        jsonl_path = Path(jsonl)
        print(f"Loading from JSONL: {jsonl_path}")
        texts = []
        with open(jsonl_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        if subset:
            texts = texts[:subset]

        report = analyzer.analyze_texts(texts, name=jsonl_path.stem)
    else:
        # Load from registry
        print(f"Loading dataset: {dataset}")
        train_dataset, _, text_column, _ = load_dataset_from_registry(dataset)

        if subset:
            train_dataset = train_dataset.select(range(min(subset, len(train_dataset))))

        texts = [ex[text_column] for ex in train_dataset]
        report = analyzer.analyze_texts(texts, name=dataset)

    # Print report
    print(report.format_report())

    if recommendations:
        rec = analyzer.recommend_tokenizer_size(report)
        print("\n" + rec.format())


def execute_train_tokenizer(
    dataset: Optional[str] = None,
    jsonl: Optional[str] = None,
    vocab_size: int = 4096,
    output_dir: Optional[str] = None,
    subset: Optional[int] = None,
) -> None:
    """Train a custom BPE tokenizer."""
    from common.data.token_analyzer import (
        train_custom_tokenizer,
        train_custom_tokenizer_from_file,
    )
    from common.data import get_models_dir

    # Determine output directory
    if output_dir:
        tokenizer_output = Path(output_dir)
    else:
        tokenizer_output = get_models_dir() / "tokenizers"

    print(f"Training BPE tokenizer with vocab_size={vocab_size}")
    print("-" * 50)

    if jsonl:
        # Train from JSONL file(s)
        jsonl_paths = [jsonl] if isinstance(jsonl, str) else jsonl
        print(f"Training from JSONL file(s): {jsonl_paths}")

        tokenizer = train_custom_tokenizer_from_file(
            jsonl_paths=jsonl_paths,
            vocab_size=vocab_size,
            output_dir=tokenizer_output,
            subset=subset,
        )
    else:
        # Train from registry dataset
        print(f"Training from dataset: {dataset}")

        tokenizer = train_custom_tokenizer(
            dataset_name=dataset,
            vocab_size=vocab_size,
            output_dir=tokenizer_output,
            subset=subset,
        )

    print("-" * 50)
    print(f"Tokenizer saved to: {tokenizer_output}")


def execute_pretokenize(
    dataset: Optional[str] = None,
    jsonl: Optional[str] = None,
    val_jsonl: Optional[str] = None,
    tokenizer: str = "gpt2",
    max_length: int = 1024,
    max_tokens: Optional[int] = None,
    batch_size: int = 32,
) -> None:
    """Pre-tokenize a dataset for faster training startup.

    Args:
        dataset: Registry dataset name (e.g., 'tinystories')
        jsonl: Path to JSONL file with {"text": ...} format
        val_jsonl: Optional validation JSONL file
        tokenizer: Tokenizer name or path
        max_length: Maximum sequence length
        max_tokens: Optional limit on total tokens to process (e.g., 10_000_000)
        batch_size: Batch size for tokenization
    """
    from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
    from common.data import get_models_dir
    from common.data.dataset_registry import (
        load_dataset_from_registry,
        get_tokenized_cache_path,
        get_tokenizer_cache_key,
        create_lm_dataloader,
    )

    def estimate_tokens(texts: list[str], avg_chars_per_token: float = 4.0) -> int:
        """Estimate total tokens from text list."""
        total_chars = sum(len(t) for t in texts)
        return int(total_chars / avg_chars_per_token)

    def sample_to_token_limit(texts: list[str], max_tok: int, avg_chars_per_token: float = 4.0) -> list[str]:
        """Sample texts to stay within token limit."""
        estimated = estimate_tokens(texts, avg_chars_per_token)
        if estimated <= max_tok:
            return texts

        # Calculate sampling ratio
        ratio = max_tok / estimated
        sample_size = max(1, int(len(texts) * ratio))
        print(f"  Sampling {sample_size:,} of {len(texts):,} texts to stay within {max_tok:,} token limit")
        return texts[:sample_size]

    # Load tokenizer
    if tokenizer == 'gpt2':
        print(f"Loading GPT-2 tokenizer (vocab_size=50257)")
        tok = GPT2TokenizerFast.from_pretrained('gpt2')
        tok.pad_token = tok.eos_token
    else:
        tokenizer_path = get_models_dir() / 'tokenizers' / tokenizer
        if not tokenizer_path.exists():
            tokenizer_path = get_models_dir() / tokenizer
        if not tokenizer_path.exists():
            tokenizer_path = Path(tokenizer)

        print(f"Loading custom tokenizer from: {tokenizer_path}")
        tok = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({'pad_token': '<pad>'})

    print(f"Vocab size: {len(tok)}")

    if jsonl:
        # Handle JSONL files
        jsonl_path = Path(jsonl)
        print(f"\n{'='*60}")
        print(f"Pre-tokenizing JSONL: {jsonl_path}")
        print(f"Tokenizer: {tokenizer}")
        print(f"Max length: {max_length}")
        if max_tokens:
            print(f"Max tokens: {max_tokens:,}")
        print(f"{'='*60}\n")

        # Load texts from JSONL
        print(f"Loading texts from: {jsonl_path}")
        texts = []
        with open(jsonl_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        print(f"  Loaded {len(texts):,} texts")

        # Apply token limit if specified
        if max_tokens:
            texts = sample_to_token_limit(texts, max_tokens)

        train_dataset = Dataset.from_dict({"text": texts})
        cache_key = get_tokenizer_cache_key(tok, max_length)
        cache_path = jsonl_path.parent / f"{jsonl_path.stem}_tokenized" / cache_key

        print(f"\nCache will be saved to: {cache_path}")
        print(f"\nTokenizing train split...")
        start_time = time.time()

        create_lm_dataloader(
            train_dataset, tok, "text",
            max_length=max_length, batch_size=batch_size,
            shuffle=False, cache_path=cache_path, split_name="train",
        )

        print(f"  Train tokenization completed in {time.time() - start_time:.1f}s")

        if val_jsonl:
            val_path = Path(val_jsonl)
            print(f"\nLoading validation texts from: {val_path}")
            val_texts = []
            with open(val_path) as f:
                for line in f:
                    val_texts.append(json.loads(line)["text"])
            print(f"  Loaded {len(val_texts):,} texts")

            val_dataset = Dataset.from_dict({"text": val_texts})
            print(f"\nTokenizing validation split...")
            start_time = time.time()

            create_lm_dataloader(
                val_dataset, tok, "text",
                max_length=max_length, batch_size=batch_size,
                shuffle=False, cache_path=cache_path, split_name="validation",
            )

            print(f"  Validation tokenization completed in {time.time() - start_time:.1f}s")

    else:
        # Handle registry datasets
        print(f"\n{'='*60}")
        print(f"Pre-tokenizing dataset: {dataset}")
        print(f"Tokenizer: {tokenizer}")
        print(f"Max length: {max_length}")
        if max_tokens:
            print(f"Max tokens: {max_tokens:,}")
        print(f"{'='*60}\n")

        print(f"\nLoading dataset: {dataset}")
        train_dataset, val_dataset, text_column, dataset_path = load_dataset_from_registry(dataset)
        print(f"  Train examples: {len(train_dataset):,}")
        if val_dataset:
            print(f"  Val examples: {len(val_dataset):,}")

        # Apply token limit if specified
        if max_tokens:
            train_texts = train_dataset[text_column]
            train_texts = sample_to_token_limit(train_texts, max_tokens)
            train_dataset = Dataset.from_dict({text_column: train_texts})
            print(f"  Train examples after sampling: {len(train_dataset):,}")

            if val_dataset:
                # Sample validation proportionally (10% of train sample size)
                val_sample_size = max(1, len(train_texts) // 10)
                val_texts = val_dataset[text_column][:val_sample_size]
                val_dataset = Dataset.from_dict({text_column: val_texts})
                print(f"  Val examples after sampling: {len(val_dataset):,}")

        cache_path = get_tokenized_cache_path(dataset_path, tok, max_length)
        print(f"\nCache will be saved to: {cache_path}")

        print(f"\nTokenizing train split...")
        start_time = time.time()

        create_lm_dataloader(
            train_dataset, tok, text_column,
            max_length=max_length, batch_size=batch_size,
            shuffle=False, cache_path=cache_path, split_name="train",
        )

        print(f"  Train tokenization completed in {time.time() - start_time:.1f}s")

        if val_dataset:
            print(f"\nTokenizing validation split...")
            start_time = time.time()

            create_lm_dataloader(
                val_dataset, tok, text_column,
                max_length=max_length, batch_size=batch_size,
                shuffle=False, cache_path=cache_path, split_name="validation",
            )

            print(f"  Validation tokenization completed in {time.time() - start_time:.1f}s")

    print(f"\n{'='*60}")
    print(f"Pre-tokenization complete!")
    print(f"Cache location: {cache_path}")
    print(f"{'='*60}\n")


def execute_download_fineweb(
    tokens: Optional[int] = None,
) -> None:
    """Download a FineWeb sample dataset."""
    from common.cli._internal.download_fineweb import download_small_sample

    target_tokens = tokens or 10_000_000
    download_small_sample(target_tokens=target_tokens)


def execute_fineweb_index(
    start_shard: int = 0,
    num_shards: Optional[int] = None,
    status: bool = False,
) -> None:
    """Build FineWeb domain index."""
    from common.cli._internal.build_domain_index import build_index, show_status

    if status:
        show_status()
    else:
        build_index(start_shard=start_shard, num_shards=num_shards)


def execute_query_domains(
    top_domains: Optional[int] = None,
    tld: Optional[str] = None,
    domain_contains: Optional[str] = None,
    sql: Optional[str] = None,
) -> None:
    """Query FineWeb domain index."""
    from common.cli._internal.query_domain_index import (
        get_connection,
        show_summary,
        top_domains as show_top_domains,
        top_tlds,
        filter_by_tld,
        filter_by_domain_pattern,
        run_sql,
    )

    con = get_connection()

    # If no specific query, show summary
    if not any([top_domains, tld, domain_contains, sql]):
        show_summary(con)
        top_tlds(con, 15)
        show_top_domains(con, 30)
        return

    if top_domains:
        show_top_domains(con, top_domains)

    if tld:
        filter_by_tld(con, tld)

    if domain_contains:
        filter_by_domain_pattern(con, domain_contains)

    if sql:
        run_sql(con, sql)


def execute_fineweb_extract(
    corpus: Optional[str] = None,
    target_tokens: Optional[int] = None,
    preview: bool = False,
) -> None:
    """Extract domain corpus from FineWeb."""
    from common.cli._internal.extract_corpus import (
        extract_corpus,
        preview_corpora,
        CORPUS_CONFIGS,
        DEFAULT_TARGET_TOKENS,
    )
    import random

    target = target_tokens or DEFAULT_TARGET_TOKENS
    random.seed(42)

    if preview:
        preview_corpora(target)
        return

    corpora_to_extract = [corpus] if corpus else list(CORPUS_CONFIGS.keys())

    for corpus_name in corpora_to_extract:
        print("\n" + "=" * 60)
        print(f"Extracting corpus: {corpus_name}")
        print("=" * 60)
        extract_corpus(corpus_name, target)


# Registry mapping wizard names to executor functions
EXECUTORS = {
    "download_dataset": execute_download_dataset,
    "download_model": execute_download_model,
    "analyze": execute_analyze,
    "train_tokenizer": execute_train_tokenizer,
    "pretokenize": execute_pretokenize,
    "download_fineweb": execute_download_fineweb,
    "fineweb_index": execute_fineweb_index,
    "query_domains": execute_query_domains,
    "fineweb_extract": execute_fineweb_extract,
}

"""Pre-tokenize datasets for faster training.

This module provides functions to pre-tokenize datasets and save them in
HuggingFace Arrow format for efficient loading during training.

Example:
    from common.data import pretokenize_dataset

    # Pre-tokenize fineweb-edu with a custom tokenizer
    output_path = pretokenize_dataset(
        'fineweb-edu-10b',
        tokenizer_path='assets/datasets/HuggingFaceFW/fineweb/tokenizers/combined_bpe_32768',
        max_length=1024,
    )

    # Load for training
    from datasets import load_from_disk
    dataset = load_from_disk(output_path)
"""

import os
from pathlib import Path
from typing import Optional, Union

from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from .dataset_registry import load_dataset_from_registry, get_datasets_dir


def pretokenize_dataset(
    dataset_name: str,
    tokenizer_path: Union[str, Path],
    max_length: int = 1024,
    output_dir: Optional[Path] = None,
    num_proc: Optional[int] = None,
    batch_size: int = 1000,
    show_progress: bool = True,
) -> Path:
    """
    Pre-tokenize a dataset from the registry and save to disk.

    Tokenizes all examples and saves as HuggingFace Arrow format for
    efficient memory-mapped loading during training.

    Args:
        dataset_name: Name of dataset in registry (e.g., 'fineweb-edu-10b')
        tokenizer_path: Path to tokenizer directory or HuggingFace model name
        max_length: Maximum sequence length (tokens will be truncated/padded)
        output_dir: Where to save. Default: alongside dataset with tokenizer suffix
        num_proc: Number of processes for parallel tokenization. Default: CPU count
        batch_size: Batch size for tokenization
        show_progress: Whether to show progress bars

    Returns:
        Path to the saved pre-tokenized dataset directory

    Example:
        from common.data import pretokenize_dataset

        output = pretokenize_dataset(
            'fineweb-edu-10b',
            tokenizer_path='combined_bpe_32768',
            max_length=1024,
        )
        # Saved to: assets/datasets/.../fineweb-edu-sample-10BT_pretokenized/combined_bpe_32768_len1024/
    """
    if num_proc is None:
        num_proc = os.cpu_count() or 1

    # Load tokenizer
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.is_absolute():
        # Check common locations
        candidates = [
            tokenizer_path,
            get_datasets_dir() / "HuggingFaceFW" / "fineweb" / "tokenizers" / tokenizer_path,
            get_datasets_dir().parent / "models" / "tokenizers" / tokenizer_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                tokenizer_path = candidate
                break

    if show_progress:
        print(f"Loading tokenizer from: {tokenizer_path}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})

    if show_progress:
        print(f"  Vocab size: {len(tokenizer):,}")

    # Load dataset
    if show_progress:
        print(f"Loading dataset: {dataset_name}")

    train_dataset, val_dataset, text_column, dataset_path = load_dataset_from_registry(dataset_name)

    if show_progress:
        print(f"  Train examples: {len(train_dataset):,}")
        if val_dataset:
            print(f"  Val examples: {len(val_dataset):,}")

    # Determine output path
    if output_dir is None:
        tokenizer_name = tokenizer_path.name
        output_dir = dataset_path.parent / f"{dataset_path.name}_pretokenized" / f"{tokenizer_name}_len{max_length}"
    else:
        output_dir = Path(output_dir)

    # Define tokenization function
    def tokenize_function(examples):
        """Tokenize batch of examples."""
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors=None,
        )
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    # Tokenize train set
    if show_progress:
        print(f"\nTokenizing train set with {num_proc} processes...")

    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train" if show_progress else None,
    )

    # Save train set
    train_output = output_dir / "train"
    train_output.parent.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"Saving train to: {train_output}")

    train_tokenized.save_to_disk(str(train_output))

    if show_progress:
        print(f"  Saved {len(train_tokenized):,} tokenized examples")

    # Tokenize validation set if it exists
    if val_dataset is not None:
        if show_progress:
            print(f"\nTokenizing validation set...")

        val_tokenized = val_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val" if show_progress else None,
        )

        val_output = output_dir / "validation"
        if show_progress:
            print(f"Saving validation to: {val_output}")

        val_tokenized.save_to_disk(str(val_output))

        if show_progress:
            print(f"  Saved {len(val_tokenized):,} tokenized examples")

    if show_progress:
        print(f"\nPre-tokenization complete!")
        print(f"Output directory: {output_dir}")

    return output_dir


def load_pretokenized_dataset(
    pretokenized_path: Union[str, Path],
    split: str = "train",
) -> Dataset:
    """
    Load a pre-tokenized dataset from disk.

    Args:
        pretokenized_path: Path to the pre-tokenized dataset directory
        split: Which split to load ('train' or 'validation')

    Returns:
        HuggingFace Dataset with tokenized data

    Example:
        from common.data import load_pretokenized_dataset

        train_ds = load_pretokenized_dataset(
            'assets/datasets/.../fineweb-edu-sample-10BT_pretokenized/combined_bpe_32768_len1024',
            split='train',
        )
    """
    from datasets import load_from_disk

    pretokenized_path = Path(pretokenized_path)
    split_path = pretokenized_path / split

    if not split_path.exists():
        available = [d.name for d in pretokenized_path.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"Split '{split}' not found at {pretokenized_path}. "
            f"Available splits: {available}"
        )

    return load_from_disk(str(split_path))


def get_pretokenized_path(
    dataset_name: str,
    tokenizer_name: str,
    max_length: int,
    datasets_dir: Optional[Path] = None,
) -> Path:
    """
    Get the expected path for a pre-tokenized dataset.

    Args:
        dataset_name: Name of dataset in registry
        tokenizer_name: Name of the tokenizer directory
        max_length: Sequence length used for tokenization
        datasets_dir: Override datasets directory

    Returns:
        Path where pre-tokenized data should be located
    """
    from .dataset_registry import get_dataset_config

    config = get_dataset_config(dataset_name)
    datasets_dir = datasets_dir or get_datasets_dir()

    dataset_path = datasets_dir / config['path']
    return dataset_path.parent / f"{dataset_path.name}_pretokenized" / f"{tokenizer_name}_len{max_length}"

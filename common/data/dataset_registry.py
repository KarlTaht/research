"""Dataset registry for unified data loading across projects."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset

from .hf_utils import get_datasets_dir


# Registry of supported datasets with their configurations
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    'tinystories': {
        'path': 'roneneldan/TinyStories',
        'text_column': 'text',
        'train_split': 'train',
        'val_split': 'validation',
        'description': 'TinyStories - simple children\'s stories for language modeling',
    },
    'tiny-textbooks': {
        'path': 'nampdn-ai/tiny-textbooks',
        'text_column': 'textbook',  # Use the textbook column, not text
        'train_split': 'train',
        'val_split': 'test',
        'description': 'Tiny textbooks for educational language modeling',
    },
    'tiny-strange-textbooks': {
        'path': 'nampdn-ai/tiny-strange-textbooks',
        'text_column': 'textbook',
        'train_split': 'train',
        'val_split': None,  # No validation split
        'description': 'Strange textbooks for diverse language modeling',
    },
    'tiny-codes': {
        'path': 'nampdn-ai/tiny-codes',
        'text_column': 'code',
        'train_split': 'train',
        'val_split': None,
        'description': 'Tiny code snippets for code language modeling',
    },
}


def get_dataset_config(name: str) -> Dict[str, Any]:
    """
    Get configuration for a registered dataset.

    Args:
        name: Dataset name (key in DATASET_REGISTRY)

    Returns:
        Dataset configuration dict

    Raises:
        ValueError: If dataset is not registered
    """
    if name not in DATASET_REGISTRY:
        available = ', '.join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]


def load_dataset_from_registry(
    dataset_name: str,
    datasets_dir: Optional[Path] = None,
) -> Tuple[Dataset, Optional[Dataset], str]:
    """
    Load a dataset from the registry.

    Args:
        dataset_name: Name of dataset in registry
        datasets_dir: Override path for datasets directory

    Returns:
        Tuple of (train_dataset, val_dataset, text_column)
        val_dataset may be None if no validation split exists
    """
    config = get_dataset_config(dataset_name)
    datasets_dir = datasets_dir or get_datasets_dir()

    # Build path from registry config
    dataset_path = datasets_dir / config['path']

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Run: python tools/download_hf_dataset.py --name {config['path']}"
        )

    # Load dataset
    ds = load_from_disk(str(dataset_path))

    train_split = config['train_split']
    val_split = config.get('val_split')

    train_dataset = ds[train_split]
    val_dataset = ds[val_split] if val_split and val_split in ds else None

    return train_dataset, val_dataset, config['text_column']


def create_lm_dataloader(
    dataset: Dataset,
    tokenizer,
    text_column: str,
    max_length: int = 512,
    batch_size: int = 16,
    shuffle: bool = True,
    subset_size: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for language modeling from a HuggingFace dataset.

    Args:
        dataset: HuggingFace Dataset
        tokenizer: Tokenizer to use (e.g., GPT2TokenizerFast)
        text_column: Name of column containing text
        max_length: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle data
        subset_size: If set, only use this many examples
        num_workers: Number of data loading workers

    Returns:
        DataLoader yielding dicts with 'input_ids' and 'labels'
    """
    # Optionally subset
    if subset_size is not None and subset_size < len(dataset):
        dataset = dataset.select(range(subset_size))

    def tokenize_function(examples):
        """Tokenize and prepare for language modeling."""
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors=None,  # Return lists for map()
        )
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Set format for PyTorch
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'labels'])

    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def load_training_data(
    dataset_name: str,
    tokenizer,
    max_length: int = 512,
    batch_size: int = 16,
    subset_size: Optional[int] = None,
    val_subset_size: Optional[int] = None,
    num_workers: int = 0,
    datasets_dir: Optional[Path] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Load training and validation data for a registered dataset.

    This is the main entry point for loading data from the registry.

    Args:
        dataset_name: Name of dataset in registry (e.g., 'tinystories')
        tokenizer: Tokenizer to use (e.g., GPT2TokenizerFast)
        max_length: Maximum sequence length
        batch_size: Batch size
        subset_size: If set, only use this many training examples
        val_subset_size: If set, only use this many validation examples
        num_workers: Number of data loading workers
        datasets_dir: Override path for datasets directory

    Returns:
        Tuple of (train_loader, val_loader)
        val_loader may be None if no validation split exists

    Example:
        from transformers import GPT2TokenizerFast
        from common.data import load_training_data

        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        train_loader, val_loader = load_training_data(
            'tinystories',
            tokenizer,
            max_length=512,
            batch_size=16,
            subset_size=10000,
        )
    """
    # Load raw datasets
    train_dataset, val_dataset, text_column = load_dataset_from_registry(
        dataset_name, datasets_dir
    )

    # Create train dataloader
    train_loader = create_lm_dataloader(
        train_dataset,
        tokenizer,
        text_column,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=True,
        subset_size=subset_size,
        num_workers=num_workers,
    )

    # Create validation dataloader (if available)
    val_loader = None
    if val_dataset is not None:
        val_loader = create_lm_dataloader(
            val_dataset,
            tokenizer,
            text_column,
            max_length=max_length,
            batch_size=batch_size,
            shuffle=False,
            subset_size=val_subset_size,
            num_workers=num_workers,
        )

    return train_loader, val_loader


def list_datasets() -> Dict[str, str]:
    """
    List all registered datasets with descriptions.

    Returns:
        Dict mapping dataset name to description
    """
    return {name: config['description'] for name, config in DATASET_REGISTRY.items()}

"""Train custom BPE tokenizers for datasets in the registry."""

from pathlib import Path
from typing import Iterator, Optional

from .dataset_registry import load_dataset_from_registry
from .download_hf_dataset import get_models_dir


def train_custom_tokenizer(
    dataset_name: str,
    vocab_size: int,
    output_dir: Optional[Path] = None,
    subset_size: Optional[int] = None,
    min_frequency: int = 2,
    show_progress: bool = True,
) -> Path:
    """
    Train a custom BPE tokenizer optimized for a specific dataset.

    This creates a smaller, dataset-specific tokenizer that can significantly
    reduce embedding parameters while maintaining good coverage.

    Args:
        dataset_name: Name of dataset in registry
        vocab_size: Target vocabulary size
        output_dir: Where to save the tokenizer (default: assets/models/tokenizers/)
        subset_size: If set, only train on this many examples
        min_frequency: Minimum frequency for a token to be included
        show_progress: Whether to show progress

    Returns:
        Path to the saved tokenizer directory

    Example:
        from common.data import train_custom_tokenizer

        # Train a small tokenizer for TinyStories
        tokenizer_path = train_custom_tokenizer(
            'tinystories',
            vocab_size=4096,
            subset_size=50000,
        )

        # Load and use it
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

    if show_progress:
        print(f"Loading dataset: {dataset_name}")

    # Load dataset
    train_dataset, _, text_column = load_dataset_from_registry(dataset_name)

    # Subset if requested
    if subset_size is not None and subset_size < len(train_dataset):
        train_dataset = train_dataset.select(range(subset_size))

    if show_progress:
        print(f"Training tokenizer on {len(train_dataset):,} examples...")
        print(f"Target vocab size: {vocab_size:,}")

    # Get texts as iterator for memory efficiency
    def text_iterator() -> Iterator[str]:
        for i in range(len(train_dataset)):
            yield train_dataset[i][text_column]

    # Create BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Use GPT-2 style pre-tokenization (byte-level)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Train the tokenizer (Rust backend uses all available CPU cores automatically)
    import os
    num_threads = os.cpu_count() or 1

    if show_progress:
        print(f"Training with {num_threads} CPU threads available")

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        show_progress=show_progress,
    )

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=len(train_dataset))

    # Add decoder and post-processor for proper decoding
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Determine output path
    if output_dir is None:
        output_dir = get_models_dir() / "tokenizers" / f"{dataset_name}_bpe_{vocab_size}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the tokenizer
    tokenizer.save(str(output_dir / "tokenizer.json"))

    # Also save as HuggingFace compatible format
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    hf_tokenizer.save_pretrained(str(output_dir))

    if show_progress:
        print(f"Tokenizer saved to: {output_dir}")
        print(f"Actual vocab size: {hf_tokenizer.vocab_size:,}")

    return output_dir
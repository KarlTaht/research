"""Train custom BPE tokenizers from JSONL files."""

from pathlib import Path
from typing import Iterator, Optional

from .dataset_registry import get_models_dir


def train_custom_tokenizer_from_file(
    jsonl_paths: str | Path | list[str | Path],
    vocab_size: int,
    output_dir: Optional[Path] = None,
    subset_size: Optional[int] = None,
    min_frequency: int = 2,
    text_key: str = "text",
    show_progress: bool = True,
) -> Path:
    """
    Train a custom BPE tokenizer from one or more JSONL files.

    This allows training tokenizers on custom corpora that aren't in the
    dataset registry, such as the continual learning automotive/food corpora.

    Args:
        jsonl_paths: Path(s) to JSONL file(s) with {"text": "..."} entries.
            Can be a single path or a list of paths.
        vocab_size: Target vocabulary size
        output_dir: Where to save the tokenizer (default: assets/models/tokenizers/)
        subset_size: If set, only train on this many examples
        min_frequency: Minimum frequency for a token to be included
        text_key: Key in JSON objects containing the text (default: "text")
        show_progress: Whether to show progress

    Returns:
        Path to the saved tokenizer directory

    Example:
        from common.data import train_custom_tokenizer_from_file

        # Train tokenizer on single corpus
        tokenizer_path = train_custom_tokenizer_from_file(
            'data/corpus_automotive/train.jsonl',
            vocab_size=16384,
        )

        # Train on multiple corpora (combined)
        tokenizer_path = train_custom_tokenizer_from_file(
            ['data/corpus_automotive/train.jsonl', 'data/corpus_food/train.jsonl'],
            vocab_size=32768,
        )

        # Load and use it
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    """
    import json
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

    # Normalize to list of Paths
    if isinstance(jsonl_paths, (str, Path)):
        jsonl_paths = [Path(jsonl_paths)]
    else:
        jsonl_paths = [Path(p) for p in jsonl_paths]

    # Load texts from all JSONL files
    texts = []
    for jsonl_path in jsonl_paths:
        if show_progress:
            print(f"Loading texts from: {jsonl_path}")

        with open(jsonl_path) as f:
            file_count = 0
            for line in f:
                data = json.loads(line)
                texts.append(data[text_key])
                file_count += 1

        if show_progress:
            print(f"  Loaded {file_count:,} texts from {jsonl_path.name}")

    if show_progress:
        print(f"Total: {len(texts):,} texts from {len(jsonl_paths)} file(s)")

    # Subset if requested
    if subset_size is not None and subset_size < len(texts):
        texts = texts[:subset_size]

    if show_progress:
        print(f"Training tokenizer on {len(texts):,} examples...")
        print(f"Target vocab size: {vocab_size:,}")

    # Create text iterator for memory efficiency
    def text_iterator() -> Iterator[str]:
        for text in texts:
            yield text

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

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=len(texts))

    # Add decoder and post-processor for proper decoding
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Determine output path
    if output_dir is None:
        if len(jsonl_paths) > 1:
            # Multiple files: use "combined" as name
            name = "combined"
        else:
            # Single file: derive name from path
            single_path = jsonl_paths[0]
            name = single_path.stem  # e.g., "train" from "train.jsonl"
            # Try to get a better name from parent directory
            if name in ("train", "val", "test"):
                name = single_path.parent.name  # e.g., "corpus_automotive"
        output_dir = get_models_dir() / "tokenizers" / f"{name}_bpe_{vocab_size}"
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
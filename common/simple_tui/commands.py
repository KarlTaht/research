"""Command handlers for the tokenization TUI."""

import shlex
from pathlib import Path


def parse_command(text: str) -> tuple[str, dict]:
    """Parse 'command arg --flag value' into (command, {positional: arg, flag: value})."""
    try:
        parts = shlex.split(text)
    except ValueError as e:
        return "", {"error": str(e)}

    if not parts:
        return "", {}

    command = parts[0].lower()
    args = {}
    positional_idx = 0
    i = 1

    while i < len(parts):
        if parts[i].startswith("--"):
            key = parts[i][2:].replace("-", "_")  # --vocab-size -> vocab_size
            if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                value = parts[i + 1]
                # Try to convert to int
                try:
                    value = int(value)
                except ValueError:
                    pass
                args[key] = value
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            # Positional argument
            if positional_idx == 0:
                args["source"] = parts[i]
            positional_idx += 1
            i += 1

    return command, args


def cmd_analyze(source: str, tokenizer: str = "gpt2", subset: int | None = None,
                recommendations: bool = False, **kwargs):
    """Analyze token distribution in a dataset."""
    from transformers import AutoTokenizer

    from common.data.token_analyzer import TokenAnalyzer

    print(f"Analyzing '{source}' with tokenizer '{tokenizer}'...")

    tok = AutoTokenizer.from_pretrained(tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    analyzer = TokenAnalyzer(tok)
    report = analyzer.analyze_dataset(source, subset_size=subset)
    print(report.summary())

    if recommendations:
        recs = analyzer.recommend_vocab_size(report)
        print("\nRecommendations:")
        for rec in recs:
            print(rec.summary())


def cmd_train(source: str, vocab_size: int | None = None, subset: int | None = None,
              output: str | None = None, **kwargs):
    """Train a BPE tokenizer on a dataset or JSONL file."""
    if vocab_size is None:
        print("Error: --vocab-size is required")
        print("Usage: train <dataset|file.jsonl> --vocab-size 4096")
        return

    # Determine if source is a file or dataset name
    source_path = Path(source)
    if source_path.exists() and source_path.suffix == ".jsonl":
        # Train from JSONL file
        from common.data.train_tokenizer import train_custom_tokenizer_from_file

        print(f"Training {vocab_size}-token BPE tokenizer from file: {source}")
        output_path = Path(output) if output else None
        result_path = train_custom_tokenizer_from_file(
            source_path,
            vocab_size=vocab_size,
            subset_size=subset,
            output_dir=output_path,
        )
    else:
        # Train from dataset registry
        from common.data.pretokenize_dataset import train_custom_tokenizer

        print(f"Training {vocab_size}-token BPE tokenizer from dataset: {source}")
        output_path = Path(output) if output else None
        result_path = train_custom_tokenizer(
            source,
            vocab_size=vocab_size,
            subset_size=subset,
            output_dir=output_path,
        )

    print(f"\nTokenizer saved to: {result_path}")


def cmd_pretokenize(source: str, tokenizer: str | None = None, max_length: int = 1024,
                    **kwargs):
    """Pretokenize a dataset with a tokenizer."""
    if tokenizer is None:
        print("Error: --tokenizer is required")
        print("Usage: pretokenize <dataset> --tokenizer gpt2")
        return

    from common.cli._internal.pretokenize import load_tokenizer
    from common.data.dataset_registry import (
        create_lm_dataloader,
        get_tokenized_cache_path,
        load_dataset_from_registry,
    )

    print(f"Pretokenizing '{source}' with tokenizer '{tokenizer}'...")
    print(f"Max sequence length: {max_length}")

    # Load tokenizer using the CLI helper (handles custom tokenizers)
    tok = load_tokenizer(tokenizer)

    # Load dataset
    train_ds, val_ds, text_column, dataset_path = load_dataset_from_registry(source)
    print(f"  Train examples: {len(train_ds):,}")
    if val_ds:
        print(f"  Val examples: {len(val_ds):,}")

    # Get cache path
    cache_path = get_tokenized_cache_path(dataset_path, tok, max_length)
    print(f"\nCache will be saved to: {cache_path}")

    # Tokenize train split (create_lm_dataloader handles caching)
    print("\nTokenizing train split...")
    create_lm_dataloader(
        train_ds,
        tok,
        text_column,
        max_length=max_length,
        shuffle=False,
        cache_path=cache_path,
        split_name="train",
    )
    print("  Train tokenization completed")

    # Tokenize validation split
    if val_ds is not None:
        print("\nTokenizing validation split...")
        create_lm_dataloader(
            val_ds,
            tok,
            text_column,
            max_length=max_length,
            shuffle=False,
            cache_path=cache_path,
            split_name="validation",
        )
        print("  Validation tokenization completed")

    print(f"\nCached to: {cache_path}")


def cmd_help(source: str | None = None, **kwargs):
    """Show help for commands."""
    # source is the positional argument (e.g., 'help analyze' -> source='analyze')
    command = source
    help_text = {
        "analyze": """
analyze <dataset> [--tokenizer gpt2] [--subset 10000] [--recommendations]

  Analyze token distribution in a dataset.

  Arguments:
    dataset          Dataset name from registry or org/name path
    --tokenizer      Tokenizer to use (default: gpt2)
    --subset         Limit analysis to N examples
    --recommendations  Show vocab size recommendations

  Examples:
    analyze tinystories
    analyze tinystories --tokenizer bert-base-uncased --subset 5000
    analyze roneneldan/TinyStories --recommendations
""",
        "train": """
train <dataset|file.jsonl> --vocab-size N [--subset N] [--output path]

  Train a BPE tokenizer on a dataset or JSONL file.

  Arguments:
    source           Dataset name or path to .jsonl file
    --vocab-size     Target vocabulary size (required)
    --subset         Train on only N examples
    --output         Output directory for tokenizer

  Examples:
    train tinystories --vocab-size 4096
    train data/corpus.jsonl --vocab-size 8192 --subset 50000
    train tinystories --vocab-size 4096 --output ./my_tokenizer
""",
        "pretokenize": """
pretokenize <dataset> --tokenizer <name> [--max-length 1024] [--max-tokens N]

  Pretokenize a dataset for faster training.

  Arguments:
    dataset          Dataset name from registry
    --tokenizer      Tokenizer to use (required)
    --max-length     Maximum sequence length (default: 1024)
    --max-tokens     Limit total tokens in output

  Examples:
    pretokenize tinystories --tokenizer gpt2
    pretokenize tinystories --tokenizer ./my_tokenizer --max-length 512
""",
    }

    if command and command in help_text:
        print(help_text[command])
    else:
        print("""
Token TUI - Tokenization workflow tools

Commands:
  analyze      Analyze token distribution in a dataset
  train        Train a BPE tokenizer
  pretokenize  Pretokenize a dataset for training
  help         Show this help or help for a specific command
  exit         Exit the TUI

Type 'help <command>' for detailed usage.
Tab completion is available for commands, datasets, and tokenizers.
""")


COMMAND_HANDLERS = {
    "analyze": cmd_analyze,
    "train": cmd_train,
    "pretokenize": cmd_pretokenize,
    "help": cmd_help,
}


def handle_command(text: str) -> bool:
    """Handle a command. Returns True if should exit."""
    text = text.strip()
    if not text:
        return False

    command, args = parse_command(text)

    if command == "exit" or command == "quit":
        return True

    if "error" in args:
        print(f"Parse error: {args['error']}")
        return False

    handler = COMMAND_HANDLERS.get(command)
    if handler is None:
        print(f"Unknown command: {command}")
        print("Type 'help' for available commands.")
        return False

    try:
        handler(**args)
    except Exception as e:
        print(f"Error: {e}")

    return False

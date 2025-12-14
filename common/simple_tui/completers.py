"""Context-aware autocomplete for the tokenization TUI."""

from prompt_toolkit.completion import Completer, Completion

from .discovery import discover_datasets, discover_tokenizers

COMMANDS = {
    "analyze": "Analyze token distribution in a dataset",
    "train": "Train a BPE tokenizer on a dataset",
    "pretokenize": "Pretokenize a dataset with a tokenizer",
    "help": "Show help for commands",
    "exit": "Exit the TUI",
}

COMMAND_FLAGS = {
    "analyze": ["--tokenizer", "--subset", "--recommendations"],
    "train": ["--vocab-size", "--subset", "--output"],
    "pretokenize": ["--tokenizer", "--max-length", "--max-tokens"],
}


class TokenTUICompleter(Completer):
    """Context-aware completer for tokenization commands."""

    def __init__(self):
        self._datasets: list[str] | None = None
        self._tokenizers: list[str] | None = None

    @property
    def datasets(self) -> list[str]:
        if self._datasets is None:
            self._datasets = discover_datasets()
        return self._datasets

    @property
    def tokenizers(self) -> list[str]:
        if self._tokenizers is None:
            self._tokenizers = discover_tokenizers()
        return self._tokenizers

    def refresh(self):
        """Refresh cached datasets and tokenizers."""
        self._datasets = None
        self._tokenizers = None

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        words = text.split()
        word_before = document.get_word_before_cursor()

        # Position 0: Complete commands
        if len(words) == 0 or (len(words) == 1 and not text.endswith(" ")):
            for cmd, desc in COMMANDS.items():
                if cmd.startswith(word_before.lower()):
                    yield Completion(cmd, -len(word_before), display_meta=desc)
            return

        cmd = words[0].lower()
        if cmd not in COMMAND_FLAGS:
            return

        # Position 1: Complete dataset/source (after command)
        if len(words) == 1 or (len(words) == 2 and not text.endswith(" ")):
            for ds in self.datasets:
                if ds.lower().startswith(word_before.lower()):
                    yield Completion(ds, -len(word_before))
            return

        # Position 2+: Complete flags or flag values
        # Check if we're completing a flag value
        if len(words) >= 2:
            prev_word = words[-1] if text.endswith(" ") else words[-2] if len(words) >= 2 else ""

            # Complete tokenizer after --tokenizer flag
            if prev_word == "--tokenizer":
                for tok in self.tokenizers:
                    if tok.lower().startswith(word_before.lower()):
                        yield Completion(tok, -len(word_before))
                return

        # Complete flag names
        if word_before.startswith("-") or text.endswith(" "):
            prefix = word_before if word_before.startswith("-") else ""
            used_flags = {w for w in words if w.startswith("--")}
            for flag in COMMAND_FLAGS.get(cmd, []):
                if flag not in used_flags and flag.startswith(prefix):
                    yield Completion(flag, -len(prefix) if prefix else 0)

"""Main REPL application for the tokenization TUI."""

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from .commands import handle_command
from .completers import TokenTUICompleter

STYLE = Style.from_dict({
    "prompt": "#78dce8 bold",  # Cyan
})


def get_history_path() -> Path:
    """Get path for command history file."""
    cache_dir = Path.home() / ".cache" / "token-tui"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "history"


def main():
    """Main entry point for the TUI."""
    print("Token TUI - Tokenization workflow tools")
    print("Type 'help' for commands, Tab for completion, Ctrl+D to exit\n")

    completer = TokenTUICompleter()
    session = PromptSession(
        history=FileHistory(str(get_history_path())),
        completer=completer,
        style=STYLE,
        complete_while_typing=False,
    )

    while True:
        try:
            text = session.prompt([("class:prompt", "token> ")])
            should_exit = handle_command(text)
            if should_exit:
                print("Goodbye!")
                break
        except KeyboardInterrupt:
            print()  # New line after ^C
            continue
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()

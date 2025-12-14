"""Discover available datasets and tokenizers from the filesystem."""

from pathlib import Path


def get_assets_dir() -> Path:
    """Get the assets directory from the research repo."""
    return Path(__file__).parent.parent.parent / "assets"


def discover_datasets() -> list[str]:
    """Discover local datasets from assets/datasets/ and registry."""
    datasets = set()

    # Add registry names
    try:
        from common.data.dataset_registry import DATASET_REGISTRY

        datasets.update(DATASET_REGISTRY.keys())
    except ImportError:
        pass

    # Discover from filesystem
    datasets_dir = get_assets_dir() / "datasets"
    if datasets_dir.exists():
        for org_dir in datasets_dir.iterdir():
            if not org_dir.is_dir() or org_dir.name.startswith("."):
                continue
            for ds_dir in org_dir.iterdir():
                if ds_dir.is_dir() and not ds_dir.name.endswith("_tokenized"):
                    datasets.add(f"{org_dir.name}/{ds_dir.name}")

    return sorted(datasets)


def discover_tokenizers() -> list[str]:
    """Discover available tokenizers from common names and local trained ones."""
    tokenizers = [
        # Common HuggingFace tokenizers
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "bert-base-uncased",
        "bert-base-cased",
        "roberta-base",
        "t5-small",
        "t5-base",
    ]

    # Discover custom trained tokenizers
    tok_dir = get_assets_dir() / "models" / "tokenizers"
    if tok_dir.exists():
        for d in tok_dir.iterdir():
            if d.is_dir() and (d / "tokenizer.json").exists():
                tokenizers.append(d.name)

    return tokenizers


def discover_jsonl_files(directory: Path | None = None) -> list[str]:
    """Discover JSONL files in a directory."""
    if directory is None:
        directory = Path.cwd()

    jsonl_files = []
    for f in directory.glob("**/*.jsonl"):
        jsonl_files.append(str(f.relative_to(directory)))

    return sorted(jsonl_files)

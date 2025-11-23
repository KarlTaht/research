"""HuggingFace utilities for downloading datasets and models."""

from pathlib import Path
from typing import Optional, Union
import os

from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download


def download_dataset(
    name: str,
    config: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    **kwargs,
):
    """
    Download a dataset from HuggingFace Hub.

    Args:
        name: Dataset name (e.g., 'wmt14', 'imagenet-1k', 'HuggingFaceFW/fineweb')
        config: Dataset configuration/subset name (e.g., 'sample-10BT', 'de-en')
        output_dir: Where to save the dataset. If None, uses HF cache.
        split: Specific split to download (e.g., 'train', 'test'). If None, downloads all.
        cache_dir: Custom cache directory. Defaults to ~/.cache/huggingface/datasets
        **kwargs: Additional arguments passed to load_dataset()

    Returns:
        Dataset object

    Example:
        >>> from common.data.hf_utils import download_dataset
        >>> dataset = download_dataset('wmt14', split='train')
        >>> dataset = download_dataset('squad', output_dir='~/research/assets/datasets/squad')
        >>> dataset = download_dataset('HuggingFaceFW/fineweb', config='sample-100BT')
    """
    if cache_dir:
        cache_dir = Path(cache_dir).expanduser().resolve()

    if output_dir:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {name}")
    if config:
        print(f"Config: {config}")
    if split:
        print(f"Split: {split}")

    dataset = load_dataset(
        name,
        name=config,  # 'name' parameter in load_dataset is actually the config
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
        **kwargs,
    )

    if output_dir:
        print(f"Saving to: {output_dir}")
        dataset.save_to_disk(str(output_dir))

    print(f"✓ Dataset downloaded: {name}")
    return dataset


def download_model(
    repo_id: str,
    output_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    revision: str = "main",
    **kwargs,
):
    """
    Download a model from HuggingFace Hub.

    Args:
        repo_id: Model repository ID (e.g., 'bert-base-uncased', 'openai/clip-vit-base-patch32')
        output_dir: Where to save the model. If None, uses HF cache.
        filename: Specific file to download. If None, downloads entire repo.
        cache_dir: Custom cache directory. Defaults to ~/.cache/huggingface/hub
        revision: Git revision (branch, tag, or commit hash)
        **kwargs: Additional arguments passed to hf_hub_download() or snapshot_download()

    Returns:
        Path to downloaded file/directory

    Example:
        >>> from common.data.hf_utils import download_model
        >>> # Download specific file
        >>> model_path = download_model('bert-base-uncased', filename='pytorch_model.bin')
        >>> # Download entire model repo
        >>> model_dir = download_model('openai/clip-vit-base-patch32',
        ...                           output_dir='~/research/assets/models/clip')
    """
    if cache_dir:
        cache_dir = Path(cache_dir).expanduser().resolve()

    if output_dir:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model: {repo_id}")
    if filename:
        print(f"File: {filename}")

    if filename:
        # Download single file
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir) if cache_dir else None,
            revision=revision,
            **kwargs,
        )
        if output_dir:
            import shutil
            dest_path = output_dir / filename
            shutil.copy(file_path, dest_path)
            print(f"✓ Model file saved to: {dest_path}")
            return str(dest_path)
        else:
            print(f"✓ Model file downloaded to cache: {file_path}")
            return file_path
    else:
        # Download entire repo
        repo_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            revision=revision,
            **kwargs,
        )
        if output_dir:
            import shutil
            shutil.copytree(repo_path, output_dir, dirs_exist_ok=True)
            print(f"✓ Model repository saved to: {output_dir}")
            return str(output_dir)
        else:
            print(f"✓ Model repository downloaded to cache: {repo_path}")
            return repo_path


def get_default_assets_dir() -> Path:
    """Get the default assets directory for the research monorepo."""
    # Assumes this is called from ~/research
    research_root = Path(__file__).parent.parent.parent
    return research_root / "assets"


def get_datasets_dir() -> Path:
    """Get the default datasets directory."""
    return get_default_assets_dir() / "datasets"


def get_models_dir() -> Path:
    """Get the default models directory."""
    return get_default_assets_dir() / "models"

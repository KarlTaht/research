# HuggingFace Download Guide

This guide shows how to download datasets and models from HuggingFace Hub using both CLI and programmatic approaches.

## Two Ways to Download

### 1. CLI Scripts (tools/)

Use these for manual downloads from the command line.

#### Download Datasets

```bash
# Download to default location (~/research/assets/datasets/<name>)
python tools/download_hf_dataset.py --name squad

# Download specific split
python tools/download_hf_dataset.py --name wmt14 --split train

# Download with config/subset
python tools/download_hf_dataset.py --name glue --config mrpc

# Download to custom location
python tools/download_hf_dataset.py --name imagenet-1k --output /path/to/data
```

#### Download Models

```bash
# Download entire model repository
python tools/download_hf_model.py --repo-id bert-base-uncased

# Download specific file
python tools/download_hf_model.py --repo-id bert-base-uncased --filename pytorch_model.bin

# Download to custom location
python tools/download_hf_model.py --repo-id openai/clip-vit-base-patch32 --output /custom/path

# Download specific version/branch
python tools/download_hf_model.py --repo-id facebook/opt-1.3b --revision v1.0
```

### 2. Programmatic API (common/data/)

Import and use in your training scripts.

#### In Your Training Script

```python
from common.data import download_dataset, download_model

# Download dataset programmatically
dataset = download_dataset(
    name='squad',
    output_dir='~/research/assets/datasets/squad',
    split='train'
)

# Or let it use default location
dataset = download_dataset('wmt14')  # Goes to assets/datasets/

# Download model
model_path = download_model(
    repo_id='bert-base-uncased',
    output_dir='~/research/assets/models/bert-base'
)

# Download specific model file
weights_path = download_model(
    repo_id='openai/clip-vit-base-patch32',
    filename='pytorch_model.bin'
)
```

#### Helper Functions for Paths

```python
from common.data import get_datasets_dir, get_models_dir

# Get standard asset directories
datasets_dir = get_datasets_dir()  # ~/research/assets/datasets
models_dir = get_models_dir()      # ~/research/assets/models

# Use in your code
my_data_path = datasets_dir / 'my_dataset'
my_model_path = models_dir / 'my_model'
```

## Example Workflows

### Workflow 1: Manual Download, Then Use

```bash
# 1. Download dataset manually via CLI
python tools/download_hf_dataset.py --name squad

# 2. Use in training script
```

```python
# train.py
from common.data import get_datasets_dir
from datasets import load_from_disk

# Load the dataset you downloaded
dataset = load_from_disk(get_datasets_dir() / 'squad')
```

### Workflow 2: Download on Demand in Code

```python
# train.py
from common.data import download_dataset
from datasets import load_from_disk

# Download if not exists, or load if already downloaded
try:
    dataset = load_from_disk('~/research/assets/datasets/squad')
except:
    dataset = download_dataset('squad', output_dir='~/research/assets/datasets/squad')
```

### Workflow 3: Paper Implementation with Symlinks

```bash
# 1. Download dataset to centralized location
python tools/download_hf_dataset.py --name wmt14

# 2. Create symlink in your paper directory
cd ~/research/papers/attention_is_all_you_need
ln -s ../../assets/datasets/wmt14 data

# 3. Use in training script
```

```python
# papers/attention_is_all_you_need/train.py
from pathlib import Path
from datasets import load_from_disk

# Load from symlinked directory
data_dir = Path(__file__).parent / 'data'
dataset = load_from_disk(data_dir)
```

## Popular Datasets Examples

```bash
# NLP
python tools/download_hf_dataset.py --name squad
python tools/download_hf_dataset.py --name glue --config mrpc
python tools/download_hf_dataset.py --name wmt14 --config de-en

# Vision
python tools/download_hf_dataset.py --name mnist
python tools/download_hf_dataset.py --name cifar10
python tools/download_hf_dataset.py --name imagenet-1k  # Requires authentication

# Multimodal
python tools/download_hf_dataset.py --name conceptual_captions
```

## Popular Models Examples

```bash
# Language models
python tools/download_hf_model.py --repo-id bert-base-uncased
python tools/download_hf_model.py --repo-id gpt2
python tools/download_hf_model.py --repo-id facebook/opt-1.3b

# Vision models
python tools/download_hf_model.py --repo-id openai/clip-vit-base-patch32
python tools/download_hf_model.py --repo-id google/vit-base-patch16-224

# Multimodal
python tools/download_hf_model.py --repo-id openai/clip-vit-large-patch14
```

## Tips

1. **First download**: Downloads go to HuggingFace cache by default, then copied to output_dir
2. **Subsequent runs**: If file exists in cache, it's reused (faster)
3. **Large models**: Some models require authentication - use `huggingface-cli login` first
4. **Disk space**: Assets are gitignored, but can take significant space. Monitor `assets/` size.
5. **Symlinks**: Use symlinks to avoid duplicating datasets across papers/projects

## Authentication (for Private/Gated Models)

Some models/datasets require HuggingFace authentication:

```bash
# Login once
huggingface-cli login

# Then download works
python tools/download_hf_model.py --repo-id meta-llama/Llama-2-7b
```

## Caching Behavior

- **Default cache**: `~/.cache/huggingface/`
- **Custom cache**: Use `--cache-dir` flag or set `HF_HOME` environment variable
- **Output dir**: If specified, files are copied from cache to output location
- **No output dir**: Files remain in cache only

## Integration with Training

### Option 1: Pre-download Everything

```bash
# Download all assets before training
python tools/download_hf_dataset.py --name squad
python tools/download_hf_model.py --repo-id bert-base-uncased

# Then train without network dependency
python train.py
```

### Option 2: Download on Demand

```python
# train.py downloads what it needs
from common.data import download_dataset, download_model

dataset = download_dataset('squad')
model_path = download_model('bert-base-uncased')
# Continue training...
```

Choose based on your workflow:
- **Pre-download**: Better for reproducibility, offline work, cluster environments
- **On-demand**: Better for rapid prototyping, automatic dependency management

# Repository Conventions

## Directory Structure

### Projects (`projects/`)
- Each project is a self-contained research experiment
- Must have: `train.py` and/or `evaluate.py`
- Should have: `configs/` directory with YAML files
- Should have: `CLAUDE.md` or `README.md` for documentation
- May have: `tests/` for project-specific tests

### Common Package (`common/`)
- Shared code used across all projects
- Always import with absolute paths: `from common.models import BaseLanguageModel`
- Submodules:
  - `models/` - Model architectures
  - `training/` - Training infrastructure
  - `data/` - Dataset utilities
  - `utils/` - General utilities
  - `visualization/` - Plotting

### Assets (`assets/`)
- **GITIGNORED** - never committed
- `datasets/` - Downloaded and processed datasets
- `models/` - Pretrained models and checkpoints
- `outputs/` - Experiment results, logs, checkpoints

## File Naming

### Scripts
- `train.py` - Training entry point
- `evaluate.py` or `eval.py` - Evaluation script
- `validate.py` - Validation utilities
- `download_*.py` - Data download scripts

### Configs
- Located in `configs/` subdirectory
- Named by purpose: `default.yaml`, `small.yaml`, `large.yaml`
- Or by dataset: `tinystories.yaml`, `wmt14.yaml`

### Tests
- In `tests/` directory
- Named `test_*.py`
- Use pytest conventions

## Config Structure

Standard YAML config sections:

```yaml
model:
  d_model: 256
  n_heads: 4
  n_layers: 6
  vocab_size: 50257

training:
  batch_size: 32
  learning_rate: 3e-4
  max_steps: 10000

data:
  dataset: tinystories
  max_seq_len: 512
```

## Import Patterns

```python
# CORRECT - absolute imports
from common.models import BaseLanguageModel
from common.training import Evaluator
from common.utils import save_experiment

# WRONG - relative imports
from ..common.models import BaseLanguageModel
```

## Running Experiments

```bash
# From project directory
cd projects/custom_transformer
python train.py --config configs/default.yaml

# From repo root
python -m projects.custom_transformer.train --config projects/custom_transformer/configs/default.yaml
```

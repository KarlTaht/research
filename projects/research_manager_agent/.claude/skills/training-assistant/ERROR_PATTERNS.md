# Common Training Errors and Fixes

## CUDA Errors

### Out of Memory (OOM)
```
RuntimeError: CUDA out of memory
```

**Fixes:**
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision (fp16)
4. Reduce sequence length
5. Use gradient accumulation instead of large batches

**Config changes:**
```yaml
training:
  batch_size: 16  # Reduce from 32
  gradient_accumulation_steps: 2  # Maintain effective batch size
  mixed_precision: true
```

### CUDA Not Available
```
RuntimeError: CUDA is not available
```

**Fixes:**
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
3. Check CUDA version compatibility

### Device Mismatch
```
RuntimeError: Expected all tensors to be on the same device
```

**Fixes:**
1. Move model to device: `model.to(device)`
2. Move data to device: `batch = batch.to(device)`
3. Check all inputs are on same device

## Import Errors

### Module Not Found
```
ModuleNotFoundError: No module named 'common'
```

**Fixes:**
1. Activate virtual environment: `source .venv/bin/activate`
2. Install package: `uv pip install -e .`
3. Run from repo root, not project directory

### Relative Import Error
```
ImportError: attempted relative import with no known parent package
```

**Fixes:**
1. Use absolute imports: `from common.models import ...`
2. Run as module: `python -m projects.custom_transformer.train`

## Config Errors

### Missing Config Key
```
KeyError: 'model'
```

**Fixes:**
1. Check config file exists and is valid YAML
2. Verify all required sections are present
3. Use `read_config()` to inspect structure

### Invalid YAML
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Fixes:**
1. Check indentation (use spaces, not tabs)
2. Verify colons have spaces after them
3. Quote strings with special characters

## Data Errors

### Dataset Not Found
```
FileNotFoundError: Dataset 'tinystories' not found
```

**Fixes:**
1. Download dataset: `research-data download dataset --name tinystories`
2. Check symlinks are valid
3. Verify path in config

### Shape Mismatch
```
RuntimeError: size mismatch, m1: [32 x 256], m2: [512 x 256]
```

**Fixes:**
1. Check d_model matches across layers
2. Verify embedding dimension matches model dimension
3. Check sequence length truncation

## Training Instabilities

### NaN Loss
**Symptoms:** Loss becomes NaN during training

**Fixes:**
1. Reduce learning rate
2. Add gradient clipping: `max_grad_norm: 1.0`
3. Check for division by zero in custom code
4. Verify input data doesn't contain NaN

### Loss Not Decreasing
**Symptoms:** Loss stays flat or increases

**Fixes:**
1. Increase learning rate (if too low)
2. Check data loading is shuffled
3. Verify model architecture
4. Try different optimizer (AdamW)

## Environment Issues

### Wrong Python Version
```
SyntaxError: invalid syntax  # Often from f-strings or type hints
```

**Fixes:**
1. Use Python 3.10+: `python --version`
2. Activate correct environment

### Package Version Conflict
```
ImportError: cannot import name 'X' from 'Y'
```

**Fixes:**
1. Update packages: `uv pip install -U package_name`
2. Check compatible versions
3. Create fresh environment if needed

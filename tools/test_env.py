#!/usr/bin/env python3
"""Test that all project dependencies are properly installed and working."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_python_version():
    """Check Python version."""
    print("=" * 70)
    print("Python Environment")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    version = sys.version_info
    if version >= (3, 10):
        print("✓ Python version >= 3.10")
        return True
    else:
        print("✗ Python version < 3.10 (required: >= 3.10)")
        return False


def test_core_packages():
    """Test core ML packages."""
    print("\n" + "=" * 70)
    print("Core ML Packages")
    print("=" * 70)

    results = []

    # NumPy
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        results.append(False)

    # Pandas
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Pandas: {e}")
        results.append(False)

    # SciPy
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ SciPy: {e}")
        results.append(False)

    # Scikit-learn
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Scikit-learn: {e}")
        results.append(False)

    return all(results)


def test_pytorch():
    """Test PyTorch and CUDA."""
    print("\n" + "=" * 70)
    print("PyTorch & CUDA")
    print("=" * 70)

    results = []

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        results.append(True)

        # Check CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"  - CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            results.append(True)
        else:
            print("⚠ CUDA not available (CPU only)")
            results.append(True)  # Not a failure, just a warning

        # Check cuDNN
        if torch.backends.cudnn.is_available():
            print(f"✓ cuDNN available: {torch.backends.cudnn.version()}")
        else:
            print("⚠ cuDNN not available")

        # Quick tensor test
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.matmul(x, y)
        print(f"✓ Tensor operations working (CPU)")

        # CUDA tensor test
        if cuda_available:
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            print(f"✓ Tensor operations working (CUDA)")

    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        results.append(False)
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        results.append(False)

    # TorchVision
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ TorchVision: {e}")
        results.append(False)

    # TorchAudio
    try:
        import torchaudio
        print(f"✓ TorchAudio {torchaudio.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ TorchAudio: {e}")
        results.append(False)

    return all(results)


def test_huggingface():
    """Test HuggingFace libraries."""
    print("\n" + "=" * 70)
    print("HuggingFace Ecosystem")
    print("=" * 70)

    results = []

    # Transformers
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        results.append(False)

    # Datasets
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Datasets: {e}")
        results.append(False)

    # HuggingFace Hub
    try:
        import huggingface_hub
        print(f"✓ HuggingFace Hub {huggingface_hub.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ HuggingFace Hub: {e}")
        results.append(False)

    # Tokenizers
    try:
        import tokenizers
        print(f"✓ Tokenizers {tokenizers.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Tokenizers: {e}")
        results.append(False)

    return all(results)


def test_visualization():
    """Test visualization libraries."""
    print("\n" + "=" * 70)
    print("Visualization & Plotting")
    print("=" * 70)

    results = []

    # Matplotlib
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        results.append(False)

    # Seaborn
    try:
        import seaborn
        print(f"✓ Seaborn {seaborn.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Seaborn: {e}")
        results.append(False)

    # Plotly
    try:
        import plotly
        print(f"✓ Plotly {plotly.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Plotly: {e}")
        results.append(False)

    return all(results)


def test_experiment_tracking():
    """Test experiment tracking tools."""
    print("\n" + "=" * 70)
    print("Experiment Tracking")
    print("=" * 70)

    results = []

    # TensorBoard
    try:
        import tensorboard
        print(f"✓ TensorBoard {tensorboard.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ TensorBoard: {e}")
        results.append(False)

    # Weights & Biases
    try:
        import wandb
        print(f"✓ Weights & Biases {wandb.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Weights & Biases: {e}")
        results.append(False)

    return all(results)


def test_utilities():
    """Test utility packages."""
    print("\n" + "=" * 70)
    print("Utilities")
    print("=" * 70)

    results = []

    # TQDM
    try:
        import tqdm
        print(f"✓ TQDM {tqdm.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ TQDM: {e}")
        results.append(False)

    # PyYAML
    try:
        import yaml
        print(f"✓ PyYAML {yaml.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        results.append(False)

    # OmegaConf
    try:
        import omegaconf
        print(f"✓ OmegaConf {omegaconf.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ OmegaConf: {e}")
        results.append(False)

    # python-dotenv
    try:
        import dotenv
        version = getattr(dotenv, "__version__", "installed")
        print(f"✓ python-dotenv {version}")
        results.append(True)
    except ImportError as e:
        print(f"✗ python-dotenv: {e}")
        results.append(False)

    return all(results)


def test_dev_tools():
    """Test development tools."""
    print("\n" + "=" * 70)
    print("Development Tools")
    print("=" * 70)

    results = []

    # Jupyter
    try:
        import jupyter
        print(f"✓ Jupyter (meta-package)")
        results.append(True)
    except ImportError as e:
        print(f"✗ Jupyter: {e}")
        results.append(False)

    # IPython
    try:
        import IPython
        print(f"✓ IPython {IPython.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ IPython: {e}")
        results.append(False)

    # Black
    try:
        import black
        print(f"✓ Black {black.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Black: {e}")
        results.append(False)

    # Ruff
    try:
        import ruff
        version = getattr(ruff, "__version__", "installed")
        print(f"✓ Ruff {version}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Ruff: {e}")
        results.append(False)

    # Pytest
    try:
        import pytest
        print(f"✓ Pytest {pytest.__version__}")
        results.append(True)
    except ImportError as e:
        print(f"✗ Pytest: {e}")
        results.append(False)

    # MyPy
    try:
        import mypy
        try:
            version = mypy.version.__version__
        except AttributeError:
            version = getattr(mypy, "__version__", "installed")
        print(f"✓ MyPy {version}")
        results.append(True)
    except ImportError as e:
        print(f"✗ MyPy: {e}")
        results.append(False)

    return all(results)


def test_common_package():
    """Test that the common package can be imported."""
    print("\n" + "=" * 70)
    print("Project Package")
    print("=" * 70)

    try:
        import common
        print(f"✓ common package can be imported")

        # Test submodules
        from common.data import download_dataset, download_model
        print(f"✓ common.data imports working")

        from common.data import get_datasets_dir, get_models_dir
        datasets_dir = get_datasets_dir()
        models_dir = get_models_dir()
        print(f"✓ Helper functions working:")
        print(f"  - Datasets dir: {datasets_dir}")
        print(f"  - Models dir: {models_dir}")

        return True
    except ImportError as e:
        print(f"✗ common package: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("█" * 70)
    print("  ML RESEARCH ENVIRONMENT TEST")
    print("█" * 70)

    results = []

    # Run all tests
    results.append(("Python Version", test_python_version()))
    results.append(("Core Packages", test_core_packages()))
    results.append(("PyTorch & CUDA", test_pytorch()))
    results.append(("HuggingFace", test_huggingface()))
    results.append(("Visualization", test_visualization()))
    results.append(("Experiment Tracking", test_experiment_tracking()))
    results.append(("Utilities", test_utilities()))
    results.append(("Dev Tools", test_dev_tools()))
    results.append(("Project Package", test_common_package()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

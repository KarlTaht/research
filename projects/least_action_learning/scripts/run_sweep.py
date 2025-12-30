#!/usr/bin/env python3
"""Run a sweep of experiments from a sweep config file."""

import argparse
import yaml
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def load_sweep_config(config_path: str) -> dict:
    """Load sweep config with base + experiments."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, experiment: dict) -> dict:
    """Merge experiment config over base config."""
    merged = base.copy()
    merged.update(experiment)
    return merged


def run_experiment(config: dict, project_root: Path, dry_run: bool = False) -> bool:
    """Run a single experiment. Returns True if successful."""
    name = config.get("name", "unnamed")

    # Build command line arguments
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "train.py"),
    ]

    # Add all config values as CLI arguments
    for key, value in config.items():
        if key == "name":
            cmd.extend(["--name", str(value)])
        elif value is None:
            continue
        elif key == "p":
            cmd.extend(["--p", str(value)])
        elif key == "operation":
            cmd.extend(["--operation", str(value)])
        elif key == "train_frac":
            cmd.extend(["--train-frac", str(value)])
        elif key == "model_type":
            cmd.extend(["--model-type", str(value)])
        elif key == "hidden_dim":
            cmd.extend(["--hidden-dim", str(value)])
        elif key == "n_layers":
            cmd.extend(["--n-layers", str(value)])
        elif key == "n_heads":
            cmd.extend(["--n-heads", str(value)])
        elif key == "epochs":
            cmd.extend(["--epochs", str(value)])
        elif key == "lr":
            cmd.extend(["--lr", f"{float(value):.10g}"])
        elif key == "weight_decay":
            cmd.extend(["--weight-decay", f"{float(value):.10g}"])
        elif key == "beta1":
            cmd.extend(["--beta1", f"{float(value):.10g}"])
        elif key == "beta2":
            cmd.extend(["--beta2", f"{float(value):.10g}"])
        elif key == "eps":
            cmd.extend(["--eps", f"{float(value):.10g}"])
        elif key == "routing_regularizer":
            cmd.extend(["--routing-reg", str(value) if value else "none"])
        elif key == "lambda_routing":
            cmd.extend(["--lambda-routing", str(value)])
        elif key == "lambda_spectral":
            cmd.extend(["--lambda-spectral", str(value)])
        elif key == "log_every":
            cmd.extend(["--log-every", str(value)])
        elif key == "save_routing_every":
            cmd.extend(["--save-routing-every", str(value)])
        elif key == "seed":
            cmd.extend(["--seed", str(value)])
        elif key == "warmup_epochs":
            cmd.extend(["--warmup-epochs", str(value)])

    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"{'='*60}")

    if dry_run:
        print(f"[DRY RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        return True

    start_time = datetime.now()
    result = subprocess.run(cmd, cwd=project_root.parent.parent)
    elapsed = datetime.now() - start_time

    if result.returncode == 0:
        print(f"\n✓ {name} completed in {elapsed}")
        return True
    else:
        print(f"\n✗ {name} failed with return code {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run experiment sweep")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to sweep YAML config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run only a specific experiment by name")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start from experiment index (0-based)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    sweep_config = load_sweep_config(args.config)

    base_config = sweep_config.get("base", {})
    experiments = sweep_config.get("experiments", [])

    if not experiments:
        print("No experiments found in config")
        return

    # Filter to specific experiment if requested
    if args.experiment:
        experiments = [e for e in experiments if e.get("name") == args.experiment]
        if not experiments:
            print(f"Experiment '{args.experiment}' not found")
            return

    # Start from index
    experiments = experiments[args.start_from:]

    print(f"Sweep: {len(experiments)} experiments")
    print(f"Base config: p={base_config.get('p')}, epochs={base_config.get('epochs')}")

    successful = 0
    failed = 0

    for i, exp in enumerate(experiments):
        config = merge_configs(base_config, exp)
        print(f"\n[{i+1}/{len(experiments)}] ", end="")

        if run_experiment(config, project_root, dry_run=args.dry_run):
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Sweep complete: {successful} succeeded, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

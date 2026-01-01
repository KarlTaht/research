"""Migrate existing Parquet experiments to TensorBoard format.

Usage:
    # Migrate all experiments
    python -m common.utils.migrate_to_tensorboard

    # Migrate specific experiments
    python -m common.utils.migrate_to_tensorboard --experiments exp1 exp2

    # Preview without writing
    python -m common.utils.migrate_to_tensorboard --dry-run
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from .experiment_storage import get_experiments_dir, list_experiments, load_experiment
from .tensorboard_writer import get_tensorboard_dir


def migrate_experiment(
    experiment_name: str,
    parquet_dir: Optional[Path] = None,
    tensorboard_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> bool:
    """
    Migrate a single Parquet experiment to TensorBoard format.

    Args:
        experiment_name: Name of experiment (without .parquet extension)
        parquet_dir: Source directory for Parquet files
        tensorboard_dir: Destination directory for TensorBoard logs
        dry_run: If True, print what would be done without writing

    Returns:
        True if migration succeeded, False otherwise
    """
    if parquet_dir is None:
        parquet_dir = get_experiments_dir()
    if tensorboard_dir is None:
        tensorboard_dir = get_tensorboard_dir()

    try:
        df = load_experiment(experiment_name, parquet_dir)
    except FileNotFoundError:
        print(f"  Skipping {experiment_name}: not found")
        return False

    print(f"  Migrating {experiment_name} ({len(df)} rows)...")

    if dry_run:
        print(f"    Would write to: {tensorboard_dir / experiment_name}")
        return True

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=str(tensorboard_dir / experiment_name))

    # Log hyperparameters if available
    if "meta_model_config" in df.columns:
        try:
            first_row = df.iloc[0]
            model_config_str = first_row.get("meta_model_config")
            train_config_str = first_row.get("meta_train_config")

            hparams = {}
            if model_config_str and pd.notna(model_config_str):
                model_config = json.loads(model_config_str)
                for k, v in model_config.items():
                    if isinstance(v, (int, float, str, bool)):
                        hparams[f"model/{k}"] = v

            if train_config_str and pd.notna(train_config_str):
                train_config = json.loads(train_config_str)
                for k, v in train_config.items():
                    if isinstance(v, (int, float, str, bool)):
                        hparams[f"train/{k}"] = v

            if hparams:
                writer.add_text("hyperparameters", json.dumps(hparams, indent=2), 0)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    # Determine step column (newer format has global_step, older has epoch only)
    has_global_step = "global_step" in df.columns

    # Log each row as scalars
    for idx, row in df.iterrows():
        # Use global_step if available, otherwise use epoch as step
        if has_global_step:
            step = int(row["global_step"])
        else:
            step = int(row.get("epoch", idx))

        # Training metrics
        if pd.notna(row.get("train_loss")):
            writer.add_scalar("train/loss", row["train_loss"], step)
        if pd.notna(row.get("train_perplexity")):
            writer.add_scalar("train/perplexity", row["train_perplexity"], step)
        if pd.notna(row.get("learning_rate")):
            writer.add_scalar("train/learning_rate", row["learning_rate"], step)
        if pd.notna(row.get("grad_norm")):
            writer.add_scalar("train/grad_norm", row["grad_norm"], step)

        # Validation metrics
        if pd.notna(row.get("val_loss")):
            writer.add_scalar("val/loss", row["val_loss"], step)
        if pd.notna(row.get("val_perplexity")):
            writer.add_scalar("val/perplexity", row["val_perplexity"], step)

        # Performance metrics
        if pd.notna(row.get("tokens_per_second")):
            writer.add_scalar("perf/tokens_per_second", row["tokens_per_second"], step)
        if pd.notna(row.get("batch_time_ms")):
            writer.add_scalar("perf/batch_time_ms", row["batch_time_ms"], step)
        if pd.notna(row.get("approximate_tflops")):
            writer.add_scalar("perf/tflops", row["approximate_tflops"], step)

    writer.close()
    print(f"    Wrote TensorBoard logs to: {tensorboard_dir / experiment_name}")
    return True


def migrate_all_experiments(
    experiments: Optional[List[str]] = None,
    parquet_dir: Optional[Path] = None,
    tensorboard_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> dict:
    """
    Migrate multiple experiments from Parquet to TensorBoard.

    Args:
        experiments: List of experiment names (None = all)
        parquet_dir: Source directory
        tensorboard_dir: Destination directory
        dry_run: Preview mode

    Returns:
        Dict with 'migrated' and 'failed' counts
    """
    if parquet_dir is None:
        parquet_dir = get_experiments_dir()
    if tensorboard_dir is None:
        tensorboard_dir = get_tensorboard_dir()

    if experiments is None:
        experiments = list_experiments(parquet_dir)

    print(f"Migrating {len(experiments)} experiments...")
    print(f"  Source: {parquet_dir}")
    print(f"  Destination: {tensorboard_dir}")

    migrated = 0
    failed = 0

    for exp_name in experiments:
        success = migrate_experiment(exp_name, parquet_dir, tensorboard_dir, dry_run)
        if success:
            migrated += 1
        else:
            failed += 1

    print(f"\nMigration complete: {migrated} succeeded, {failed} failed")
    return {"migrated": migrated, "failed": failed}


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Parquet experiments to TensorBoard format"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Specific experiments to migrate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing files",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Source Parquet directory",
    )
    parser.add_argument(
        "--dest",
        type=str,
        help="Destination TensorBoard directory",
    )

    args = parser.parse_args()

    migrate_all_experiments(
        experiments=args.experiments,
        parquet_dir=Path(args.source) if args.source else None,
        tensorboard_dir=Path(args.dest) if args.dest else None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

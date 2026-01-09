#!/usr/bin/env python3
"""Training script for least action learning experiments."""

import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root.parent.parent))

from projects.least_action_learning.src.trainer import TrainerConfig, Trainer
from projects.least_action_learning.src.visualize import save_all_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description="Train least action learning models")

    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Data arguments
    parser.add_argument("--p", type=int, default=113, help="Prime modulus")
    parser.add_argument("--operation", type=str, default="add", choices=["add", "multiply", "both"])
    parser.add_argument("--train-frac", type=float, default=0.3)

    # Model arguments
    parser.add_argument("--model-type", type=str, default="routed",
                        choices=["baseline", "routed", "single_head", "transformer", "routed_transformer"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-ffn-heads", type=int, default=None,
                        help="Number of parallel FFN heads for routed_transformer (default: same as n_heads)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup epochs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--extra-epochs", type=int, default=0,
                        help="Add extra epochs on top of config (use with --resume)")

    # Regularization arguments
    parser.add_argument("--routing-reg", type=str, default="entropy",
                        choices=["entropy", "sparsity", "gini", "none"])
    parser.add_argument("--lambda-routing", type=float, default=0.01)
    parser.add_argument("--lambda-spectral", type=float, default=0.0)

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--name", type=str, default="")

    # Logging
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-routing-every", type=int, default=1000,
                        help="Save routing snapshots every N epochs (for visualizer)")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # Start with defaults
    config_dict = {}

    # Load from YAML if provided
    if args.config:
        config_dict = load_config(args.config)

    # Override with command line arguments
    cli_overrides = {
        "p": args.p,
        "operation": args.operation,
        "train_frac": args.train_frac,
        "model_type": args.model_type,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "n_ffn_heads": args.n_ffn_heads,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "eps": args.eps,
        "warmup_epochs": args.warmup_epochs,
        "routing_regularizer": args.routing_reg if args.routing_reg != "none" else None,
        "lambda_routing": args.lambda_routing,
        "lambda_spectral": args.lambda_spectral,
        "log_every": args.log_every,
        "save_routing_every": args.save_routing_every,
        "seed": args.seed,
        "name": args.name,
    }

    # Only override if explicitly set (non-default)
    for key, value in cli_overrides.items():
        if key not in config_dict:
            config_dict[key] = value

    # Create config
    config = TrainerConfig(**config_dict)

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        name = config.name or f"{config.model_type}_{config.p}_{config.operation}"
        output_dir = project_root / "outputs" / name

    # Handle resume
    start_epoch = 0
    if args.resume:
        checkpoint_path = output_dir / "final.pt"
        if not checkpoint_path.exists():
            checkpoint_path = output_dir / "best.pt"
        if checkpoint_path.exists():
            import torch
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Determine last trained epoch
            if "epoch" in checkpoint and checkpoint["epoch"] is not None:
                # New checkpoint format with explicit epoch
                start_epoch = checkpoint["epoch"] + 1
            elif checkpoint_path.name == "final.pt" and "config" in checkpoint:
                # Old final.pt: training completed, use config.epochs
                saved_config = checkpoint["config"]
                start_epoch = saved_config.get("epochs", 0)
            else:
                # Fall back to best_epoch (may be inaccurate)
                start_epoch = checkpoint.get("best_epoch", 0) + 1

            print(f"Resuming from checkpoint: {checkpoint_path}")
            print(f"  Resuming at epoch: {start_epoch}, Best test acc: {checkpoint.get('best_test_acc', 0):.4f}")
        else:
            print(f"No checkpoint found in {output_dir}, starting fresh")

    # Add extra epochs if requested
    if args.extra_epochs > 0:
        config.epochs = config.epochs + args.extra_epochs
        print(f"Extended training to {config.epochs} epochs")

    # Train
    trainer = Trainer(config, output_dir)

    # Load checkpoint if resuming
    if args.resume and (output_dir / "final.pt").exists():
        trainer.load_checkpoint("final.pt")
    elif args.resume and (output_dir / "best.pt").exists():
        trainer.load_checkpoint("best.pt")

    history = trainer.train(start_epoch=start_epoch)

    # Save visualizations
    save_all_visualizations(
        history=history,
        model=trainer.model,
        dataset=trainer.dataset,
        p=config.p,
        device=trainer.device,
        output_dir=output_dir / "plots",
    )

    # Print summary
    grokking_step = history.get_grokking_step()
    if grokking_step:
        print(f"\nGrokking occurred at step {grokking_step}")
    else:
        print("\nGrokking threshold (95%) not reached")


if __name__ == "__main__":
    main()

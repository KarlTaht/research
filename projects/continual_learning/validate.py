#!/usr/bin/env python3
"""Validate trained models with interactive chat mode.

Usage:
    python validate.py --checkpoint checkpoints/automotive_baseline/best.pt --config configs/automotive.yaml --chat
    python validate.py --checkpoint checkpoints/food_baseline/latest.pt --config configs/food.yaml --prompt "To make"
    python validate.py --checkpoint checkpoints/automotive_baseline/best.pt --config configs/automotive.yaml --chat --temperature 0.5
"""

import argparse
import torch
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import TorchTransformer
from train import load_tokenizer


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: torch.device,
                                tokenizer_override: str = None):
    """Load model and tokenizer from checkpoint + config.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        config_path: Path to YAML config file (for model architecture)
        device: torch device (cuda/cpu)
        tokenizer_override: Optional tokenizer name to override config

    Returns:
        Tuple of (model, tokenizer, checkpoint_dict)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load config for model architecture
    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build model config from YAML
    model_config = {
        'vocab_size': None,  # Will be set from tokenizer
        'd_model': config['model']['d_model'],
        'n_heads': config['model']['n_heads'],
        'n_blocks': config['model']['n_blocks'],
        'd_ffn': config['model']['d_ffn'],
        'max_seq_len': config['model']['max_seq_len'],
    }

    # Get tokenizer name (priority: CLI override > config)
    if tokenizer_override:
        tokenizer_name = tokenizer_override
    else:
        tokenizer_name = config['data'].get('tokenizer', 'combined_bpe_32768')

    print(f"Using tokenizer: {tokenizer_name}")
    tokenizer = load_tokenizer(tokenizer_name)

    # Set vocab size from tokenizer
    model_config['vocab_size'] = len(tokenizer)

    # Create and load model
    model = TorchTransformer(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    return model, tokenizer, checkpoint


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_length: int = 100,
                      temperature: float = 0.8, top_k: int = 50, device: str = 'cuda') -> str:
    """Generate text continuation from a prompt using autoregressive sampling."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids

    for _ in range(max_length - input_ids.size(1)):
        # Forward pass
        outputs = model(generated)
        logits = outputs['logits']

        # Get next token logits (last position)
        next_logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k is not None and top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')

        # Sample
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        generated = torch.cat([generated, next_token], dim=1)

        # Stop on EOS
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

        # Stop if max_seq_len reached
        if generated.size(1) >= model.max_seq_len:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def chat_mode(model, tokenizer, device, max_length: int = 100,
              temperature: float = 0.8, top_k: int = 50):
    """Interactive chat loop."""
    print("\n" + "=" * 50)
    print("CHAT MODE")
    print("=" * 50)
    print("Type your prompt and press Enter.")
    print("Commands: 'quit' to exit, 'temp X' to change temperature")
    print(f"Settings: max_length={max_length}, temperature={temperature}, top_k={top_k}")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input("You: ").strip()

            # Handle commands
            if prompt.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            if prompt.lower().startswith('temp '):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"Temperature set to {temperature}")
                except (IndexError, ValueError):
                    print("Usage: temp 0.5")
                continue
            if not prompt:
                continue

            # Generate response
            response = generate_response(
                model, tokenizer, prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            print(f"Model: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate trained TorchTransformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate.py --checkpoint checkpoints/automotive_baseline/best.pt --config configs/automotive.yaml --chat
  python validate.py --checkpoint checkpoints/food_baseline/latest.pt --config configs/food.yaml --prompt "To make"
  python validate.py --checkpoint checkpoints/automotive_baseline/best.pt --config configs/automotive.yaml --prompt "The engine" --temperature 0.5
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.pt)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file (for model architecture)')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Override tokenizer name (default: from config)')
    parser.add_argument('--chat', action='store_true',
                        help='Interactive chat mode')
    parser.add_argument('--prompt', type=str,
                        help='Single prompt to generate from')
    parser.add_argument('--max-length', type=int, default=100,
                        help='Max generation length (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling (default: 50)')
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model, tokenizer, checkpoint = load_model_from_checkpoint(
        args.checkpoint, args.config, device, args.tokenizer
    )

    # Print model info
    info = model.get_model_info()
    print(f"Model: {info['parameters']:,} params ({info['parameters_millions']:.1f}M)")

    # Print checkpoint metrics if available
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        val_loss = metrics.get('val_loss')
        val_ppl = metrics.get('val_perplexity')
        if val_loss is not None:
            print(f"Checkpoint: epoch={checkpoint.get('epoch', '?')}, "
                  f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")

    # Run mode
    if args.chat:
        chat_mode(model, tokenizer, device, args.max_length, args.temperature, args.top_k)
    elif args.prompt:
        response = generate_response(
            model, tokenizer, args.prompt,
            args.max_length, args.temperature, args.top_k, device
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {response}")
    else:
        parser.print_help()
        print("\n[!] Specify --chat for interactive mode or --prompt 'text' for single generation")


if __name__ == '__main__':
    main()

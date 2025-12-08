#!/usr/bin/env python3
"""
Overfit test: Validate CustomTransformer can memorize 3 short sequences.

This is a sanity check to verify the model can learn. If it can't overfit
3 sequences, something is fundamentally broken.

Usage:
    python overfit.py
"""

import torch
from transformers import GPT2TokenizerFast

from common.models.custom_transfromer.wrapper import CustomTransformerWrapper


# Sequences to memorize (~100 tokens each)
SEQUENCES = [
    "Once upon a time there was a little cat named Whiskers who lived in a cozy cottage by the sea. Every morning, Whiskers would wake up early and watch the sunrise over the ocean waves. The golden light made his fur shimmer beautifully.",

    "The little girl went to the park to play with her friends on a sunny afternoon. They played on the swings and slides until the sun began to set. Her mother called her home for dinner, and she waved goodbye to everyone with a big smile.",

    "One day a boy named Tom found a magic hat in his grandmother's dusty attic. When he put it on, he could suddenly understand what animals were saying. The mice told him secrets, and the birds sang him songs only he could comprehend.",

    "In a faraway kingdom there lived a wise old wizard who knew all the secrets of the universe. People came from distant lands to ask him questions about life and magic. He always answered with riddles that made them think deeply.",

    "The brave knight rode through the dark forest on his white horse seeking the dragon's lair. He carried a sword forged by the greatest blacksmith in the realm. The trees whispered warnings, but he pressed on without fear in his heart.",

    "A curious rabbit named Clover discovered a hidden garden behind the old stone wall. Inside grew flowers of every color imaginable, and butterflies danced in the warm breeze. She visited every day to tend to her secret paradise.",

    "The old lighthouse keeper had watched over the rocky shore for fifty long years without fail. Every night he climbed the spiral stairs to light the beacon that guided ships safely home. Sailors blessed his name in their prayers.",

    "Deep beneath the ocean waves there existed a magnificent city made entirely of coral and pearls. Mermaids and mermen swam through its streets, trading treasures from sunken ships. Music echoed through the underwater kingdom constantly.",

    "The tiny mouse outsmarted the hungry fox by hiding in a hollow log near the stream. She waited patiently until the fox grew tired and wandered away to search elsewhere. Then she scurried home to her family with the cheese.",

    "When winter came to the mountain village, everyone gathered around the fireplace to share stories. Grandmother told tales of ancient heroes and magical creatures that roamed the earth long ago. The children listened with wonder in their eyes.",
]

# Prompts to test generation (first few words of each sequence)
PROMPTS = [
    "Once upon a time there was a little cat",
    "The little girl went to the park",
    "One day a boy named Tom found",
    "In a faraway kingdom there lived",
    "The brave knight rode through the",
    "A curious rabbit named Clover discovered",
    "The old lighthouse keeper had watched",
    "Deep beneath the ocean waves there",
    "The tiny mouse outsmarted the hungry",
    "When winter came to the mountain",
]

# Training config (larger model for longer sequences)
CONFIG = {
    'd_model': 128,
    'n_blocks': 4,
    'n_heads': 4,
    'd_ffn': 256,
    'max_seq_len': 128,
    'learning_rate': 0.01,
    'max_steps': 5000,
    'target_loss': 0.01,
    'log_every': 50,
}


def main():
    print("=" * 50)
    print("OVERFIT TEST")
    print("=" * 50)
    print("\nSequences to memorize:")
    for i, seq in enumerate(SEQUENCES, 1):
        print(f"  {i}. \"{seq}\"")

    # Load tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize sequences (with EOS token at end)
    print("Tokenizing sequences...")
    tokenized = []
    for seq in SEQUENCES:
        tokens = tokenizer.encode(seq, return_tensors='pt')
        # Append EOS token so model learns to stop
        eos = torch.tensor([[tokenizer.eos_token_id]])
        tokens = torch.cat([tokens, eos], dim=1)
        tokenized.append(tokens)
        print(f"  \"{seq[:30]}...\" -> {tokens.shape[1]} tokens (with EOS)")

    # Find max length and pad
    max_len = max(t.shape[1] for t in tokenized)
    padded = []
    for t in tokenized:
        if t.shape[1] < max_len:
            padding = torch.full((1, max_len - t.shape[1]), tokenizer.pad_token_id)
            t = torch.cat([t, padding], dim=1)
        padded.append(t)

    # Stack into batch
    batch = torch.cat(padded, dim=0)  # [3, max_len]
    print(f"\nBatch shape: {batch.shape}")

    # Create model
    print("\nInitializing model...")
    model = CustomTransformerWrapper(
        vocab_size=len(tokenizer),
        max_seq_len=CONFIG['max_seq_len'],
        n_blocks=CONFIG['n_blocks'],
        n_heads=CONFIG['n_heads'],
        d_model=CONFIG['d_model'],
        d_ffn=CONFIG['d_ffn'],
        dtype=torch.float32,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {model.device}")

    # Move batch to device
    batch = batch.to(model.device)
    labels = batch.clone()

    # Training loop
    print(f"\nTraining (lr={CONFIG['learning_rate']}, target_loss={CONFIG['target_loss']})...")
    print("-" * 40)

    for step in range(1, CONFIG['max_steps'] + 1):
        result = model.train_step(
            batch,
            labels,
            learning_rate=CONFIG['learning_rate'],
            max_grad_norm=1.0,
        )

        loss = result['loss']

        if step % CONFIG['log_every'] == 0 or step == 1:
            print(f"Step {step:4d}: loss = {loss:.6f}")

        if loss < CONFIG['target_loss']:
            print(f"\nTarget loss reached at step {step}!")
            break

    if loss >= CONFIG['target_loss']:
        print(f"\nWarning: Did not reach target loss. Final loss: {loss:.6f}")

    # Validation: Generate from prompts
    print("\n" + "=" * 50)
    print("VALIDATION")
    print("=" * 50)

    passed = 0
    failed = 0

    for prompt, expected in zip(PROMPTS, SEQUENCES):
        # Tokenize prompt
        prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

        # Generate with greedy decoding (top_k=1)
        generated_tokens = model.generate(
            prompt_tokens,
            max_length=len(tokenizer.encode(expected)) + 5,  # A bit of slack
            temperature=0.01,  # Near-deterministic
            top_k=1,  # Greedy
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # Check match - exact match or starts with expected (model learned content)
        gen_clean = generated_text.strip()
        exp_clean = expected.strip()

        if gen_clean == exp_clean:
            status = "✓ PASS (exact)"
            passed += 1
        elif gen_clean.startswith(exp_clean):
            status = "✓ PASS (starts with expected, continues after)"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1

        print(f"\nPrompt: \"{prompt}\"")
        print(f"Expected: \"{expected}\"")
        print(f"Generated: \"{generated_text}\"")
        print(f"Status: {status}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}/{len(SEQUENCES)}")
    print(f"Failed: {failed}/{len(SEQUENCES)}")

    if failed == 0:
        print("\n🎉 All sequences memorized successfully!")
        return 0
    else:
        print("\n❌ Some sequences failed to memorize.")
        return 1


if __name__ == "__main__":
    exit(main())

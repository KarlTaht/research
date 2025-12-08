# Encoder-Decoder Transformer

Validation project for the **ReferenceTransformer** - a pure tensor operation implementation of the encoder-decoder Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017).

## Purpose

This project validates the low-level transformer implementation by training on a real sequence-to-sequence task: **dialogue summarization** using the SAMSum dataset.

## Architecture

**ReferenceTransformer** (Encoder-Decoder)
- All components built with pure tensor operations (no `nn.Linear`, `nn.LayerNorm`, etc.)
- Manual embeddings, attention mechanisms, layer normalization, and feed-forward networks
- Encoder processes source (dialogue)
- Decoder generates target (summary) with cross-attention to encoder

See `common/models/reference_transformer/` for implementation details.

## Dataset

**SAMSum** - Samsung Dialogue Summarization Corpus
- **Size**: 16,369 messenger-like conversations
  - Train: 14,732
  - Validation: 818
  - Test: 819
- **Task**: Dialogue → Summary
- **Format**: `{'dialogue': '...', 'summary': '...', 'id': '...'}`
- **Source**: Human-annotated by linguists fluent in English

**Example:**
```
Dialogue:
Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Hannah: Thanks!
Amanda: Sorry, can't find it.

Summary:
Amanda can't find Betty's number for Hannah.
```

## Setup

### 1. Download Dataset

```bash
# From repo root
python tools/download_hf_dataset.py --name samsum
```

### 2. Create Data Symlink

```bash
cd projects/encoder_decoder_transformer
ln -s ../../assets/datasets/samsum data
```

### 3. Activate Environment

```bash
source .venv/bin/activate
```

## Usage

### Training

```bash
# From project directory
python train.py --config config.yaml

# Or from repo root
python -m projects.encoder_decoder_transformer.train
```

### Evaluation

```bash
# Load checkpoint and generate summaries
python evaluate.py --checkpoint ../assets/models/encoder_decoder_transformer_epoch_10.pt

# Evaluate on specific split
python evaluate.py --checkpoint <path> --split test
```

## Configuration

See `config.yaml` for hyperparameters:
- **Model**: 256-dim, 8 heads, 4 encoder/decoder layers
- **Training**: Batch size 16, LR 0.0001, 10 epochs
- **Data**: Max dialogue 256 tokens, max summary 64 tokens

## Implementation Details

### Input Format (Sequence Packing)

Since `BaseLanguageModel` expects single `input_ids`, we pack dialogue and summary:

```python
# Format: [dialogue tokens] [SEP] [summary tokens]
dialogue_ids = tokenizer(dialogue).input_ids
summary_ids = tokenizer(summary).input_ids
input_ids = dialogue_ids + [SEP_TOKEN] + summary_ids
```

The model splits on SEP token internally:
- Encoder processes dialogue
- Decoder generates summary with cross-attention

### Tokenizer

**GPT2TokenizerFast** (50,257 vocab)
- Same as SimpleLSTM project for consistency
- SEP token: `vocab_size - 1` (50256)

### Metrics

- **Perplexity**: Measures model's prediction confidence
- **Loss**: Cross-entropy on summary tokens only
- **Generation quality**: Visual inspection of generated summaries

## Experiment Tracking

Results are automatically saved to `assets/outputs/experiments/`:

```python
# Query experiments
python tools/query_experiments.py --list
python tools/query_experiments.py --best perplexity --top 5
```

## Expected Results

First validation run should achieve:
- **Perplexity**: < 50 after 10 epochs (untrained: ~50K)
- **Loss**: < 4.0
- **Quality**: Basic coherence in generated summaries

## Future Work

See `FUTURE_EXPERIMENTS.md` for planned experiments:
- Testing encoder-decoder on "incorrect" raw text task
- Implementing decoder-only variant
- Comparing architectures on same dataset

## Project Structure

```
projects/encoder_decoder_transformer/
├── README.md                    # This file
├── FUTURE_EXPERIMENTS.md        # Future experiment ideas
├── config.yaml                  # Hyperparameters
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── tests/                       # Test suite
│   ├── test_evaluate.py         # Checkpoint loading tests
│   ├── test_training_pipeline.py # Training pipeline tests
│   └── test_tokenization_encoding.py # Tokenization and encoding tests
└── data -> ../../assets/datasets/samsum  # Dataset symlink
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [SAMSum Dataset](https://huggingface.co/datasets/samsum)
- [SAMSum Paper](https://arxiv.org/abs/1911.12237) (Gliwa et al., 2019)

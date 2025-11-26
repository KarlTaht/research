# Future Experiments

This document outlines experimental ideas to explore beyond the initial validation of the ReferenceTransformer.

---

## Experiment 1: Raw Text Next-Token Prediction with Encoder-Decoder

### Motivation

What happens when we use an encoder-decoder Transformer (designed for paired sequences) on a raw text corpus (single sequences)? This is architecturally "incorrect" but provides valuable learning about:
- Why architecture-dataset alignment matters
- Practical implications of design choices
- Trade-offs between encoder-decoder and decoder-only models

### Hypothesis

Using the encoder-decoder ReferenceTransformer for standard next-token prediction on raw text (tiny-textbooks) will:
1. **Work** (training will succeed, loss will decrease)
2. **Be suboptimal** (learning efficiency and quality will suffer)
3. **Demonstrate** the importance of matching architecture to task

### Setup

**Dataset**: `nampdn-ai/tiny-textbooks`
- 420K synthetic textbook documents
- Single text field (no source-target pairs)
- Educational content

**Preprocessing Strategy**: Arbitrary split
```python
# Take each document and split at midpoint
text = "The quick brown fox jumps over the lazy dog"
midpoint = len(tokens) // 2

# Create fake source-target pair
source = tokens[:midpoint]  # "The quick brown fox"
target = tokens[midpoint:]  # "jumps over the lazy dog"

# Pack for encoder-decoder
input_ids = source + [SEP] + target
```

**Input Format**: `[first_half] [SEP] [second_half]`

### Questions to Explore

1. **Training Efficiency**
   - How much slower is training compared to decoder-only?
   - What's the effective token utilization? (~50% vs 100%)

2. **Attention Patterns**
   - What does the encoder learn from the "fake" source?
   - Does cross-attention find meaningful patterns in arbitrary splits?

3. **Generation Quality**
   - Can it generate coherent text when given arbitrary "source" prompts?
   - How does perplexity compare to decoder-only baseline?

4. **Computational Overhead**
   - What's the FLOPs cost of encoder + cross-attention vs decoder-only?
   - Memory usage comparison?

### Expected Outcomes

**Confirmed Hypotheses:**
- ✓ Training works (loss decreases)
- ✓ Model learns language patterns
- ✓ Can generate text at inference

**Predicted Issues:**
- ❌ **50% token waste**: Only learning from second half (target)
- ❌ **Encoder redundancy**: Bidirectional attention on "fake" source adds no value
- ❌ **Generation awkwardness**: Requires arbitrary "source" at inference
- ❌ **Lower quality**: Decoder-only should significantly outperform

**Quantitative Predictions:**
- Perplexity: 2-3x worse than decoder-only
- Training time: 1.5-2x longer per epoch
- Memory usage: 1.3-1.5x higher

### Baseline Comparison

Train decoder-only GPT-style model on same data:

```python
# Decoder-only: learns from EVERY token
input_ids = [token_0, token_1, token_2, ..., token_n]
labels    = [token_1, token_2, token_3, ..., token_n+1]

# Encoder-decoder: learns from ~50% of tokens
source = [token_0, ..., token_k]
target = [token_k+1, ..., token_n]
```

Compare:
- Perplexity on held-out test set
- Generation quality (human eval)
- Training curves (loss over time)
- Computational efficiency

### Implementation Plan

**Phase 1: Data Preparation**
1. Download tiny-textbooks
2. Implement splitting preprocessor:
   - Tokenize full text
   - Split at 50% mark
   - Handle edge cases (very short docs)
3. Create train/val/test splits

**Phase 2: Training**
1. Use existing ReferenceTransformer (no modifications)
2. Train with same hyperparameters as SAMSum baseline
3. Track metrics: loss, perplexity, training time, memory

**Phase 3: Decoder-Only Baseline**
1. Implement `ReferenceDecoderOnly` (simpler variant)
   - Remove encoder
   - Remove cross-attention
   - Keep causal self-attention + FFN
2. Train on same data (but using full sequences)
3. Compare metrics

**Phase 4: Analysis**
1. Compare learning curves
2. Visualize attention patterns
3. Generate sample text from both models
4. Quantify computational overhead

### Metrics to Track

| Metric | Encoder-Decoder | Decoder-Only | Expected Winner |
|--------|----------------|--------------|-----------------|
| Perplexity | ? | ? | Decoder-Only |
| Training tokens/sec | ? | ? | Decoder-Only |
| GPU memory (GB) | ? | ? | Decoder-Only |
| Parameters | ? | ? | Similar |
| Generation quality | ? | ? | Decoder-Only |

### Educational Value

This experiment demonstrates:

1. **Architecture Matters**: Right tool for the right job
2. **Efficiency**: Token utilization and computational costs
3. **Attention Mechanisms**: What cross-attention learns (or doesn't)
4. **Design Choices**: Why GPT uses decoder-only for language modeling
5. **Empirical Validation**: Test intuitions with real experiments

### Success Criteria

Experiment is successful if we can:
- ✓ Quantify the inefficiency of using encoder-decoder for raw text
- ✓ Demonstrate decoder-only superiority on this task
- ✓ Understand attention patterns in "incorrect" setup
- ✓ Document clear guidelines for architecture selection

### Timeline

- **Data prep**: 1 hour
- **Encoder-decoder training**: 2-4 hours (10 epochs)
- **Decoder-only implementation**: 2 hours
- **Decoder-only training**: 2-4 hours
- **Analysis**: 2-3 hours
- **Total**: 1-2 days

---

## Experiment 2: Decoder-Only Variant Implementation

### Goal

Create `ReferenceDecoderOnly` - a simplified transformer using only the decoder stack for autoregressive language modeling.

### Architecture

```python
class ReferenceDecoderOnly(BaseLanguageModel):
    """GPT-style decoder-only transformer."""

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout):
        # Embedding + positional encoding
        self.embedding = nn.Parameter(...)
        self.pos_encoding = PositionalEncoding(d_model)

        # Decoder stack (no encoder, no cross-attention)
        self.decoder_layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_projection = nn.Parameter(...)

    def forward(self, input_ids, labels=None):
        # Embed
        x = self.embedding[input_ids]
        x = self.pos_encoding(x)

        # Causal mask (prevent attending to future)
        mask = create_causal_mask(x.size(1))

        # Decode
        for layer in self.decoder_layers:
            x = layer(x, mask)  # No encoder output!

        # Project to vocab
        logits = x @ self.output_projection

        # Loss: predict token i+1 from tokens 0:i
        if labels is not None:
            loss = cross_entropy(logits[:, :-1], labels[:, 1:])

        return {'logits': logits, 'loss': loss}
```

**Key Differences from Encoder-Decoder:**
- ❌ No encoder
- ❌ No cross-attention in decoder layers
- ✓ Only causal self-attention
- ✓ Simpler, faster, more memory efficient

### Benefits

- Simpler architecture (easier to understand/debug)
- Better suited for language modeling tasks
- Can use ANY text corpus (no paired data needed)
- Matches GPT/phi/TinyStories models

---

## Experiment 3: Attention Visualization

### Goal

Visualize what the encoder-decoder learns on summarization task vs raw text task.

### Approach

Extract and visualize attention weights:
1. **Encoder self-attention**: What parts of dialogue/source attend to each other?
2. **Decoder self-attention**: How does summary generation attend to previous tokens?
3. **Cross-attention**: Which dialogue parts are most important for each summary token?

**Tools:**
- Matplotlib heatmaps
- Interactive attention visualization
- Compare patterns between SAMSum (correct) and tiny-textbooks (incorrect)

### Expected Insights

**SAMSum (correct usage):**
- Cross-attention should highlight relevant dialogue turns
- Encoder should cluster semantically related utterances

**Tiny-textbooks (incorrect usage):**
- Cross-attention patterns may be random/diffuse
- Encoder attention may not find meaningful structure

---

## Experiment 4: Model Scaling Study

### Goal

Understand how model size affects performance on SAMSum.

### Variants

| Config | d_model | n_layers | n_heads | Parameters | Training Time |
|--------|---------|----------|---------|------------|---------------|
| Tiny   | 128     | 2        | 4       | ~500K      | Fast         |
| Small  | 256     | 4        | 8       | ~2M        | Medium       |
| Medium | 512     | 6        | 8       | ~15M       | Slow         |
| Large  | 768     | 8        | 12      | ~50M       | Very slow    |

### Metrics

- Perplexity vs model size
- Generation quality vs parameters
- Diminishing returns analysis

---

## Experiment 5: Transfer Learning

### Goal

Pre-train on tiny-textbooks (raw text), fine-tune on SAMSum (summarization).

### Hypothesis

Even "incorrect" pre-training on raw text should provide useful language understanding for downstream summarization task.

### Approach

1. Train decoder-only on tiny-textbooks (language modeling)
2. Add encoder + cross-attention (convert to encoder-decoder)
3. Fine-tune on SAMSum
4. Compare to training from scratch

### Expected Result

Pre-training should help, even though architectures differ.

---

## Notes

These experiments are exploratory and educational. They're designed to build intuition about:
- Architecture design choices
- When to use encoder-decoder vs decoder-only
- How models learn from different data structures
- Practical implications of theoretical concepts

Priority: **Experiment 1** (raw text with encoder-decoder) - most educational value.

# CustomTransformer

## Purpose
This is an **educational, low-level implementation** of a decoder-only transformer.

## Design Philosophy
- **Depth of understanding over efficiency** — code prioritizes clarity and learning
- **Pure tensor operations** — minimal use of high-level PyTorch modules
- **Explicit over implicit** — operations are spelled out rather than abstracted

## Guidance for AI Assistants
When providing advice on this codebase:
- Do NOT optimize for performance or production-readiness
- DO explain the "why" behind each operation
- DO use concrete examples with small dimensions
- DO relate new concepts to existing code patterns (e.g., the FCNN)
- Keep explanations concise and conversational

## Current Session Plan

### Phase 1: Bug Fixes ✓
1. ~~**Apply W_o output projection** — currently initialized but unused in attention~~
2. ~~**Cache causal mask** — create once in `__init__` instead of every forward pass~~
3. ~~**Device consistency** — ensure dynamically created tensors match input device~~

### Phase 2: Multi-Block Architecture ✓
- Add `n_blocks` configuration parameter
- Reshape weights to 3D: `[n_blocks, ...]` for Q, K, V, W_o, W1, W2, layer norm params
- Loop over blocks in decoder, passing block index to attention/FFN

### Phase 3: Manual Gradient Descent (Incremental)
Build backprop layer-by-layer, testing each against PyTorch autograd:
1. Output projection (simplest linear layer)
2. FFN (linear + GELU nonlinearity)
3. Layer norm
4. Attention (most complex — softmax, multi-head reshapes)
5. Embeddings

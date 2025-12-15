# Continual Learning

The main goal of this project is to mimic catastropic forgetting that happens with models, while building on fundamentals.

## Smaller Contributions
- New TorchTransformer.py implementation using higher level abstractions than the custom transformer
- Two corpuses of data extracted from a fine-web sample for food and automotive data
- Pre-tokenized 32K vocabs (~98% coverage on the corpora)

## E2E Worklfow

### Dataset Selection
1. Choose root dataset (FineWeb is one option)
2. Build index of dataset (e.g. URLs from fineweb)
3. Determine key corpora to extract, extract them (note, this should be based on planned model size)
4. Make sure the directory hierarchy makes sense :) 

### Tokenization
1. Analyze corpora to determine a good vocab size
2. Train tokenizer (this part is usually fast)
3. (Optional) Pre-tokenize the dataset - useful if you're going to do multiple training rains

### Pre-training Setup and Validation
1. Generate a config with your model config, dataset, and tokenizer
2. Do a smaller validation run first to make sure loss is decreasing as spected
3. Validate things make sense with a checkpoint

### Continual learning forgetting
This part isn't defined yet, but the general idea:

Consider three baselines
- **Sequential A** Train(A) -> Evaluate(A) -> Train(B) -> Evaluate(A,B)
- **Sequential B** Train(B) -> Evaluate(B) -> Train(A) -> Evaluate(A,B)
- **Combined** Train(A,B) -> Evaluate(A,B)

Then continual learning attempts can take a few directions
- **Option 1**: Training Process
  - Data Interleaving
- **Option 2**: Architectural
  - Conceptual Idea: Malleability and breakage. Find a way to track the "geometry" of the model. Look at when the manifold is "bending" too much from new training. Fracture the model apart to partition the knowledge apart and build new shapes. 



## Up next

1. Validation of the model on a larger set
```python validate.py --checkpoint checkpoints/tinystories_torch_validation/best.pt \
    --config configs/tinystories.yaml --chat```
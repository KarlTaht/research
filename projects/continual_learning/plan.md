# Continual Learning Experiment Playbook

A hands-on experiment to build intuition for catastrophic forgetting in transformers, using QA pairs as a richer evaluation signal than perplexity alone.

## Overview

**Goal:** Measure how a small transformer forgets previously learned content when trained on new data, and test interventions to mitigate forgetting.

**Key innovation:** Instead of just tracking perplexity, generate QA pairs from training data to probe whether the model can actually *answer questions* about content it's seen.

**Hardware:** RTX 5060 Ti 16GB (or equivalent)
**Time estimate:** ~1 week to baseline results

---

## Phase 1: Data Preparation (Day 1)

### 1.1 Subset your FineWeb data

You don't need 400GB. Pull out two distinct slices:
- **Corpus A:** ~500MB of text (filter for a specific domain—science articles, history, etc.)
- **Corpus B:** ~500MB of different domain (news, fiction, or different language for sharp distribution shift)

### 1.2 Generate QA pairs from Corpus A

This is your eval signal. For each document (or chunk), generate simple factual questions that can only be answered from that text.

**Options:**
- Use Claude API to generate QA pairs from chunks (expensive but high quality)
- Use a smaller local model (Mistral 7B, Llama 8B) to generate them
- Rule-based extraction: named entities, dates, numbers → cloze-style questions ("The treaty was signed in ___")

**Target:** ~5-10k QA pairs spanning your Corpus A content.

### 1.3 Format your data

```
training_data/
  corpus_a/
    train.jsonl      # {"text": "..."}
    val.jsonl
  corpus_b/
    train.jsonl
    val.jsonl
  eval/
    qa_pairs.jsonl   # {"context": "...", "question": "...", "answer": "..."}
```

---

## Phase 2: Tokenizer (Day 1-2)

### 2.1 Train a BPE tokenizer from scratch

Use `tokenizers` library (HuggingFace). Train on a mix of both corpora so your vocab covers both distributions.

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=16384, 
    special_tokens=["<pad>", "<eos>", "<bos>"]
)
tokenizer.train(files=["corpus_a.txt", "corpus_b.txt"], trainer=trainer)
tokenizer.save("tokenizer.json")
```

16k vocab is fine for this scale. Smaller than GPT-2's 50k but sufficient.

---

## Phase 3: Model Architecture (Day 2)

### 3.1 Define a minimal transformer

Target ~20M parameters:

| Component | Dimension |
|-----------|-----------|
| Layers | 6 |
| Embedding dim | 512 |
| Heads | 8 |
| FFN hidden | 2048 |
| Context length | 512 |
| Vocab size | 16384 |

This gives you roughly 20-25M parameters. Adjust layers or embedding dim to hit your target.

### 3.2 Implementation checklist

- [ ] Token embedding + learned positional embedding
- [ ] Transformer blocks (pre-norm, RMSNorm or LayerNorm)
- [ ] Causal attention mask
- [ ] Output projection (weight-tie with embedding)
- [ ] No dropout for now (simpler, and you're not overfitting at this scale)

---

## Phase 4: Training Infrastructure (Day 2-3)

### 4.1 Data loader

Stream chunks from your jsonl files. Pack sequences to context length (512) with `<eos>` separators. Don't waste compute on padding.

### 4.2 Training loop essentials

```python
optimizer = AdamW(
    model.parameters(), 
    lr=3e-4, 
    betas=(0.9, 0.95), 
    weight_decay=0.1
)
```

- **Warmup:** 500-1000 steps linear warmup
- **Schedule:** Cosine decay to 10% of peak LR
- **Batch size:** Start with 32-64 sequences, increase if VRAM allows
- **Gradient clipping:** 1.0

### 4.3 Logging

Track per-step:
- Training loss
- Validation loss (both corpora)
- Learning rate

Log every 50-100 steps. Use wandb or just write to CSV.

---

## Phase 5: Evaluation Harness (Day 3-4)

### 5.1 Perplexity eval

Simple—run forward pass on held-out data, compute cross-entropy, exponentiate.

```python
def compute_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch["input_ids"])
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                batch["input_ids"][:, 1:].reshape(-1),
                reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += batch["input_ids"][:, 1:].numel()
    return math.exp(total_loss / total_tokens)
```

### 5.2 Knowledge evaluation (beyond memorization)

**Key insight:** A base model won't answer questions in QA format—it's only trained on next-token prediction. Instead, use discriminative evaluation: measure whether the model assigns higher probability to correct completions than incorrect ones.

#### Test types (ordered by difficulty)

| Type | What it tests | Example |
|------|---------------|---------|
| **Direct** | Exact fact recall | "The treaty was signed in" → Paris |
| **Paraphrase** | Same fact, different surface form | "The 1842 treaty was finalized in" → Paris |
| **Inverted** | Fact queried from different angle | "The treaty signed in Paris occurred in" → 1842 |
| **Relational** | Combining multiple facts | "Pierre Curie's wife was born in" → Warsaw |
| **Negation** | Reasoning about contrasts | "The region that does NOT have mild winters is" → northern |
| **Temporal** | Ordering and causation | "Before the acquisition, the company" → went public |

#### Test suite format

```json
{
  "source_text": "Marie Curie was born in Warsaw in 1867. She moved to Paris in 1891 and married Pierre Curie in 1895.",
  "tests": [
    {
      "type": "direct",
      "prompt": "Marie Curie was born in",
      "answer": "Warsaw",
      "distractors": ["Paris", "London", "Berlin"]
    },
    {
      "type": "paraphrase",
      "prompt": "The birthplace of Marie Curie was",
      "answer": "Warsaw",
      "distractors": ["Paris", "London", "Berlin"]
    },
    {
      "type": "inverted",
      "prompt": "The scientist born in Warsaw in 1867 was",
      "answer": "Marie Curie",
      "distractors": ["Pierre Curie", "Albert Einstein", "Isaac Newton"]
    },
    {
      "type": "relational",
      "prompt": "Pierre Curie's wife was born in",
      "answer": "Warsaw",
      "distractors": ["Paris", "London", "Berlin"]
    },
    {
      "type": "temporal",
      "prompt": "Marie Curie moved to Paris after being born in",
      "answer": "Warsaw",
      "distractors": ["Paris", "London", "Berlin"]
    },
    {
      "type": "temporal_order",
      "prompt": "Before marrying Pierre Curie, Marie",
      "answer": "moved to Paris",
      "distractors": ["was born", "won Nobel Prize", "died"]
    }
  ]
}
```

#### Discriminative evaluation code

```python
def score_completion(model, tokenizer, prompt: str, completion: str) -> float:
    """Return negative log likelihood of completion given prompt."""
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion)
    full_ids = torch.tensor([prompt_ids + completion_ids])
    
    with torch.no_grad():
        logits = model(full_ids)
    
    # Only score the completion tokens
    completion_logits = logits[0, len(prompt_ids)-1:-1]
    completion_targets = torch.tensor(completion_ids)
    
    loss = F.cross_entropy(completion_logits, completion_targets, reduction='mean')
    return loss.item()


def eval_test(model, tokenizer, test: dict) -> dict:
    """Evaluate a single test. Returns scores and whether model got it right."""
    answer_score = score_completion(model, tokenizer, test["prompt"], test["answer"])
    
    distractor_scores = [
        score_completion(model, tokenizer, test["prompt"], d)
        for d in test["distractors"]
    ]
    
    # Model "passes" if correct answer has lowest NLL (highest probability)
    correct = answer_score < min(distractor_scores)
    
    # Margin: how much better is the correct answer?
    margin = min(distractor_scores) - answer_score
    
    return {
        "correct": correct,
        "answer_nll": answer_score,
        "best_distractor_nll": min(distractor_scores),
        "margin": margin
    }


def eval_test_suite(model, tokenizer, test_suite: list[dict]) -> dict:
    """Evaluate full test suite, breaking down by test type."""
    results_by_type = defaultdict(list)
    
    for item in test_suite:
        for test in item["tests"]:
            result = eval_test(model, tokenizer, test)
            result["type"] = test["type"]
            results_by_type[test["type"]].append(result)
    
    summary = {}
    for test_type, results in results_by_type.items():
        summary[test_type] = {
            "accuracy": sum(r["correct"] for r in results) / len(results),
            "avg_margin": sum(r["margin"] for r in results) / len(results),
            "n": len(results)
        }
    
    summary["overall"] = {
        "accuracy": sum(r["correct"] for rs in results_by_type.values() for r in rs) / 
                   sum(len(rs) for rs in results_by_type.values()),
        "n": sum(len(rs) for rs in results_by_type.values())
    }
    
    return summary
```

### 5.3 Generating test suites with Claude Haiku

Use Claude Haiku (cheap, fast) to generate rich test suites from your training passages.

#### Setup

```python
import anthropic
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

def generate_test_suite(passage: str, max_retries: int = 3) -> dict | None:
    """Generate a test suite for a single passage using Claude Haiku."""
    
    prompt = f"""Analyze this passage and generate a test suite to evaluate whether a language model has learned the information—not just memorized exact phrases.

<passage>
{passage}
</passage>

Generate tests in these categories:

1. **direct**: Complete a sentence with a fact stated explicitly
2. **paraphrase**: Same fact, but rephrase the prompt (different words, same meaning)  
3. **inverted**: Query the fact from a different angle (e.g., if passage says "A is B", test "B is" → A)
4. **relational**: Requires combining 2+ facts from the passage
5. **temporal**: Tests ordering of events or causal relationships (if applicable)
6. **negation**: Tests understanding of contrasts or what's NOT true (if applicable)

For each test:
- The prompt should be an incomplete sentence that the model completes
- The answer should be 1-4 words
- Include 3 plausible distractors (wrong answers that would be reasonable guesses)
- Distractors should be the same "type" as the answer (if answer is a city, distractors are cities)

Generate 6-10 tests total, prioritizing variety of test types over quantity.

Respond with JSON only, no other text:
{{
  "tests": [
    {{
      "type": "direct|paraphrase|inverted|relational|temporal|negation",
      "prompt": "The incomplete sentence to complete",
      "answer": "correct completion",
      "distractors": ["wrong1", "wrong2", "wrong3"],
      "reasoning": "Brief note on what this tests"
    }}
  ]
}}"""

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-20250414",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON from response
            content = response.content[0].text
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            result["source_text"] = passage
            return result
            
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            if attempt == max_retries - 1:
                print(f"Failed to parse response for passage: {passage[:50]}...")
                return None
            continue
    
    return None


def generate_test_suites_batch(passages: list[str], max_workers: int = 10) -> list[dict]:
    """Generate test suites for multiple passages in parallel."""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_test_suite, p) for p in passages]
        
        for future in tqdm(futures, desc="Generating test suites"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    return results
```

#### Running the generation pipeline

```python
import json
from pathlib import Path

# Load your corpus A passages
# Chunk them to ~200-500 tokens each for good test generation
passages = load_and_chunk_corpus("training_data/corpus_a/train.jsonl", chunk_size=400)

# Sample if you have too many (5-10k tests is plenty)
if len(passages) > 2000:
    passages = random.sample(passages, 2000)

# Generate test suites (will take a while, ~$5-10 for 2000 passages with Haiku)
print(f"Generating test suites for {len(passages)} passages...")
test_suites = generate_test_suites_batch(passages, max_workers=10)

# Save
output_path = Path("training_data/eval/test_suites.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    for suite in test_suites:
        f.write(json.dumps(suite) + "\n")

print(f"Generated {len(test_suites)} test suites")
print(f"Total tests: {sum(len(s['tests']) for s in test_suites)}")

# Print breakdown by type
from collections import Counter
type_counts = Counter(t["type"] for s in test_suites for t in s["tests"])
print("\nTests by type:")
for t, count in type_counts.most_common():
    print(f"  {t}: {count}")
```

#### Cost estimate

- Claude Haiku: ~$0.25 per million input tokens, ~$1.25 per million output tokens
- Average passage: ~400 tokens input
- Average response: ~600 tokens output
- Per passage: ~$0.0001 input + ~$0.00075 output ≈ $0.00085
- **2000 passages ≈ $1.70**

Very cheap. You can afford to be generous.

#### Quality control

After generation, run a quick sanity check:

```python
def validate_test_suite(suite: dict) -> list[str]:
    """Return list of issues with a test suite."""
    issues = []
    
    for test in suite.get("tests", []):
        # Check required fields
        for field in ["type", "prompt", "answer", "distractors"]:
            if field not in test:
                issues.append(f"Missing field: {field}")
        
        # Check distractor count
        if len(test.get("distractors", [])) != 3:
            issues.append(f"Expected 3 distractors, got {len(test.get('distractors', []))}")
        
        # Check answer isn't in distractors
        if test.get("answer") in test.get("distractors", []):
            issues.append(f"Answer '{test['answer']}' is also a distractor")
        
        # Check answer length
        if len(test.get("answer", "").split()) > 5:
            issues.append(f"Answer too long: '{test['answer']}'")
    
    return issues

# Validate all
issues_by_suite = [(i, validate_test_suite(s)) for i, s in enumerate(test_suites)]
problematic = [(i, issues) for i, issues in issues_by_suite if issues]
print(f"{len(problematic)} suites have issues")
```

### 5.4 Metrics to track

| Metric | Description |
|--------|-------------|
| `ppl_A` | Perplexity on Corpus A validation |
| `ppl_B` | Perplexity on Corpus B validation |
| `accuracy_direct` | % correct on direct fact completion |
| `accuracy_paraphrase` | % correct on paraphrased prompts |
| `accuracy_inverted` | % correct on inverted queries |
| `accuracy_relational` | % correct on relational (multi-fact) queries |
| `accuracy_temporal` | % correct on temporal ordering |
| `accuracy_negation` | % correct on negation/contrast |
| `accuracy_overall` | Overall accuracy across all test types |
| `avg_margin` | Average probability margin (correct vs best distractor) |

#### Forgetting analysis by test type

The most interesting output will be plotting accuracy by test type over training steps on Corpus B:

```python
import matplotlib.pyplot as plt

def plot_forgetting_by_type(eval_history: list[dict], save_path: str):
    """
    eval_history: list of {"step": int, "metrics": dict} from periodic evaluation
    """
    steps = [e["step"] for e in eval_history]
    test_types = ["direct", "paraphrase", "inverted", "relational", "temporal", "negation"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for test_type in test_types:
        accuracies = [e["metrics"].get(f"accuracy_{test_type}", {}).get("accuracy", None) 
                      for e in eval_history]
        # Filter None values
        valid = [(s, a) for s, a in zip(steps, accuracies) if a is not None]
        if valid:
            ax.plot([v[0] for v in valid], [v[1] for v in valid], label=test_type, marker='o')
    
    ax.set_xlabel("Training steps on Corpus B")
    ax.set_ylabel("Accuracy")
    ax.set_title("Knowledge Retention by Test Type")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

**Hypothesis to test:** Relational and temporal tests will degrade faster than direct tests—the model loses the ability to flexibly apply knowledge before it loses the raw associations.

---

## Phase 6: Run the Experiment (Day 4)

### 6.1 Baseline training

1. Train on Corpus A until validation loss plateaus (~2-3 hours)
2. Save checkpoint: `model_a_baseline.pt`
3. Record all metrics:
   - `ppl_A_baseline`
   - `qa_accuracy_baseline`
   - `qa_answer_ppl_baseline`

### 6.2 Measure forgetting

1. Load `model_a_baseline.pt`
2. Continue training on Corpus B until validation loss plateaus
3. Save checkpoint: `model_a_then_b.pt`
4. Re-evaluate all metrics on Corpus A:
   - `ppl_A_after`
   - `qa_accuracy_after`
   - `qa_answer_ppl_after`

### 6.3 Track the forgetting curve

**Critical:** Don't just measure endpoints. Every N steps on Corpus B, evaluate:
- `ppl_A`
- `qa_accuracy`

Plot these over training steps. You want to see:
- How fast does the model forget?
- Is it gradual or a cliff?
- Does QA accuracy degrade faster/slower than perplexity?

---

## Phase 7: Interventions (Day 5+)

Now you have a baseline forgetting curve. Try things:

### 7.1 Replay

Mix in X% of Corpus A batches while training on B.

```python
def get_batch():
    if random.random() < replay_ratio:
        return next(corpus_a_loader)
    else:
        return next(corpus_b_loader)
```

Try 1%, 5%, 10%. Plot forgetting curves for each.

**Question to answer:** How much replay do you need to prevent forgetting?

### 7.2 Learning rate

Does lower LR on Corpus B reduce forgetting?

Try training on B with:
- 3e-4 (same as A)
- 1e-4
- 3e-5

**Trade-off:** Slower learning on B vs. better retention of A.

### 7.3 Freeze layers

Freeze early layers, only fine-tune later layers on B.

```python
for i, layer in enumerate(model.layers):
    if i < freeze_until:
        for param in layer.parameters():
            param.requires_grad = False
```

Try freezing first 2, 3, 4 layers. Does this preserve A knowledge?

### 7.4 EWC (Elastic Weight Consolidation)

More involved but worth trying:

1. After training on A, compute Fisher information for each parameter
2. During B training, add penalty for deviating from A weights, scaled by Fisher

```python
# After training on A
fisher = {}
for name, param in model.named_parameters():
    fisher[name] = param.grad.data.clone().pow(2)

# During B training
ewc_loss = 0
for name, param in model.named_parameters():
    ewc_loss += (fisher[name] * (param - params_a[name]).pow(2)).sum()
    
total_loss = ce_loss + ewc_lambda * ewc_loss
```

---

## What Success Looks Like

By the end of this experiment, you should have:

1. **Intuition** for how fast forgetting happens in small transformers
2. **Quantitative curves** showing the phenomenon (both perplexity and QA-based)
3. **Baseline results** for naive replay and other simple interventions
4. **A working experimental setup** you can extend

### Questions you'll be able to answer:

- How many steps on new data before the model "forgets" old content?
- Does perplexity or QA accuracy degrade first?
- How much replay is needed to maintain X% of original performance?
- Do different types of knowledge (factual QA vs. general fluency) forget at different rates?

---

## Future Directions

Once you have this working, natural next questions:

1. **Selective replay:** What if you only replay examples the model is starting to forget?
2. **Learned memory:** What if the model could decide what to write to a separate memory module?
3. **Retrieval-augmented continual learning:** Instead of replaying raw data, retrieve compressed representations
4. **Different forgetting for different knowledge types:** Do facts forget faster than skills?

---

## Time Estimate

| Phase | Time |
|-------|------|
| Data prep (subset corpora) | 2-3 hours |
| Test suite generation (Claude Haiku) | 2-3 hours (mostly waiting) |
| Tokenizer | 1 hour |
| Model implementation | 3-4 hours |
| Training loop | 2-3 hours |
| Eval harness | 3-4 hours |
| Baseline experiment | 3-4 hours |
| Intervention experiments | Ongoing |

**Total to baseline results:** ~20-24 hours of focused work (roughly one weekend + evenings)

**Cost:** ~$2-5 for Claude Haiku API calls for test generation

---

## Appendix A: Chunking Passages for Test Generation

```python
def chunk_corpus(input_path: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Chunk corpus into passages suitable for test generation.
    
    Args:
        input_path: Path to jsonl file with {"text": "..."} entries
        chunk_size: Target tokens per chunk (approximate, using whitespace)
        overlap: Token overlap between chunks for context continuity
    """
    chunks = []
    
    with open(input_path) as f:
        for line in f:
            doc = json.loads(line)
            text = doc["text"]
            words = text.split()
            
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk = " ".join(words[start:end])
                
                # Only keep chunks with enough content
                if len(chunk.split()) >= chunk_size * 0.5:
                    chunks.append(chunk)
                
                start = end - overlap
    
    return chunks


def filter_chunks_for_facts(chunks: list[str], min_entities: int = 2) -> list[str]:
    """
    Filter to chunks likely to contain testable facts.
    Simple heuristic: must contain numbers, dates, or capitalized words (entities).
    """
    import re
    
    def has_facts(chunk: str) -> bool:
        # Contains numbers (years, quantities, etc.)
        has_numbers = bool(re.search(r'\b\d{4}\b|\b\d+\.\d+\b|\b\d+%', chunk))
        
        # Contains likely named entities (capitalized words not at sentence start)
        words = chunk.split()
        capitalized = sum(1 for i, w in enumerate(words) 
                         if w[0].isupper() and i > 0 and words[i-1][-1] not in '.!?')
        has_entities = capitalized >= min_entities
        
        return has_numbers or has_entities
    
    return [c for c in chunks if has_facts(c)]
```

## Appendix B: Alternative Test Generation (Rule-Based)

If you want to avoid API costs entirely, here's a simpler rule-based approach for direct/cloze tests:

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_cloze_tests(passage: str) -> list[dict]:
    """Extract simple cloze-style tests using NER."""
    doc = nlp(passage)
    tests = []
    
    for sent in doc.sents:
        for ent in sent.ents:
            # Create cloze by blanking the entity
            prompt = sent.text[:ent.start_char - sent.start_char]
            answer = ent.text
            
            # Skip if prompt is too short
            if len(prompt.split()) < 3:
                continue
            
            # Generate distractors of same entity type
            distractors = get_distractors_for_type(ent.label_, answer)
            
            tests.append({
                "type": "direct",
                "prompt": prompt.strip(),
                "answer": answer,
                "distractors": distractors
            })
    
    return tests


def get_distractors_for_type(ent_type: str, answer: str) -> list[str]:
    """Return plausible distractors based on entity type."""
    distractor_pools = {
        "DATE": ["1847", "1923", "1965", "2001", "1776", "1889"],
        "GPE": ["London", "Paris", "Berlin", "Tokyo", "Rome", "Madrid"],  # places
        "PERSON": ["John Smith", "Marie Laurent", "Hans Weber", "James Wilson"],
        "ORG": ["Microsoft", "Harvard", "United Nations", "Reuters"],
        "CARDINAL": ["three", "seven", "twelve", "hundred", "thousand"],
        "MONEY": ["$500", "$1 million", "€200", "£50,000"],
        "PERCENT": ["15%", "42%", "8%", "73%"],
    }
    
    pool = distractor_pools.get(ent_type, ["unknown", "other", "none"])
    # Remove answer if it's in pool
    pool = [d for d in pool if d.lower() != answer.lower()]
    
    return random.sample(pool, min(3, len(pool)))
```

This won't give you relational or temporal tests, but it's free and fast. You could use this for a first pass and then augment with Claude Haiku for the harder test types.

## Appendix C: Full Evaluation Loop

```python
def run_full_evaluation(
    model, 
    tokenizer, 
    corpus_a_val_path: str,
    corpus_b_val_path: str,
    test_suites_path: str
) -> dict:
    """Run complete evaluation suite and return all metrics."""
    
    model.eval()
    metrics = {}
    
    # Perplexity on both corpora
    print("Computing perplexity...")
    metrics["ppl_A"] = compute_perplexity(
        model, 
        create_dataloader(corpus_a_val_path, tokenizer, batch_size=32)
    )
    metrics["ppl_B"] = compute_perplexity(
        model,
        create_dataloader(corpus_b_val_path, tokenizer, batch_size=32)
    )
    
    # Knowledge tests
    print("Running knowledge tests...")
    test_suites = load_test_suites(test_suites_path)
    test_results = eval_test_suite(model, tokenizer, test_suites)
    
    # Flatten test results into metrics
    for test_type, results in test_results.items():
        if test_type == "overall":
            metrics["accuracy_overall"] = results["accuracy"]
        else:
            metrics[f"accuracy_{test_type}"] = results["accuracy"]
            metrics[f"margin_{test_type}"] = results["avg_margin"]
    
    return metrics


def training_loop_with_eval(
    model,
    tokenizer,
    train_loader,
    optimizer,
    scheduler,
    eval_every: int = 500,
    eval_config: dict = None,
) -> list[dict]:
    """Training loop that periodically evaluates and logs forgetting."""
    
    eval_history = []
    step = 0
    
    for batch in train_loader:
        # Training step
        model.train()
        optimizer.zero_grad()
        
        logits = model(batch["input_ids"])
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            batch["input_ids"][:, 1:].reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        step += 1
        
        # Periodic evaluation
        if step % eval_every == 0:
            print(f"\n=== Evaluation at step {step} ===")
            metrics = run_full_evaluation(
                model, tokenizer,
                eval_config["corpus_a_val"],
                eval_config["corpus_b_val"],
                eval_config["test_suites"]
            )
            metrics["step"] = step
            metrics["train_loss"] = loss.item()
            
            eval_history.append(metrics)
            
            # Print summary
            print(f"  ppl_A: {metrics['ppl_A']:.2f}")
            print(f"  ppl_B: {metrics['ppl_B']:.2f}")
            print(f"  accuracy_overall: {metrics['accuracy_overall']:.3f}")
            print(f"  accuracy_relational: {metrics.get('accuracy_relational', 'N/A')}")
    
    return eval_history
```

"""Advanced evaluation framework with qualitative metrics and LLM coherence scoring."""

import os
import re
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

# Anthropic SDK for Claude Haiku
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class AdvancedEvaluator:
    """
    Extended evaluator with qualitative text analysis and LLM-based coherence scoring.

    Features:
    - Standard loss and perplexity evaluation
    - Text generation sampling
    - Simple heuristics (sentence length, vocab diversity, repetition rate)
    - Claude Haiku coherence evaluation (optional, requires ANTHROPIC_API_KEY)

    Example:
        evaluator = AdvancedEvaluator(
            model=model,
            tokenizer=tokenizer,
            device='cuda',
        )

        # Full evaluation
        results = evaluator.full_evaluation(
            val_dataloader,
            generation_prompts=['Once upon a time', 'The little girl'],
            evaluate_coherence=True,
        )

        print(f"Perplexity: {results['perplexity']:.2f}")
        print(f"Vocab diversity: {results['heuristic_vocab_diversity']:.3f}")
        print(f"Coherence score: {results.get('claude_coherence_score_avg', 'N/A')}")
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        anthropic_api_key: Optional[str] = None,
    ):
        """
        Initialize AdvancedEvaluator.

        Args:
            model: Language model (CustomTransformerWrapper or BaseLanguageModel)
            tokenizer: Tokenizer (e.g., GPT2TokenizerFast)
            device: Device for computation
            anthropic_api_key: API key for Claude Haiku (uses ANTHROPIC_API_KEY env var if not provided)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Set up Anthropic client if available
        self.anthropic_client = None
        if HAS_ANTHROPIC:
            api_key = anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                print("Anthropic client initialized for coherence evaluation")
            else:
                print("No ANTHROPIC_API_KEY found - coherence evaluation disabled")
        else:
            print("anthropic package not installed - coherence evaluation disabled")

    @torch.no_grad()
    def evaluate_perplexity(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute loss and perplexity on dataset.

        Args:
            dataloader: DataLoader yielding {'input_ids', 'labels'} dicts
            max_batches: Maximum batches to evaluate (None = all)

        Returns:
            Dict with 'loss', 'perplexity', 'num_tokens', 'num_batches'
        """
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)

            outputs = self.model.forward(input_ids, labels=labels)

            # Get loss (handle both tensor and dict returns)
            if isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                loss = outputs

            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(min(avg_loss, 20))  # Cap to avoid overflow

        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'num_tokens': total_tokens,
            'num_batches': num_batches,
        }

    @torch.no_grad()
    def generate_samples(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        num_samples_per_prompt: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Generate text samples from prompts.

        Args:
            prompts: List of prompt strings
            max_length: Maximum generation length (including prompt)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = no filtering)
            num_samples_per_prompt: Number of samples per prompt

        Returns:
            List of dicts with 'prompt', 'generated', 'continuation' keys
        """
        samples = []

        for prompt in prompts:
            for _ in range(num_samples_per_prompt):
                # Tokenize prompt
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

                # Generate
                if hasattr(self.model, 'generate'):
                    generated_ids = self.model.generate(
                        input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k if top_k > 0 else None,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    # Fallback for models without generate method
                    generated_ids = self._simple_generate(
                        input_ids, max_length, temperature, top_k
                    )

                # Decode
                generated_text = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                )

                samples.append({
                    'prompt': prompt,
                    'generated': generated_text,
                    'continuation': generated_text[len(prompt):].strip(),
                })

        return samples

    def _simple_generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """Simple autoregressive generation for models without generate()."""
        generated = input_ids

        for _ in range(max_length - input_ids.size(1)):
            outputs = self.model.forward(generated)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            next_token_logits = logits[:, -1, :] / max(temperature, 1e-7)

            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs.float(), num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if self.tokenizer.eos_token_id and (next_token == self.tokenizer.eos_token_id).all():
                break

        return generated

    def compute_text_heuristics(self, samples: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Compute simple heuristics on generated text.

        Metrics:
        - avg_sentence_length: Average words per sentence
        - vocab_diversity: Unique words / total words (type-token ratio)
        - avg_word_length: Average characters per word
        - repetition_rate: Fraction of repeated bigrams (sign of degeneration)

        Args:
            samples: List of generation samples from generate_samples()

        Returns:
            Dict of heuristic metrics
        """
        # Combine all continuations
        all_text = ' '.join([s['continuation'] for s in samples if s['continuation']])

        if not all_text.strip():
            return {
                'avg_sentence_length': 0.0,
                'vocab_diversity': 0.0,
                'avg_word_length': 0.0,
                'repetition_rate': 1.0,
            }

        # Tokenize into words
        words = re.findall(r'\b\w+\b', all_text.lower())

        if not words:
            return {
                'avg_sentence_length': 0.0,
                'vocab_diversity': 0.0,
                'avg_word_length': 0.0,
                'repetition_rate': 1.0,
            }

        # Split into sentences
        sentences = re.split(r'[.!?]+', all_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Sentence length
        if sentences:
            sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
            avg_sentence_length = float(np.mean(sentence_lengths))
        else:
            avg_sentence_length = len(words)  # Treat as one sentence

        # Vocabulary diversity (type-token ratio)
        unique_words = set(words)
        vocab_diversity = len(unique_words) / len(words)

        # Average word length
        avg_word_length = float(np.mean([len(w) for w in words]))

        # Repetition rate (repeated bigrams)
        if len(words) > 1:
            bigrams = list(zip(words[:-1], words[1:]))
            bigram_counts = Counter(bigrams)
            repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
            repetition_rate = repeated_bigrams / len(bigram_counts) if bigram_counts else 0.0
        else:
            repetition_rate = 0.0

        return {
            'avg_sentence_length': avg_sentence_length,
            'vocab_diversity': vocab_diversity,
            'avg_word_length': avg_word_length,
            'repetition_rate': repetition_rate,
        }

    def evaluate_coherence_with_claude(
        self,
        samples: List[Dict[str, str]],
        max_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Use Claude Haiku to evaluate text coherence.

        Requires ANTHROPIC_API_KEY environment variable or constructor arg.

        Args:
            samples: List of generation samples
            max_samples: Maximum samples to evaluate (API cost control)

        Returns:
            Dict with:
                - coherence_scores: List of 1-5 scores per sample
                - coherence_score_avg: Average coherence score
                - issues: Summary of common issues
                - raw_response: Full Claude response
                - error: Error message if evaluation failed
        """
        if not self.anthropic_client:
            return {
                'coherence_scores': [],
                'coherence_score_avg': None,
                'issues': 'Anthropic client not available',
                'error': 'Missing API key or anthropic package',
            }

        # Filter to samples with non-empty continuations
        valid_samples = [s for s in samples if s['continuation'].strip()]
        samples_to_evaluate = valid_samples[:max_samples]

        if not samples_to_evaluate:
            return {
                'coherence_scores': [],
                'coherence_score_avg': None,
                'issues': 'No valid samples to evaluate',
            }

        # Build prompt for Claude
        prompt = """You are evaluating text generated by a language model being trained on children's stories (TinyStories dataset).

For each generated text sample below, rate its coherence on a scale of 1-5:
1 = Completely incoherent, random words with no meaning
2 = Some recognizable phrases but no clear narrative or meaning
3 = Partially coherent, attempts at story structure but inconsistent or confused
4 = Mostly coherent, minor issues with grammar, logic, or flow
5 = Fully coherent, reads like a natural children's story

Also briefly note any specific issues you observe (repetition, topic drift, grammatical errors, etc).

Generated samples:
"""
        for i, sample in enumerate(samples_to_evaluate, 1):
            continuation = sample['continuation'][:500]  # Limit length
            prompt += f"\n--- Sample {i} ---\n"
            prompt += f"Prompt: {sample['prompt']}\n"
            prompt += f"Generated: {continuation}{'...' if len(sample['continuation']) > 500 else ''}\n"

        prompt += """

Please respond in this exact format:
SCORES: [score1, score2, ...]
AVERAGE: [average score to 1 decimal]
ISSUES: [brief summary of common issues across samples]
"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Parse response
            scores = []
            avg_score = None
            issues = ""

            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('SCORES:'):
                    # Extract numbers from the line
                    numbers = re.findall(r'\d+', line)
                    scores = [int(n) for n in numbers if 1 <= int(n) <= 5]
                elif line.startswith('AVERAGE:'):
                    numbers = re.findall(r'[\d.]+', line)
                    if numbers:
                        avg_score = float(numbers[0])
                elif line.startswith('ISSUES:'):
                    issues = line.replace('ISSUES:', '').strip()

            # Calculate average if not parsed
            if avg_score is None and scores:
                avg_score = sum(scores) / len(scores)

            return {
                'coherence_scores': scores,
                'coherence_score_avg': avg_score,
                'issues': issues,
                'raw_response': response_text,
            }

        except Exception as e:
            return {
                'coherence_scores': [],
                'coherence_score_avg': None,
                'issues': '',
                'error': str(e),
            }

    def full_evaluation(
        self,
        val_dataloader: DataLoader,
        generation_prompts: Optional[List[str]] = None,
        max_eval_batches: int = 100,
        max_generation_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        evaluate_coherence: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full evaluation pipeline.

        Combines:
        1. Perplexity evaluation on validation data
        2. Text generation with sample prompts
        3. Heuristic analysis of generated text
        4. Claude Haiku coherence evaluation (optional)

        Args:
            val_dataloader: Validation data loader
            generation_prompts: Prompts for text generation (defaults provided)
            max_eval_batches: Max batches for perplexity eval
            max_generation_length: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            evaluate_coherence: Whether to use Claude Haiku for coherence

        Returns:
            Dict with all evaluation metrics
        """
        results = {}

        # 1. Perplexity evaluation
        print("Evaluating perplexity...")
        perplexity_metrics = self.evaluate_perplexity(val_dataloader, max_eval_batches)
        results.update(perplexity_metrics)

        # 2. Generate samples
        if generation_prompts is None:
            generation_prompts = [
                "Once upon a time",
                "The little girl",
                "One day, a boy named",
                "In a small village",
            ]

        print("Generating samples...")
        samples = self.generate_samples(
            generation_prompts,
            max_length=max_generation_length,
            temperature=temperature,
            top_k=top_k,
        )
        results['generated_samples'] = samples

        # 3. Text heuristics
        print("Computing heuristics...")
        heuristics = self.compute_text_heuristics(samples)
        results.update({f'heuristic_{k}': v for k, v in heuristics.items()})

        # 4. Claude coherence (optional)
        if evaluate_coherence and self.anthropic_client:
            print("Evaluating coherence with Claude Haiku...")
            coherence = self.evaluate_coherence_with_claude(samples)
            results.update({f'claude_{k}': v for k, v in coherence.items()})
        elif evaluate_coherence:
            results['claude_error'] = 'Anthropic client not available'

        return results

    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of evaluation results."""
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)

        print(f"\nPerplexity Metrics:")
        print(f"  Loss:       {results.get('loss', 'N/A'):.4f}")
        print(f"  Perplexity: {results.get('perplexity', 'N/A'):.2f}")
        print(f"  Tokens:     {results.get('num_tokens', 'N/A'):,}")

        print(f"\nText Heuristics:")
        print(f"  Avg sentence length: {results.get('heuristic_avg_sentence_length', 'N/A'):.1f} words")
        print(f"  Vocab diversity:     {results.get('heuristic_vocab_diversity', 'N/A'):.3f}")
        print(f"  Avg word length:     {results.get('heuristic_avg_word_length', 'N/A'):.1f} chars")
        print(f"  Repetition rate:     {results.get('heuristic_repetition_rate', 'N/A'):.3f}")

        if 'claude_coherence_score_avg' in results:
            print(f"\nClaude Coherence:")
            print(f"  Average score: {results['claude_coherence_score_avg']:.1f}/5")
            print(f"  Scores:        {results.get('claude_coherence_scores', [])}")
            if results.get('claude_issues'):
                print(f"  Issues:        {results['claude_issues']}")

        if results.get('generated_samples'):
            print(f"\nSample Generations:")
            for i, sample in enumerate(results['generated_samples'][:2], 1):
                print(f"  [{i}] Prompt: \"{sample['prompt']}\"")
                continuation = sample['continuation'][:100]
                print(f"      Generated: \"{continuation}{'...' if len(sample['continuation']) > 100 else ''}\"")

        print("=" * 50)

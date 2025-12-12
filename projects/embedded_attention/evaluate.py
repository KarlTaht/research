"""Evaluation with LongMemEval benchmark."""

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from common.utils import save_experiment


@dataclass
class EvalMetrics:
    """Evaluation metrics for retrieval and generation."""

    accuracy: float  # Fraction of correct answers
    precision_at_k: float  # Retrieved chunks that were relevant
    recall: float  # Relevant chunks that were retrieved
    mrr: float  # Mean reciprocal rank
    latency_ms: float  # Average retrieval latency


def load_longmemeval(split: str = "test"):
    """Load LongMemEval dataset from HuggingFace.

    Args:
        split: Dataset split to load ('train', 'validation', 'test')

    Returns:
        HuggingFace dataset
    """
    from datasets import load_dataset

    return load_dataset("xiaowu0162/LongMemEval", split=split)


def check_answer(response: str, expected: str, threshold: float = 0.8) -> bool:
    """Check if response contains the expected answer.

    Uses fuzzy matching to handle slight variations in phrasing.

    Args:
        response: Generated response
        expected: Expected answer
        threshold: Minimum similarity for match

    Returns:
        True if answer is considered correct
    """
    # Simple containment check for now
    response_lower = response.lower()
    expected_lower = expected.lower()

    # Exact containment
    if expected_lower in response_lower:
        return True

    # Word overlap check
    expected_words = set(expected_lower.split())
    response_words = set(response_lower.split())

    if len(expected_words) == 0:
        return True

    overlap = len(expected_words & response_words) / len(expected_words)
    return overlap >= threshold


def compute_mrr(results: list[dict]) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        results: List of result dicts with 'rank_of_relevant' key

    Returns:
        MRR score
    """
    if not results:
        return 0.0

    reciprocal_ranks = []
    for r in results:
        rank = r.get("rank_of_relevant", 0)
        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_on_longmemeval(
    config: dict,
    split: str = "test",
    max_examples: Optional[int] = None,
    verbose: bool = False,
) -> EvalMetrics:
    """Run evaluation on LongMemEval benchmark.

    Args:
        config: Configuration dict with retrieval parameters
        split: Dataset split to evaluate on
        max_examples: Maximum examples to evaluate (None for all)
        verbose: Print progress

    Returns:
        EvalMetrics with aggregate scores
    """
    from .core.builder import ConversationBuilder, ConversationConfig
    from .core.generator import DummyGenerator

    dataset = load_longmemeval(split)
    results = []

    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    # Create a reusable generator
    generator = DummyGenerator()

    for i, example in enumerate(dataset):
        if verbose and i % 10 == 0:
            print(f"Evaluating example {i}/{len(dataset)}")

        # Create fresh conversation for each example using builder
        conv_config = ConversationConfig(
            db_path=":memory:",
            embedding_model=config.get("embedding_model", "BAAI/bge-small-en-v1.5"),
            semantic_top_k=config.get("semantic_top_k", 5),
            min_similarity=config.get("min_similarity", 0.7),
        )

        conv = (
            ConversationBuilder(conv_config)
            .with_generator(generator)
            .build()
        )

        # Load conversation history
        history = example.get("history", example.get("conversation", []))
        for turn in history:
            role = turn.get("role", turn.get("speaker", "user"))
            content = turn.get("content", turn.get("text", ""))
            conv.add_turn(content, role)

        # Perform retrieval
        question = example.get("question", example.get("query", ""))
        start = time.perf_counter()
        retrieved = conv.retriever.retrieve(
            query=question,
            conversation_id=conv.conversation_id,
        )
        latency = (time.perf_counter() - start) * 1000

        # Evaluate retrieval quality (retrieved is now list[ScoredChunk])
        retrieved_ids = {sc.chunk.id for sc in retrieved}
        relevant_ids = set(example.get("relevant_turn_ids", []))

        # Calculate precision and recall
        if retrieved_ids and relevant_ids:
            precision = len(retrieved_ids & relevant_ids) / len(retrieved_ids)
            recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
        elif not relevant_ids:
            precision = 1.0 if not retrieved_ids else 0.0
            recall = 1.0
        else:
            precision = 0.0
            recall = 0.0

        # Generate response and check correctness
        response, _ = conv.chat_with_retrieval_info(question)
        expected = example.get("answer", example.get("expected", ""))
        correct = check_answer(response, expected) if expected else True

        results.append(
            {
                "task_type": example.get("task_type", "unknown"),
                "correct": correct,
                "precision": precision,
                "recall": recall,
                "latency_ms": latency,
                "num_retrieved": len(retrieved),
                "num_relevant": len(relevant_ids),
            }
        )

        conv.store.close()

    df = pd.DataFrame(results)

    return EvalMetrics(
        accuracy=df["correct"].mean(),
        precision_at_k=df["precision"].mean(),
        recall=df["recall"].mean(),
        mrr=compute_mrr(results),
        latency_ms=df["latency_ms"].mean(),
    )


def analyze_by_task_type(results_df: pd.DataFrame) -> pd.DataFrame:
    """Break down performance by LongMemEval task type.

    Args:
        results_df: DataFrame with evaluation results

    Returns:
        DataFrame with per-task-type metrics
    """
    task_types = [
        "information_extraction",
        "multi_session_reasoning",
        "temporal_reasoning",
        "knowledge_update",
        "abstention",
    ]

    task_metrics = []
    for task in task_types:
        subset = results_df[results_df["task_type"] == task]
        if len(subset) > 0:
            task_metrics.append(
                {
                    "task_type": task,
                    "count": len(subset),
                    "accuracy": subset["correct"].mean(),
                    "precision": subset["precision"].mean(),
                    "recall": subset["recall"].mean(),
                    "latency_ms": subset["latency_ms"].mean(),
                }
            )

    return pd.DataFrame(task_metrics)


def run_evaluation(
    configs: Optional[list[dict]] = None,
    output_dir: str = "assets/outputs/embedded_attention/eval",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run full evaluation suite with multiple configurations.

    Args:
        configs: List of configuration dicts to evaluate
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        DataFrame with all results
    """
    if configs is None:
        configs = [
            {"semantic_top_k": 3, "min_similarity": 0.6},
            {"semantic_top_k": 5, "min_similarity": 0.7},
            {"semantic_top_k": 10, "min_similarity": 0.5},
        ]

    all_results = []

    for i, config in enumerate(configs):
        if verbose:
            print(f"\nEvaluating config {i + 1}/{len(configs)}: {config}")

        metrics = evaluate_on_longmemeval(config, verbose=verbose)

        all_results.append(
            {
                "config": str(config),
                "semantic_top_k": config.get("semantic_top_k", 5),
                "min_similarity": config.get("min_similarity", 0.7),
                "accuracy": metrics.accuracy,
                "precision": metrics.precision_at_k,
                "recall": metrics.recall,
                "mrr": metrics.mrr,
                "latency_ms": metrics.latency_ms,
            }
        )

    df = pd.DataFrame(all_results)

    # Save results
    save_experiment(
        "embedded_attention_longmemeval",
        df,
        metadata={
            "embedding_model": "bge-small-en-v1.5",
            "benchmark": "LongMemEval",
        },
    )

    if verbose:
        print("\n=== Evaluation Results ===")
        print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate on LongMemEval")
    parser.add_argument(
        "--max-examples", type=int, default=None, help="Max examples to evaluate"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Semantic search top-k"
    )
    parser.add_argument(
        "--min-sim", type=float, default=0.7, help="Minimum similarity threshold"
    )
    args = parser.parse_args()

    config = {"semantic_top_k": args.top_k, "min_similarity": args.min_sim}

    metrics = evaluate_on_longmemeval(
        config, max_examples=args.max_examples, verbose=True
    )

    print("\n=== Results ===")
    print(f"Accuracy: {metrics.accuracy:.2%}")
    print(f"Precision@k: {metrics.precision_at_k:.2%}")
    print(f"Recall: {metrics.recall:.2%}")
    print(f"MRR: {metrics.mrr:.3f}")
    print(f"Latency: {metrics.latency_ms:.1f}ms")

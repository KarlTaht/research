"""Token analyzer for understanding dataset vocabulary and recommending tokenizer sizes.

This module analyzes token distributions in datasets to help design efficient tokenizers
for small-scale experiments. Key features:
- CDF analysis: How many unique tokens needed to cover X% of the dataset?
- Edge case detection: Special characters, hyphens, numbers, etc.
- Tokenizer size recommendations for different coverage targets

Example:
    from common.data import TokenAnalyzer
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    analyzer = TokenAnalyzer(tokenizer)

    # Analyze a dataset
    report = analyzer.analyze_dataset('tinystories', subset_size=10000)
    print(report.summary())

    # Get recommendations
    recommendations = analyzer.recommend_vocab_size(target_coverage=0.99)
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
import re
import numpy as np
from datasets import Dataset

from .dataset_registry import load_dataset_from_registry, get_dataset_config, DATASET_REGISTRY
from .download_hf_dataset import get_models_dir


@dataclass
class EdgeCaseStats:
    """Statistics about edge case tokens in the dataset."""

    # Token counts by category
    single_char_tokens: Dict[int, int] = field(default_factory=dict)  # token_id -> count
    punctuation_tokens: Dict[int, int] = field(default_factory=dict)
    number_tokens: Dict[int, int] = field(default_factory=dict)
    hyphenated_tokens: Dict[int, int] = field(default_factory=dict)
    whitespace_tokens: Dict[int, int] = field(default_factory=dict)
    rare_tokens: Dict[int, int] = field(default_factory=dict)  # below threshold

    # Aggregate stats
    total_single_char_occurrences: int = 0
    total_punctuation_occurrences: int = 0
    total_number_occurrences: int = 0
    total_hyphenated_occurrences: int = 0
    total_whitespace_occurrences: int = 0
    total_rare_occurrences: int = 0

    def summary(self) -> str:
        """Return a formatted summary of edge case statistics."""
        lines = [
            "Edge Case Statistics:",
            f"  Single character tokens: {len(self.single_char_tokens):,} unique, {self.total_single_char_occurrences:,} occurrences",
            f"  Punctuation tokens: {len(self.punctuation_tokens):,} unique, {self.total_punctuation_occurrences:,} occurrences",
            f"  Number tokens: {len(self.number_tokens):,} unique, {self.total_number_occurrences:,} occurrences",
            f"  Hyphenated tokens: {len(self.hyphenated_tokens):,} unique, {self.total_hyphenated_occurrences:,} occurrences",
            f"  Whitespace tokens: {len(self.whitespace_tokens):,} unique, {self.total_whitespace_occurrences:,} occurrences",
            f"  Rare tokens: {len(self.rare_tokens):,} unique, {self.total_rare_occurrences:,} occurrences",
        ]
        return "\n".join(lines)

@dataclass
class CoveragePoint:
    """A point on the coverage CDF curve."""
    num_tokens: int  # Number of unique tokens
    coverage: float  # Fraction of total tokens covered (0-1)
    cumulative_count: int  # Cumulative token count


@dataclass
class TokenAnalysisReport:
    """Complete analysis report for a dataset."""

    dataset_name: str
    tokenizer_name: str

    # Basic stats
    total_tokens: int  # Total token occurrences in dataset
    unique_tokens: int  # Number of unique token IDs used
    vocab_size: int  # Tokenizer vocabulary size
    vocab_utilization: float  # unique_tokens / vocab_size

    # Frequency distribution
    token_frequencies: Counter  # token_id -> count

    # CDF data
    coverage_cdf: List[CoveragePoint]

    # Edge cases
    edge_cases: EdgeCaseStats

    # Coverage milestones
    tokens_for_90_coverage: int = 0
    tokens_for_95_coverage: int = 0
    tokens_for_99_coverage: int = 0
    tokens_for_999_coverage: int = 0

    def summary(self) -> str:
        """Return a formatted summary of the analysis."""
        lines = [
            f"=" * 60,
            f"Token Analysis Report: {self.dataset_name}",
            f"=" * 60,
            f"",
            f"Tokenizer: {self.tokenizer_name}",
            f"Tokenizer vocab size: {self.vocab_size:,}",
            f"",
            f"Dataset Statistics:",
            f"  Total tokens: {self.total_tokens:,}",
            f"  Unique tokens used: {self.unique_tokens:,}",
            f"  Vocabulary utilization: {self.vocab_utilization:.2%}",
            f"",
            f"Coverage Milestones (tokens needed for X% coverage):",
            f"  90% coverage: {self.tokens_for_90_coverage:,} tokens",
            f"  95% coverage: {self.tokens_for_95_coverage:,} tokens",
            f"  99% coverage: {self.tokens_for_99_coverage:,} tokens",
            f"  99.9% coverage: {self.tokens_for_999_coverage:,} tokens",
            f"",
            self.edge_cases.summary(),
            f"",
            f"Top 20 Most Frequent Tokens:",
        ]

        for token_id, count in self.token_frequencies.most_common(20):
            pct = count / self.total_tokens * 100
            lines.append(f"  {token_id:6d}: {count:8,} ({pct:5.2f}%)")

        return "\n".join(lines)

    def get_coverage_at_vocab_size(self, vocab_size: int) -> float:
        """Get coverage percentage for a given vocabulary size."""
        for point in self.coverage_cdf:
            if point.num_tokens >= vocab_size:
                return point.coverage
        return 1.0  # Full coverage if vocab_size exceeds all points

    def get_vocab_size_for_coverage(self, target_coverage: float) -> int:
        """Get minimum vocabulary size needed for target coverage."""
        for point in self.coverage_cdf:
            if point.coverage >= target_coverage:
                return point.num_tokens
        return self.unique_tokens  # Need all tokens

@dataclass
class TokenizerRecommendation:
    """Recommendation for tokenizer configuration."""

    target_coverage: float
    recommended_vocab_size: int
    actual_coverage: float
    tokens_saved: int  # Compared to full GPT-2 vocab
    embedding_reduction: float  # Percentage reduction in embedding params

    def summary(self) -> str:
        """Return formatted recommendation."""
        return (
            f"Target: {self.target_coverage:.1%} coverage\n"
            f"  Recommended vocab size: {self.recommended_vocab_size:,}\n"
            f"  Actual coverage: {self.actual_coverage:.2%}\n"
            f"  Tokens saved vs GPT-2: {self.tokens_saved:,}\n"
            f"  Embedding param reduction: {self.embedding_reduction:.1%}"
        )

class TokenAnalyzer:
    """Analyzes token distributions in datasets for tokenizer optimization."""

    def __init__(self, tokenizer, rare_threshold: int = 10):
        """
        Initialize the analyzer.

        Args:
            tokenizer: HuggingFace tokenizer (e.g., GPT2TokenizerFast)
            rare_threshold: Tokens with count below this are considered "rare"
        """
        self.tokenizer = tokenizer
        self.rare_threshold = rare_threshold
        self.vocab_size = len(tokenizer)

        # Cache for decoded tokens (for edge case detection)
        self._decode_cache: Dict[int, str] = {}

    def _decode_token(self, token_id: int) -> str:
        """Decode a token ID to its string representation with caching."""
        if token_id not in self._decode_cache:
            try:
                self._decode_cache[token_id] = self.tokenizer.decode([token_id])
            except Exception:
                self._decode_cache[token_id] = ""
        return self._decode_cache[token_id]

    def _classify_token(self, token_id: int, token_str: str) -> List[str]:
        """Classify a token into edge case categories."""
        categories = []

        # Single character
        if len(token_str) == 1:
            categories.append('single_char')

        # Whitespace (including space-prefixed tokens like ' the')
        if token_str.isspace() or (token_str.startswith(' ') and len(token_str) == 1):
            categories.append('whitespace')

        # Punctuation-heavy (more than half punctuation)
        stripped = token_str.strip()
        if stripped:
            punct_count = sum(1 for c in stripped if not c.isalnum() and not c.isspace())
            if punct_count > len(stripped) / 2:
                categories.append('punctuation')

        # Contains hyphen
        if '-' in token_str or '\u2013' in token_str or '\u2014' in token_str:
            categories.append('hyphenated')

        # Number tokens (contains digits)
        if any(c.isdigit() for c in token_str):
            categories.append('number')

        return categories

    def analyze_tokens(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> Tuple[Counter, int]:
        """
        Tokenize texts and count token frequencies.

        Args:
            texts: List of text strings to analyze
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (token_frequencies Counter, total_tokens int)
        """
        token_counts = Counter()
        total_tokens = 0

        # Process in batches for efficiency
        batch_size = 1000
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Tokenize batch
            encoded = self.tokenizer(
                batch,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )

            # Count tokens
            for input_ids in encoded['input_ids']:
                token_counts.update(input_ids)
                total_tokens += len(input_ids)

            if show_progress and (i // batch_size) % 10 == 0:
                print(f"  Processed {min(i + batch_size, len(texts)):,}/{len(texts):,} texts...")

        return token_counts, total_tokens

    def compute_coverage_cdf(
        self,
        token_frequencies: Counter,
        total_tokens: int,
    ) -> List[CoveragePoint]:
        """
        Compute the cumulative distribution function of token coverage.

        The CDF shows what fraction of the dataset is covered by the N most
        frequent tokens.

        Args:
            token_frequencies: Counter of token_id -> count
            total_tokens: Total number of token occurrences

        Returns:
            List of CoveragePoint sorted by num_tokens
        """
        # Sort tokens by frequency (descending)
        sorted_tokens = token_frequencies.most_common()

        cdf_points = []
        cumulative = 0

        for i, (token_id, count) in enumerate(sorted_tokens, 1):
            cumulative += count
            coverage = cumulative / total_tokens

            # Sample points to avoid huge lists
            # Always include: first 100, then every 10th, then every 100th, then every 1000th
            if (i <= 100 or
                i % 10 == 0 and i <= 1000 or
                i % 100 == 0 and i <= 10000 or
                i % 1000 == 0 or
                i == len(sorted_tokens)):
                cdf_points.append(CoveragePoint(
                    num_tokens=i,
                    coverage=coverage,
                    cumulative_count=cumulative,
                ))

        return cdf_points

    def detect_edge_cases(
        self,
        token_frequencies: Counter,
    ) -> EdgeCaseStats:
        """
        Detect and categorize edge case tokens.

        Categories:
        - Single character tokens
        - Punctuation-heavy tokens
        - Number tokens (contain digits)
        - Hyphenated tokens
        - Whitespace tokens
        - Rare tokens (below threshold)

        Args:
            token_frequencies: Counter of token_id -> count

        Returns:
            EdgeCaseStats with categorized tokens
        """
        stats = EdgeCaseStats()

        for token_id, count in token_frequencies.items():
            token_str = self._decode_token(token_id)
            categories = self._classify_token(token_id, token_str)

            if 'single_char' in categories:
                stats.single_char_tokens[token_id] = count
                stats.total_single_char_occurrences += count

            if 'punctuation' in categories:
                stats.punctuation_tokens[token_id] = count
                stats.total_punctuation_occurrences += count

            if 'number' in categories:
                stats.number_tokens[token_id] = count
                stats.total_number_occurrences += count

            if 'hyphenated' in categories:
                stats.hyphenated_tokens[token_id] = count
                stats.total_hyphenated_occurrences += count

            if 'whitespace' in categories:
                stats.whitespace_tokens[token_id] = count
                stats.total_whitespace_occurrences += count

            if count < self.rare_threshold:
                stats.rare_tokens[token_id] = count
                stats.total_rare_occurrences += count

        return stats

    def _find_coverage_milestone(
        self,
        cdf_points: List[CoveragePoint],
        target: float,
    ) -> int:
        """Find number of tokens needed for target coverage."""
        for point in cdf_points:
            if point.coverage >= target:
                return point.num_tokens
        return cdf_points[-1].num_tokens if cdf_points else 0

    def analyze_dataset(
        self,
        dataset_name: str,
        subset_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> TokenAnalysisReport:
        """
        Analyze token distribution in a registered dataset.

        Args:
            dataset_name: Name of dataset in registry
            subset_size: If set, only analyze this many examples
            show_progress: Whether to show progress

        Returns:
            TokenAnalysisReport with full analysis
        """
        if show_progress:
            print(f"Loading dataset: {dataset_name}")

        # Load dataset
        train_dataset, _, text_column, _ = load_dataset_from_registry(dataset_name)

        # Subset if requested
        if subset_size is not None and subset_size < len(train_dataset):
            train_dataset = train_dataset.select(range(subset_size))

        if show_progress:
            print(f"Analyzing {len(train_dataset):,} examples...")

        # Get texts
        texts = train_dataset[text_column]

        # Analyze tokens
        token_frequencies, total_tokens = self.analyze_tokens(texts, show_progress)

        # Compute CDF
        if show_progress:
            print("Computing coverage CDF...")
        cdf_points = self.compute_coverage_cdf(token_frequencies, total_tokens)

        # Detect edge cases
        if show_progress:
            print("Detecting edge cases...")
        edge_cases = self.detect_edge_cases(token_frequencies)

        # Build report
        report = TokenAnalysisReport(
            dataset_name=dataset_name,
            tokenizer_name=getattr(self.tokenizer, 'name_or_path', 'unknown'),
            total_tokens=total_tokens,
            unique_tokens=len(token_frequencies),
            vocab_size=self.vocab_size,
            vocab_utilization=len(token_frequencies) / self.vocab_size,
            token_frequencies=token_frequencies,
            coverage_cdf=cdf_points,
            edge_cases=edge_cases,
            tokens_for_90_coverage=self._find_coverage_milestone(cdf_points, 0.90),
            tokens_for_95_coverage=self._find_coverage_milestone(cdf_points, 0.95),
            tokens_for_99_coverage=self._find_coverage_milestone(cdf_points, 0.99),
            tokens_for_999_coverage=self._find_coverage_milestone(cdf_points, 0.999),
        )

        if show_progress:
            print("Analysis complete!")

        return report

    def analyze_texts(
        self,
        texts: List[str],
        name: str = "custom",
        show_progress: bool = True,
    ) -> TokenAnalysisReport:
        """
        Analyze token distribution in arbitrary texts.

        Args:
            texts: List of text strings to analyze
            name: Name for the analysis (for reporting)
            show_progress: Whether to show progress

        Returns:
            TokenAnalysisReport with full analysis
        """
        if show_progress:
            print(f"Analyzing {len(texts):,} texts...")

        # Analyze tokens
        token_frequencies, total_tokens = self.analyze_tokens(texts, show_progress)

        # Compute CDF
        cdf_points = self.compute_coverage_cdf(token_frequencies, total_tokens)

        # Detect edge cases
        edge_cases = self.detect_edge_cases(token_frequencies)

        # Build report
        report = TokenAnalysisReport(
            dataset_name=name,
            tokenizer_name=getattr(self.tokenizer, 'name_or_path', 'unknown'),
            total_tokens=total_tokens,
            unique_tokens=len(token_frequencies),
            vocab_size=self.vocab_size,
            vocab_utilization=len(token_frequencies) / self.vocab_size,
            token_frequencies=token_frequencies,
            coverage_cdf=cdf_points,
            edge_cases=edge_cases,
            tokens_for_90_coverage=self._find_coverage_milestone(cdf_points, 0.90),
            tokens_for_95_coverage=self._find_coverage_milestone(cdf_points, 0.95),
            tokens_for_99_coverage=self._find_coverage_milestone(cdf_points, 0.99),
            tokens_for_999_coverage=self._find_coverage_milestone(cdf_points, 0.999),
        )

        return report

    def recommend_vocab_size(
        self,
        report: TokenAnalysisReport,
        target_coverages: List[float] = [0.90, 0.95, 0.99, 0.999],
        base_vocab_size: int = 50257,  # GPT-2 vocab size
    ) -> List[TokenizerRecommendation]:
        """
        Generate tokenizer size recommendations for different coverage targets.

        Args:
            report: TokenAnalysisReport from analysis
            target_coverages: Coverage levels to recommend for
            base_vocab_size: Baseline vocab size for comparison (e.g., GPT-2)

        Returns:
            List of TokenizerRecommendation for each coverage target
        """
        recommendations = []

        for target in target_coverages:
            vocab_needed = report.get_vocab_size_for_coverage(target)
            actual_coverage = report.get_coverage_at_vocab_size(vocab_needed)

            tokens_saved = base_vocab_size - vocab_needed
            embedding_reduction = tokens_saved / base_vocab_size

            recommendations.append(TokenizerRecommendation(
                target_coverage=target,
                recommended_vocab_size=vocab_needed,
                actual_coverage=actual_coverage,
                tokens_saved=tokens_saved,
                embedding_reduction=embedding_reduction,
            ))

        return recommendations

    def compare_datasets(
        self,
        reports: List[TokenAnalysisReport],
    ) -> str:
        """
        Compare token distributions across multiple datasets.

        Args:
            reports: List of TokenAnalysisReport from different datasets

        Returns:
            Formatted comparison string
        """
        lines = [
            "=" * 80,
            "Dataset Comparison",
            "=" * 80,
            "",
        ]

        # Header
        header = f"{'Dataset':<25} {'Total Tokens':>15} {'Unique':>10} {'Util%':>8} {'95%Cov':>8} {'99%Cov':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for report in reports:
            lines.append(
                f"{report.dataset_name:<25} "
                f"{report.total_tokens:>15,} "
                f"{report.unique_tokens:>10,} "
                f"{report.vocab_utilization:>7.1%} "
                f"{report.tokens_for_95_coverage:>8,} "
                f"{report.tokens_for_99_coverage:>8,}"
            )

        lines.append("")
        lines.append("Note: 95%Cov and 99%Cov show tokens needed for that coverage level")

        return "\n".join(lines)

    def export_cdf_data(
        self,
        report: TokenAnalysisReport,
        output_path: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Export CDF data for plotting.

        Args:
            report: TokenAnalysisReport to export
            output_path: If provided, save to this path as CSV

        Returns:
            CSV string if output_path is None, else None
        """
        lines = ["num_tokens,coverage,cumulative_count"]

        for point in report.coverage_cdf:
            lines.append(f"{point.num_tokens},{point.coverage:.6f},{point.cumulative_count}")

        csv_content = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(csv_content)
            return None

        return csv_content

    def get_essential_tokens(
        self,
        report: TokenAnalysisReport,
        coverage: float = 0.99,
    ) -> List[Tuple[int, str, int]]:
        """
        Get the most essential tokens for a given coverage level.

        Args:
            report: TokenAnalysisReport from analysis
            coverage: Target coverage (0-1)

        Returns:
            List of (token_id, token_string, count) tuples
        """
        vocab_needed = report.get_vocab_size_for_coverage(coverage)

        essential = []
        for token_id, count in report.token_frequencies.most_common(vocab_needed):
            token_str = self._decode_token(token_id)
            essential.append((token_id, token_str, count))

        return essential








def analyze_and_recommend(
    dataset_name: str,
    subset_size: Optional[int] = None,
    target_coverage: float = 0.99,
    show_progress: bool = True,
) -> Tuple[TokenAnalysisReport, List[TokenizerRecommendation]]:
    """
    Convenience function to analyze a dataset and get tokenizer recommendations.

    Args:
        dataset_name: Name of dataset in registry
        subset_size: If set, only analyze this many examples
        target_coverage: Primary coverage target for recommendations
        show_progress: Whether to show progress

    Returns:
        Tuple of (TokenAnalysisReport, List[TokenizerRecommendation])

    Example:
        from common.data import analyze_and_recommend

        report, recommendations = analyze_and_recommend('tinystories', subset_size=10000)
        print(report.summary())
        for rec in recommendations:
            print(rec.summary())
    """
    from transformers import GPT2TokenizerFast

    # Use GPT-2 tokenizer as baseline
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    analyzer = TokenAnalyzer(tokenizer)
    report = analyzer.analyze_dataset(dataset_name, subset_size, show_progress)

    # Generate recommendations including the target coverage
    coverages = [0.90, 0.95, 0.99, 0.999]
    if target_coverage not in coverages:
        coverages.append(target_coverage)
        coverages.sort()

    recommendations = analyzer.recommend_vocab_size(report, coverages)

    return report, recommendations

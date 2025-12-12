"""Context assembly with token budget management.

Delegates chunk selection to RelevanceScorer for consistent scoring.
Handles formatting of selected chunks into LLM-ready messages.
"""

from dataclasses import dataclass
from typing import Optional

from .chunk import Chunk, ScoredChunk
from .scoring import RelevanceScorer


@dataclass
class ContextConfig:
    """Configuration for context assembly.

    Attributes:
        total_budget: Total context window size in tokens.
        reserved_for_generation: Tokens reserved for model output.
        reserved_for_current: Tokens reserved for current user message.
    """

    total_budget: int = 8192
    reserved_for_generation: int = 1024
    reserved_for_current: int = 1000

    @property
    def retrieval_budget(self) -> int:
        """Token budget available for retrieved chunks."""
        return self.total_budget - self.reserved_for_generation - self.reserved_for_current


class ContextAssembler:
    """Assembles retrieved chunks into a context for the LLM.

    Delegates chunk selection to the shared RelevanceScorer to ensure
    consistent scoring behavior with the Retriever. Handles formatting
    into the HuggingFace messages format.

    Example:
        >>> assembler = ContextAssembler(scorer)
        >>> messages = assembler.select_and_format(scored_chunks, "Hello!", now)
        >>> response = generator.generate(messages)
    """

    def __init__(self, scorer: RelevanceScorer, config: Optional[ContextConfig] = None):
        """Initialize the context assembler.

        Args:
            scorer: RelevanceScorer for chunk selection (shared with Retriever).
            config: Context configuration (uses defaults if not provided).
        """
        self.scorer = scorer
        self.config = config or ContextConfig()

    def select_and_format(
        self,
        scored_chunks: list[ScoredChunk],
        current_message: str,
        reference_time: float,
    ) -> list[dict]:
        """Select chunks within budget and format as messages.

        Args:
            scored_chunks: Pre-scored chunks from Retriever.
            current_message: The current user message.
            reference_time: Unix timestamp for age formatting.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        # Delegate selection to scorer
        selected = self.scorer.select_by_budget(
            scored_chunks, self.config.retrieval_budget
        )

        return self._format_messages(selected, current_message, reference_time)

    def _format_messages(
        self,
        selected: list[ScoredChunk],
        current_message: str,
        reference_time: float,
    ) -> list[dict]:
        """Format selected chunks as HuggingFace messages.

        Args:
            selected: Chunks selected within budget.
            current_message: The current user message.
            reference_time: Unix timestamp for age formatting.

        Returns:
            List of message dicts.
        """
        messages = []

        if selected:
            context_parts = []
            for sc in selected:
                age = self._format_age(reference_time - sc.chunk.timestamp)
                header = f"[{sc.chunk.role.upper()} - {age}]"
                context_parts.append(f"{header}\n{sc.chunk.content}")

            messages.append({
                "role": "system",
                "content": "Relevant conversation history:\n\n" + "\n\n".join(context_parts),
            })

        messages.append({"role": "user", "content": current_message})
        return messages

    def format_context_string(
        self,
        scored_chunks: list[ScoredChunk],
        current_message: str,
        reference_time: float,
    ) -> str:
        """Format chunks into a plain string context.

        Alternative format for models that don't use message-based input.

        Args:
            scored_chunks: Pre-scored chunks from Retriever.
            current_message: The current user message.
            reference_time: Unix timestamp for age formatting.

        Returns:
            Formatted context string.
        """
        selected = self.scorer.select_by_budget(
            scored_chunks, self.config.retrieval_budget
        )

        parts = []

        if selected:
            parts.append("[RETRIEVED CONTEXT]")
            for sc in selected:
                age = self._format_age(reference_time - sc.chunk.timestamp)
                parts.append(f"[{sc.chunk.role.upper()} - {age}]")
                parts.append(sc.chunk.content)
                parts.append("")
            parts.append("[END RETRIEVED CONTEXT]\n")

        parts.append("[CURRENT MESSAGE]")
        parts.append(current_message)

        return "\n".join(parts)

    def _format_age(self, seconds: float) -> str:
        """Format age in human-readable form.

        Args:
            seconds: Age in seconds.

        Returns:
            Human-readable age string.
        """
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds / 3600)}h ago"
        else:
            return f"{int(seconds / 86400)}d ago"

    def estimate_context_tokens(
        self,
        scored_chunks: list[ScoredChunk],
        current_message: str,
        reference_time: float,
        tokenizer,
    ) -> dict:
        """Estimate token usage for context.

        Args:
            scored_chunks: Pre-scored chunks.
            current_message: Current user message.
            reference_time: Unix timestamp.
            tokenizer: Tokenizer with encode() method.

        Returns:
            Dict with token counts for each component.
        """
        selected = self.scorer.select_by_budget(
            scored_chunks, self.config.retrieval_budget
        )

        messages = self._format_messages(selected, current_message, reference_time)

        system_tokens = 0
        user_tokens = 0

        for msg in messages:
            tokens = len(tokenizer.encode(msg["content"]))
            if msg["role"] == "system":
                system_tokens += tokens
            elif msg["role"] == "user":
                user_tokens += tokens

        return {
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
            "total_tokens": system_tokens + user_tokens,
            "budget_remaining": self.config.retrieval_budget - system_tokens - user_tokens,
            "num_chunks_selected": len(selected),
        }

"""Thin orchestration layer for memory-augmented conversation.

The Conversation class coordinates between components but contains no
business logic. All scoring is done by RelevanceScorer, all splitting
by TextChunker, etc.

For creating conversations, use ConversationBuilder from builder.py.
"""

import json
import time
from typing import Optional
from uuid import uuid4

from .chunk import Chunk, ScoredChunk
from .chunk_store import ChunkStore
from .chunking import TextChunker
from .context_assembler import ContextAssembler
from .embedder import Embedder
from .generator import Generator
from .retriever import Retriever


class Conversation:
    """Thin orchestration layer for memory-augmented conversation.

    Coordinates the flow between components:
    1. Chunker splits content into segments
    2. Embedder encodes segments
    3. Store persists chunks
    4. Retriever finds relevant context
    5. Assembler formats context
    6. Generator produces responses

    All business logic lives in the individual components.
    Use ConversationBuilder to create configured instances.

    Example:
        >>> from embedded_attention.builder import create_conversation
        >>> conv = create_conversation()
        >>> response = conv.chat("Hello!")
    """

    def __init__(
        self,
        store: ChunkStore,
        chunker: TextChunker,
        embedder: Embedder,
        retriever: Retriever,
        assembler: ContextAssembler,
        generator: Generator,
        conversation_id: Optional[str] = None,
        conversation_group_id: Optional[str] = None,
    ):
        """Initialize a conversation.

        Args:
            store: ChunkStore for persisting and querying chunks.
            chunker: TextChunker for splitting long content.
            embedder: Embedder for encoding text.
            retriever: Retriever for hybrid retrieval strategy.
            assembler: ContextAssembler for formatting context.
            generator: Generator for producing responses.
            conversation_id: Optional ID (auto-generated if None).
            conversation_group_id: Optional group ID for cross-conversation retrieval.
        """
        self.store = store
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.assembler = assembler
        self.generator = generator

        self.conversation_id = conversation_id or str(uuid4())
        self.conversation_group_id = conversation_group_id
        self._last_response = ""

    def chat(self, user_message: str) -> str:
        """Process a user message and generate a response.

        This is the main entry point for conversation turns:
        1. Store the user message
        2. Retrieve relevant context
        3. Format context and generate response
        4. Store the response

        Args:
            user_message: The user's input text.

        Returns:
            Generated assistant response.
        """
        now = time.time()

        # 1. Store user turn
        self._store_turn(user_message, "user", now)

        # 2. Retrieve relevant context
        scored_chunks = self.retriever.retrieve(
            query=user_message,
            conversation_id=self.conversation_id,
            conversation_group_id=self.conversation_group_id,
            recent_context=self._last_response,
            reference_time=now,
        )

        # 3. Format context and generate
        messages = self.assembler.select_and_format(scored_chunks, user_message, now)
        response = self.generator.generate(messages)

        # 4. Store response
        self._store_turn(response, "assistant", now)
        self._last_response = response

        return response

    def chat_with_retrieval_info(self, user_message: str) -> tuple[str, dict]:
        """Chat with detailed retrieval information returned.

        Useful for debugging and analysis of retrieval behavior.

        Args:
            user_message: The user's input text.

        Returns:
            Tuple of (response, retrieval_info dict).
        """
        now = time.time()

        # Store user message
        self._store_turn(user_message, "user", now)

        # Retrieve with detailed breakdown
        retrieval_info = self.retriever.retrieve_with_breakdown(
            query=user_message,
            conversation_id=self.conversation_id,
            conversation_group_id=self.conversation_group_id,
            recent_context=self._last_response,
            reference_time=now,
        )

        # Format context and generate
        messages = self.assembler.select_and_format(
            retrieval_info["merged"], user_message, now
        )
        response = self.generator.generate(messages)

        # Store response
        self._store_turn(response, "assistant", now)
        self._last_response = response

        return response, {
            "retrieval": {
                "semantic": [(sc.chunk.id, sc.score) for sc in retrieval_info["semantic"]],
                "recent": [(sc.chunk.id, sc.score) for sc in retrieval_info["recent"]],
                "linked": [(sc.chunk.id, sc.score) for sc in retrieval_info["linked"]],
                "merged": [(sc.chunk.id, sc.score) for sc in retrieval_info["merged"]],
            },
        }

    def add_turn(self, content: str, role: str) -> list[Chunk]:
        """Add a turn to the conversation without generating a response.

        Useful for loading conversation history.

        Args:
            content: The text content of the turn.
            role: The role ('user', 'assistant', 'tool_call', 'tool_result').

        Returns:
            List of Chunk objects created.
        """
        return self._store_turn(content, role, time.time())

    def _store_turn(self, content: str, role: str, timestamp: float) -> list[Chunk]:
        """Split content into chunks and store.

        Args:
            content: Text to store.
            role: Role of the speaker.
            timestamp: Unix timestamp.

        Returns:
            List of created chunks.
        """
        segments = self.chunker.split(content)
        chunks = []

        for seg in segments:
            chunk = Chunk(
                role=role,
                content=seg["content"],
                embedding=self.embedder.embed(seg["content"]),
                timestamp=timestamp,
                token_count=self.chunker.count_tokens(seg["content"]),
                conversation_id=self.conversation_id,
                conversation_group_id=self.conversation_group_id,
                segment_index=seg["segment_index"],
                total_segments=seg["total_segments"],
            )
            self.store.add(chunk)
            chunks.append(chunk)

        return chunks

    def handle_tool_call(self, tool_call: dict, result: str) -> tuple[Chunk, Chunk]:
        """Store a tool call and its result as linked chunks.

        Tool calls and results are stored with a parent-child relationship
        so that when one is retrieved, the other is also included.

        Args:
            tool_call: Dict representing the tool call (will be JSON serialized).
            result: The result from executing the tool.

        Returns:
            Tuple of (tool_call_chunk, tool_result_chunk).
        """
        now = time.time()

        # Store tool call
        call_content = json.dumps(tool_call, indent=2)
        call_chunk = Chunk(
            role="tool_call",
            content=call_content,
            embedding=self.embedder.embed(call_content),
            timestamp=now,
            token_count=self.chunker.count_tokens(call_content),
            conversation_id=self.conversation_id,
            conversation_group_id=self.conversation_group_id,
        )
        self.store.add(call_chunk)

        # Store result linked to call
        result_chunk = Chunk(
            role="tool_result",
            content=result,
            embedding=self.embedder.embed(result),
            timestamp=now,
            token_count=self.chunker.count_tokens(result),
            conversation_id=self.conversation_id,
            conversation_group_id=self.conversation_group_id,
            parent_chunk_id=call_chunk.id,
        )
        self.store.add(result_chunk)

        return call_chunk, result_chunk

    def get_history(self) -> list[Chunk]:
        """Get all chunks in this conversation, ordered chronologically.

        Returns:
            List of all chunks in timestamp order.
        """
        return self.store.get_conversation_chunks(self.conversation_id)

    def get_stats(self) -> dict:
        """Get statistics about this conversation.

        Returns:
            Dict with conversation statistics.
        """
        chunks = self.get_history()

        role_counts: dict[str, int] = {}
        total_tokens = 0

        for chunk in chunks:
            role_counts[chunk.role] = role_counts.get(chunk.role, 0) + 1
            total_tokens += chunk.token_count

        return {
            "conversation_id": self.conversation_id,
            "conversation_group_id": self.conversation_group_id,
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "role_counts": role_counts,
            "duration_seconds": (
                (chunks[-1].timestamp - chunks[0].timestamp) if len(chunks) > 1 else 0
            ),
        }

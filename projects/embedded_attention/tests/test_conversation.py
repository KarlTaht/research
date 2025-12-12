"""Integration tests for Conversation and ConversationBuilder."""

import pytest


class TestConversationBuilder:
    """Tests for ConversationBuilder and create_conversation."""

    def test_create_conversation_defaults(self):
        """Test creating conversation with defaults."""
        from projects.embedded_attention.core.builder import create_conversation

        conv = create_conversation()

        assert conv is not None
        assert conv.conversation_id is not None
        assert conv.store is not None
        assert conv.retriever is not None

    def test_create_conversation_with_db_path(self):
        """Test creating conversation with custom db path."""
        from projects.embedded_attention.core.builder import create_conversation

        conv = create_conversation(db_path=":memory:")

        assert conv is not None

    def test_builder_fluent_api(self):
        """Test ConversationBuilder fluent API."""
        from projects.embedded_attention.core.builder import (
            ConversationBuilder,
            ConversationConfig,
        )

        config = ConversationConfig(
            db_path=":memory:",
            semantic_top_k=10,
            min_similarity=0.5,
        )

        conv = (
            ConversationBuilder(config)
            .with_retrieval_config(semantic_top_k=15)
            .build()
        )

        assert conv.retriever.config.semantic_top_k == 15


class TestConversation:
    """Integration tests for Conversation."""

    @pytest.fixture
    def conversation(self):
        """Provide a fresh conversation for each test."""
        from projects.embedded_attention.core.builder import create_conversation

        return create_conversation(db_path=":memory:")

    def test_add_turn(self, conversation):
        """Test adding a turn to conversation."""
        chunks = conversation.add_turn("Hello, world!", "user")

        assert len(chunks) >= 1
        assert chunks[0].role == "user"
        assert "Hello" in chunks[0].content

    def test_get_history(self, conversation):
        """Test retrieving conversation history."""
        conversation.add_turn("Message 1", "user")
        conversation.add_turn("Response 1", "assistant")

        history = conversation.get_history()

        assert len(history) >= 2

    def test_get_stats(self, conversation):
        """Test getting conversation stats."""
        conversation.add_turn("Test message", "user")

        stats = conversation.get_stats()

        assert stats["conversation_id"] == conversation.conversation_id
        assert stats["total_chunks"] >= 1
        assert "user" in stats["role_counts"]

    def test_chat_stores_both_turns(self, conversation):
        """Test that chat stores both user message and response."""
        initial_count = len(conversation.get_history())

        response = conversation.chat("Hello!")

        final_count = len(conversation.get_history())

        # Should have added user message and assistant response
        assert final_count > initial_count
        assert response is not None

    def test_chat_with_retrieval_info(self, conversation):
        """Test chat_with_retrieval_info returns info dict."""
        # First add some context
        conversation.add_turn("My name is Alice.", "user")

        response, info = conversation.chat_with_retrieval_info("What is my name?")

        assert response is not None
        assert "retrieval" in info
        assert "merged" in info["retrieval"]

    def test_handle_tool_call(self, conversation):
        """Test storing tool calls and results."""
        tool_call = {"tool": "search", "query": "weather"}
        result = "It's sunny today."

        call_chunk, result_chunk = conversation.handle_tool_call(tool_call, result)

        assert call_chunk.role == "tool_call"
        assert result_chunk.role == "tool_result"
        assert result_chunk.parent_chunk_id == call_chunk.id


class TestMemoryRecall:
    """Tests for memory retrieval functionality."""

    @pytest.fixture
    def conversation(self):
        """Provide conversation with some stored facts."""
        from projects.embedded_attention.core.builder import create_conversation

        conv = create_conversation(
            db_path=":memory:",
            min_similarity=0.3,  # Lower threshold for testing
        )

        # Store some facts
        conv.add_turn("My favorite color is blue.", "user")
        conv.add_turn("I noted that your favorite color is blue.", "assistant")
        conv.add_turn("The secret code is ALPHA-7492.", "user")
        conv.add_turn("I'll remember the code.", "assistant")

        return conv

    def test_retrieves_relevant_facts(self, conversation):
        """Test that relevant facts are retrieved."""
        retrieved = conversation.retriever.retrieve(
            query="What is my favorite color?",
            conversation_id=conversation.conversation_id,
        )

        # Should retrieve something
        assert len(retrieved) > 0

        # Check if "blue" is in retrieved content
        all_content = " ".join(sc.chunk.content for sc in retrieved)
        assert "blue" in all_content.lower()

    def test_different_query_retrieves_different_facts(self, conversation):
        """Test that different queries retrieve different facts."""
        color_retrieved = conversation.retriever.retrieve(
            query="favorite color",
            conversation_id=conversation.conversation_id,
        )
        code_retrieved = conversation.retriever.retrieve(
            query="secret code",
            conversation_id=conversation.conversation_id,
        )

        color_content = " ".join(sc.chunk.content for sc in color_retrieved)
        code_content = " ".join(sc.chunk.content for sc in code_retrieved)

        # Different queries should prioritize different content
        assert "blue" in color_content.lower() or "color" in color_content.lower()
        assert "code" in code_content.lower() or "ALPHA" in code_content

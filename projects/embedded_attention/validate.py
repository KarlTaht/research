#!/usr/bin/env python3
"""Interactive validation TUI for Embedded Attention.

Provides guided workflows for testing and debugging:
- Chat mode: Interactive conversation with retrieval
- Retrieval inspection: See what chunks are retrieved and why
- Store exploration: Browse stored chunks
- Scoring analysis: Understand relevance scoring
- Sanity tests: Overfit-style validation

Usage:
    python -m projects.embedded_attention.validate
    python -m projects.embedded_attention.validate --db memory.duckdb
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


MAIN_MENU = """
+==============================================================+
|           EMBEDDED ATTENTION - Interactive Validator         |
+==============================================================+
|  [c] Chat          - Interactive conversation with memory    |
|  [r] Retrieval     - Inspect retrieval for a query           |
|  [s] Store         - Browse chunks in database               |
|  [a] Analyze       - Score breakdown for query               |
|  [t] Test          - Run sanity checks                       |
|  [o] Overfit       - Test memory recall (like overfit.py)    |
|  [q] Quit          - Exit                                    |
+==============================================================+
"""

CHAT_HELP = """
Chat Mode Commands:
  /info    - Show last retrieval details
  /stats   - Show conversation stats
  /clear   - Clear conversation history (start fresh)
  /back    - Return to main menu
"""


def interactive_prompt(prompt: str, options: list[str], default: str = None) -> str:
    """Wait for user input with validation."""
    try:
        cmd = input(prompt).strip().lower()
        if cmd == '' and default:
            return default
        if cmd in options:
            return cmd
        return default or options[0]
    except (EOFError, KeyboardInterrupt):
        return 'q'


def format_age(seconds: float) -> str:
    """Format age as human-readable string."""
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    else:
        return f"{int(seconds / 86400)}d ago"


def truncate(text: str, max_len: int = 50) -> str:
    """Truncate text and add ellipsis."""
    text = text.replace('\n', ' ')
    if len(text) > max_len:
        return text[:max_len-3] + "..."
    return text


def chat_mode(conv, verbose: bool = True):
    """Interactive chat with retrieval info display."""
    print("\n" + "=" * 60)
    print("CHAT MODE")
    print("=" * 60)
    print(CHAT_HELP)

    last_retrieval_info = None

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.startswith('/'):
            cmd = user_input[1:].lower()
            if cmd == 'back':
                break
            elif cmd == 'info':
                if last_retrieval_info:
                    print_retrieval_info(last_retrieval_info)
                else:
                    print("  No retrieval info yet. Send a message first.")
            elif cmd == 'stats':
                stats = conv.get_stats()
                print("\n--- Conversation Stats ---")
                print(f"  Conversation ID: {stats['conversation_id'][:8]}...")
                print(f"  Total chunks: {stats['total_chunks']}")
                print(f"  Total tokens: {stats['total_tokens']}")
                print(f"  Roles: {stats['role_counts']}")
            elif cmd == 'clear':
                print("  (Note: Cannot clear in-memory. Starting new conversation would require restart)")
            else:
                print(f"  Unknown command: /{cmd}")
            continue

        # Process chat with retrieval info
        response, info = conv.chat_with_retrieval_info(user_input)
        last_retrieval_info = info

        print(f"\nAssistant: {response}")

        if verbose:
            merged = info.get('retrieval', {}).get('merged', [])
            print(f"  [Retrieved {len(merged)} chunks]")


def print_retrieval_info(info: dict):
    """Print detailed retrieval information."""
    print("\n--- Last Retrieval Details ---")
    retrieval = info.get('retrieval', {})

    for source in ['semantic', 'recent', 'linked', 'merged']:
        chunks = retrieval.get(source, [])
        if chunks:
            print(f"\n  {source.upper()} ({len(chunks)} chunks):")
            for chunk_id, score in chunks[:5]:
                print(f"    {chunk_id[:12]}... score={score:.3f}")


def retrieval_mode(conv):
    """Inspect retrieval results for a query."""
    print("\n" + "=" * 60)
    print("RETRIEVAL INSPECTION")
    print("=" * 60)
    print("Enter queries to see what chunks would be retrieved.")
    print("Type /back to return to main menu.\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query == '/back':
            break

        if not query:
            continue

        # Get retrieval with breakdown
        breakdown = conv.retriever.retrieve_with_breakdown(
            query=query,
            conversation_id=conv.conversation_id,
            conversation_group_id=conv.conversation_group_id,
            reference_time=time.time(),
        )

        print(f"\n{'Source':<12} | {'Role':<10} | {'Score':>7} | {'Sim':>7} | Content Preview")
        print("-" * 80)

        for source in ['semantic', 'recent', 'linked']:
            chunks = breakdown.get(source, [])
            for sc in chunks[:5]:
                content_preview = truncate(sc.chunk.content, 30)
                print(f"{source:<12} | {sc.chunk.role:<10} | {sc.score:>7.3f} | "
                      f"{sc.similarity:>7.3f} | {content_preview}")

        merged = breakdown.get('merged', [])
        print(f"\n  Total merged results: {len(merged)} chunks")
        print()


def store_mode(conv):
    """Browse stored chunks."""
    print("\n" + "=" * 60)
    print("STORE BROWSER")
    print("=" * 60)

    chunks = conv.get_history()
    print(f"Total chunks in conversation: {len(chunks)}\n")

    if not chunks:
        print("  No chunks stored yet. Chat first to add some!")
        return

    # Show pagination
    page_size = 10
    page = 0
    total_pages = (len(chunks) + page_size - 1) // page_size

    while True:
        start = page * page_size
        end = min(start + page_size, len(chunks))

        print(f"\n--- Page {page + 1}/{total_pages} (showing {start+1}-{end} of {len(chunks)}) ---\n")
        print(f"{'#':>3} | {'Role':<10} | {'Age':<10} | {'Tokens':>6} | Content Preview")
        print("-" * 70)

        for i, chunk in enumerate(chunks[start:end], start=start):
            age = format_age(time.time() - chunk.timestamp)
            content_preview = truncate(chunk.content, 35)
            print(f"{i+1:>3} | {chunk.role:<10} | {age:<10} | {chunk.token_count:>6} | {content_preview}")

        print("\n[n]ext page | [p]rev page | [#] view chunk | [b]ack")
        cmd = input("> ").strip().lower()

        if cmd == 'b' or cmd == 'back':
            break
        elif cmd == 'n' or cmd == 'next':
            if page < total_pages - 1:
                page += 1
        elif cmd == 'p' or cmd == 'prev':
            if page > 0:
                page -= 1
        elif cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                print("\n" + "=" * 60)
                print(f"CHUNK #{idx+1} DETAILS")
                print("=" * 60)
                print(f"  ID: {chunk.id}")
                print(f"  Role: {chunk.role}")
                print(f"  Timestamp: {chunk.timestamp} ({format_age(time.time() - chunk.timestamp)})")
                print(f"  Token count: {chunk.token_count}")
                print(f"  Segment: {chunk.segment_index + 1}/{chunk.total_segments}")
                print(f"\nContent:")
                print("-" * 40)
                print(chunk.content)
                print("-" * 40)


def analyze_mode(conv):
    """Detailed scoring analysis."""
    print("\n" + "=" * 60)
    print("SCORING ANALYSIS")
    print("=" * 60)
    print("Enter a query to see detailed score breakdown for each chunk.")
    print("Type /back to return to main menu.\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query == '/back':
            break

        if not query:
            continue

        now = time.time()
        retrieved = conv.retriever.retrieve(
            query=query,
            conversation_id=conv.conversation_id,
            reference_time=now,
        )

        if not retrieved:
            print("  No chunks retrieved. Store some content first!")
            continue

        print(f"\n{'Role':<10} | {'Age':>10} | {'Similarity':>10} | {'Recency Wt':>10} | {'Final':>8}")
        print("-" * 60)

        for sc in retrieved[:15]:
            age_secs = now - sc.chunk.timestamp
            age_str = format_age(age_secs)

            # Compute recency weight from score/similarity ratio
            if sc.similarity > 0:
                recency_wt = sc.score / sc.similarity
            else:
                recency_wt = 1.0

            print(f"{sc.chunk.role:<10} | {age_str:>10} | {sc.similarity:>10.4f} | "
                  f"{recency_wt:>10.4f} | {sc.score:>8.4f}")

        print(f"\n  Total retrieved: {len(retrieved)} chunks")

        # Show formula
        print("\n  Scoring formula: final_score = similarity * recency_weight")
        print("  Where recency_weight = 1 / (1 + age_hours * decay_rate)")
        print()


def test_mode(conv):
    """Quick sanity checks."""
    print("\n" + "=" * 60)
    print("SANITY TESTS")
    print("=" * 60)
    print("Running component tests...\n")

    tests = [
        ("Import core modules", _test_imports),
        ("Embedder initialization", lambda c: _test_embedder(c)),
        ("Store a message", lambda c: c.add_turn("Test message for sanity check", "user")),
        ("Retrieve chunks", lambda c: c.retriever.retrieve("test", c.conversation_id)),
        ("Generate response", lambda c: c.chat("Hello, this is a test")),
        ("Get conversation stats", lambda c: c.get_stats()),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn == _test_imports:
                result = test_fn()
            else:
                result = test_fn(conv)
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")


def _test_imports():
    """Test that all core modules can be imported."""
    from .core import (
        Chunk, ScoredChunk, TextChunker, ChunkStore, Embedder,
        RelevanceScorer, Retriever, ContextAssembler, Conversation,
        ConversationBuilder, create_conversation
    )
    return True


def _test_embedder(conv):
    """Test embedder produces valid output."""
    emb = conv.embedder.embed("Test text")
    assert emb is not None
    assert len(emb) > 0
    return emb


def overfit_mode(conv):
    """Test memory recall - like overfit.py for transformers.

    This mode:
    1. Stores a specific fact
    2. Asks about it
    3. Verifies retrieval works correctly
    """
    print("\n" + "=" * 60)
    print("OVERFIT / MEMORY RECALL TEST")
    print("=" * 60)
    print("This tests whether the RAG system can recall stored facts.\n")

    # Test cases: (statement to store, query, expected_in_response)
    test_cases = [
        ("My favorite color is blue.", "What is my favorite color?", "blue"),
        ("The secret code is ALPHA-7492.", "What was the secret code?", "ALPHA-7492"),
        ("I have a dog named Max.", "What is my pet's name?", "Max"),
        ("The meeting is scheduled for Tuesday at 3pm.", "When is the meeting?", "Tuesday"),
        ("My email address is test@example.com.", "What is my email?", "test@example.com"),
    ]

    print("Running memory recall tests...\n")

    passed = 0
    failed = 0

    for statement, query, expected in test_cases:
        # Store the fact
        conv.add_turn(statement, "user")
        conv.add_turn(f"I'll remember that: {statement}", "assistant")

        # Query for retrieval
        retrieved = conv.retriever.retrieve(
            query=query,
            conversation_id=conv.conversation_id,
        )

        # Check if the fact was retrieved
        retrieved_content = " ".join(sc.chunk.content for sc in retrieved)
        found = expected.lower() in retrieved_content.lower()

        if found:
            print(f"  [PASS] '{truncate(query, 40)}' -> found '{expected}'")
            passed += 1
        else:
            print(f"  [FAIL] '{truncate(query, 40)}' -> expected '{expected}'")
            print(f"         Retrieved: {truncate(retrieved_content, 60)}")
            failed += 1

    print(f"\n  Memory recall: {passed}/{len(test_cases)} tests passed")

    if passed == len(test_cases):
        print("\n  SUCCESS: All facts were retrieved correctly!")
    elif passed > 0:
        print("\n  PARTIAL: Some facts retrieved. Check similarity thresholds.")
    else:
        print("\n  FAILURE: No facts retrieved. Check embedding/retrieval pipeline.")

    # Show retrieval settings
    print(f"\n  Current settings:")
    print(f"    semantic_top_k: {conv.retriever.config.semantic_top_k}")
    print(f"    min_similarity: {conv.retriever.config.min_similarity}")
    print(f"    recent_n: {conv.retriever.config.recent_n}")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive validation TUI for Embedded Attention'
    )
    parser.add_argument('--db', type=str, default=':memory:',
                        help='Database path (default: :memory:)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show extra retrieval info in chat mode')
    args = parser.parse_args()

    print("\nInitializing Embedded Attention...")

    try:
        from .core.builder import create_conversation
        conv = create_conversation(db_path=args.db)
        print(f"  Database: {args.db}")
        print(f"  Conversation ID: {conv.conversation_id[:8]}...")
        print(f"  Embedding model: {conv.embedder.model_name}")
        print("  Ready!")
    except Exception as e:
        print(f"\n  ERROR: Failed to initialize: {e}")
        print("\n  Make sure you've installed dependencies:")
        print("    uv pip install -e '.[embedded_attention]'")
        return 1

    while True:
        print(MAIN_MENU)
        cmd = interactive_prompt("Select option: ", ['c', 'r', 's', 'a', 't', 'o', 'q'], 'c')

        if cmd == 'c':
            chat_mode(conv, verbose=args.verbose)
        elif cmd == 'r':
            retrieval_mode(conv)
        elif cmd == 's':
            store_mode(conv)
        elif cmd == 'a':
            analyze_mode(conv)
        elif cmd == 't':
            test_mode(conv)
        elif cmd == 'o':
            overfit_mode(conv)
        elif cmd == 'q':
            print("\nGoodbye!")
            break

    return 0


if __name__ == '__main__':
    sys.exit(main())

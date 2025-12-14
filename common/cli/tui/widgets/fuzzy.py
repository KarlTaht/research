"""Fuzzy matching utilities for autocomplete widgets."""


def fuzzy_match(pattern: str, text: str) -> tuple[bool, int]:
    """Check if pattern fuzzy-matches text, with score.

    Characters in pattern must appear in text in order, but not necessarily
    consecutively. Higher score = better match.

    Args:
        pattern: The search pattern (e.g., "ts")
        text: The text to match against (e.g., "TinyStories")

    Returns:
        Tuple of (matches, score). Score is higher for better matches:
        - Consecutive matches score higher
        - Matches at word boundaries score higher
        - Exact prefix matches score highest
    """
    if not pattern:
        return True, 0

    pattern = pattern.lower()
    text_lower = text.lower()

    # Exact prefix match gets highest score
    if text_lower.startswith(pattern):
        return True, 1000 + len(pattern)

    # Exact substring match gets high score
    if pattern in text_lower:
        # Score based on position (earlier = better)
        pos = text_lower.find(pattern)
        return True, 500 - pos

    # Fuzzy character-by-character match
    pattern_idx = 0
    score = 0
    prev_match_idx = -1

    for i, char in enumerate(text_lower):
        if pattern_idx < len(pattern) and char == pattern[pattern_idx]:
            # Match found
            pattern_idx += 1

            # Bonus for consecutive matches
            if prev_match_idx == i - 1:
                score += 10
            else:
                score += 1

            # Bonus for word boundary matches (after / or space or at start)
            if i == 0 or text[i - 1] in "/ -_":
                score += 20

            prev_match_idx = i

    if pattern_idx == len(pattern):
        return True, score
    return False, 0


def fuzzy_filter(pattern: str, items: list[str], limit: int = 10) -> list[str]:
    """Filter and sort items by fuzzy match score.

    Args:
        pattern: The search pattern
        items: List of strings to filter
        limit: Maximum number of results to return

    Returns:
        List of matching items, sorted by score (best first)
    """
    if not pattern:
        return items[:limit]

    scored = []
    for item in items:
        matches, score = fuzzy_match(pattern, item)
        if matches:
            scored.append((score, item))

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])

    return [item for score, item in scored[:limit]]

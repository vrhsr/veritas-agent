"""
Context Window Budget Manager
Manages token budget allocation across all context components.
Applies priority-based truncation when total exceeds context limit.

Token budget (8K total context):
- System prompt:        500 tokens  (always kept)
- Retrieved chunks:    2,500 tokens (trimmed first)
- Conversation history: 1,500 tokens (oldest dropped first)
- Long-term memories:    800 tokens (summarized if too long)
- Reasoning output:    1,500 tokens (preserved)
- Buffer:               700 tokens  (safety margin)
"""
import os
from typing import List, Optional

from utils.token_counter import count_tokens

CONTEXT_LIMIT = int(os.getenv("CONTEXT_WINDOW_LIMIT", "8000"))
CHUNK_BUDGET = 2500
MEMORY_BUDGET = 800
HISTORY_BUDGET = 1500


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token budget (approximate by chars)."""
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text
    # Approximate: 1 token ≈ 4 chars
    char_limit = max_tokens * 4
    return text[:char_limit] + "... [truncated]"


def build_context_within_budget(
    chunks: List[dict],
    memories: List[str],
    history: List[dict],
    max_tokens: int = CHUNK_BUDGET,
) -> str:
    """
    Build a context string from retrieved chunks within token budget.
    Applies priority-based truncation:
    1. Start with all chunks
    2. If over budget, reduce to top-3 chunks
    3. If still over budget, truncate each chunk to key sentences
    """
    if not chunks:
        return "No relevant context retrieved."

    # Try full top-5 chunks
    full_context = _format_chunks(chunks[:5])
    if count_tokens(full_context) <= max_tokens:
        return full_context

    # Trim to top-3
    reduced_context = _format_chunks(chunks[:3])
    if count_tokens(reduced_context) <= max_tokens:
        return reduced_context

    # Truncate each chunk
    truncated_parts = []
    budget_per_chunk = max_tokens // min(len(chunks[:3]), 3)
    for i, chunk in enumerate(chunks[:3]):
        text = chunk.get("text", "")
        truncated = truncate_to_budget(text, budget_per_chunk)
        truncated_parts.append(f"[Source {i+1}: {chunk.get('source', 'unknown')}]\n{truncated}")

    return "\n\n".join(truncated_parts)


def _format_chunks(chunks: List[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        score = chunk.get("score", 0)
        parts.append(f"[Source {i+1}: {source} (score: {score:.3f})]\n{text}")
    return "\n\n".join(parts)


def build_memory_context(memories: List[str], max_tokens: int = MEMORY_BUDGET) -> str:
    """Build memory context within budget, summarizing if needed."""
    if not memories:
        return ""

    full = "\n".join(f"- {m}" for m in memories)
    if count_tokens(full) <= max_tokens:
        return full

    # Use first N memories that fit
    result_parts = []
    used = 0
    for m in memories:
        tokens = count_tokens(m)
        if used + tokens > max_tokens:
            break
        result_parts.append(f"- {m[:200]}")  # Trim each memory to 200 chars
        used += tokens

    return "\n".join(result_parts)

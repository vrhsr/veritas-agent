"""
Token counting utility using tiktoken.
"""
import os
import tiktoken

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# tiktoken uses cl100k_base for gpt-4o-mini, gpt-4, etc.
try:
    _enc = tiktoken.encoding_for_model(MODEL)
except KeyError:
    _enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    if not text:
        return 0
    try:
        return len(_enc.encode(text))
    except Exception:
        # Fallback: rough estimate (4 chars per token)
        return len(text) // 4

"""
Document Fetcher Tool
Given a URL, fetches and chunks the content on the fly.
Used when web search returns a highly relevant URL but snippet is insufficient.
"""
import os
import re
from typing import List

import httpx
from bs4 import BeautifulSoup

from utils.logger import get_logger

logger = get_logger(__name__)
CHUNK_SIZE = 400  # words per chunk
MAX_CHUNKS = 5
FETCH_TIMEOUT = 5.0


def fetch_document(url: str, chunk_size: int = CHUNK_SIZE) -> List[dict]:
    """
    Fetches a URL, extracts clean text, and returns chunked results.
    Returns list of {text, source, score} dicts.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research-assistant-bot)"}
        response = httpx.get(url, timeout=FETCH_TIMEOUT, headers=headers, follow_redirects=True)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        # Chunk by words
        words = text.split()
        chunks = []
        for i in range(0, min(len(words), chunk_size * MAX_CHUNKS), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "source": url,
                "score": 0.75,
                "retrieval_method": "doc_fetcher",
            })

        logger.info("doc_fetcher", url=url[:80], chunks=len(chunks))
        return chunks[:MAX_CHUNKS]

    except httpx.TimeoutException:
        logger.warning("doc_fetcher_timeout", url=url)
        return []
    except Exception as e:
        logger.warning("doc_fetcher_error", url=url, error=str(e))
        return []

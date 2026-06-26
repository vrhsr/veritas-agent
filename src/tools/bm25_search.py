"""
BM25 Keyword Search Tool
Sparse retrieval using BM25Okapi — excellent for queries with specific numbers,
dates, model names, and proper nouns.
"""
import os
import pickle
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from utils.logger import get_logger

logger = get_logger(__name__)
BM25_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl"))

_bm25_index: BM25Okapi | None = None
_bm25_docs: list[dict] | None = None


def _load_index():
    global _bm25_index, _bm25_docs
    if _bm25_index is not None:
        return

    if not BM25_INDEX_PATH.exists():
        logger.warning("bm25_index_not_found", path=str(BM25_INDEX_PATH))
        _bm25_index = None
        _bm25_docs = []
        return

    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    _bm25_index = data["index"]
    _bm25_docs = data["docs"]
    logger.info("bm25_index_loaded", num_docs=len(_bm25_docs))


def bm25_search(query: str, top_k: int = 10) -> List[dict]:
    """
    BM25Okapi sparse retrieval.
    Returns list of {text, source, score} dicts sorted by BM25 score descending.
    """
    _load_index()

    if _bm25_index is None or not _bm25_docs:
        logger.warning("bm25_search_skipped_no_index")
        return []

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    # Sort by score descending
    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for idx, score in ranked[:top_k]:
        if score > 0:
            doc = _bm25_docs[idx]
            results.append({
                "text": doc.get("text", ""),
                "source": doc.get("source", "bm25"),
                "score": float(score),
                "retrieval_method": "bm25",
            })

    logger.debug("bm25_search", query=query[:60], results=len(results))
    return results

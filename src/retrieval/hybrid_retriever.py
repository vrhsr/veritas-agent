"""
Hybrid Retriever
Combines BM25 (sparse) + FAISS (dense) with Reciprocal Rank Fusion (RRF),
then reranks with a cross-encoder for final scoring.

Architecture:
    Query
      ├── BM25 keyword search (sparse)   → ranked list A
      └── FAISS semantic search (dense)  → ranked list B
                ↓
        RRF Fusion: score(d) = Σ 1/(k + rank(d)), k=60
                ↓
        Cross-encoder reranking
        (cross-encoder/ms-marco-MiniLM-L-6-v2)
                ↓
        Top-K chunks with confidence scores
"""
import os
from typing import List, Optional

from sentence_transformers import CrossEncoder

from tools.bm25_search import bm25_search
from tools.vector_search import vector_search
from utils.logger import get_logger

logger = get_logger(__name__)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DISABLE_RERANKER = os.getenv("DISABLE_RERANKER", "false").lower() == "true"
RRF_K = 60  # Standard RRF constant — calibrated to maximize fusion quality
DEFAULT_TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "5"))
CANDIDATE_K = 20  # Number of candidates before reranking

_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("loading_cross_encoder", model=RERANKER_MODEL)
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


def _rrf_score(rankings: List[List[dict]], k: int = RRF_K) -> List[dict]:
    """
    Reciprocal Rank Fusion across multiple ranked lists.
    Each document's RRF score = Σ 1/(k + rank) across all lists where it appears.
    Deduplicates by text content.
    """
    doc_scores: dict[str, float] = {}
    doc_store: dict[str, dict] = {}

    for ranked_list in rankings:
        for rank, doc in enumerate(ranked_list):
            key = doc["text"][:100]  # Use text prefix as dedup key
            rrf = 1.0 / (k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0.0) + rrf
            if key not in doc_store:
                doc_store[key] = doc

    # Sort by accumulated RRF score
    sorted_keys = sorted(doc_scores, key=lambda x: doc_scores[x], reverse=True)

    results = []
    for key in sorted_keys:
        doc = doc_store[key].copy()
        doc["rrf_score"] = round(doc_scores[key], 6)
        results.append(doc)

    return results


def hybrid_retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    threshold_override: Optional[float] = None,
) -> List[dict]:
    """
    Main hybrid retrieval entry point.

    Steps:
    1. Run BM25 and FAISS in parallel
    2. Fuse rankings with RRF
    3. Rerank top candidates with cross-encoder
    4. Return top_k results

    Args:
        query: The search query string
        top_k: Number of final results to return
        threshold_override: Override FAISS similarity threshold (for widened retry search)
    """
    # ── Step 1: Run both retrievers ──────────────────────────────────────────
    bm25_kwargs = {}
    vector_kwargs = {}
    if threshold_override is not None:
        vector_kwargs["threshold"] = threshold_override

    bm25_results = bm25_search(query, top_k=CANDIDATE_K, **bm25_kwargs)
    vector_results = vector_search(query, top_k=CANDIDATE_K, **vector_kwargs)

    logger.debug(
        "hybrid_retrieve_raw",
        bm25_count=len(bm25_results),
        vector_count=len(vector_results),
    )

    if not bm25_results and not vector_results:
        logger.warning("hybrid_retrieve_no_results", query=query[:60])
        return []

    # ── Step 2: RRF Fusion ───────────────────────────────────────────────────
    fused = _rrf_score([bm25_results, vector_results])
    candidates = fused[:CANDIDATE_K]

    if not candidates:
        return []

    # ── Step 3: Cross-encoder Reranking ─────────────────────────────────────
    if DISABLE_RERANKER:
        logger.info("cross_encoder_disabled_using_rrf_scores")
        for doc in candidates:
            doc["score"] = doc.get("rrf_score", 0.0)
        return candidates[:top_k]

    try:
        cross_encoder = _get_cross_encoder()
        pairs = [(query, doc["text"][:512]) for doc in candidates]
        ce_scores = cross_encoder.predict(pairs)

        for doc, score in zip(candidates, ce_scores):
            doc["cross_encoder_score"] = float(score)
            # Final score: blend RRF + cross-encoder
            doc["score"] = float(score)

        candidates.sort(key=lambda x: x.get("cross_encoder_score", 0), reverse=True)
        logger.debug("cross_encoder_reranking_done", candidates=len(candidates))

    except Exception as e:
        logger.warning("cross_encoder_error", error=str(e))
        # Fall back to RRF scores
        for doc in candidates:
            doc["score"] = doc.get("rrf_score", 0.0)

    return candidates[:top_k]

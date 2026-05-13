"""
OOD (Out-of-Distribution) Detection Gate
Detects queries that the corpus cannot answer before wasting retry tokens.

Algorithm:
  1. Embed the query with the same model used for corpus indexing
  2. Run a FAISS search over the corpus index (top-1)
  3. If max similarity < threshold, query is OOD — skip retries, respond with uncertainty

Rationale:
  Without this gate, OOD queries burn MAX_RETRIES * (retrieval + reasoning) tokens
  each time. Empirically this was the source of P99 cost spikes (8x average).
  The gate adds ~50ms but saves the full retry loop cost for unanswereable queries.
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

logger = get_logger(__name__)

FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OOD_THRESHOLD = float(os.getenv("OOD_THRESHOLD", "0.30"))  # Below this = OOD
OOD_SAMPLE_K = 1  # Just need top-1 similarity

_ood_index: Optional[faiss.Index] = None
_ood_embed_model: Optional[SentenceTransformer] = None


def _get_embed_model() -> SentenceTransformer:
    global _ood_embed_model
    if _ood_embed_model is None:
        _ood_embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _ood_embed_model


def _get_corpus_index() -> Optional[faiss.Index]:
    """Lazy-load the corpus FAISS index for OOD detection."""
    global _ood_index
    if _ood_index is not None:
        return _ood_index

    index_file = FAISS_INDEX_PATH / "index.faiss"
    if not index_file.exists():
        logger.warning("ood_index_not_found", path=str(index_file))
        return None

    try:
        _ood_index = faiss.read_index(str(index_file))
        logger.info("ood_index_loaded", ntotal=_ood_index.ntotal)
        return _ood_index
    except Exception as e:
        logger.warning("ood_index_load_error", error=str(e))
        return None


def is_out_of_distribution(query: str) -> tuple[bool, float]:
    """
    Check if a query is out of distribution relative to the corpus.

    Returns:
        (is_ood, max_similarity)
        is_ood=True means the corpus likely cannot answer this query.

    Strategy: embed query, find nearest neighbor in corpus index.
    If cosine similarity < OOD_THRESHOLD, corpus has no relevant content.
    """
    index = _get_corpus_index()
    if index is None or index.ntotal == 0:
        # If we can't load index, assume in-distribution (fail open)
        return False, 1.0

    try:
        model = _get_embed_model()
        embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)
        distances, _ = index.search(embedding, OOD_SAMPLE_K)
        max_sim = float(distances[0][0])

        is_ood = max_sim < OOD_THRESHOLD
        logger.info(
            "ood_check",
            query_preview=query[:60],
            max_similarity=round(max_sim, 4),
            threshold=OOD_THRESHOLD,
            is_ood=is_ood,
        )
        return is_ood, max_sim

    except Exception as e:
        logger.warning("ood_check_error", error=str(e))
        return False, 1.0  # Fail open — don't block queries on OOD error

"""
FAISS Vector Search Tool
Dense retrieval using sentence-transformers/all-MiniLM-L6-v2 embeddings.
Excellent for paraphrased queries and semantic similarity.
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List

import faiss
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

logger = get_logger(__name__)
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_faiss_index: faiss.Index | None = None
_faiss_docs: list[dict] | None = None
_embed_model: SentenceTransformer | None = None


def _load_resources():
    global _faiss_index, _faiss_docs, _embed_model

    if _embed_model is None:
        logger.info("loading_embedding_model", model=EMBEDDING_MODEL)
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)

    if _faiss_index is not None:
        return

    index_file = FAISS_INDEX_PATH / "index.faiss"
    docs_file = FAISS_INDEX_PATH / "docs.jsonl"

    if not index_file.exists():
        logger.warning("faiss_index_not_found", path=str(index_file))
        _faiss_index = None
        _faiss_docs = []
        return

    _faiss_index = faiss.read_index(str(index_file))

    _faiss_docs = []
    with open(docs_file, "r", encoding="utf-8") as f:
        for line in f:
            _faiss_docs.append(json.loads(line.strip()))

    logger.info("faiss_index_loaded", num_docs=len(_faiss_docs), index_size=_faiss_index.ntotal)


def get_embedding(text: str) -> np.ndarray:
    """Get normalized embedding vector for a text string."""
    _load_resources()
    embedding = _embed_model.encode([text], normalize_embeddings=True)
    return embedding.astype(np.float32)


def vector_search(query: str, top_k: int = 10, threshold: float = 0.3) -> List[dict]:
    """
    FAISS dense retrieval.
    Returns list of {text, source, score} dicts sorted by cosine similarity descending.
    """
    _load_resources()

    if _faiss_index is None or not _faiss_docs:
        logger.warning("vector_search_skipped_no_index")
        return []

    query_embedding = get_embedding(query)
    distances, indices = _faiss_index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        score = float(dist)  # Already normalized cosine similarity
        if score < threshold:
            continue
        doc = _faiss_docs[idx]
        results.append({
            "text": doc.get("text", ""),
            "source": doc.get("source", "faiss"),
            "score": score,
            "retrieval_method": "vector",
        })

    logger.debug("vector_search", query=query[:60], results=len(results))
    return results

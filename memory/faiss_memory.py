"""
FAISS Long-Term Memory
Stores important facts, user preferences, and validated Q&A pairs across sessions.
Enables semantic cache — similar past questions return cached answers.

Per-user namespacing: each user_id gets a separate FAISS collection directory,
preventing cross-user memory leakage (privacy fix).
"""
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Optional

import faiss
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

logger = get_logger(__name__)
FAISS_MEMORY_BASE = Path(os.getenv("FAISS_MEMORY_PATH", "data/faiss_memory"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dim
SEMANTIC_CACHE_THRESHOLD = 0.92  # Only return cache hit if very similar

# Shared embedding model across all FAISSMemory instances (lazy-loaded once)
_shared_embed_model: Optional[SentenceTransformer] = None


def _get_shared_embed_model() -> SentenceTransformer:
    global _shared_embed_model
    if _shared_embed_model is None:
        _shared_embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _shared_embed_model


class FAISSMemory:
    """
    Per-user namespaced FAISS memory store.
    Each user_id gets its own index directory under FAISS_MEMORY_BASE/<user_id>/.
    This prevents cross-user memory leakage — a privacy and relevance requirement.
    """

    def __init__(self, user_id: str = "global"):
        # Sanitize user_id to safe directory name
        safe_uid = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        self._user_id = safe_uid
        self._memory_path = FAISS_MEMORY_BASE / safe_uid
        self._index: Optional[faiss.Index] = None
        self._memories: list[dict] = []
        self._load()

    def _embed(self, text: str) -> np.ndarray:
        model = _get_shared_embed_model()
        embedding = model.encode([text], normalize_embeddings=True)
        return embedding.astype(np.float32)

    def _load(self):
        self._memory_path.mkdir(parents=True, exist_ok=True)
        index_file = self._memory_path / "memory.faiss"
        docs_file = self._memory_path / "memories.jsonl"

        if index_file.exists() and docs_file.exists():
            try:
                self._index = faiss.read_index(str(index_file))
                with open(docs_file, "r", encoding="utf-8") as f:
                    self._memories = [json.loads(l) for l in f if l.strip()]
                logger.info("faiss_memory_loaded", user_id=self._user_id, count=len(self._memories))
                return
            except Exception as e:
                logger.warning("faiss_memory_load_error", user_id=self._user_id, error=str(e))

        # Create new flat index for memory (cosine similarity via normalized vectors)
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._memories = []
        logger.info("faiss_memory_initialized_new", user_id=self._user_id)

    def _save(self):
        self._memory_path.mkdir(parents=True, exist_ok=True)
        index_file = self._memory_path / "memory.faiss"
        docs_file = self._memory_path / "memories.jsonl"
        faiss.write_index(self._index, str(index_file))
        with open(docs_file, "w", encoding="utf-8") as f:
            for mem in self._memories:
                f.write(json.dumps(mem) + "\n")

    def add_memory(self, text: str, metadata: Optional[dict] = None) -> None:
        """Add a new memory to this user's FAISS index."""
        try:
            embedding = self._embed(text)
            self._index.add(embedding)
            self._memories.append({
                "text": text,
                "metadata": {**(metadata or {}), "user_id": self._user_id},
            })
            self._save()
            logger.debug("faiss_memory_added", user_id=self._user_id, text_preview=text[:80])
        except Exception as e:
            logger.warning("faiss_memory_add_error", user_id=self._user_id, error=str(e))

    def search_memories(self, query: str, top_k: int = 3) -> List[str]:
        """
        Semantic search over this user's long-term memory only.
        Namespacing is enforced at load time — each instance only holds one user's data.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        try:
            embedding = self._embed(query)
            k = min(top_k, self._index.ntotal)
            distances, indices = self._index.search(embedding, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1 or dist < 0.5:
                    continue
                mem = self._memories[idx]
                results.append(mem["text"])

            return results
        except Exception as e:
            logger.warning("faiss_memory_search_error", user_id=self._user_id, error=str(e))
            return []

    def check_semantic_cache(self, query: str) -> Optional[str]:
        """
        Check if a very similar query has been answered before for this user.
        Returns cached answer if similarity > threshold, else None.
        """
        if self._index is None or self._index.ntotal == 0:
            return None

        try:
            embedding = self._embed(query)
            distances, indices = self._index.search(embedding, 1)

            if distances[0][0] >= SEMANTIC_CACHE_THRESHOLD and indices[0][0] != -1:
                mem = self._memories[indices[0][0]]
                if mem["text"].startswith("Q:"):
                    parts = mem["text"].split("\nA:", 1)
                    if len(parts) == 2:
                        logger.info("semantic_cache_hit", user_id=self._user_id, score=float(distances[0][0]))
                        return parts[1].strip()
        except Exception as e:
            logger.warning("faiss_cache_check_error", user_id=self._user_id, error=str(e))

        return None

    @property
    def size(self) -> int:
        """Number of memories stored for this user."""
        return self._index.ntotal if self._index else 0

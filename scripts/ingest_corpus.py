"""
ingest_corpus.py
Builds BM25 and FAISS indexes from the downloaded corpus.

Pipeline:
  1. Load all JSONL files from data/corpus/
  2. Chunk each document into ~200-word passages
  3. Build BM25Okapi index → save to data/bm25_index.pkl
  4. Embed all chunks with sentence-transformers → build FAISS index
  5. Save FAISS index + docs to data/faiss_index/

Run: python scripts/ingest_corpus.py
"""
import json
import os
import pickle
import re
from pathlib import Path
from typing import List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CORPUS_DIR = Path("data/corpus")
BM25_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl"))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 200   # words per chunk
CHUNK_OVERLAP = 40 # word overlap between chunks
EMBEDDING_DIM = 384
BATCH_SIZE = 64    # Embedding batch size


def load_corpus_docs() -> List[dict]:
    """Load all JSONL files from the corpus directory."""
    docs = []
    for jsonl_file in CORPUS_DIR.glob("*.jsonl"):
        print(f"Loading: {jsonl_file.name}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    print(f"Loaded {len(docs)} documents")
    return docs


def chunk_document(doc: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[dict]:
    """
    Chunk a document into overlapping word windows.
    Returns list of chunk dicts with text, source, doc_id.
    """
    text = doc.get("text", "")
    source = doc.get("source", doc.get("id", "unknown"))
    title = doc.get("title", "")
    corpus_type = doc.get("corpus_type", "unknown")

    # Clean text
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    words = text.split()

    # Prepend title to first chunk
    if title and not text.startswith(title):
        words = title.split() + words

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "text": chunk_text,
            "source": source,
            "doc_id": doc.get("id", source),
            "title": title,
            "corpus_type": corpus_type,
            "chunk_index": len(chunks),
        })
        if end >= len(words):
            break
        start += chunk_size - overlap  # Slide with overlap

    return chunks


def build_bm25_index(chunks: List[dict]) -> None:
    """Build and save BM25Okapi index."""
    print(f"\nBuilding BM25 index over {len(chunks)} chunks...")

    tokenized = [doc["text"].lower().split() for doc in tqdm(chunks, desc="Tokenizing")]
    index = BM25Okapi(tokenized)

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"index": index, "docs": chunks}, f)

    print(f"BM25 index saved -> {BM25_INDEX_PATH} ({len(chunks)} docs)")


def build_faiss_index(chunks: List[dict]) -> None:
    """Build and save FAISS flat IP index with normalized embeddings."""
    print(f"\nBuilding FAISS index over {len(chunks)} chunks...")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)

    # Embed in batches
    all_embeddings = []
    texts = [c["text"] for c in chunks]
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embeddings shape: {all_embeddings.shape}")

    # Build FAISS flat index (inner product = cosine on normalized vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(all_embeddings)

    # Save
    faiss.write_index(index, str(FAISS_INDEX_PATH / "index.faiss"))

    # Save docs as JSONL
    with open(FAISS_INDEX_PATH / "docs.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"FAISS index saved -> {FAISS_INDEX_PATH} ({index.ntotal} vectors)")


def print_stats(chunks: List[dict]) -> None:
    corpus_types = {}
    for c in chunks:
        ct = c.get("corpus_type", "unknown")
        corpus_types[ct] = corpus_types.get(ct, 0) + 1

    avg_words = sum(len(c["text"].split()) for c in chunks) / max(len(chunks), 1)
    print(f"\nCorpus Statistics:")
    print(f"  Total chunks: {len(chunks):,}")
    print(f"  Avg words/chunk: {avg_words:.0f}")
    for ct, count in corpus_types.items():
        print(f"  {ct}: {count:,} chunks")


if __name__ == "__main__":
    print("=" * 60)
    print("Corpus Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load
    docs = load_corpus_docs()
    if not docs:
        print("ERROR: No corpus documents found. Run scripts/download_data.py first.")
        exit(1)

    # Step 2: Chunk
    print(f"\nChunking {len(docs)} documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    all_chunks = []
    for doc in tqdm(docs, desc="Chunking"):
        all_chunks.extend(chunk_document(doc))

    print_stats(all_chunks)

    # Step 3: BM25
    build_bm25_index(all_chunks)

    # Step 4: FAISS
    build_faiss_index(all_chunks)

    print("\n[OK] Ingestion complete! Both indexes are ready.")
    print(f"   BM25  -> {BM25_INDEX_PATH}")
    print(f"   FAISS -> {FAISS_INDEX_PATH}/")

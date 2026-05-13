"""
download_data.py
Downloads the domain corpus:
  - arXiv ML/AI abstracts via the arxiv Python library
  - Wikipedia ML articles via the wikipedia library

Saves to data/corpus/ as JSONL files.
Run: python scripts/download_data.py
"""
import json
import os
import time
from pathlib import Path

import arxiv
import wikipedia

CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# ── arXiv config ─────────────────────────────────────────────────────────────
ARXIV_QUERIES = [
    "large language models",
    "transformer architecture attention",
    "BERT GPT language model",
    "LoRA QLoRA parameter efficient fine-tuning",
    "RLHF reinforcement learning human feedback",
    "retrieval augmented generation RAG",
    "chain of thought prompting reasoning",
    "diffusion models generative AI",
    "multimodal vision language models",
    "in-context learning few-shot",
    "instruction tuning alignment",
    "knowledge distillation neural network",
    "neural architecture search",
    "federated learning privacy",
    "graph neural networks",
]
ARXIV_MAX_PER_QUERY = 50  # abstracts per query

# ── Wikipedia config ──────────────────────────────────────────────────────────
WIKIPEDIA_ARTICLES = [
    "BERT (language model)",
    "GPT-4",
    "Transformer (deep learning architecture)",
    "Attention mechanism",
    "Generative pre-trained transformer",
    "Large language model",
    "Reinforcement learning from human feedback",
    "Low-rank adaptation",
    "Retrieval-augmented generation",
    "Word2vec",
    "Recurrent neural network",
    "Convolutional neural network",
    "Backpropagation",
    "Stochastic gradient descent",
    "Overfitting",
    "Regularization (mathematics)",
    "Dropout (neural networks)",
    "Batch normalization",
    "Variational autoencoder",
    "Generative adversarial network",
]


def download_arxiv_abstracts():
    """Download arXiv abstracts and save as JSONL."""
    out_path = CORPUS_DIR / "arxiv_abstracts.jsonl"
    seen_ids = set()

    # Resume from existing file
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                seen_ids.add(doc.get("id", ""))
        print(f"[arxiv] Resuming — {len(seen_ids)} existing abstracts found")

    total_new = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for query in ARXIV_QUERIES:
            print(f"[arxiv] Querying: {query!r}")
            try:
                client = arxiv.Client()
                search = arxiv.Search(
                    query=query,
                    max_results=ARXIV_MAX_PER_QUERY,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                for result in client.results(search):
                    if result.entry_id in seen_ids:
                        continue
                    seen_ids.add(result.entry_id)
                    doc = {
                        "id": result.entry_id,
                        "title": result.title,
                        "text": f"{result.title}\n\n{result.summary}",
                        "source": result.entry_id,
                        "authors": [a.name for a in result.authors[:5]],
                        "published": str(result.published),
                        "categories": result.categories,
                        "corpus_type": "arxiv",
                    }
                    f.write(json.dumps(doc) + "\n")
                    total_new += 1
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"[arxiv] Error on query {query!r}: {e}")

    print(f"[arxiv] Done — {total_new} new abstracts saved to {out_path}")


def download_wikipedia_articles():
    """Download Wikipedia ML articles and save as JSONL."""
    out_path = CORPUS_DIR / "wikipedia_articles.jsonl"
    wikipedia.set_lang("en")

    seen_titles = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                seen_titles.add(doc.get("title", ""))
        print(f"[wikipedia] Resuming — {len(seen_titles)} existing articles found")

    total_new = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for title in WIKIPEDIA_ARTICLES:
            if title in seen_titles:
                print(f"[wikipedia] Skip (already exists): {title}")
                continue
            try:
                page = wikipedia.page(title, auto_suggest=False)
                doc = {
                    "id": f"wiki:{page.pageid}",
                    "title": page.title,
                    "text": page.content[:8000],  # First 8000 chars
                    "source": page.url,
                    "corpus_type": "wikipedia",
                }
                f.write(json.dumps(doc) + "\n")
                total_new += 1
                print(f"[wikipedia] Saved: {page.title}")
                time.sleep(0.2)
            except wikipedia.exceptions.DisambiguationError as e:
                # Try first option
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    doc = {
                        "id": f"wiki:{page.pageid}",
                        "title": page.title,
                        "text": page.content[:8000],
                        "source": page.url,
                        "corpus_type": "wikipedia",
                    }
                    f.write(json.dumps(doc) + "\n")
                    total_new += 1
                except Exception:
                    print(f"[wikipedia] Skipping {title} (disambiguation)")
            except Exception as e:
                print(f"[wikipedia] Error: {title}: {e}")

    print(f"[wikipedia] Done — {total_new} new articles saved to {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading corpus data...")
    print("=" * 60)
    download_arxiv_abstracts()
    print()
    download_wikipedia_articles()
    print()
    print("Download complete. Run scripts/ingest_corpus.py next.")

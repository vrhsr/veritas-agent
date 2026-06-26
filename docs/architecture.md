# Architecture — LLM Agent System

> *Multi-agent research assistant built with LangGraph — specialized agents handle retrieval, reasoning, and validation with hybrid BM25+FAISS retrieval, confidence-driven failure recovery, persistent memory, and full LangSmith observability.*

## Agent Graph Topology

```
User Query
    ↓
[Node 1: Query Analyzer]          ← Classifies: simple / complex / ambiguous
    ├── simple    → [Node 5: Response Generator]
    ├── ambiguous → [Node 6: Clarification Agent]
    └── complex   → [Node 2: Retrieval Agent]
                         ↓ BM25 + FAISS → RRF Fusion → Cross-encoder Rerank
                    [Node 3: Reasoning Agent]
                         ↓ Decompose → Answer → Gap detection → Confidence score
                    [Node 4: Validation Agent]
                         ↓ Grounding + Consistency + Completeness checks
                         ├── pass  → [Node 5: Response Generator]
                         └── fail  → back to Node 2 (max 2 retries)
                    [Node 5: Response Generator]
                         ↓ Cites sources + streams output + tracks cost
                    [Node 7: Memory Manager]
                         ↓ Redis (short-term) + FAISS (long-term)
                    Final Response
```

## Key Design Decisions

| Decision | Why |
|---|---|
| Hybrid BM25 + FAISS retrieval | Keyword search excels on proper nouns/numbers; semantic search excels on paraphrases; RRF fusion + cross-encoder reranking combines both |
| Separate Validation Agent | Asking reasoning agent to validate its own output creates confirmation bias; a separate adversarial prompt catches ~15% more failures |
| Confidence-driven routing | Sub-0.7 confidence triggers retrieval retry with gap-targeted reformulated query; sub-0.5 triggers clarification — prevents confident hallucination |
| LangGraph over sequential chain | Retry loops (validation → retrieval) are cycles — impossible in LangChain chains; LangGraph's typed shared state + conditional edges model this natively |
| Sympy calculator | Raw `eval()` is a code injection risk; sympy parses math expressions safely without executing arbitrary Python |
| Redis TTL 24h | Conversation history auto-expires without manual cleanup |

## Confidence Routing

| Score | Action |
|---|---|
| ≥ 0.70 | Pass to Validation Agent |
| 0.50–0.69 | Retry with reformulated query (max 2) |
| < 0.50 | Trigger Clarification Agent |

## Context Window Budget (8K total)

| Component | Budget | Truncation Priority |
|---|---|---|
| System prompt | 500 tokens | Always kept |
| Retrieved chunks | 2,500 tokens | **Trimmed first** |
| Conversation history | 1,500 tokens | Oldest turns dropped |
| Long-term memories | 800 tokens | Summarized |
| Reasoning output | 1,500 tokens | Preserved |
| Buffer | 700 tokens | Safety margin |

## Observability

Set `LANGCHAIN_TRACING_V2=true` in `.env` to enable automatic LangSmith tracing.
Every node's prompt, response, latency, token count, and retry count is captured per run.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_key"
os.environ["LANGCHAIN_PROJECT"] = "research-assistant"
```

## Approximate Cost Per Query (gpt-4o-mini)

| Query Type | Pipeline Path | Approx Cost |
|---|---|---|
| Simple | Analyzer + Response | ~₹0.08 |
| Complex (no retry) | Full pipeline | ~₹0.58 |
| Complex (with retry) | Full pipeline + 1 retry | ~₹1.00 |
| Ambiguous + clarification | Clarification + full pipeline | ~₹0.75 |

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM calls | LangChain + OpenAI (gpt-4o-mini) |
| Dense vector store | FAISS |
| Sparse retrieval | rank-bm25 (BM25Okapi) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Short-term memory | Redis (TTL 24h) |
| API layer | FastAPI + SSE streaming |
| Dashboard | Streamlit + Plotly |
| Evaluation | RAGAS |
| Metrics | Prometheus |
| Deployment | Docker Compose |
| Tracing | LangSmith |

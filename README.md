# LLM Agent System — Multi-Agent Research Assistant

> *"I built a multi-agent research assistant using LangGraph where specialized agents handle retrieval, reasoning, and validation — with hybrid BM25+FAISS retrieval, confidence-driven failure recovery, persistent memory across sessions, and full observability via LangSmith tracing."*

## Architecture

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

## Project Structure

```
├── agent/
│   ├── state.py               # AgentState TypedDict — contract between all nodes
│   ├── graph.py               # LangGraph definition with all conditional edges
│   └── nodes/
│       ├── query_analyzer.py  # Node 1: classify + route
│       ├── retrieval_agent.py # Node 2: tool orchestration
│       ├── reasoning_agent.py # Node 3: decompose + confidence
│       ├── validation_agent.py# Node 4: grounding/consistency/completeness
│       ├── response_generator.py # Node 5: format + cite + stream
│       ├── clarification_agent.py # Node 6: targeted clarification
│       └── memory_manager.py  # Node 7: Redis + FAISS persistence
├── tools/
│   ├── bm25_search.py         # BM25Okapi sparse retrieval
│   ├── vector_search.py       # FAISS dense retrieval
│   ├── web_search.py          # Serper API + DuckDuckGo fallback
│   ├── calculator.py          # sympy safe calculator
│   └── doc_fetcher.py         # URL → chunks
├── retrieval/
│   └── hybrid_retriever.py    # RRF fusion + cross-encoder reranking
├── memory/
│   ├── redis_memory.py        # Short-term (TTL 24h, last 10 turns)
│   └── faiss_memory.py        # Long-term (semantic cache, user facts)
├── prompts/v1/                # Versioned system prompts
├── serving/
│   ├── main.py                # FastAPI: /query, /stream/query, /health, /metrics
│   └── schemas.py             # Pydantic request/response models
├── dashboard/app.py           # Streamlit metrics dashboard
├── scripts/
│   ├── download_data.py       # Fetch arXiv + Wikipedia corpus
│   ├── ingest_corpus.py       # Build BM25 + FAISS indexes
│   └── evaluate.py            # RAGAS evaluation pipeline
├── tests/
│   ├── test_nodes.py          # Unit tests (routing logic, node functions)
│   ├── test_retrieval.py      # BM25, RRF, calculator, context manager
│   └── test_graph.py          # Full integration test (mocked LLMs)
├── docker-compose.yml         # Redis + API + Dashboard
└── Dockerfile
```

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and (optional) LANGCHAIN_API_KEY
```

### 3. Download corpus
```bash
python scripts/download_data.py
```

### 4. Build indexes
```bash
python scripts/ingest_corpus.py
```

### 5. Run the API
```bash
# Development (local)
uvicorn serving.main:app --reload --port 8000

# Or with Docker (includes Redis automatically)
docker-compose up
```

### 6. Run the dashboard
```bash
streamlit run dashboard/app.py
```

### 7. Run tests
```bash
pytest tests/ -v
```

### 8. Evaluate
```bash
python scripts/evaluate.py --max-queries 20
```

## API Usage

### Synchronous query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does LoRA reduce memory compared to full fine-tuning?", "session_id": "user-123"}'
```

### Streaming query (SSE)
```bash
curl -X POST http://localhost:8000/stream/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain chain-of-thought prompting.", "session_id": "user-123"}'
```

### Health check
```bash
curl http://localhost:8000/health
```

## Evaluation Targets

| Metric | Target | How Measured |
|---|---|---|
| Task completion rate | > 85% | % of queries with validated answer |
| RAGAS Faithfulness | > 0.80 | Automated on 100 test queries |
| Context Precision | > 0.70 | RAGAS metric |
| Tool selection accuracy | > 80% | Human labeled 50 queries |
| p50 End-to-end latency | < 4s | Measured per run |
| Avg cost per query | < $0.01 | gpt-4o-mini token tracking |

## Context Window Budget (8K total)

| Component | Budget | Truncation Priority |
|---|---|---|
| System prompt | 500 tokens | Always kept |
| Retrieved chunks | 2,500 tokens | **Trimmed first** |
| Conversation history | 1,500 tokens | Oldest turns dropped |
| Long-term memories | 800 tokens | Summarized |
| Reasoning output | 1,500 tokens | Preserved |
| Buffer | 700 tokens | Safety margin |

## Confidence Routing

| Score | Action |
|---|---|
| ≥ 0.70 | Pass to Validation Agent |
| 0.50–0.69 | Retry with reformulated query (max 2) |
| < 0.50 | Trigger Clarification Agent |

## Observability

Set `LANGCHAIN_TRACING_V2=true` in `.env` to enable automatic LangSmith tracing.
Every node's prompt, response, latency, token count, and retry count is captured per run.

## LangSmith Setup

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_key"
os.environ["LANGCHAIN_PROJECT"] = "research-assistant"
```

## Approximate Cost Per Query (gpt-4o-mini)

| Query Type | Pipeline Path | Approx Cost |
|---|---|---|
| Simple | Analyzer + Response | ~$0.001 |
| Complex (no retry) | Full pipeline | ~$0.007 |
| Complex (with retry) | Full pipeline + 1 retry | ~$0.012 |
| Ambiguous + clarification | Clarification + full pipeline | ~$0.009 |

## Tech Stack

- **LangGraph** — agent orchestration with typed state and conditional edges
- **LangChain + OpenAI** — LLM calls (gpt-4o-mini)
- **FAISS** — dense vector store for corpus + long-term memory
- **rank-bm25** — sparse keyword retrieval
- **sentence-transformers** — embeddings (all-MiniLM-L6-v2)
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — reranking
- **Redis** — short-term session memory (TTL 24h)
- **FastAPI + SSE** — REST + streaming API
- **Streamlit + Plotly** — live metrics dashboard
- **RAGAS** — automated RAG evaluation
- **Prometheus** — metrics endpoint
- **Docker Compose** — containerized deployment
- **LangSmith** — full observability and tracing

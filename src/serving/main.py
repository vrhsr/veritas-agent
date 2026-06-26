"""
FastAPI Serving Layer
Exposes the agent graph via REST and SSE streaming endpoints.

Endpoints:
    POST /query           → synchronous, returns full JSON response
    POST /stream/query    → streams tokens as Server-Sent Events
    GET  /health          → liveness check
    GET  /metrics         → Prometheus-compatible metrics
"""
import os
from dotenv import load_dotenv

load_dotenv()
import json
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog

from agent.graph import run_query, astream_query
from agent.nodes.clarification_agent import inject_clarification_answer
from serving.schemas import QueryRequest, QueryResponse, ClarificationRequest
from utils.logger import get_logger
from utils.cost import format_cost

logger = get_logger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
QUERY_COUNTER = Counter("agent_queries_total", "Total queries processed", ["query_type"])
QUERY_LATENCY = Histogram(
    "agent_query_latency_seconds",
    "Query end-to-end latency",
    buckets=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0],
)
RETRY_COUNTER = Counter("agent_retries_total", "Total retrieval retries triggered")
VALIDATION_COUNTER = Counter("agent_validations_total", "Validation outcomes", ["result"])
COST_HISTOGRAM = Histogram(
    "agent_cost_inr",
    "Cost per query in USD",
    buckets=[0.0001, 0.001, 0.005, 0.01, 0.02, 0.05],
)

# In-memory metrics store for Streamlit dashboard
_metrics_store: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("agent_api_starting", version="1.0.0")
    # Warm up embedding model on startup
    try:
        from tools.vector_search import get_embedding
        get_embedding("warmup")
        logger.info("embedding_model_warmed_up")
    except Exception as e:
        logger.warning("warmup_failed", error=str(e))
    yield
    logger.info("agent_api_shutting_down")


app = FastAPI(
    title="Research Assistant Agent API",
    description="Multi-agent RAG system with hybrid retrieval, confidence-driven validation, and persistent memory.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (demo UI)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Serve the demo UI."""
    demo_path = _static_dir / "demo.html"
    if demo_path.exists():
        return FileResponse(str(demo_path), media_type="text/html")
    return JSONResponse({"message": "Veritas Agent API", "docs": "/docs"})


@app.get("/api/architecture")
async def architecture():
    """Return the agent graph topology for the interactive diagram."""
    return {
        "nodes": [
            {"id": "memory_loader", "label": "Memory Loader", "description": "Load Redis history + FAISS long-term memories"},
            {"id": "query_analyzer", "label": "Query Analyzer", "description": "Classify: simple / complex / ambiguous"},
            {"id": "retrieval_agent", "label": "Retrieval Agent", "description": "BM25 + FAISS → RRF fusion → Cross-encoder rerank"},
            {"id": "reasoning_agent", "label": "Reasoning Agent", "description": "Decompose → Answer → Gap detection → Confidence score"},
            {"id": "validation_agent", "label": "Validation Agent", "description": "Grounding + Consistency + Completeness checks"},
            {"id": "response_generator", "label": "Response Generator", "description": "Format + cite sources + stream output"},
            {"id": "clarification_agent", "label": "Clarification Agent", "description": "Generate targeted clarification questions"},
            {"id": "memory_manager", "label": "Memory Manager", "description": "Persist to Redis (short-term) + FAISS (long-term)"},
        ],
        "edges": [
            {"from": "memory_loader", "to": "query_analyzer", "label": "always"},
            {"from": "query_analyzer", "to": "retrieval_agent", "label": "simple/complex"},
            {"from": "query_analyzer", "to": "clarification_agent", "label": "ambiguous"},
            {"from": "retrieval_agent", "to": "response_generator", "label": "simple"},
            {"from": "retrieval_agent", "to": "reasoning_agent", "label": "complex"},
            {"from": "reasoning_agent", "to": "validation_agent", "label": "confidence ≥ 0.7"},
            {"from": "reasoning_agent", "to": "retrieval_agent", "label": "confidence 0.5-0.69 (retry)"},
            {"from": "reasoning_agent", "to": "clarification_agent", "label": "confidence < 0.5"},
            {"from": "validation_agent", "to": "response_generator", "label": "pass"},
            {"from": "validation_agent", "to": "retrieval_agent", "label": "fail + retry < max"},
            {"from": "response_generator", "to": "memory_manager", "label": "always"},
            {"from": "clarification_agent", "to": "memory_manager", "label": "save & pause"},
        ],
        "confidence_routing": {
            "pass": {"threshold": 0.7, "action": "→ Validation Agent"},
            "retry": {"range": [0.5, 0.69], "action": "→ Retry retrieval (max 2)"},
            "clarify": {"threshold": 0.5, "action": "→ Clarification Agent"},
        }
    }


@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok", "timestamp": time.time(), "version": "1.0.0"}


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/json")
async def metrics_json():
    """JSON metrics for Streamlit dashboard."""
    return {"metrics": _metrics_store[-100:]}  # Last 100 runs


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Synchronous query endpoint.
    Runs the full agent graph and returns the complete response.
    """
    session_id = request.session_id or str(uuid.uuid4())
    start = time.perf_counter()

    logger.info("api_query_received", query=request.query[:80], session_id=session_id)

    try:
        # Run the graph
        result = await asyncio.get_event_loop().run_in_executor(
            None, run_query, request.query, session_id
        )

        elapsed = time.perf_counter() - start

        # Record metrics
        query_type = result.get("query_type", "unknown")
        QUERY_COUNTER.labels(query_type=query_type).inc()
        QUERY_LATENCY.observe(elapsed)
        COST_HISTOGRAM.observe(result.get("cost_inr", 0.0))
        if result.get("retry_count", 0) > 0:
            RETRY_COUNTER.inc(result["retry_count"])
        VALIDATION_COUNTER.labels(
            result="pass" if result.get("validation_passed") else "fail"
        ).inc()

        # Append to dashboard store
        run_record = {
            "timestamp": time.time(),
            "query": request.query[:100],
            "query_type": query_type,
            "latency_s": round(elapsed, 3),
            "confidence": result.get("confidence", 0.0),
            "validation_passed": result.get("validation_passed", False),
            "retry_count": result.get("retry_count", 0),
            "cost_inr": result.get("cost_inr", 0.0),
            "node_latencies": result.get("node_latencies", {}),
        }
        _metrics_store.append(run_record)

        # Handle clarification pause
        if result.get("awaiting_clarification"):
            return QueryResponse(
                session_id=session_id,
                query_type="awaiting_clarification",
                final_answer=result.get("clarification_question", ""),
                cited_sources=[],
                confidence=0.0,
                validation_passed=False,
                retry_count=0,
                cost_inr=result.get("cost_inr", 0.0),
                latency_s=round(elapsed, 3),
                node_latencies=result.get("node_latencies", {}),
                awaiting_clarification=True,
            )

        logger.info(
            "api_query_done",
            session_id=session_id,
            latency_s=round(elapsed, 3),
            cost=format_cost(result.get("cost_inr", 0.0)),
            confidence=result.get("confidence", 0.0),
        )

        return QueryResponse(
            session_id=session_id,
            query_type=result.get("query_type", ""),
            final_answer=result.get("final_answer", ""),
            cited_sources=result.get("cited_sources", []),
            confidence=result.get("confidence", 0.0),
            validation_passed=result.get("validation_passed", False),
            retry_count=result.get("retry_count", 0),
            cost_inr=result.get("cost_inr", 0.0),
            latency_s=round(elapsed, 3),
            node_latencies=result.get("node_latencies", {}),
            awaiting_clarification=False,
        )

    except Exception as e:
        logger.error("api_query_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/stream/query")
async def stream_query_endpoint(request: QueryRequest):
    """
    Streaming query endpoint using Server-Sent Events.
    Streams tokens as they generate.
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("api_stream_query", query=request.query[:80], session_id=session_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for token in astream_query(request.query, session_id):
                yield f"data: {json.dumps({'token': token, 'session_id': session_id})}\n\n"
            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return EventSourceResponse(event_generator())


@app.post("/clarify")
async def clarify_endpoint(request: ClarificationRequest):
    """
    Inject a clarification answer and continue the graph.
    Called after /query returns awaiting_clarification=True.
    """
    logger.info("api_clarify", session_id=request.session_id)

    # In production, reconstruct state from Redis using session_id
    # For demo, run fresh with clarification injected into query
    enriched_query = f"{request.original_query} [Clarification: {request.answer}]"

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, run_query, enriched_query, request.session_id
        )

        return QueryResponse(
            session_id=request.session_id,
            query_type=result.get("query_type", ""),
            final_answer=result.get("final_answer", ""),
            cited_sources=result.get("cited_sources", []),
            confidence=result.get("confidence", 0.0),
            validation_passed=result.get("validation_passed", False),
            retry_count=result.get("retry_count", 0),
            cost_inr=result.get("cost_inr", 0.0),
            latency_s=0.0,
            node_latencies=result.get("node_latencies", {}),
            awaiting_clarification=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

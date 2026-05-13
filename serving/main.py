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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
    "agent_cost_usd",
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


# ── Endpoints ────────────────────────────────────────────────────────────────

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
        COST_HISTOGRAM.observe(result.get("cost_usd", 0.0))
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
            "cost_usd": result.get("cost_usd", 0.0),
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
                cost_usd=result.get("cost_usd", 0.0),
                latency_s=round(elapsed, 3),
                node_latencies=result.get("node_latencies", {}),
                awaiting_clarification=True,
            )

        logger.info(
            "api_query_done",
            session_id=session_id,
            latency_s=round(elapsed, 3),
            cost=format_cost(result.get("cost_usd", 0.0)),
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
            cost_usd=result.get("cost_usd", 0.0),
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
            cost_usd=result.get("cost_usd", 0.0),
            latency_s=0.0,
            node_latencies=result.get("node_latencies", {}),
            awaiting_clarification=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

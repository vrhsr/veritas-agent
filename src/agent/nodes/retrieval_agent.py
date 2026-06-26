"""
Node 2: Retrieval Agent
Decides HOW to retrieve — selects from 5 tools, orders calls based on query type.
Tools: BM25 search, FAISS vector search, web search, calculator, document fetcher.
Uses hybrid retrieval (RRF fusion + cross-encoder reranking).

On retry runs, enforces a 40% chunk diversity constraint:
If the new retrieval returns < 40% novel chunks vs the previous attempt,
the corpus genuinely doesn't have better information — the graph should stop retrying.
This is tracked via the `low_chunk_diversity` flag in state.
"""
import time
import os
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from tools.bm25_search import bm25_search
from tools.vector_search import vector_search
from tools.web_search import web_search
from tools.calculator import safe_calculate
from tools.doc_fetcher import fetch_document
from retrieval.hybrid_retriever import hybrid_retrieve
from utils.logger import get_logger
from utils.cost import estimate_cost
from utils.token_counter import count_tokens

logger = get_logger(__name__)
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "prompts/v1"))
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "5"))
TOOL_TIMEOUT = float(os.getenv("TOOL_TIMEOUT_S", "5.0"))


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def retrieval_agent_node(state: AgentState) -> AgentState:
    """
    Node 2: Orchestrates retrieval using the appropriate tools.
    Builds a reformulated query if this is a retry (uses gaps from reasoning).
    """
    start = time.perf_counter()
    retry_count = state.get("retry_count", 0)
    query = state["original_query"]

    # On retry, reformulate query to target specific gaps
    if retry_count > 0 and state.get("gaps"):
        gaps_str = "; ".join(state["gaps"][:3])
        query = f"{query} [Focus on: {gaps_str}]"
        logger.info("retrieval_reformulated_query", retry=retry_count, query=query[:100])

    required_tools = state.get("required_tools", ["vector_search", "bm25_search"])
    retrieval_strategy = "hybrid"
    chunks: List[dict] = []

    # ── Hybrid retrieval (BM25 + FAISS with RRF fusion + reranking) ──────────
    if "vector_search" in required_tools or "bm25_search" in required_tools:
        try:
            # On second retry, widen search by reducing similarity threshold
            threshold_override = 0.2 if retry_count >= 2 else None
            chunks = hybrid_retrieve(query, top_k=TOP_K, threshold_override=threshold_override)
            retrieval_strategy = "hybrid"
            logger.info("hybrid_retrieval_done", chunks_returned=len(chunks))
        except Exception as e:
            logger.warning("hybrid_retrieval_error", error=str(e))
            state["error_log"] = state.get("error_log", []) + [f"Hybrid retrieval error: {e}"]

    # ── Web search (if needed or if hybrid returned nothing) ─────────────────
    if "web_search" in required_tools or (not chunks and retry_count >= 1):
        try:
            web_results = web_search(query, max_results=3)
            if web_results:
                chunks.extend(web_results)
                retrieval_strategy = "web" if not chunks else "hybrid+web"
                logger.info("web_search_done", results=len(web_results))
        except Exception as e:
            logger.warning("web_search_error", error=str(e))
            state["error_log"] = state.get("error_log", []) + [f"Web search error: {e}"]

    # ── Calculator (if math required) ────────────────────────────────────────
    calculation_result = None
    if "calculator" in required_tools:
        try:
            calc_query = _extract_math_expression(query)
            if calc_query:
                calculation_result = safe_calculate(calc_query)
                logger.info("calculator_result", result=calculation_result)
        except Exception as e:
            logger.warning("calculator_error", error=str(e))
            state["error_log"] = state.get("error_log", []) + [f"Calculator error: {e}"]

    # ── Document fetcher (if URL present in query) ────────────────────────────
    if "doc_fetcher" in required_tools:
        url = _extract_url(query)
        if url:
            try:
                doc_chunks = fetch_document(url)
                chunks.extend(doc_chunks[:2])
                logger.info("doc_fetcher_done", url=url, chunks=len(doc_chunks))
            except Exception as e:
                logger.warning("doc_fetcher_error", error=str(e))

    # Inject calculation result as a pseudo-chunk if available
    if calculation_result:
        chunks.insert(0, {
            "text": f"Mathematical calculation result: {calculation_result}",
            "source": "calculator",
            "score": 1.0,
        })

    # ── Chunk diversity check on retry ──────────────────────────────────────
    diversity_ok = True
    if retry_count > 0 and chunks:
        prev_chunks = state.get("retrieved_chunks", [])
        prev_texts = {c["text"][:80] for c in prev_chunks}
        new_texts = {c["text"][:80] for c in chunks}
        novel = new_texts - prev_texts
        diversity_ratio = len(novel) / len(new_texts) if new_texts else 0.0
        diversity_ok = diversity_ratio >= 0.40
        logger.info(
            "chunk_diversity_check",
            retry=retry_count,
            total=len(new_texts),
            novel=len(novel),
            ratio=round(diversity_ratio, 2),
            diversity_ok=diversity_ok,
        )
        if not diversity_ok:
            logger.warning(
                "low_chunk_diversity",
                reason="< 40% novel chunks — corpus likely exhausted on this query",
            )

    elapsed = time.perf_counter() - start

    updates = {
        "retrieved_chunks": chunks,
        "retrieval_strategy": retrieval_strategy,
        "low_chunk_diversity": not diversity_ok,
        "node_latencies": {**state.get("node_latencies", {}), "retrieval_agent": round(elapsed, 3)},
    }
    return {**state, **updates}


def _extract_math_expression(query: str) -> str | None:
    """Heuristic: look for arithmetic-looking substrings in the query."""
    import re
    match = re.search(r"[\d\.\s\+\-\*\/\(\)\^%]+", query)
    if match:
        expr = match.group().strip()
        if any(op in expr for op in ["+", "-", "*", "/", "^", "%"]) and len(expr) > 2:
            return expr
    return None


def _extract_url(query: str) -> str | None:
    """Extract the first URL from the query string."""
    import re
    match = re.search(r"https?://[^\s]+", query)
    return match.group() if match else None


def route_after_retrieval(state: AgentState) -> str:
    """Conditional edge: simple queries skip reasoning and go straight to response generator."""
    if state.get("query_type") == "simple":
        return "response_generator"
    return "reasoning_agent"

"""
AgentState — the shared state object that flows through every node in the LangGraph.
This is the typed contract between all agents. No hidden data passing, no global variables.
"""
from typing import TypedDict, Optional, List, Annotated
import operator


class AgentState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    original_query: str
    session_id: str

    # ── Query Analysis (Node 1) ───────────────────────────────────────────────
    query_type: str               # "simple" | "complex" | "ambiguous"
    named_entities: List[str]
    required_tools: List[str]
    complexity_score: float

    # ── Clarification (Node 6) ────────────────────────────────────────────────
    clarification_question: str
    clarification_answer: str
    awaiting_clarification: bool
    intent_slots: dict             # {"task": ..., "entity": ..., "metric": ..., "timeframe": ...}

    # ── Retrieval (Node 2) ────────────────────────────────────────────────────
    retrieved_chunks: List[dict]  # [{text, source, score}]
    retrieval_strategy: str       # "hybrid" | "web" | "cached"
    low_chunk_diversity: bool     # True if retry returned < 40% novel chunks
    ood_max_similarity: float     # Max cosine similarity to out-of-domain set

    # ── Reasoning (Node 3) ────────────────────────────────────────────────────
    sub_questions: List[str]
    sub_answers: List[str]
    gaps: List[str]
    reasoning_output: str
    confidence: float

    # ── Validation (Node 4) ───────────────────────────────────────────────────
    validation_passed: bool
    validation_failures: List[str]
    retry_count: int

    # ── Response (Node 5) ─────────────────────────────────────────────────────
    final_answer: str
    cited_sources: List[str]

    # ── Memory (Node 7) ───────────────────────────────────────────────────────
    relevant_memories: List[str]

    # ── Observability ─────────────────────────────────────────────────────────
    cost_usd: float
    node_latencies: dict          # {node_name: seconds}
    token_usage: dict             # {node_name: {input, output}}
    error_log: List[str]
    prompt_version: str

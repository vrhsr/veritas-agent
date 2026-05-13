"""
LangGraph Graph Definition
Wires all 7 nodes together with conditional edges.
Supports cycles (validation → retrieval retry loop).
"""
import os
import uuid
from typing import AsyncGenerator

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.query_analyzer import query_analyzer_node, route_after_query_analyzer
from agent.nodes.retrieval_agent import retrieval_agent_node, route_after_retrieval
from agent.nodes.reasoning_agent import reasoning_agent_node, route_after_reasoning
from agent.nodes.validation_agent import validation_agent_node, route_after_validation
from agent.nodes.response_generator import response_generator_node, astream_response_generator
from agent.nodes.clarification_agent import clarification_agent_node
from agent.nodes.memory_manager import memory_manager_node, load_relevant_memories
from utils.logger import get_logger

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """
    Builds and compiles the full LangGraph agent graph.

    Graph topology:
        memory_loader → query_analyzer
            ├── ambiguous → clarification_agent → retrieval_agent
            └── simple/complex → retrieval_agent
                                   ├── simple → response_generator
                                   └── complex → reasoning_agent
                                                 ├── confidence ≥ 0.7 → validation_agent
                                                 ├── confidence 0.5–0.69 → retrieval_agent (retry)
                                                 └── confidence < 0.5  → clarification_agent
                         validation_agent
                              ├── pass → response_generator
                              ├── fail + retry < max → retrieval_agent
                              └── fail + retry = max → response_generator
                         response_generator → memory_manager → END
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("memory_loader", load_relevant_memories)
    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("retrieval_agent", retrieval_agent_node)
    graph.add_node("reasoning_agent", reasoning_agent_node)
    graph.add_node("validation_agent", validation_agent_node)
    graph.add_node("response_generator", response_generator_node)
    graph.add_node("clarification_agent", clarification_agent_node)
    graph.add_node("memory_manager", memory_manager_node)

    # ── Entry point ────────────────────────────────────────────────────────────
    graph.set_entry_point("memory_loader")

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_edge("memory_loader", "query_analyzer")

    # Node 1 → conditional routing
    graph.add_conditional_edges(
        "query_analyzer",
        route_after_query_analyzer,
        {
            "clarification_agent": "clarification_agent",
            "retrieval_agent": "retrieval_agent",
        },
    )

    # Clarification → memory_manager (saves question to Redis)
    graph.add_edge("clarification_agent", "memory_manager")

    # Retrieval → reasoning or response (if simple)
    graph.add_conditional_edges(
        "retrieval_agent",
        route_after_retrieval,
        {
            "response_generator": "response_generator",
            "reasoning_agent": "reasoning_agent",
        },
    )

    # Node 3 → conditional routing
    graph.add_conditional_edges(
        "reasoning_agent",
        route_after_reasoning,
        {
            "validation_agent": "validation_agent",
            "retrieval_agent": "retrieval_agent",
            "clarification_agent": "clarification_agent",
        },
    )

    # Node 4 → conditional routing
    graph.add_conditional_edges(
        "validation_agent",
        route_after_validation,
        {
            "response_generator": "response_generator",
            "retrieval_agent": "retrieval_agent",
        },
    )

    # Response → memory → END
    graph.add_edge("response_generator", "memory_manager")
    graph.add_edge("memory_manager", END)

    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────────────

_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def _default_state(query: str, session_id: str | None = None) -> AgentState:
    return AgentState(
        original_query=query,
        session_id=session_id or str(uuid.uuid4()),
        query_type="",
        named_entities=[],
        required_tools=[],
        complexity_score=0.0,
        clarification_question="",
        clarification_answer="",
        awaiting_clarification=False,
        intent_slots={"task": None, "entity": None, "metric": None, "timeframe": None},
        retrieved_chunks=[],
        retrieval_strategy="",
        low_chunk_diversity=False,
        ood_max_similarity=1.0,
        sub_questions=[],
        sub_answers=[],
        gaps=[],
        reasoning_output="",
        confidence=0.0,
        validation_passed=False,
        validation_failures=[],
        retry_count=0,
        final_answer="",
        cited_sources=[],
        relevant_memories=[],
        cost_usd=0.0,
        node_latencies={},
        token_usage={},
        error_log=[],
        prompt_version=os.getenv("PROMPT_VERSION", "v1"),
    )


def run_query(query: str, session_id: str | None = None) -> AgentState:
    """Synchronous query execution. Returns final state."""
    graph = _get_graph()
    initial_state = _default_state(query, session_id)
    logger.info("graph_run_start", query=query[:80], session_id=initial_state["session_id"])
    result = graph.invoke(initial_state)
    logger.info("graph_run_done", cost_usd=result.get("cost_usd", 0), answer_len=len(result.get("final_answer", "")))
    return result


async def astream_query(query: str, session_id: str | None = None) -> AsyncGenerator[str, None]:
    """
    Async streaming query execution.
    Runs the graph through all nodes except response_generator,
    then streams the final response token by token.
    """
    graph = _get_graph()
    initial_state = _default_state(query, session_id)

    # Run all nodes up to (but not including) response_generator
    # We use the graph normally — response_generator_node runs synchronously
    # but we then stream separately for the response.
    # For full streaming: run graph with streaming=True config.
    result = graph.invoke(initial_state, config={"recursion_limit": 25})

    # Stream the response
    async for token in astream_response_generator(result):
        yield token

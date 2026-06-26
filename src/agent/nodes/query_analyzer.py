"""
Node 1: Query Analyzer
Classifies incoming query and extracts structured metadata before any LLM work happens.
Routes: simple → response_generator | complex → retrieval_agent | ambiguous → clarification_agent
"""
import json
import time
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from utils.logger import get_logger
from utils.cost import estimate_cost
from utils.token_counter import count_tokens

logger = get_logger(__name__)

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "prompts/v1"))


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def query_analyzer_node(state: AgentState) -> AgentState:
    """
    Node 1: Intent Slot Filler + Query Classifier.

    Two-phase operation:
    1. Load current intent slots from Redis for this session.
    2. Ask the LLM to update slots from the latest message AND assess completeness.
    3. Merge slots back to Redis. If all required slots filled, rewrite original_query
       to the resolved_query so downstream nodes always receive a clear, full query.
    """
    start = time.perf_counter()
    logger.info("query_analyzer", query=state["original_query"], session=state["session_id"])

    # Load intent slot state persisted from previous turns
    from memory.redis_memory import RedisMemory
    _redis = RedisMemory()
    current_slots = _redis.get_intent(state["session_id"])

    llm = ChatOpenAI(
        model=os.getenv("FAST_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini")),
        temperature=0,
        response_format={"type": "json_object"},
    )

    system_prompt = load_prompt("query_analyzer")

    # Inject conversation history + current slot state into the user message
    context_parts = []
    memories = state.get("relevant_memories", [])
    if memories:
        context_parts.append("Conversation History:\n" + "\n".join(memories[-3:]))

    # Only include non-null slots to save tokens
    filled_slots = {k: v for k, v in current_slots.items() if v is not None}
    if filled_slots:
        context_parts.append(f"Current Intent Slots (already known): {json.dumps(filled_slots)}")

    context_parts.append(f"Latest User Message: {state['original_query']}")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="\n\n".join(context_parts)),
    ]

    response = llm.invoke(messages)
    raw = response.content

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("query_analyzer_json_parse_error", raw=raw[:200])
        parsed = {
            "query_type": "complex",
            "intent_slots": current_slots,
            "resolved_query": None,
            "named_entities": [],
            "required_tools": ["vector_search", "bm25_search"],
            "complexity_score": 0.7,
        }

    elapsed = time.perf_counter() - start
    input_tokens = count_tokens(str(messages))
    output_tokens = count_tokens(raw)
    cost = estimate_cost(input_tokens, output_tokens)

    query_type = parsed.get("query_type", "complex")
    new_slots = parsed.get("intent_slots", {})
    resolved_query = parsed.get("resolved_query")

    # Persist updated slot state to Redis
    _redis.set_intent(state["session_id"], new_slots)

    # If the LLM produced a clean resolved_query, use it as the working query.
    # This means "clients" → "Prioritize clients by fastest growth potential"
    working_query = state["original_query"]
    if resolved_query and query_type != "ambiguous":
        working_query = resolved_query
        logger.info(
            "query_resolved",
            original=state["original_query"][:80],
            resolved=resolved_query[:80],
        )
        # Clear the intent slots so they don't bleed into the next conversation turn
        _redis.clear_intent(state["session_id"])

    logger.info(
        "query_analyzer_done",
        query_type=query_type,
        complexity=parsed.get("complexity_score"),
        slots=new_slots,
        latency_s=round(elapsed, 3),
    )

    updates: dict = {
        "original_query": working_query,
        "query_type": query_type,
        "intent_slots": {**current_slots, **new_slots},
        "named_entities": parsed.get("named_entities", []),
        "required_tools": parsed.get("required_tools", ["vector_search", "bm25_search"]),
        "complexity_score": float(parsed.get("complexity_score", 0.5)),
        "cost_inr": state.get("cost_inr", 0.0) + cost,
        "prompt_version": PROMPT_VERSION,
        "node_latencies": {**state.get("node_latencies", {}), "query_analyzer": round(elapsed, 3)},
        "token_usage": {
            **state.get("token_usage", {}),
            "query_analyzer": {"input": input_tokens, "output": output_tokens},
        },
    }
    return {**state, **updates}


def route_after_query_analyzer(state: AgentState) -> str:
    """Conditional edge: ambiguous → clarification, all else → retrieval."""
    if state.get("query_type") == "ambiguous":
        return "clarification_agent"
    return "retrieval_agent"


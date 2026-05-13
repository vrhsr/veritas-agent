"""
Node 7: Memory Manager
Runs after every successful response.
Short-term: Redis (conversation history, TTL 24h)
Long-term: FAISS (user facts, Q&A semantic cache, preferences)

Key behaviors:
- Per-user FAISS namespacing: user_id = session_id prefix (prevents cross-user leakage)
- Selective LLM extraction: runs a lightweight LLM call to extract facts worth keeping,
  rather than blindly storing the raw Q+A. This keeps long-term memory signal-dense.
- Only stores validated, high-confidence answers (≥ 0.7).
"""
import time
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from memory.redis_memory import RedisMemory
from memory.faiss_memory import FAISSMemory
from utils.logger import get_logger

logger = get_logger(__name__)

TOP_K_MEMORY = int(os.getenv("TOP_K_MEMORY", "3"))
_redis_memory: RedisMemory | None = None
_faiss_cache: dict[str, FAISSMemory] = {}  # user_id → FAISSMemory instance


def _get_redis() -> RedisMemory:
    global _redis_memory
    if _redis_memory is None:
        _redis_memory = RedisMemory()
    return _redis_memory


def _get_faiss(user_id: str) -> FAISSMemory:
    """Returns a per-user FAISSMemory instance, cached in memory."""
    if user_id not in _faiss_cache:
        _faiss_cache[user_id] = FAISSMemory(user_id=user_id)
    return _faiss_cache[user_id]


def _extract_memorable_facts(query: str, answer: str) -> list[str]:
    """
    Run a lightweight LLM call to extract facts worth storing in long-term memory.
    Filters out pleasantries, transient context, and volatile information.
    Returns a list of distilled fact strings (may be empty if nothing is worth keeping).
    """
    try:
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
        system = (
            "You extract facts worth remembering from a Q&A exchange for future personalization. "
            "Extract ONLY durable, user-specific facts: preferences, domain context, entities the user cares about, or key findings. "
            "Ignore pleasantries, transient details, and anything likely to change. "
            "If nothing is worth keeping, return an empty list. "
            "Respond ONLY with a JSON array of short fact strings. Example: "
            '[\"User works in fintech\", \"User prefers bullet-point answers\"]'
        )
        user_msg = f"Q: {query}\nA: {answer[:800]}"
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])

        import json
        raw = response.content.strip()
        # Handle markdown-fenced JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        facts = json.loads(raw)
        if isinstance(facts, list):
            return [str(f) for f in facts if f]
        return []
    except Exception as e:
        logger.warning("memory_extraction_error", error=str(e))
        return []


def memory_manager_node(state: AgentState) -> AgentState:
    """
    Node 7: Persists current interaction to Redis + FAISS memory stores.
    Uses selective LLM extraction for long-term memory — only durable facts are kept.
    """
    start = time.perf_counter()
    session_id = state.get("session_id", "default")
    # User ID = session_id (in production this would be an authenticated user ID)
    user_id = session_id
    logger.info("memory_manager", session_id=session_id, user_id=user_id)

    redis = _get_redis()
    faiss = _get_faiss(user_id)

    # ── Short-term: save turn to Redis history ───────────────────────────────
    try:
        answer = state.get("final_answer") or state.get("clarification_question", "")
        turn = {
            "query": state["original_query"],
            "answer": answer,
            "confidence": state.get("confidence", 0.0),
        }
        redis.push_turn(session_id, turn)
    except Exception as e:
        logger.warning("redis_save_error", error=str(e))

    # ── Long-term: selective extraction → FAISS (per-user namespace) ─────────
    try:
        if state.get("validation_passed") and state.get("confidence", 0.0) >= 0.7:
            query = state["original_query"]
            answer = state.get("final_answer", "")

            # LLM extracts only the facts worth keeping, not the raw Q+A
            facts = _extract_memorable_facts(query, answer)

            if facts:
                for fact in facts:
                    faiss.add_memory(fact, metadata={
                        "session_id": session_id,
                        "confidence": state.get("confidence", 0.0),
                        "sources": state.get("cited_sources", []),
                        "origin_query": query[:100],
                    })
                logger.info("memory_facts_stored", user_id=user_id, count=len(facts))
            else:
                # Fallback: store a compact Q&A entry for semantic cache
                qa_text = f"Q: {query}\nA: {answer[:400]}"
                faiss.add_memory(qa_text, metadata={
                    "session_id": session_id,
                    "confidence": state.get("confidence", 0.0),
                })
                logger.debug("memory_qa_stored_fallback", user_id=user_id)
    except Exception as e:
        logger.warning("faiss_save_error", error=str(e))

    elapsed = time.perf_counter() - start
    logger.info("memory_manager_done", latency_s=round(elapsed, 3), faiss_size=faiss.size)

    updates = {
        "node_latencies": {**state.get("node_latencies", {}), "memory_manager": round(elapsed, 3)},
    }
    return {**state, **updates}


def load_relevant_memories(state: AgentState) -> AgentState:
    """
    Entry node: loads relevant memories before the query analyzer runs.
    Retrieves from Redis (conversation history) and user-scoped FAISS (long-term facts).
    Per-user namespacing ensures user A's memories are never surfaced to user B.
    """
    session_id = state.get("session_id", "default")
    user_id = session_id
    query = state.get("original_query", "")
    memories: list[str] = []

    redis = _get_redis()
    faiss = _get_faiss(user_id)

    # Recent conversation turns from Redis
    try:
        history = redis.get_history(session_id, max_turns=5)
        for turn in history[-3:]:  # Last 3 turns
            memories.append(
                f"Previous Q: {turn.get('query', '')}\n"
                f"Previous A: {turn.get('answer', '')[:200]}"
            )
    except Exception as e:
        logger.warning("redis_load_error", error=str(e))

    # Semantic memories from user's private FAISS index
    try:
        faiss_memories = faiss.search_memories(query, top_k=TOP_K_MEMORY)
        memories.extend(faiss_memories)
    except Exception as e:
        logger.warning("faiss_load_error", error=str(e))

    logger.debug("memory_loaded", session_id=session_id, count=len(memories))
    return {**state, "relevant_memories": memories[:TOP_K_MEMORY + 3]}

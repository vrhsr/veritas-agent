"""
Node 3: Reasoning Agent
Core differentiator: decomposes question into sub-questions, answers each from context,
identifies gaps, produces a confidence score that drives the rest of the pipeline.

Routing decisions:
  confidence >= 0.7  → validation_agent
  confidence 0.5–0.69 AND not OOD → retrieval_agent (retry with reformulated query)
  confidence < 0.5 OR OOD query  → clarification_agent
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
from utils.context_manager import build_context_within_budget
from utils.ood_detector import is_out_of_distribution

logger = get_logger(__name__)
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "prompts/v1"))
CONFIDENCE_PASS = float(os.getenv("CONFIDENCE_THRESHOLD_PASS", "0.7"))
CONFIDENCE_CLARIFY = float(os.getenv("CONFIDENCE_THRESHOLD_CLARIFY", "0.5"))


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def reasoning_agent_node(state: AgentState) -> AgentState:
    """
    Node 3: Decomposes → answers sub-questions → identifies gaps → rates confidence.
    Outputs structured JSON with sub_questions, sub_answers, gaps, final_answer, confidence.
    """
    start = time.perf_counter()
    logger.info("reasoning_agent", query=state["original_query"], retry=state.get("retry_count", 0))

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    system_prompt = load_prompt("reasoning_agent")

    # Build context string within token budget
    context_str = build_context_within_budget(
        chunks=state.get("retrieved_chunks", []),
        memories=state.get("relevant_memories", []),
        history=[],  # conversation history injected separately
        max_tokens=2500,
    )

    # Include memories
    memory_str = ""
    if state.get("relevant_memories"):
        memory_str = "\n\nRelevant User Context:\n" + "\n".join(
            f"- {m}" for m in state["relevant_memories"][:3]
        )

    user_message = (
        f"Question: {state['original_query']}\n\n"
        f"Retrieved Context:\n{context_str}"
        f"{memory_str}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw = response.content

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("reasoning_json_parse_error", raw=raw[:300])
        parsed = {
            "sub_questions": [state["original_query"]],
            "sub_answers": ["Unable to parse structured reasoning."],
            "gaps": ["JSON parse failed"],
            "final_answer": raw,
            "confidence": 0.4,
        }

    confidence = float(parsed.get("confidence", 0.4))
    elapsed = time.perf_counter() - start
    input_tokens = count_tokens(str(messages))
    output_tokens = count_tokens(raw)
    cost = estimate_cost(input_tokens, output_tokens)

    logger.info(
        "reasoning_agent_done",
        confidence=confidence,
        sub_questions=len(parsed.get("sub_questions", [])),
        gaps=len(parsed.get("gaps", [])),
        latency_s=round(elapsed, 3),
    )

    updates = {
        "sub_questions": parsed.get("sub_questions", []),
        "sub_answers": parsed.get("sub_answers", []),
        "gaps": parsed.get("gaps", []),
        "reasoning_output": parsed.get("final_answer", ""),
        "confidence": confidence,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
        "node_latencies": {**state.get("node_latencies", {}), "reasoning_agent": round(elapsed, 3)},
        "token_usage": {
            **state.get("token_usage", {}),
            "reasoning_agent": {"input": input_tokens, "output": output_tokens},
        },
    }
    return {**state, **updates}


def route_after_reasoning(state: AgentState) -> str:
    """
    Conditional edge after reasoning:
    >= 0.7                          → validation_agent
    0.5–0.69, not OOD, retry < max → retrieval_agent (reformulated query)
    OOD query detected              → clarification_agent (skip retries entirely)
    < 0.5 or max retries hit        → clarification_agent

    OOD gate: if the corpus has no relevant content (max cosine similarity < 0.30),
    retrying is pointless — skip straight to clarification/uncertainty response.
    This prevents the P99 cost spiral from OOD queries burning max retry tokens.
    """
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = int(os.getenv("MAX_RETRIES", "2"))

    if confidence >= CONFIDENCE_PASS:
        return "validation_agent"

    # OOD gate: check before deciding to retry
    if confidence < CONFIDENCE_PASS and retry_count == 0:
        query = state.get("original_query", "")
        ood, max_sim = is_out_of_distribution(query)
        if ood:
            logger.info(
                "ood_gate_triggered",
                confidence=confidence,
                max_corpus_similarity=round(max_sim, 4),
                reason="corpus cannot answer — skipping retries",
            )
            return "clarification_agent"

    if confidence >= CONFIDENCE_CLARIFY and retry_count < max_retries:
        logger.info("routing_retry", confidence=confidence, retry=retry_count)
        return "retrieval_agent"
    else:
        return "clarification_agent"

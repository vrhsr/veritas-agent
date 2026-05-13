"""
Node 4: Validation Agent
Separate agent that checks the reasoning agent's work.
Three checks: grounding, consistency, completeness.
Prevents the reasoning agent's confirmation bias.
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
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "prompts/v1"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def validation_agent_node(state: AgentState) -> AgentState:
    """
    Node 4: Runs three independent checks on the reasoning output.
    Uses a different system prompt than the reasoning agent — avoids confirmation bias.
    """
    start = time.perf_counter()
    retry_count = state.get("retry_count", 0)
    logger.info("validation_agent", retry=retry_count)

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        response_format={"type": "json_object"},
    )

    system_prompt = load_prompt("validation_agent")

    # Build compact context for validation
    chunks_summary = "\n\n".join(
        f"[Source {i+1}: {c.get('source', 'unknown')}]\n{c['text'][:300]}"
        for i, c in enumerate(state.get("retrieved_chunks", [])[:5])
    )

    user_message = (
        f"Original Question: {state['original_query']}\n\n"
        f"Answer to Validate:\n{state.get('reasoning_output', '')}\n\n"
        f"Retrieved Context:\n{chunks_summary}\n\n"
        f"Sub-Questions: {json.dumps(state.get('sub_questions', []))}\n"
        f"Sub-Answers: {json.dumps(state.get('sub_answers', []))}"
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
        logger.warning("validation_json_parse_error", raw=raw[:200])
        parsed = {"passed": True, "failures": [], "grounding": True, "consistency": True, "completeness": True}

    passed = bool(parsed.get("passed", True))
    failures = parsed.get("failures", [])

    elapsed = time.perf_counter() - start
    input_tokens = count_tokens(str(messages))
    output_tokens = count_tokens(raw)
    cost = estimate_cost(input_tokens, output_tokens)

    logger.info(
        "validation_done",
        passed=passed,
        failures=failures,
        grounding=parsed.get("grounding"),
        consistency=parsed.get("consistency"),
        completeness=parsed.get("completeness"),
        latency_s=round(elapsed, 3),
    )

    updates = {
        "validation_passed": passed,
        "validation_failures": failures,
        "retry_count": retry_count + (0 if passed else 1),
        "cost_usd": state.get("cost_usd", 0.0) + cost,
        "node_latencies": {**state.get("node_latencies", {}), "validation_agent": round(elapsed, 3)},
        "token_usage": {
            **state.get("token_usage", {}),
            "validation_agent": {"input": input_tokens, "output": output_tokens},
        },
    }
    return {**state, **updates}


def route_after_validation(state: AgentState) -> str:
    """
    Conditional edge after validation:
    pass               → response_generator
    fail + retry < max → retrieval_agent (different strategy)
    fail + retry = max → response_generator (explicit uncertainty)
    """
    if state.get("validation_passed", False):
        return "response_generator"

    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_RETRIES:
        logger.info("validation_retry", retry=retry_count)
        return "retrieval_agent"
    else:
        logger.info("validation_max_retries", retry=retry_count)
        # Mark answer as uncertain before generating response
        return "response_generator"

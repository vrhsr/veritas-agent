"""
Node 5: Response Generator
Formats validated answer with inline citations, confidence disclosure, cost tracking.
Supports both synchronous and streaming (async generator) output.
"""
import time
import os
from pathlib import Path
from typing import AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from utils.logger import get_logger
from utils.cost import estimate_cost
from utils.token_counter import count_tokens

logger = get_logger(__name__)
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "prompts/v1"))
BORDERLINE_LOW = 0.7
BORDERLINE_HIGH = 0.75


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def response_generator_node(state: AgentState) -> AgentState:
    """
    Node 5: Formats the final answer with citations, confidence disclosure,
    and cost tracking. For streaming, use astream_response_generator.
    """
    start = time.perf_counter()
    logger.info("response_generator", validation_passed=state.get("validation_passed"))

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.3,
    )

    system_prompt = load_prompt("response_generator")

    # Build source citations
    sources = state.get("retrieved_chunks", [])
    cited_sources = list({c.get("source", "unknown") for c in sources if c.get("source")})

    sources_str = "\n".join(
        f"[{i+1}] {c.get('source', 'unknown')}: {c['text'][:150]}..."
        for i, c in enumerate(sources[:5])
    )

    # Confidence disclosure for borderline cases
    confidence = state.get("confidence", 1.0)
    confidence_note = ""
    if BORDERLINE_LOW <= confidence <= BORDERLINE_HIGH:
        confidence_note = (
            f"\n\n⚠️ Note: My confidence in this answer is {confidence:.0%}. "
            "Some aspects may benefit from additional verification."
        )

    # Max retries exceeded — explicit uncertainty
    validation_failed_final = (
        not state.get("validation_passed", True)
        and state.get("retry_count", 0) >= int(os.getenv("MAX_RETRIES", "2"))
    )
    uncertainty_note = ""
    if validation_failed_final:
        failures = state.get("validation_failures", [])
        uncertainty_note = (
            "\n\n⚠️ I was unable to fully validate this answer after multiple retrieval attempts. "
            f"Issues detected: {', '.join(failures)}. "
            "Please verify key claims independently."
        )

    reasoning = state.get("reasoning_output", "")
    user_message = (
        f"Question: {state['original_query']}\n\n"
        f"Reasoning and Answer:\n{reasoning}\n\n"
        f"Sources:\n{sources_str}\n\n"
        f"Format the answer with numbered inline citations matching the sources above."
        f"{confidence_note}{uncertainty_note}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    final_answer = response.content.strip()

    elapsed = time.perf_counter() - start
    input_tokens = count_tokens(str(messages))
    output_tokens = count_tokens(final_answer)
    cost = estimate_cost(input_tokens, output_tokens)
    total_cost = state.get("cost_usd", 0.0) + cost

    logger.info(
        "response_generator_done",
        answer_len=len(final_answer),
        total_cost_usd=round(total_cost, 6),
        latency_s=round(elapsed, 3),
    )

    updates = {
        "final_answer": final_answer,
        "cited_sources": cited_sources,
        "cost_usd": total_cost,
        "node_latencies": {**state.get("node_latencies", {}), "response_generator": round(elapsed, 3)},
        "token_usage": {
            **state.get("token_usage", {}),
            "response_generator": {"input": input_tokens, "output": output_tokens},
        },
    }
    return {**state, **updates}


async def astream_response_generator(state: AgentState) -> AsyncGenerator[str, None]:
    """
    Async streaming version — yields tokens as they arrive.
    Used by the /stream/query FastAPI endpoint via SSE.
    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        streaming=True,
    )

    system_prompt = load_prompt("response_generator")
    sources = state.get("retrieved_chunks", [])
    sources_str = "\n".join(
        f"[{i+1}] {c.get('source', 'unknown')}: {c['text'][:150]}..."
        for i, c in enumerate(sources[:5])
    )

    confidence = state.get("confidence", 1.0)
    confidence_note = ""
    if BORDERLINE_LOW <= confidence <= BORDERLINE_HIGH:
        confidence_note = f"\n\n⚠️ Confidence: {confidence:.0%} — borderline, please verify."

    user_message = (
        f"Question: {state['original_query']}\n\n"
        f"Reasoning and Answer:\n{state.get('reasoning_output', '')}\n\n"
        f"Sources:\n{sources_str}\n\n"
        f"Format with numbered inline citations.{confidence_note}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content

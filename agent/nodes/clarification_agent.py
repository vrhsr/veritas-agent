"""
Node 6: Clarification Agent
Triggered when query is ambiguous OR reasoning confidence < 0.5.
Generates a targeted, single-turn clarification question.
"""
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


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def clarification_agent_node(state: AgentState) -> AgentState:
    """
    Node 6: Identifies the dimension of ambiguity and generates a targeted binary/
    multiple-choice clarification question. Max 1-turn clarification limit.
    """
    start = time.perf_counter()
    logger.info("clarification_agent", query=state["original_query"])

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.2,
    )

    system_prompt = load_prompt("clarification_agent")

    context_parts = [f"User Query: {state['original_query']}"]
    if state.get("gaps"):
        context_parts.append(f"Identified Gaps: {', '.join(state['gaps'])}")
    if state.get("relevant_memories"):
        context_parts.append(f"User Context from Memory:\n" + "\n".join(state["relevant_memories"][:2]))

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="\n\n".join(context_parts)),
    ]

    response = llm.invoke(messages)
    clarification_question = response.content.strip()

    elapsed = time.perf_counter() - start
    input_tokens = count_tokens(str(messages))
    output_tokens = count_tokens(clarification_question)
    cost = estimate_cost(input_tokens, output_tokens)

    logger.info("clarification_generated", question=clarification_question[:100], latency_s=round(elapsed, 3))

    updates = {
        "clarification_question": clarification_question,
        "awaiting_clarification": True,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
        "node_latencies": {**state.get("node_latencies", {}), "clarification_agent": round(elapsed, 3)},
        "token_usage": {
            **state.get("token_usage", {}),
            "clarification_agent": {"input": input_tokens, "output": output_tokens},
        },
    }
    return {**state, **updates}


def inject_clarification_answer(state: AgentState, answer: str) -> AgentState:
    """
    Called by the serving layer to inject the user's clarification answer back into state.
    Appends the answer to the original query and clears the awaiting flag.
    """
    enriched_query = f"{state['original_query']} [Clarification: {answer}]"
    return {
        **state,
        "original_query": enriched_query,
        "clarification_answer": answer,
        "awaiting_clarification": False,
    }

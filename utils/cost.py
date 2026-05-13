"""
Token cost estimation for gpt-4o-mini pricing.
Updated: May 2025 pricing (verify at platform.openai.com/pricing)
"""

# gpt-4o-mini pricing (per 1M tokens)
PRICING = {
    "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
}

DEFAULT_MODEL = "gpt-4o-mini"


def estimate_cost(input_tokens: int, output_tokens: int, model: str | None = None) -> float:
    """
    Estimate INR cost for a given number of input/output tokens.
    Returns cost in INR (assuming 1 USD = 84 INR).
    """
    model = model or DEFAULT_MODEL
    prices = PRICING.get(model, PRICING[DEFAULT_MODEL])
    cost_usd = (input_tokens * prices["input"]) + (output_tokens * prices["output"])
    cost_inr = cost_usd * 84.0
    return round(cost_inr, 8)


def format_cost(cost_inr: float) -> str:
    """Human-readable cost string."""
    if cost_inr < 0.01:
        return f"₹{cost_inr * 100:.4f}p"  # paise
    return f"₹{cost_inr:.4f}"

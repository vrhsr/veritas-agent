"""
Safe Calculator Tool
Uses sympy for safe expression parsing — NOT raw eval().
Handles arithmetic, percentages, and symbolic math.
"""
from typing import Optional
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

from utils.logger import get_logger

logger = get_logger(__name__)

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def safe_calculate(expression: str) -> str:
    """
    Safely evaluates a mathematical expression using sympy.
    Returns a string result or raises ValueError on invalid input.

    Examples:
        safe_calculate("2 + 2")          → "4"
        safe_calculate("(1/60) * 100")   → "5/3"
        safe_calculate("sqrt(144)")      → "12"
        safe_calculate("sin(pi/2)")      → "1"
    """
    # Sanitize: only allow math characters
    allowed_chars = set("0123456789+-*/().^% abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
    sanitized = "".join(c for c in expression if c in allowed_chars).strip()

    if not sanitized:
        raise ValueError(f"Empty or invalid expression: {expression!r}")

    # Block dangerous function names
    blocked = ["__", "import", "exec", "eval", "open", "os", "sys", "subprocess"]
    for b in blocked:
        if b in sanitized.lower():
            raise ValueError(f"Blocked keyword in expression: {b!r}")

    try:
        result = parse_expr(sanitized, transformations=TRANSFORMATIONS)
        evaluated = sympy.simplify(result)

        # Return numeric result if possible, else symbolic
        if evaluated.is_number:
            float_val = float(evaluated)
            # Return int if it's a whole number
            if float_val == int(float_val):
                return str(int(float_val))
            return f"{float_val:.6g}"
        else:
            return str(evaluated)

    except Exception as e:
        logger.warning("calculator_parse_error", expression=sanitized, error=str(e))
        raise ValueError(f"Could not evaluate expression: {sanitized!r}. Error: {e}")

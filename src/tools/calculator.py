"""Mathematical toolset with symbolic and numeric capabilities."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .utils import SanitizationError, safe_numeric_eval, sanitize_math_expression

try:  # pragma: no cover - behavior validated via unit tests with monkeypatching
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None  # type: ignore[assignment]


class ToolUnavailableError(RuntimeError):
    """Raised when an optional dependency required for a tool is missing."""


def _ok(result: Any, method: str, **metadata: Any) -> Dict[str, Any]:
    return {"ok": True, "result": result, "method": method, "metadata": metadata}


def _error(message: str, method: str, **metadata: Any) -> Dict[str, Any]:
    return {"ok": False, "error": message, "method": method, "metadata": metadata}


def evaluate_expression(expression: str, variables: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    try:
        if sp is not None:
            expr = sp.sympify(sanitize_math_expression(expression).replace("^", "**"))
            subs = {sp.Symbol(k): v for k, v in (variables or {}).items()}
            value = expr.evalf(subs=subs)
            return _ok(float(value), "sympy", expression=expression)
        value = safe_numeric_eval(expression, variables=variables)
        return _ok(value, "safe_numeric_eval", expression=expression)
    except Exception as exc:
        return _error(str(exc), "evaluate_expression", expression=expression)


def simplify_expression(expression: str) -> Dict[str, Any]:
    try:
        if sp is None:
            return _error("SymPy is required for symbolic simplification.", "simplify_expression")
        expr = sp.sympify(sanitize_math_expression(expression).replace("^", "**"))
        simplified = sp.simplify(expr)
        return _ok(str(simplified), "sympy", expression=expression)
    except Exception as exc:
        return _error(str(exc), "simplify_expression", expression=expression)


def solve_equation(equation: str, variable: str = "x") -> Dict[str, Any]:
    try:
        if sp is None:
            return _error("SymPy is required for equation solving.", "solve_equation")
        symbol = sp.Symbol(variable)
        sanitized = sanitize_math_expression(equation)
        if "=" in sanitized:
            left, right = sanitized.split("=", 1)
            expr = sp.Eq(sp.sympify(left.replace("^", "**")), sp.sympify(right.replace("^", "**")))
        else:
            expr = sp.Eq(sp.sympify(sanitized.replace("^", "**")), 0)
        solutions = sp.solve(expr, symbol)
        return _ok([str(s) for s in solutions], "sympy", variable=variable)
    except Exception as exc:
        return _error(str(exc), "solve_equation", equation=equation, variable=variable)


def differentiate_expression(expression: str, variable: str = "x", order: int = 1) -> Dict[str, Any]:
    try:
        if sp is None:
            return _error("SymPy is required for symbolic differentiation.", "differentiate_expression")
        expr = sp.sympify(sanitize_math_expression(expression).replace("^", "**"))
        symbol = sp.Symbol(variable)
        derivative = sp.diff(expr, symbol, order)
        return _ok(str(derivative), "sympy", variable=variable, order=order)
    except Exception as exc:
        return _error(str(exc), "differentiate_expression", expression=expression)


def integrate_expression(
    expression: str,
    variable: str = "x",
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        if sp is None:
            return _error("SymPy is required for symbolic integration.", "integrate_expression")
        expr = sp.sympify(sanitize_math_expression(expression).replace("^", "**"))
        symbol = sp.Symbol(variable)

        if lower is not None and upper is not None:
            result = sp.integrate(expr, (symbol, lower, upper))
            return _ok(float(result.evalf()), "sympy", definite=True, variable=variable)

        result = sp.integrate(expr, symbol)
        return _ok(str(result), "sympy", definite=False, variable=variable)
    except Exception as exc:
        return _error(str(exc), "integrate_expression", expression=expression)


def numerical_integration(expression: str, variable: str, lower: float, upper: float, steps: int = 1000) -> Dict[str, Any]:
    try:
        sanitized = sanitize_math_expression(expression)
        if steps <= 0:
            raise ValueError("steps must be positive")
        h = (upper - lower) / float(steps)
        total = 0.0

        for i in range(steps + 1):
            x_val = lower + i * h
            fx = safe_numeric_eval(sanitized.replace(variable, str(x_val)))
            weight = 4 if i % 2 == 1 else 2
            if i in (0, steps):
                weight = 1
            total += weight * fx

        result = (h / 3.0) * total
        return _ok(result, "simpson", lower=lower, upper=upper, steps=steps)
    except Exception as exc:
        return _error(str(exc), "numerical_integration", expression=expression)


def solve_ode(equation: str, function: str = "y") -> Dict[str, Any]:
    try:
        if sp is None:
            return _error("SymPy is required for ODE solving.", "solve_ode")

        x = sp.Symbol("x")
        y = sp.Function(function)
        sanitized = sanitize_math_expression(equation)

        if "=" in sanitized:
            left, right = sanitized.split("=", 1)
            ode_eq = sp.Eq(sp.sympify(left.replace("^", "**")), sp.sympify(right.replace("^", "**")))
        else:
            ode_eq = sp.Eq(sp.sympify(sanitized.replace("^", "**")), 0)

        solution = sp.dsolve(ode_eq)
        return _ok(str(solution), "sympy", function=function)
    except SanitizationError as exc:
        return _error(str(exc), "solve_ode")
    except Exception as exc:
        return _error(str(exc), "solve_ode", equation=equation)

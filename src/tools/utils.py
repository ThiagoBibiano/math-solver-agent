"""Security and utility helpers for tool execution."""

from __future__ import annotations

import ast
import math
import re
from typing import Dict, Optional

_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
}

_ALLOWED_NAMES: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
}

MATH_EXPR_REGEX = re.compile(r"^[a-zA-Z0-9_+\-*/^().,=\s]+$")

_UNICODE_REPLACEMENTS = {
    "−": "-",
    "–": "-",
    "—": "-",
    "×": "*",
    "÷": "/",
    "·": "*",
    "∙": "*",
    "⁄": "/",
}

_SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
}


class SanitizationError(ValueError):
    """Raised when an input expression does not pass validation."""


def sanitize_math_expression(expression: str, max_length: int = 400, blocked_patterns: Optional[list] = None) -> str:
    normalized = _normalize_math_unicode(expression.strip())
    if not normalized:
        raise SanitizationError("Expression cannot be empty.")
    if len(normalized) > max_length:
        raise SanitizationError("Expression exceeds max length of {} characters.".format(max_length))
    if not MATH_EXPR_REGEX.match(normalized):
        raise SanitizationError("Expression contains unsupported characters.")

    patterns = blocked_patterns or ["__", "import", "exec", "eval"]
    lowered = normalized.lower()
    for pattern in patterns:
        if pattern.lower() in lowered:
            raise SanitizationError("Expression contains blocked pattern '{}'.".format(pattern))

    return normalized


def _normalize_math_unicode(expression: str) -> str:
    if not expression:
        return expression

    for source, target in _UNICODE_REPLACEMENTS.items():
        expression = expression.replace(source, target)

    result_chars = []
    i = 0
    while i < len(expression):
        char = expression[i]
        if char in _SUPERSCRIPT_MAP:
            superscript_tokens = []
            while i < len(expression) and expression[i] in _SUPERSCRIPT_MAP:
                superscript_tokens.append(_SUPERSCRIPT_MAP[expression[i]])
                i += 1
            result_chars.append("^" + "".join(superscript_tokens))
            continue

        result_chars.append(char)
        i += 1

    return "".join(result_chars)


def safe_numeric_eval(expression: str, variables: Optional[Dict[str, float]] = None) -> float:
    """Evaluates arithmetic expressions using a constrained AST interpreter."""

    variables = variables or {}
    sanitized = sanitize_math_expression(expression)
    parsed = ast.parse(sanitized.replace("^", "**"), mode="eval")

    for node in ast.walk(parsed):
        if type(node) not in _ALLOWED_AST_NODES:
            raise SanitizationError("Unsupported syntax: {}".format(type(node).__name__))

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in variables:
                return float(variables[node.id])
            if node.id in _ALLOWED_NAMES:
                return float(_ALLOWED_NAMES[node.id])
            raise SanitizationError("Unknown symbol '{}'".format(node.id))
        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand)
            return -val if isinstance(node.op, ast.USub) else +val
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise SanitizationError("Unsupported operator.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise SanitizationError("Only direct function calls are allowed.")
            fn_name = node.func.id
            if fn_name not in _ALLOWED_FUNCS:
                raise SanitizationError("Function '{}' is not allowed.".format(fn_name))
            args = [_eval(arg) for arg in node.args]
            return float(_ALLOWED_FUNCS[fn_name](*args))
        raise SanitizationError("Unsupported expression component.")

    return float(_eval(parsed))

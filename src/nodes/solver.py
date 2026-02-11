"""Solver node: executes tool calls planned by the generative pipeline."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

from src.agents.state import AgentState, append_trace
from src.llm import GenerativeMathClient
from src.tools.calculator import (
    differentiate_expression,
    evaluate_expression,
    integrate_expression,
    numerical_integration,
    simplify_expression,
    solve_equation,
    solve_ode,
)


_TOOL_HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "differentiate_expression": differentiate_expression,
    "integrate_expression": integrate_expression,
    "solve_equation": solve_equation,
    "simplify_expression": simplify_expression,
    "evaluate_expression": evaluate_expression,
    "numerical_integration": numerical_integration,
    "solve_ode": solve_ode,
}


def solve_problem(
    state: AgentState,
    llm_client: Optional[GenerativeMathClient] = None,
    prompt_pack: Optional[Dict[str, str]] = None,
) -> AgentState:
    """Executes one planned tool-call against deterministic math tools.

    Args:
        state: Current agent state with `tool_call`.
        llm_client: Unused in this stage (kept for node signature parity).
        prompt_pack: Unused in this stage (kept for node signature parity).

    Returns:
        Updated state with tool execution outputs.
    """
    del llm_client, prompt_pack  # solver now consumes already-converted tool syntax.

    started = time.perf_counter()
    state["iterations"] = state.get("iterations", 0) + 1

    tool_call = state.get("tool_call", {})
    if not isinstance(tool_call, dict) or not tool_call.get("name"):
        state["status"] = "failed_solving"
        state["should_refine"] = False
        state.setdefault("errors", []).append("Missing tool_call in state. Conversion stage is required.")
        return state

    tool_name = str(tool_call.get("name"))
    tool_args = tool_call.get("args", {})
    if not isinstance(tool_args, dict):
        tool_args = {}

    result, confidence = _execute_tool_call(tool_name, tool_args)

    state["tool_result"] = result
    state["symbolic_result"] = _result_to_text(result)
    state["numeric_result"] = _try_extract_numeric_result(result)
    state["status"] = "solved"

    if not result.get("ok"):
        state.setdefault("errors", []).append(str(result.get("error", "Tool execution failed.")))
        state["should_refine"] = True

    elapsed = time.perf_counter() - started
    metrics = state.setdefault("metrics", {})
    metrics["solving_seconds"] = elapsed

    append_trace(
        state,
        node="solver",
        summary="Executed tool '{}' at iteration {}".format(tool_name, state.get("iterations", 0)),
        confidence=confidence,
    )
    return state


def _execute_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """Dispatches tool call to handler and returns confidence score.

    Args:
        tool_name: Registered tool handler name.
        tool_args: Keyword arguments passed to handler.

    Returns:
        Tuple with handler result dictionary and confidence score.
    """
    handler = _TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return (
            {
                "ok": False,
                "error": "Unsupported tool '{}'".format(tool_name),
                "method": "solver_dispatch",
                "metadata": {"tool_name": tool_name},
            },
            0.1,
        )

    try:
        result = handler(**tool_args)
    except TypeError as exc:
        result = {
            "ok": False,
            "error": "Invalid tool args for '{}': {}".format(tool_name, exc),
            "method": "solver_dispatch",
            "metadata": {"tool_name": tool_name, "tool_args": tool_args},
        }

    confidence = 0.88 if result.get("ok") else 0.2
    return result, confidence


def _result_to_text(result: Dict[str, Any]) -> str:
    """Formats tool result payload into a textual result value.

    Args:
        result: Tool result dictionary.

    Returns:
        Result string or error string.
    """
    if result.get("ok"):
        return str(result.get("result"))
    return str(result.get("error") or "Tool execution failed.")


def _try_extract_numeric_result(result: Dict[str, Any]) -> Optional[float]:
    """Attempts to coerce tool result into a float value.

    Args:
        result: Tool result dictionary.

    Returns:
        Float value when coercion succeeds; otherwise None.
    """
    if not result.get("ok"):
        return None
    value = result.get("result")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

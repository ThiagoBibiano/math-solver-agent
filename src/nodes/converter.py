"""Conversion node: transforms natural language analysis into tool-call syntax."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.agents.state import AgentState, append_trace
from src.llm import GenerativeMathClient
from src.tools.utils import SanitizationError, sanitize_math_expression

_ALLOWED_TOOLS = {
    "differentiate_expression",
    "integrate_expression",
    "solve_equation",
    "simplify_expression",
    "evaluate_expression",
    "numerical_integration",
    "solve_ode",
}

_TOOL_ALIASES = {
    "differentiate": "differentiate_expression",
    "derivative": "differentiate_expression",
    "integrate": "integrate_expression",
    "integral": "integrate_expression",
    "solve": "solve_equation",
    "equation": "solve_equation",
    "simplify": "simplify_expression",
    "evaluate": "evaluate_expression",
    "numeric_integration": "numerical_integration",
    "ode": "solve_ode",
}


def convert_to_tool_syntax(
    state: AgentState,
    llm_client: Optional[GenerativeMathClient] = None,
    prompt_pack: Optional[Dict[str, str]] = None,
) -> AgentState:
    """Generates and normalizes tool-call syntax from LLM planning output.

    Args:
        state: Current agent state.
        llm_client: Generative client used to produce conversion payload.
        prompt_pack: Prompt templates for conversion stage.

    Returns:
        Updated state with `tool_plan` and normalized `tool_call`.
    """
    started = time.perf_counter()

    if llm_client is None or not llm_client.is_available:
        state["status"] = "failed_precondition"
        state.setdefault("errors", []).append("LLM is required for conversion/planning stage.")
        return state

    analysis = {
        "domain": state.get("domain"),
        "constraints": state.get("constraints", []),
        "complexity_score": state.get("complexity_score"),
        "plan": state.get("plan", []),
    }

    payload = llm_client.plan_tool_execution(
        problem=str(state.get("normalized_problem") or state.get("problem", "")),
        analysis=analysis,
        prompt_pack=prompt_pack,
        image_input=state.get("visual_input", {}),
        iteration=int(state.get("iterations", 0)) + 1,
    )

    tool_call = _normalize_tool_call(payload.get("tool_call", {}))
    if not tool_call:
        state["status"] = "failed_planning"
        state.setdefault("errors", []).append("LLM did not return a valid tool_call payload.")
        return state

    state["strategy"] = str(payload.get("strategy") or "symbolic").strip().lower()
    state["tool_plan"] = payload
    state["tool_call"] = tool_call
    state["plan"] = _coerce_plan(payload.get("plan"), fallback=state.get("plan", []))

    normalized_problem = str(payload.get("normalized_problem") or "").strip()
    if normalized_problem:
        state["normalized_problem"] = normalized_problem

    state["status"] = "converted"
    metrics = state.setdefault("metrics", {})
    metrics["conversion_seconds"] = time.perf_counter() - started

    append_trace(
        state,
        node="converter",
        summary="Planned tool '{}' with AI conversion".format(tool_call.get("name")),
        confidence=0.85,
    )
    return state


def _normalize_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and normalizes tool-call payload.

    Args:
        tool_call: Raw tool-call dictionary emitted by the LLM.

    Returns:
        Normalized tool-call dictionary or empty dict if invalid.
    """
    if not isinstance(tool_call, dict):
        return {}

    raw_name = str(tool_call.get("name", "")).strip()
    if not raw_name:
        return {}

    name = _TOOL_ALIASES.get(raw_name.lower(), raw_name)
    if name not in _ALLOWED_TOOLS:
        return {}

    args = tool_call.get("args", {})
    if not isinstance(args, dict):
        args = {}

    normalized_args: Dict[str, Any] = {}
    for key, value in args.items():
        normalized_args[str(key)] = value

    if "expression" in normalized_args and isinstance(normalized_args["expression"], str):
        try:
            normalized_args["expression"] = sanitize_math_expression(normalized_args["expression"], max_length=1000)
        except SanitizationError:
            return {}

    if "equation" in normalized_args and isinstance(normalized_args["equation"], str):
        try:
            normalized_args["equation"] = sanitize_math_expression(normalized_args["equation"], max_length=1000)
        except SanitizationError:
            return {}

    return {"name": name, "args": normalized_args}


def _coerce_plan(raw_plan: Any, fallback: Any) -> list:
    """Converts plan payload into a normalized list of non-empty strings.

    Args:
        raw_plan: Candidate plan payload.
        fallback: Fallback plan payload.

    Returns:
        Normalized plan list.
    """
    if isinstance(raw_plan, list):
        cleaned = [str(step).strip() for step in raw_plan if str(step).strip()]
        if cleaned:
            return cleaned
    if isinstance(fallback, list):
        return [str(step).strip() for step in fallback if str(step).strip()]
    return []

"""Problem analysis node: fully generative classification and planning context."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.agents.state import AgentState, append_trace
from src.llm import GenerativeMathClient
from src.tools.utils import SanitizationError, sanitize_math_expression

_ALLOWED_DOMAINS = {"calculo_i", "calculo_ii", "calculo_iii", "edo", "algebra_linear"}


def analyze_problem(
    state: AgentState,
    llm_client: Optional[GenerativeMathClient] = None,
    prompt_pack: Optional[Dict[str, str]] = None,
) -> AgentState:
    """Runs generative analysis and populates core planning fields.

    Args:
        state: Current agent state.
        llm_client: Generative client used for analysis.
        prompt_pack: Prompt templates for analyzer stage.

    Returns:
        Updated state with domain, constraints, complexity, and plan.
    """
    started = time.perf_counter()
    problem = state.get("problem", "")

    if llm_client is None or not llm_client.is_available:
        state["status"] = "failed_precondition"
        state.setdefault("errors", []).append("LLM is required for generative analysis stage.")
        return state

    try:
        state["normalized_problem"] = sanitize_math_expression(problem, max_length=1000)
    except SanitizationError:
        state["normalized_problem"] = problem.strip()

    ai_analysis = llm_client.analyze_problem(
        problem=problem,
        prompt_pack=prompt_pack,
        prompt_variant=state.get("prompt_variant"),
        image_input=state.get("visual_input", {}),
    )
    _apply_ai_analysis(state, ai_analysis)

    if state.get("status") == "failed_analysis":
        return state

    state["status"] = "analyzed"

    elapsed = time.perf_counter() - started
    metrics = state.setdefault("metrics", {})
    metrics["analysis_seconds"] = elapsed

    append_trace(
        state,
        node="analyzer",
        summary="Domain '{}' with {} constraints{}".format(
            state.get("domain"),
            len(state.get("constraints", [])),
            " (AI-driven)",
        ),
        confidence=max(0.6, 1.0 - state.get("complexity_score", 0.0) / 2.0),
    )
    return state


def _apply_ai_analysis(state: AgentState, ai_analysis: Dict[str, Any]) -> None:
    """Applies normalized LLM analysis payload to state in-place.

    Args:
        state: Mutable agent state.
        ai_analysis: LLM analysis dictionary.
    """
    if not ai_analysis:
        state["status"] = "failed_analysis"
        state.setdefault("errors", []).append("LLM did not return analysis payload.")
        return

    domain = str(ai_analysis.get("domain", "")).strip()
    if domain not in _ALLOWED_DOMAINS:
        state["status"] = "failed_analysis"
        state.setdefault("errors", []).append("LLM returned unsupported or missing domain.")
        return
    state["domain"] = domain

    constraints = ai_analysis.get("constraints")
    if isinstance(constraints, list):
        cleaned = [str(item).strip() for item in constraints if str(item).strip()]
        state["constraints"] = sorted(set(cleaned))
    else:
        state["constraints"] = []

    complexity = ai_analysis.get("complexity_score")
    try:
        if complexity is not None:
            numeric = float(complexity)
            state["complexity_score"] = max(0.0, min(1.0, numeric))
        else:
            state["complexity_score"] = 0.5
    except (TypeError, ValueError):
        state["complexity_score"] = 0.5

    plan = ai_analysis.get("plan")
    if isinstance(plan, list):
        cleaned_plan = [str(step).strip() for step in plan if str(step).strip()]
        state["plan"] = cleaned_plan or ["Planejamento não detalhado pela LLM"]
    else:
        state["plan"] = ["Planejamento não detalhado pela LLM"]

    normalized_problem = str(ai_analysis.get("normalized_problem", "")).strip()
    if normalized_problem:
        state["normalized_problem"] = normalized_problem

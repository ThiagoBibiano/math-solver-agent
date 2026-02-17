"""Verification node for plausibility, convergence, and loop control."""

from __future__ import annotations

import math
import re
import time
from typing import Dict, Optional

from src.agents.state import AgentState, append_trace
from src.llm import GenerativeMathClient


def _is_plausible(result: str) -> bool:
    """Checks whether a solver result looks like a valid mathematical output.

    Args:
        result: Solver output string.

    Returns:
        True when output does not contain common error/invalid markers.
    """
    if result is None:
        return False
    lowered = str(result).strip().lower()
    if not lowered:
        return False

    error_markers = [
        "error",
        "traceback",
        "unsupported",
        "missing",
        "required",
        "failed",
        "invalid",
        "exception",
        "cannot",
    ]
    if any(token in lowered for token in error_markers):
        return False
    if re.search(r"\b(?:nan|inf|infinity)\b", lowered):
        return False
    return True


def _signature(state: AgentState) -> str:
    """Builds a compact signature for loop/divergence detection.

    Args:
        state: Current agent state.

    Returns:
        String signature derived from strategy and outputs.
    """
    return "{}|{}|{}".format(
        state.get("strategy", ""),
        state.get("symbolic_result", ""),
        state.get("numeric_result", ""),
    )


def _normalize_latex_delimiters(text: str) -> str:
    """Normalizes LaTeX delimiters to dollar style expected by UI.

    Converts:
    - `\\( ... \\)` -> `$ ... $`
    - `\\[ ... \\]` -> `$$ ... $$`
    """
    if not text:
        return ""

    normalized = str(text)
    normalized = normalized.replace("·", r" \cdotp ")
    normalized = _replace_latex_delimited_blocks(normalized, open_token=r"\[", close_token=r"\]", left="$$", right="$$")
    normalized = _replace_latex_delimited_blocks(normalized, open_token=r"\(", close_token=r"\)", left="$", right="$")
    return normalized


def _replace_latex_delimited_blocks(text: str, open_token: str, close_token: str, left: str, right: str) -> str:
    """Replaces balanced LaTeX delimiter pairs while preserving unmatched tokens."""
    output: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        start = text.find(open_token, index)
        if start < 0:
            output.append(text[index:])
            break

        output.append(text[index:start])
        closing = _find_matching_delimiter(text, start + len(open_token), open_token, close_token)
        if closing < 0:
            output.append(text[start:])
            break

        inner = text[start + len(open_token) : closing]
        output.append("{}{}{}".format(left, inner, right))
        index = closing + len(close_token)

    return "".join(output)


def _find_matching_delimiter(text: str, start: int, open_token: str, close_token: str) -> int:
    """Finds the closing delimiter index for a potentially nested block."""
    depth = 1
    cursor = start
    while cursor < len(text):
        next_open = text.find(open_token, cursor)
        next_close = text.find(close_token, cursor)
        if next_close < 0:
            return -1

        if 0 <= next_open < next_close:
            depth += 1
            cursor = next_open + len(open_token)
            continue

        depth -= 1
        if depth == 0:
            return next_close
        cursor = next_close + len(close_token)
    return -1


def verify_solution(
    state: AgentState,
    llm_client: Optional[GenerativeMathClient] = None,
    prompt_pack: Optional[Dict[str, str]] = None,
) -> AgentState:
    """Verifies solution plausibility and produces final explanation content.

    Args:
        state: Current agent state after solver execution.
        llm_client: Optional generative client for pedagogical explanation.
        prompt_pack: Prompt templates for verifier stage.

    Returns:
        Updated state with verification status and explanation.
    """
    started = time.perf_counter()

    checks: Dict[str, bool] = {}
    notes = []

    symbolic_result = state.get("symbolic_result")
    numeric_result = state.get("numeric_result")

    checks["plausible_symbolic"] = _is_plausible(str(symbolic_result))
    checks["numeric_finite"] = numeric_result is None or math.isfinite(float(numeric_result))
    checks["constraints_considered"] = len(state.get("constraints", [])) >= 0

    success = all(checks.values())

    signatures = state.setdefault("seen_signatures", [])
    sig = _signature(state)
    signatures.append(sig)

    divergence_failures = state.get("convergence_failures", 0)
    if signatures.count(sig) > 1:
        divergence_failures += 1
        notes.append("Repeated solution signature detected.")
    state["convergence_failures"] = divergence_failures

    max_iterations = int(state.get("max_iterations", 4))
    iterations = int(state.get("iterations", 0))
    patience = 2

    should_refine = (not success) and iterations < max_iterations and divergence_failures < patience
    state["should_refine"] = should_refine

    if success:
        state["status"] = "verified"
    elif should_refine:
        state["status"] = "refine_required"
        notes.append("Verification failed; another iteration will be attempted.")
    else:
        state["status"] = "failed_verification"
        notes.append("Verification did not converge within configured limits.")

    confidence = sum(1 for ok in checks.values() if ok) / float(max(1, len(checks)))
    state["verification"] = {
        "success": success,
        "checks": checks,
        "notes": notes,
        "confidence": round(confidence, 4),
    }

    explanation_lines = [
        "Domínio classificado: {}".format(state.get("domain", "desconhecido")),
        "Estratégia utilizada: {}".format(state.get("strategy", "desconhecida")),
        "Resultado principal: {}".format(state.get("symbolic_result")),
        "Validação: {}".format("aprovada" if success else "pendente/reprovada"),
    ]
    if notes:
        explanation_lines.append("Observações: {}".format(" ".join(notes)))

    deterministic_explanation = _normalize_latex_delimiters("\n".join(explanation_lines))
    state["explanation"] = deterministic_explanation

    if llm_client is not None and llm_client.is_available:
        ai_explanation = llm_client.generate_explanation(
            state_summary={
                "problem": state.get("problem"),
                "domain": state.get("domain"),
                "strategy": state.get("strategy"),
                "result": state.get("symbolic_result"),
                "verification": state.get("verification"),
                "constraints": state.get("constraints", []),
                "notes": notes,
            },
            prompt_pack=prompt_pack,
        )
        if ai_explanation:
            state["explanation"] = _normalize_latex_delimiters(ai_explanation)

    elapsed = time.perf_counter() - started
    metrics = state.setdefault("metrics", {})
    metrics["verification_seconds"] = elapsed

    append_trace(
        state,
        node="verifier",
        summary="Verification {}{}".format(
            "passed" if success else "failed",
            " (AI explanation)" if llm_client is not None and llm_client.is_available else "",
        ),
        confidence=confidence,
    )

    return state

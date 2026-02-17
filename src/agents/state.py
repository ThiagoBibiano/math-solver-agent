"""Typed state contracts for MathSolverAgent orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict


class DecisionTrace(TypedDict, total=False):
    node: str
    summary: str
    confidence: float
    timestamp: str


class VerificationReport(TypedDict, total=False):
    success: bool
    checks: Dict[str, bool]
    notes: List[str]
    confidence: float


class AgentState(TypedDict, total=False):
    session_id: str
    problem: str
    normalized_problem: str
    domain: str
    constraints: List[str]
    complexity_score: float
    strategy: str
    prompt_variant: Optional[str]
    llm: Dict[str, Any]
    ocr: Dict[str, Any]
    visual_input: Dict[str, Any]
    forced_tool_call: Dict[str, Any]
    tool_plan: Dict[str, Any]
    tool_call: Dict[str, Any]
    tool_result: Dict[str, Any]
    plan: List[str]
    iterations: int
    max_iterations: int
    timeout_seconds: int
    symbolic_result: Optional[str]
    numeric_result: Optional[float]
    verification: VerificationReport
    explanation: str
    warnings: List[str]
    errors: List[str]
    status: str
    should_refine: bool
    seen_signatures: List[str]
    convergence_failures: int
    metrics: Dict[str, float]
    decision_trace: List[DecisionTrace]
    artifacts: List[Dict[str, Any]]
    created_at: str
    updated_at: str


def build_initial_state(problem: str, session_id: str, max_iterations: int, timeout_seconds: int) -> AgentState:
    """Creates the initial agent state snapshot.

    Args:
        problem: Raw problem statement.
        session_id: Session identifier.
        max_iterations: Maximum allowed refinement iterations.
        timeout_seconds: Default timeout budget in seconds.

    Returns:
        Initialized `AgentState`.
    """
    now = datetime.utcnow().isoformat() + "Z"
    return AgentState(
        session_id=session_id,
        problem=problem,
        normalized_problem=problem.strip(),
        constraints=[],
        complexity_score=0.0,
        strategy="pending",
        prompt_variant=None,
        llm={},
        ocr={},
        visual_input={},
        forced_tool_call={},
        tool_plan={},
        tool_call={},
        tool_result={},
        plan=[],
        iterations=0,
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
        symbolic_result=None,
        numeric_result=None,
        verification=VerificationReport(success=False, checks={}, notes=[], confidence=0.0),
        explanation="",
        warnings=[],
        errors=[],
        status="initialized",
        should_refine=False,
        seen_signatures=[],
        convergence_failures=0,
        metrics={},
        decision_trace=[],
        artifacts=[],
        created_at=now,
        updated_at=now,
    )


def append_trace(state: AgentState, node: str, summary: str, confidence: float) -> None:
    """Appends a decision-trace entry to state.

    Args:
        state: Mutable agent state.
        node: Node name producing the trace.
        summary: Short event summary.
        confidence: Confidence score in range [0, 1].
    """
    trace = state.setdefault("decision_trace", [])
    trace.append(
        DecisionTrace(
            node=node,
            summary=summary,
            confidence=max(0.0, min(confidence, 1.0)),
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    )
    state["updated_at"] = datetime.utcnow().isoformat() + "Z"

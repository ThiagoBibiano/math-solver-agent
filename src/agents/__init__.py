"""Agent package entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .graph import MathSolverAgent

__all__ = ["MathSolverAgent"]


def __getattr__(name: str) -> Any:
    if name == "MathSolverAgent":
        from .graph import MathSolverAgent

        return MathSolverAgent
    raise AttributeError("module '{}' has no attribute '{}'".format(__name__, name))

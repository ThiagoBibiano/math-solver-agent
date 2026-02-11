"""Export helpers for LaTeX and Jupyter formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.agents.state import AgentState


def export_latex(state: AgentState, output_path: str) -> str:
    content = [
        r"\\section*{MathSolverAgent Result}",
        r"\\textbf{Session:} " + str(state.get("session_id", "-")) + r"\\",
        r"\\textbf{Status:} " + str(state.get("status", "-")) + r"\\",
        r"\\textbf{Domain:} " + str(state.get("domain", "-")) + r"\\",
        r"\\textbf{Strategy:} " + str(state.get("strategy", "-")) + r"\\",
        r"\\textbf{Result:} $" + str(state.get("symbolic_result", "-")) + r"$\\",
        r"\\textbf{Explanation:} " + str(state.get("explanation", "")),
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(content), encoding="utf-8")
    return str(output)


def export_notebook(state: AgentState, output_path: str) -> str:
    notebook: Dict[str, Any] = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MathSolverAgent Result\\n",
                    "- Session: {}\\n".format(state.get("session_id", "-")),
                    "- Status: {}\\n".format(state.get("status", "-")),
                    "- Domain: {}\\n".format(state.get("domain", "-")),
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Result\\n",
                    "```\\n{}\\n```\\n".format(state.get("symbolic_result", "")),
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Explanation\\n", state.get("explanation", "")],
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output)

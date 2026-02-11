"""Visualization tools for 2D/3D mathematical plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .utils import safe_numeric_eval, sanitize_math_expression

try:  # pragma: no cover
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def plot_function_2d(expression: str, output_path: str, x_min: float = -10.0, x_max: float = 10.0, points: int = 400) -> Dict[str, Any]:
    if plt is None:
        return {
            "ok": False,
            "error": "matplotlib is required for plotting.",
            "method": "plot_function_2d",
            "metadata": {},
        }

    try:
        sanitized = sanitize_math_expression(expression)
        xs = [x_min + (x_max - x_min) * i / float(points - 1) for i in range(points)]
        ys = [safe_numeric_eval(sanitized, {"x": x}) for x in xs]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, linewidth=2.0)
        ax.set_title("f(x) = {}".format(expression))
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)

        return {"ok": True, "result": str(output), "method": "matplotlib", "metadata": {"points": points}}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "method": "plot_function_2d", "metadata": {}}

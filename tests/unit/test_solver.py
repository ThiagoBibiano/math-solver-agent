import unittest
from unittest.mock import patch

from src.agents.state import build_initial_state
from src.nodes.solver import solve_problem


class SolverTestCase(unittest.TestCase):
    def test_solver_executes_tool_call(self) -> None:
        state = build_initial_state(problem="2+2", session_id="s1", max_iterations=2, timeout_seconds=5)
        state["tool_call"] = {"name": "evaluate_expression", "args": {"expression": "2+2"}}

        out = solve_problem(state)

        self.assertEqual(out["status"], "solved")
        self.assertEqual(out["symbolic_result"], "4.0")
        self.assertEqual(out["numeric_result"], 4.0)

    def test_solver_fails_without_tool_call(self) -> None:
        state = build_initial_state(problem="2+2", session_id="s2", max_iterations=2, timeout_seconds=5)

        out = solve_problem(state)

        self.assertEqual(out["status"], "failed_solving")
        self.assertGreater(len(out.get("errors", [])), 0)

    def test_solver_stores_plot_artifact(self) -> None:
        state = build_initial_state(problem="plot", session_id="s3", max_iterations=2, timeout_seconds=5)
        state["tool_call"] = {"name": "plot_function_2d", "args": {"expression": "x", "output_path": "artifacts/p.png"}}

        with patch("src.nodes.solver._TOOL_HANDLERS", {"plot_function_2d": lambda **_: {"ok": True, "result": "artifacts/p.png"}}):
            out = solve_problem(state)

        self.assertEqual(out["status"], "solved")
        self.assertEqual(len(out.get("artifacts", [])), 1)
        self.assertEqual(out["artifacts"][0]["path"], "artifacts/p.png")


if __name__ == "__main__":
    unittest.main()

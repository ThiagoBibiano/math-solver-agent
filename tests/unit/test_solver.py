import unittest

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


if __name__ == "__main__":
    unittest.main()

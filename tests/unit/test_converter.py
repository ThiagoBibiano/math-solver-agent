import unittest

from src.agents.state import build_initial_state
from src.nodes.converter import convert_to_tool_syntax


class _FakeLLM:
    is_available = True

    def plan_tool_execution(self, problem, analysis, iteration, prompt_pack=None, image_input=None):
        return {
            "strategy": "numeric",
            "normalized_problem": "2+2",
            "plan": ["Converter para tool call", "Executar evaluate"],
            "tool_call": {"name": "evaluate_expression", "args": {"expression": "2+2"}},
        }


class ConverterNodeTestCase(unittest.TestCase):
    def test_converter_builds_tool_call(self) -> None:
        state = build_initial_state(problem="Calcule 2+2", session_id="c1", max_iterations=2, timeout_seconds=5)
        state["domain"] = "calculo_i"
        state["plan"] = ["analise"]

        out = convert_to_tool_syntax(state, llm_client=_FakeLLM(), prompt_pack={"system": "x"})

        self.assertEqual(out["status"], "converted")
        self.assertEqual(out["tool_call"]["name"], "evaluate_expression")

    def test_converter_forces_plot_tool_call_when_marker_is_present(self) -> None:
        state = build_initial_state(
            problem=(
                "Use contexto anterior.\n"
                "[PLOT_REQUEST]\n"
                "expression: x^2 - 1\n"
                "[/PLOT_REQUEST]"
            ),
            session_id="plot-session",
            max_iterations=2,
            timeout_seconds=5,
        )
        state["domain"] = "calculo_i"

        out = convert_to_tool_syntax(state, llm_client=_FakeLLM(), prompt_pack={"system": "x"})

        self.assertEqual(out["status"], "converted")
        self.assertEqual(out["tool_call"]["name"], "plot_function_2d")
        self.assertEqual(str(out["tool_call"]["args"]["expression"]).replace(" ", ""), "x^2-1")
        self.assertIn("output_path", out["tool_call"]["args"])
        self.assertTrue(str(out["tool_call"]["args"]["output_path"]).startswith("artifacts/plots/"))

    def test_converter_uses_structured_forced_tool_call_without_llm(self) -> None:
        state = build_initial_state(problem="Pedido de follow-up", session_id="plot-structured", max_iterations=2, timeout_seconds=5)
        state["domain"] = "calculo_i"
        state["normalized_problem"] = "texto reescrito sem marcador"
        state["forced_tool_call"] = {"name": "plot_function_2d", "args": {"expression": "x^2 + 2*x + 1"}}

        out = convert_to_tool_syntax(state, llm_client=None, prompt_pack={"system": "x"})

        self.assertEqual(out["status"], "converted")
        self.assertEqual(out["tool_call"]["name"], "plot_function_2d")
        self.assertEqual(str(out["tool_call"]["args"]["expression"]).replace(" ", ""), "x^2+2*x+1")
        self.assertIn("output_path", out["tool_call"]["args"])


if __name__ == "__main__":
    unittest.main()

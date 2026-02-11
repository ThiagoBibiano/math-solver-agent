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


if __name__ == "__main__":
    unittest.main()

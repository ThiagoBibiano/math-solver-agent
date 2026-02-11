import unittest

from src.agents.state import build_initial_state
from src.nodes.analyzer import analyze_problem


class _FakeLLM:
    is_available = True

    def analyze_problem(self, problem, prompt_pack=None, prompt_variant=None, image_input=None):
        return {
            "domain": "calculo_i",
            "constraints": ["x>0"],
            "complexity_score": 0.2,
            "plan": ["Analisar", "Converter", "Executar"],
            "normalized_problem": "x^2",
        }


class AnalyzerNodeTestCase(unittest.TestCase):
    def test_analyzer_uses_llm_payload(self) -> None:
        state = build_initial_state(
            problem="Calcule a derivada de x^2 com x > 0",
            session_id="test-session",
            max_iterations=3,
            timeout_seconds=10,
        )

        out = analyze_problem(state, llm_client=_FakeLLM(), prompt_pack={"system": "x"})

        self.assertEqual(out["domain"], "calculo_i")
        self.assertIn("x>0", out["constraints"])
        self.assertEqual(out["normalized_problem"], "x^2")
        self.assertEqual(out["status"], "analyzed")


if __name__ == "__main__":
    unittest.main()

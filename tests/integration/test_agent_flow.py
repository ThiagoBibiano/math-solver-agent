import tempfile
import unittest
from unittest.mock import patch

from src.agents.graph import MathSolverAgent
from src.utils.config_loader import GraphConfig, RuntimeSettings


class _FakeGenerativeClient:
    def __init__(self, config=None):
        self.config = config or {}
        self.is_available = True

    def describe(self):
        return {
            "enabled": True,
            "provider": "fake",
            "model": "fake-model",
            "available": True,
            "reason": None,
            "multimodal_enabled": True,
            "api_key_env": "NVIDIA_API_KEY",
            "api_key_present": True,
        }

    def analyze_problem(self, problem, prompt_pack=None, prompt_variant=None, image_input=None):
        return {
            "domain": "calculo_i",
            "constraints": [],
            "complexity_score": 0.1,
            "plan": ["analisar", "converter", "executar", "verificar"],
            "normalized_problem": problem.strip() or "2+2",
        }

    def plan_tool_execution(self, problem, analysis, iteration, prompt_pack=None, image_input=None):
        return {
            "strategy": "numeric",
            "normalized_problem": problem or "2+2",
            "plan": ["gerar tool_call", "executar evaluate_expression"],
            "tool_call": {"name": "evaluate_expression", "args": {"expression": "2+2"}},
        }

    def generate_explanation(self, state_summary, prompt_pack=None):
        return "Explicacao gerada por LLM fake"


class _UnavailableGenerativeClient(_FakeGenerativeClient):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.is_available = False

    def describe(self):
        payload = super().describe()
        payload["available"] = False
        payload["reason"] = "missing_api_key"
        payload["api_key_present"] = False
        return payload


class AgentFlowIntegrationTestCase(unittest.TestCase):
    def test_flow_runs_end_to_end_with_fake_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GraphConfig(
                version="1.0.0",
                runtime=RuntimeSettings(
                    max_iterations=3,
                    divergence_patience=2,
                    checkpoint_dir=tmpdir,
                    default_timeout_seconds=5,
                    operation_timeouts={"analysis": 3, "planning": 3, "solving": 3, "verification": 3},
                ),
                llm={"require_available": True},
            )

            with patch("src.agents.graph.GenerativeMathClient", _FakeGenerativeClient), patch(
                "src.agents.graph.load_graph_config", return_value=config
            ), patch(
                "src.agents.graph.load_prompts_config",
                return_value={"experiments": {"ab_testing": {"enabled": True, "variants": {"A": {}, "B": {}}}}},
            ), patch(
                "src.agents.graph.load_prompts_registry",
                return_value={
                    "analyzer": {"system": "x"},
                    "converter": {"system": "x"},
                    "solver": {"system": "x"},
                    "verifier": {"system": "x"},
                },
            ):
                agent = MathSolverAgent()
                result = agent.solve(problem="2+2")

            self.assertEqual(result["status"], "verified")
            self.assertTrue(result.get("session_id"))
            self.assertIn(result.get("prompt_variant"), {"A", "B"})
            self.assertEqual(result.get("symbolic_result"), "4.0")

    def test_fails_fast_when_llm_unavailable_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GraphConfig(
                version="1.0.0",
                runtime=RuntimeSettings(
                    max_iterations=2,
                    divergence_patience=2,
                    checkpoint_dir=tmpdir,
                    default_timeout_seconds=5,
                    operation_timeouts={"analysis": 3, "planning": 3, "solving": 3, "verification": 3},
                ),
                llm={"require_available": True},
            )

            with patch("src.agents.graph.GenerativeMathClient", _UnavailableGenerativeClient), patch(
                "src.agents.graph.load_graph_config", return_value=config
            ), patch(
                "src.agents.graph.load_prompts_config",
                return_value={"experiments": {"ab_testing": {"enabled": True, "variants": {"A": {}, "B": {}}}}},
            ), patch(
                "src.agents.graph.load_prompts_registry",
                return_value={"analyzer": {"system": "x"}, "converter": {"system": "x"}, "solver": {"system": "x"}},
            ):
                agent = MathSolverAgent()
                result = agent.solve(problem="2+2")

            self.assertEqual(result["status"], "failed_precondition")
            self.assertGreater(len(result.get("errors", [])), 0)

    def test_image_input_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GraphConfig(
                version="1.0.0",
                runtime=RuntimeSettings(
                    max_iterations=2,
                    divergence_patience=2,
                    checkpoint_dir=tmpdir,
                    default_timeout_seconds=5,
                    operation_timeouts={"analysis": 3, "planning": 3, "solving": 3, "verification": 3},
                ),
                llm={"require_available": True},
            )

            with patch("src.agents.graph.GenerativeMathClient", _FakeGenerativeClient), patch(
                "src.agents.graph.load_graph_config", return_value=config
            ), patch(
                "src.agents.graph.load_prompts_config",
                return_value={"experiments": {"ab_testing": {"enabled": True, "variants": {"A": {}, "B": {}}}}},
            ), patch(
                "src.agents.graph.load_prompts_registry",
                return_value={"analyzer": {"system": "x"}, "converter": {"system": "x"}, "solver": {"system": "x"}},
            ):
                agent = MathSolverAgent()
                result = agent.solve(
                    problem="",
                    image_base64="aGVsbG8=",
                    image_media_type="image/png",
                )

            self.assertEqual(result.get("visual_input", {}).get("image_media_type"), "image/png")


if __name__ == "__main__":
    unittest.main()

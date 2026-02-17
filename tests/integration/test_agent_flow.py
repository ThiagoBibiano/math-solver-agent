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
        provider = str(self.config.get("provider", "fake"))
        model = str(self.config.get("model", "fake-model"))
        model_profile = str(self.config.get("model_profile", "fake_profile"))
        supports_multimodal = bool(self.config.get("supports_multimodal", self.config.get("multimodal_enabled", True)))
        return {
            "enabled": True,
            "provider": provider,
            "model": model,
            "model_profile": model_profile,
            "available": True,
            "reason": None,
            "multimodal_enabled": supports_multimodal,
            "supports_multimodal": supports_multimodal,
            "api_key_env": str(self.config.get("api_key_env", "NVIDIA_API_KEY")),
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


class _FakeAnalyzerDropsMarkerClient(_FakeGenerativeClient):
    def analyze_problem(self, problem, prompt_pack=None, prompt_variant=None, image_input=None):
        del prompt_pack, prompt_variant, image_input
        return {
            "domain": "calculo_i",
            "constraints": [],
            "complexity_score": 0.2,
            "plan": ["analisar", "converter", "executar", "verificar"],
            "normalized_problem": "pedido normalizado sem marcador explicito",
        }


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

    def test_request_llm_overrides_are_applied(self) -> None:
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
                llm={"require_available": True, "provider": "nvidia", "model_profile": "kimi_k2_5", "model": "moonshotai/kimi-k2.5"},
            )

            with patch("src.agents.graph.GenerativeMathClient", _FakeGenerativeClient), patch(
                "src.agents.graph.load_graph_config", return_value=config
            ), patch(
                "src.agents.graph.load_prompts_config",
                return_value={"experiments": {"ab_testing": {"enabled": False}}},
            ), patch(
                "src.agents.graph.load_prompts_registry",
                return_value={"analyzer": {"system": "x"}, "converter": {"system": "x"}, "solver": {"system": "x"}},
            ):
                agent = MathSolverAgent()
                result = agent.solve(
                    problem="2+2",
                    llm_overrides={
                        "provider": "maritaca",
                        "model_profile": "sabia_4",
                        "model": "sabia-4",
                        "api_key_env": "MARITACA_API_KEY",
                    },
                )

            self.assertEqual(result.get("llm", {}).get("provider"), "maritaca")
            self.assertEqual(result.get("llm", {}).get("model"), "sabia-4")
            self.assertEqual(result.get("llm", {}).get("api_key_env"), "MARITACA_API_KEY")

    def test_provider_override_sets_default_api_key_env(self) -> None:
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
                llm={"require_available": True, "provider": "nvidia", "model_profile": "kimi_k2_5", "api_key_env": "NVIDIA_API_KEY"},
            )

            with patch("src.agents.graph.GenerativeMathClient", _FakeGenerativeClient), patch(
                "src.agents.graph.load_graph_config", return_value=config
            ), patch(
                "src.agents.graph.load_prompts_config",
                return_value={"experiments": {"ab_testing": {"enabled": False}}},
            ), patch(
                "src.agents.graph.load_prompts_registry",
                return_value={"analyzer": {"system": "x"}, "converter": {"system": "x"}, "solver": {"system": "x"}},
            ):
                agent = MathSolverAgent()
                result = agent.solve(
                    problem="2+2",
                    llm_overrides={
                        "provider": "maritaca",
                        "model_profile": "sabiazinho_4",
                    },
                )

            self.assertEqual(result.get("llm", {}).get("provider"), "maritaca")
            self.assertEqual(result.get("llm", {}).get("api_key_env"), "MARITACA_API_KEY")

    def test_profile_override_without_explicit_model_uses_profile_model(self) -> None:
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
                llm={
                    "require_available": True,
                    "provider": "nvidia",
                    "model_profile": "kimi_k2_5",
                    "model": "moonshotai/kimi-k2.5",
                },
            )

            with patch("src.agents.graph.GenerativeMathClient", _FakeGenerativeClient), patch(
                "src.agents.graph.load_graph_config", return_value=config
            ), patch(
                "src.agents.graph.load_prompts_config",
                return_value={"experiments": {"ab_testing": {"enabled": False}}},
            ), patch(
                "src.agents.graph.load_prompts_registry",
                return_value={"analyzer": {"system": "x"}, "converter": {"system": "x"}, "solver": {"system": "x"}},
            ):
                agent = MathSolverAgent()
                result = agent.solve(
                    problem="2+2",
                    llm_overrides={
                        "provider": "maritaca",
                        "model_profile": "sabiazinho_4",
                    },
                )

            self.assertEqual(result.get("llm", {}).get("provider"), "maritaca")
            self.assertNotEqual(result.get("llm", {}).get("model"), "moonshotai/kimi-k2.5")

    def test_resume_without_checkpoint_returns_resume_not_found(self) -> None:
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
                result = agent.solve(problem="", session_id="sessao-ausente", resume=True)

            self.assertEqual(result.get("status"), "resume_not_found")
            self.assertGreater(len(result.get("errors", [])), 0)

    def test_resume_with_follow_up_prompt_keeps_context(self) -> None:
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
                return_value={"experiments": {"ab_testing": {"enabled": False}}},
            ), patch(
                "src.agents.graph.load_prompts_registry",
                return_value={"analyzer": {"system": "x"}, "converter": {"system": "x"}, "solver": {"system": "x"}},
            ):
                agent = MathSolverAgent()
                first = agent.solve(problem="2+2", session_id="sessao-memoria", resume=False)
                second = agent.solve(problem="Explique melhor o resultado.", session_id="sessao-memoria", resume=True)

            self.assertEqual(first.get("status"), "verified")
            merged_problem = str(second.get("problem", ""))
            self.assertIn("2+2", merged_problem)
            self.assertIn("Explique melhor o resultado.", merged_problem)

    def test_auto_ocr_requires_text_for_non_multimodal_with_image(self) -> None:
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
                llm={"require_available": True, "multimodal_enabled": False},
            )

            with patch("src.agents.graph.GenerativeMathClient", _FakeGenerativeClient), patch(
                "src.agents.graph.load_graph_config", return_value=config
            ), patch(
                "src.agents.graph.load_prompts_config",
                return_value={"experiments": {"ab_testing": {"enabled": False}}},
            ), patch(
                "src.agents.graph.load_prompts_registry",
                return_value={"analyzer": {"system": "x"}, "converter": {"system": "x"}, "solver": {"system": "x"}},
            ):
                agent = MathSolverAgent()
                result = agent.solve(problem="", image_base64="aGVsbG8=", image_media_type="image/png", ocr_mode="auto")

            self.assertEqual(result.get("status"), "ocr_required")

    def test_auto_ocr_accepts_pre_extracted_text(self) -> None:
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
                llm={"require_available": True, "multimodal_enabled": False},
            )

            with patch("src.agents.graph.GenerativeMathClient", _FakeGenerativeClient), patch(
                "src.agents.graph.load_graph_config", return_value=config
            ), patch(
                "src.agents.graph.load_prompts_config",
                return_value={"experiments": {"ab_testing": {"enabled": False}}},
            ), patch(
                "src.agents.graph.load_prompts_registry",
                return_value={"analyzer": {"system": "x"}, "converter": {"system": "x"}, "solver": {"system": "x"}},
            ):
                agent = MathSolverAgent()
                result = agent.solve(
                    problem="",
                    image_base64="aGVsbG8=",
                    image_media_type="image/png",
                    ocr_mode="auto",
                    ocr_text="calcule 2+2",
                )

            self.assertEqual(result.get("status"), "verified")
            self.assertTrue(bool(result.get("ocr", {}).get("used")))

    def test_forced_plot_action_survives_analyzer_normalization(self) -> None:
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

            with patch("src.agents.graph.GenerativeMathClient", _FakeAnalyzerDropsMarkerClient), patch(
                "src.agents.graph.load_graph_config", return_value=config
            ), patch(
                "src.agents.graph.load_prompts_config",
                return_value={"experiments": {"ab_testing": {"enabled": False}}},
            ), patch(
                "src.agents.graph.load_prompts_registry",
                return_value={"analyzer": {"system": "x"}, "converter": {"system": "x"}, "solver": {"system": "x"}},
            ):
                agent = MathSolverAgent()
                result = agent.solve(
                    problem=(
                        "Gere um grafico.\n"
                        "[PLOT_REQUEST]\n"
                        "expression: x^2 - 4\n"
                        "[/PLOT_REQUEST]"
                    ),
                    session_id="sessao-plot-forcado",
                    resume=False,
                )

            self.assertEqual(result.get("status"), "verified")
            self.assertEqual(result.get("tool_call", {}).get("name"), "plot_function_2d")
            self.assertTrue(bool(result.get("artifacts")))


if __name__ == "__main__":
    unittest.main()

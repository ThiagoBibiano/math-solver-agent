import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    TestClient = None

from src.api.server import create_app


class _FakeAgent:
    def solve(self, **kwargs):  # noqa: ANN003 - test double
        return {
            "session_id": "sessao-teste",
            "status": "verified",
            "domain": "calculo_i",
            "strategy": "symbolic",
            "llm": {"provider": "nvidia", "model": "moonshotai/kimi-k2.5"},
            "visual_input": {},
            "symbolic_result": "4",
            "numeric_result": 4.0,
            "verification": {"success": True},
            "explanation": "Resposta teste",
            "metrics": {"latency_ms": 12.3},
            "decision_trace": [{"node": "analyzer", "summary": "analise inicial", "confidence": 0.9}],
        }


class APIServerTestCase(unittest.TestCase):
    def test_solve_includes_decision_trace(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        with patch("src.api.server.MathSolverAgent", return_value=_FakeAgent()):
            app = create_app()
            client = TestClient(app)
            response = client.post("/v1/solve", json={"problem": "2+2"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("decision_trace", body)
        self.assertEqual(len(body["decision_trace"]), 1)
        self.assertEqual(body["decision_trace"][0]["node"], "analyzer")


if __name__ == "__main__":
    unittest.main()

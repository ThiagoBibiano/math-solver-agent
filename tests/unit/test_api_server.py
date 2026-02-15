import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    TestClient = None

from src.api.server import create_app


class _FakeAgent:
    def solve(self, **kwargs):  # noqa: ANN003 - test double
        is_resume = bool(kwargs.get("resume"))
        status = "resume_not_found" if is_resume else "verified"
        return {
            "session_id": "sessao-teste",
            "status": status,
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
            "artifacts": [],
        }

    async def solve_events(self, **kwargs):  # noqa: ANN003 - test double
        if bool(kwargs.get("resume")):
            yield {
                "type": "result",
                "data": {
                    "session_id": "sessao-teste",
                    "status": "resume_not_found",
                    "llm": {},
                    "result": None,
                    "explanation": "",
                    "decision_trace": [],
                    "artifacts": [],
                },
            }
            return
        yield {"type": "node_status", "data": {"node": "analyzer", "status": "running"}}
        yield {"type": "trace", "data": {"node": "analyzer", "summary": "analise inicial", "confidence": 0.9}}
        yield {"type": "node_status", "data": {"node": "analyzer", "status": "done"}}
        yield {
            "type": "result",
            "data": {
                "session_id": "sessao-teste",
                "status": "verified",
                "domain": "calculo_i",
                "strategy": "symbolic",
                "llm": {"provider": "nvidia", "model": "moonshotai/kimi-k2.5"},
                "has_visual_input": False,
                "result": "4",
                "numeric_result": 4.0,
                "verification": {"success": True},
                "explanation": "Resposta teste",
                "metrics": {"latency_ms": 12.3},
                "decision_trace": [{"node": "analyzer", "summary": "analise inicial", "confidence": 0.9}],
                "artifacts": [],
            },
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
        self.assertIn("artifacts", body)
        self.assertEqual(len(body["decision_trace"]), 1)
        self.assertEqual(body["decision_trace"][0]["node"], "analyzer")

    def test_solve_resume_not_found(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        with patch("src.api.server.MathSolverAgent", return_value=_FakeAgent()):
            app = create_app()
            client = TestClient(app)
            response = client.post("/v1/solve", json={"problem": "", "resume": True, "session_id": "missing"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "resume_not_found")

    def test_stream_emits_realtime_events(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        with patch("src.api.server.MathSolverAgent", return_value=_FakeAgent()):
            app = create_app()
            client = TestClient(app)
            with client.websocket_connect("/v1/solve/stream") as ws:
                ws.send_json({"problem": "2+2"})
                events = []
                for _ in range(5):
                    events.append(ws.receive_json())
                    if events[-1].get("type") == "done":
                        break

        event_types = [event.get("type") for event in events]
        self.assertIn("node_status", event_types)
        self.assertIn("trace", event_types)
        self.assertIn("result", event_types)
        self.assertIn("done", event_types)


if __name__ == "__main__":
    unittest.main()

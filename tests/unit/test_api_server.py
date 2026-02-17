import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
import time

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    TestClient = None

from src.api.server import create_app


class _FakeAgent:
    def __init__(self) -> None:
        self.last_solve_kwargs = {}
        self.last_stream_kwargs = {}
        self.last_llm_overrides = {}
        self.supports_multimodal = True
        self.checkpoints = self
        self.config = type(
            "Cfg",
            (),
            {
                "ocr": {"provider": "rapidocr", "min_confidence": 0.35},
                "runtime_control": {
                    "max_inflight_global": 4,
                    "max_queue_size": 32,
                    "queue_wait_timeout_seconds": 1,
                    "solve_hard_timeout_seconds": 1,
                },
            },
        )()

    def list_recent(self, limit=10):  # noqa: ANN001, ANN201 - test double
        return [
            {
                "session_id": "sessao-recente",
                "updated_at": "2026-01-01T00:00:00Z",
                "status": "verified",
                "domain": "calculo_i",
                "problem_preview": "2+2",
            }
        ][: int(limit)]

    def solve(self, **kwargs):  # noqa: ANN003 - test double
        self.last_solve_kwargs = dict(kwargs)
        is_resume = bool(kwargs.get("resume"))
        status = "resume_not_found" if is_resume and kwargs.get("session_id") == "missing" else "verified"
        return {
            "session_id": "sessao-teste",
            "status": status,
            "normalized_problem": kwargs.get("problem", ""),
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
            "tool_call": {"name": "evaluate_expression", "args": {"expression": "2+2"}},
        }

    def describe_request_llm(self, llm_overrides=None):  # noqa: ANN001, ANN201 - test double
        self.last_llm_overrides = dict(llm_overrides or {})
        return {
            "provider": self.last_llm_overrides.get("provider", "nvidia"),
            "model": self.last_llm_overrides.get("model", "moonshotai/kimi-k2.5"),
            "supports_multimodal": bool(self.supports_multimodal),
            "multimodal_enabled": bool(self.supports_multimodal),
        }

    async def solve_events(self, **kwargs):  # noqa: ANN003 - test double
        self.last_stream_kwargs = dict(kwargs)
        if bool(kwargs.get("resume")):
            yield {
                "type": "result",
                "data": {
                    "session_id": "sessao-teste",
                    "status": "resume_not_found",
                    "normalized_problem": "",
                    "llm": {},
                    "result": None,
                    "explanation": "",
                    "decision_trace": [],
                    "artifacts": [],
                    "tool_call": {},
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
                "normalized_problem": "2+2",
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
                "tool_call": {"name": "evaluate_expression", "args": {"expression": "2+2"}},
            },
        }


class _SlowAgent(_FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.config.runtime_control["solve_hard_timeout_seconds"] = 0.05

    def solve(self, **kwargs):  # noqa: ANN003 - test double
        time.sleep(0.2)
        return super().solve(**kwargs)


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

    def test_solve_accepts_images_list_and_analysis_only(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        fake = _FakeAgent()
        with patch("src.api.server.MathSolverAgent", return_value=fake):
            app = create_app()
            client = TestClient(app)
            response = client.post(
                "/v1/solve",
                json={
                    "problem": "",
                    "analysis_only": True,
                    "images": [{"image_base64": "aGVsbG8=", "image_media_type": "image/png"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(fake.last_solve_kwargs.get("analysis_only"))
        self.assertEqual(len(fake.last_solve_kwargs.get("images", [])), 1)

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

    def test_stream_accepts_images_and_analysis_only(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        fake = _FakeAgent()
        with patch("src.api.server.MathSolverAgent", return_value=fake):
            app = create_app()
            client = TestClient(app)
            with client.websocket_connect("/v1/solve/stream") as ws:
                ws.send_json(
                    {
                        "problem": "2+2",
                        "analysis_only": True,
                        "images": [{"image_base64": "aGVsbG8=", "image_media_type": "image/png"}],
                    }
                )
                while True:
                    if ws.receive_json().get("type") == "done":
                        break

        self.assertTrue(fake.last_stream_kwargs.get("analysis_only"))
        self.assertEqual(len(fake.last_stream_kwargs.get("images", [])), 1)

    def test_sessions_endpoint(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        with patch("src.api.server.MathSolverAgent", return_value=_FakeAgent()):
            app = create_app()
            client = TestClient(app)
            response = client.get("/v1/sessions?limit=1")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("sessions", body)
        self.assertEqual(len(body["sessions"]), 1)

    def test_export_with_resume_and_session_id(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        with TemporaryDirectory() as tmpdir:
            output_path = "{}/result.tex".format(tmpdir)
            fake = _FakeAgent()
            with patch("src.api.server.MathSolverAgent", return_value=fake):
                app = create_app()
                client = TestClient(app)
                response = client.post(
                    "/v1/export",
                    json={
                        "problem": "",
                        "session_id": "sessao-teste",
                        "resume": True,
                        "format": "latex",
                        "output_path": output_path,
                    },
                )

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["format"], "latex")
            self.assertEqual(body["file_path"], output_path)
            self.assertTrue(fake.last_solve_kwargs.get("resume"))

    def test_solve_forwards_ocr_fields(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        fake = _FakeAgent()
        with patch("src.api.server.MathSolverAgent", return_value=fake):
            app = create_app()
            client = TestClient(app)
            response = client.post(
                "/v1/solve",
                json={
                    "problem": "",
                    "ocr_mode": "on",
                    "ocr_text": "x^2 + 1 = 0",
                    "images": [{"image_base64": "aGVsbG8=", "image_media_type": "image/png"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake.last_solve_kwargs.get("ocr_mode"), "on")
        self.assertEqual(fake.last_solve_kwargs.get("ocr_text"), "x^2 + 1 = 0")

    def test_solve_off_non_multimodal_image_only_returns_422(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        fake = _FakeAgent()
        fake.supports_multimodal = False
        with patch("src.api.server.MathSolverAgent", return_value=fake):
            app = create_app()
            client = TestClient(app)
            response = client.post(
                "/v1/solve",
                json={
                    "problem": "",
                    "ocr_mode": "off",
                    "images": [{"image_base64": "aGVsbG8=", "image_media_type": "image/png"}],
                },
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("OCR mode is off", response.json().get("detail", ""))

    def test_ocr_extract_endpoint(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        fake = _FakeAgent()
        with patch("src.api.server.MathSolverAgent", return_value=fake), patch(
            "src.api.server.extract_text_from_images",
            return_value={
                "text": "x^2+1=0",
                "pages": [{"index": 1, "text": "x^2+1=0", "confidence": 0.9, "lines": []}],
                "engine": "rapidocr",
                "warnings": [],
            },
        ) as mocked_ocr:
            app = create_app()
            client = TestClient(app)
            response = client.post(
                "/v1/ocr/extract",
                json={"images": [{"image_base64": "aGVsbG8=", "image_media_type": "image/png"}]},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["engine"], "rapidocr")
        self.assertEqual(body["text"], "x^2+1=0")
        mocked_ocr.assert_called_once()

    def test_runtime_status_endpoint(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        with patch("src.api.server.MathSolverAgent", return_value=_FakeAgent()):
            app = create_app()
            client = TestClient(app)
            response = client.get("/v1/runtime/status")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("busy", body)
        self.assertIn("in_flight_total", body)
        self.assertIn("queue_depth", body)

    def test_jobs_solve_submit_and_status(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        with patch("src.api.server.MathSolverAgent", return_value=_FakeAgent()):
            app = create_app()
            with TestClient(app) as client:
                submit = client.post("/v1/jobs/solve", json={"problem": "2+2"})
                self.assertEqual(submit.status_code, 200)
                job_id = submit.json().get("job_id")
                self.assertTrue(job_id)

                deadline = time.time() + 2.0
                last_status = "queued"
                while time.time() < deadline:
                    status = client.get("/v1/jobs/{}".format(job_id))
                    self.assertEqual(status.status_code, 200)
                    body = status.json()
                    last_status = body.get("status", "")
                    if last_status in {"succeeded", "failed", "timeout", "canceled"}:
                        break
                    time.sleep(0.05)

        self.assertEqual(last_status, "succeeded")

    def test_solve_hard_timeout_returns_504(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        with patch("src.api.server.MathSolverAgent", return_value=_SlowAgent()):
            app = create_app()
            client = TestClient(app)
            response = client.post("/v1/solve", json={"problem": "2+2"})

        self.assertEqual(response.status_code, 504)

    def test_solve_returns_429_when_admission_rejects(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")

        from src.api.runtime import AdmissionRejectedError

        with patch("src.api.server.MathSolverAgent", return_value=_FakeAgent()), patch(
            "src.api.server.AdmissionController.acquire",
            side_effect=AdmissionRejectedError("Runtime queue is full.", reason="queue_full", retry_after_seconds=1),
        ):
            app = create_app()
            client = TestClient(app)
            response = client.post("/v1/solve", json={"problem": "2+2"})

        self.assertEqual(response.status_code, 429)


if __name__ == "__main__":
    unittest.main()

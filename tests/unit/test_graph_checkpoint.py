import unittest
from tempfile import TemporaryDirectory

from src.agents.graph import CheckpointStore, _state_for_checkpoint
from src.agents.state import build_initial_state


class GraphCheckpointTestCase(unittest.TestCase):
    def test_checkpoint_keeps_artifacts(self) -> None:
        state = build_initial_state(problem="plot", session_id="s-art", max_iterations=1, timeout_seconds=5)
        state["artifacts"] = [{"type": "image", "path": "artifacts/plot.png"}]
        state["visual_input"] = {"image_base64": "abc"}

        payload = _state_for_checkpoint(state)

        self.assertIn("artifacts", payload)
        self.assertEqual(payload["artifacts"][0]["path"], "artifacts/plot.png")
        self.assertNotIn("image_base64", payload.get("visual_input", {}))

    def test_checkpoint_store_lists_recent_sessions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = CheckpointStore(tmpdir)
            state = build_initial_state(problem="2+2", session_id="sessao-1", max_iterations=1, timeout_seconds=5)
            state["status"] = "verified"
            state["domain"] = "calculo_i"
            store.save(state)

            sessions = store.list_recent(limit=5)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "sessao-1")


if __name__ == "__main__":
    unittest.main()

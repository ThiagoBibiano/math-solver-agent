import unittest

from src.agents.graph import _state_for_checkpoint
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


if __name__ == "__main__":
    unittest.main()

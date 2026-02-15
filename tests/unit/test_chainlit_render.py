import unittest

try:
    from src.ui.chainlit_app import _format_final_response
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _format_final_response = None


class ChainlitRenderTestCase(unittest.TestCase):
    def test_format_response_preserves_latex_backslashes(self) -> None:
        if _format_final_response is None:
            self.skipTest("chainlit is not installed")

        response = {
            "session_id": "s1",
            "status": "verified",
            "domain": "calculo_i",
            "strategy": "symbolic",
            "llm": {"provider": "nvidia", "model": "m"},
            "result": r"$\frac{a}{b}$",
            "explanation": r"Use $\alpha + \beta$ e $$\frac{1}{2}$$",
            "verification": {},
        }
        formatted = _format_final_response(response)
        self.assertIn(r"$\frac{a}{b}$", formatted)
        self.assertIn(r"$\alpha + \beta$", formatted)
        self.assertIn(r"$$\frac{1}{2}$$", formatted)


if __name__ == "__main__":
    unittest.main()

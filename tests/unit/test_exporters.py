import tempfile
import unittest

from src.agents.state import build_initial_state
from src.utils.exporters import export_latex, export_notebook


class ExportersTestCase(unittest.TestCase):
    def test_export_latex_and_notebook(self) -> None:
        state = build_initial_state(problem="2+2", session_id="s1", max_iterations=1, timeout_seconds=5)
        state["status"] = "verified"
        state["symbolic_result"] = "4"
        state["explanation"] = "Resultado correto."

        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = export_latex(state, output_path=f"{tmpdir}/result.tex")
            nb_path = export_notebook(state, output_path=f"{tmpdir}/result.ipynb")

            with open(tex_path, "r", encoding="utf-8") as tex:
                self.assertIn("MathSolverAgent Result", tex.read())
            with open(nb_path, "r", encoding="utf-8") as nb:
                self.assertIn("nbformat", nb.read())


if __name__ == "__main__":
    unittest.main()

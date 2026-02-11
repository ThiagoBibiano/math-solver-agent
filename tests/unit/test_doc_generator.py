import tempfile
import unittest

from src.utils.config_loader import GraphConfig, RuntimeSettings
from src.utils.doc_generator import generate_config_docs


class DocGeneratorTestCase(unittest.TestCase):
    def test_generate_config_docs(self) -> None:
        graph = GraphConfig(runtime=RuntimeSettings(max_iterations=7, divergence_patience=2, checkpoint_dir="tmp", default_timeout_seconds=10))
        prompts = {"base": {"system": "abc", "user": "def", "tool": "ghi"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output = generate_config_docs(graph, prompts, f"{tmpdir}/CONFIGURATION.md")
            with open(output, "r", encoding="utf-8") as handle:
                content = handle.read()

        self.assertIn("max_iterations: 7", content)
        self.assertIn("### base", content)


if __name__ == "__main__":
    unittest.main()

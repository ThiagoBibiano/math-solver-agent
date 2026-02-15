import unittest

import yaml


class PromptsConfigTestCase(unittest.TestCase):
    def test_verifier_requires_dollar_delimiters(self) -> None:
        with open("configs/prompts.yml", "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        verifier_system = str(payload.get("registry", {}).get("verifier", {}).get("system", ""))
        self.assertIn("Use only '$...$' for inline formulas and '$$...$$' for block formulas.", verifier_system)
        self.assertIn(r"Do not use '\( ... \)' or '\[ ... \]' delimiters.", verifier_system)


if __name__ == "__main__":
    unittest.main()

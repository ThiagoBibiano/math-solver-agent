import unittest

from src.tools.calculator import evaluate_expression
from src.tools.utils import SanitizationError, sanitize_math_expression


class SpecializedMathTests(unittest.TestCase):
    def test_invariance_commutativity(self) -> None:
        left = evaluate_expression("2+3")
        right = evaluate_expression("3+2")
        self.assertTrue(left["ok"])
        self.assertTrue(right["ok"])
        self.assertAlmostEqual(float(left["result"]), float(right["result"]), places=8)

    def test_robustness_malformed_input(self) -> None:
        with self.assertRaises(SanitizationError):
            sanitize_math_expression("import os")


if __name__ == "__main__":
    unittest.main()

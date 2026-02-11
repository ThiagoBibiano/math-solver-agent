import unittest

from src.tools.calculator import evaluate_expression, numerical_integration


class CalculatorTestCase(unittest.TestCase):
    def test_evaluate_expression_fallback(self) -> None:
        result = evaluate_expression("3*(2+1)")
        self.assertTrue(result["ok"])
        self.assertAlmostEqual(float(result["result"]), 9.0, places=6)

    def test_numerical_integration_simpson(self) -> None:
        result = numerical_integration("x", variable="x", lower=0.0, upper=1.0, steps=100)
        self.assertTrue(result["ok"])
        self.assertAlmostEqual(float(result["result"]), 0.5, places=2)


if __name__ == "__main__":
    unittest.main()

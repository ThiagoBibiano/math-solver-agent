import unittest

from src.tools.utils import SanitizationError, safe_numeric_eval, sanitize_math_expression


class ToolsUtilsTestCase(unittest.TestCase):
    def test_safe_numeric_eval_simple_expression(self) -> None:
        value = safe_numeric_eval("2 + 2 * 3")
        self.assertEqual(value, 8.0)

    def test_safe_numeric_eval_with_variables(self) -> None:
        value = safe_numeric_eval("x^2 + 1", {"x": 3})
        self.assertEqual(value, 10.0)

    def test_sanitization_blocks_unsafe_pattern(self) -> None:
        with self.assertRaises(SanitizationError):
            sanitize_math_expression("__import__('os')")

    def test_sanitization_normalizes_unicode_math(self) -> None:
        normalized = sanitize_math_expression("x³ + 2×x − 1")
        self.assertEqual(normalized, "x^3 + 2*x - 1")


if __name__ == "__main__":
    unittest.main()

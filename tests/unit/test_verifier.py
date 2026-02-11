import unittest

from src.nodes.verifier import _is_plausible


class VerifierTestCase(unittest.TestCase):
    def test_is_plausible_false_for_error_messages(self) -> None:
        self.assertFalse(_is_plausible("Expression contains unsupported characters."))
        self.assertFalse(_is_plausible("SymPy is required for symbolic differentiation."))

    def test_is_plausible_true_for_valid_expression(self) -> None:
        self.assertTrue(_is_plausible("3*x**2"))


if __name__ == "__main__":
    unittest.main()

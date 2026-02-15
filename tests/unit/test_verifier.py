import unittest

from src.nodes.verifier import _is_plausible, _normalize_latex_delimiters


class VerifierTestCase(unittest.TestCase):
    def test_is_plausible_false_for_error_messages(self) -> None:
        self.assertFalse(_is_plausible("Expression contains unsupported characters."))
        self.assertFalse(_is_plausible("SymPy is required for symbolic differentiation."))

    def test_is_plausible_true_for_valid_expression(self) -> None:
        self.assertTrue(_is_plausible("3*x**2"))

    # -------------------------
    # normalize latex delimiters
    # -------------------------

    def test_normalize_empty_and_none(self) -> None:
        self.assertEqual(_normalize_latex_delimiters(""), "")
        self.assertEqual(_normalize_latex_delimiters(None), "")  # type: ignore[arg-type]

    def test_normalize_inline_single_occurrence(self) -> None:
        text = r"Resultado: \(x^2 + 1\)"
        self.assertEqual(_normalize_latex_delimiters(text), "Resultado: $x^2 + 1$")

    def test_normalize_block_single_occurrence(self) -> None:
        text = r"Passo:\n\[\frac{1}{2}\]"
        self.assertEqual(_normalize_latex_delimiters(text), r"Passo:\n$$\frac{1}{2}$$")

    def test_normalize_multiple_inline_occurrences_same_line(self) -> None:
        text = r"\(a\) + \(b\) = \(c\)"
        self.assertEqual(_normalize_latex_delimiters(text), "$a$ + $b$ = $c$")

    def test_normalize_multiple_block_occurrences(self) -> None:
        text = r"Um:\[\alpha\] Dois:\[\beta\]"
        self.assertEqual(_normalize_latex_delimiters(text), "Um:$$\\alpha$$ Dois:$$\\beta$$")

    def test_normalize_mixed_inline_and_block(self) -> None:
        text = r"Inline \(x\), bloco \[y\], inline \(z^2\)."
        self.assertEqual(_normalize_latex_delimiters(text), "Inline $x$, bloco $$y$$, inline $z^2$.")

    def test_normalize_multiline_content_inside_delimiters(self) -> None:
        text = r"Eq:\[\begin{aligned}a&=b\\c&=d\end{aligned}\]"
        expected = r"Eq:$$\begin{aligned}a&=b\\c&=d\end{aligned}$$"
        self.assertEqual(_normalize_latex_delimiters(text), expected)

    def test_normalize_preserves_non_delimiter_backslashes(self) -> None:
        # não deve alterar comandos LaTeX, só os delimitadores
        text = r"Comando: \frac{1}{2} e inline \(x\)."
        self.assertEqual(_normalize_latex_delimiters(text), r"Comando: \frac{1}{2} e inline $x$.")

    def test_normalize_replaces_unicode_middle_dot_with_cdot(self) -> None:
        text = r"Produto: \(a · b\) e \[c·d\]"
        self.assertEqual(_normalize_latex_delimiters(text), r"Produto: $a  \cdotp  b$ e $$c \cdotp d$$")

    def test_normalize_does_not_touch_existing_dollar_math(self) -> None:
        text = r"Já ok: $x^2$ e $$y$$ e também \(z\)."
        self.assertEqual(_normalize_latex_delimiters(text), r"Já ok: $x^2$ e $$y$$ e também $z$.")

    def test_normalize_is_idempotent(self) -> None:
        text = r"Mix: \(x\) \[y\] $z$ $$w$$"
        once = _normalize_latex_delimiters(text)
        twice = _normalize_latex_delimiters(once)
        self.assertEqual(once, twice)

    def test_normalize_no_delimiters_no_change(self) -> None:
        text = r"Sem matemática aqui: \frac{1}{2} literal."
        self.assertEqual(_normalize_latex_delimiters(text), text)

    def test_normalize_unmatched_opening_or_closing_is_ignored(self) -> None:
        # regex só pega pares completos
        self.assertEqual(_normalize_latex_delimiters(r"Abre só: \("), r"Abre só: \(")
        self.assertEqual(_normalize_latex_delimiters(r"Fecha só: \)"), r"Fecha só: \)")
        self.assertEqual(_normalize_latex_delimiters(r"Abre bloco: \["), r"Abre bloco: \[")
        self.assertEqual(_normalize_latex_delimiters(r"Fecha bloco: \]"), r"Fecha bloco: \]")

    def test_normalize_adjacent_delimiters(self) -> None:
        # sem espaços entre expressões
        text = r"\(a\)\(b\)\[c\]\[d\]"
        self.assertEqual(_normalize_latex_delimiters(text), r"$a$$b$$$c$$$$d$$")

    def test_normalize_handles_spaces_inside_delimiters(self) -> None:
        text = r"Calc: \(  x + 1  \) e \[  y  \]"
        self.assertEqual(_normalize_latex_delimiters(text), "Calc: $  x + 1  $ e $$  y  $$")

    # -------------------------
    # casos conhecidos "problemáticos" com regex atual
    # -------------------------

    @unittest.expectedFailure
    def test_normalize_nested_delimiters_should_be_handled_but_is_not(self) -> None:
        # Com o regex atual (.*?), isso tende a se perder (fecha no primeiro \) / \]).
        # Esse teste documenta um comportamento desejado (se você futuramente melhorar o parser).
        text = r"Nested: \(a + \(b\)\)"
        self.assertEqual(_normalize_latex_delimiters(text), r"Nested: $a + $b$$")

    @unittest.expectedFailure
    def test_normalize_delimiter_tokens_inside_math(self) -> None:
        # Se o conteúdo tiver \) literal, o regex pode fechar cedo.
        text = r"Token: \(a \) b\)"
        self.assertEqual(_normalize_latex_delimiters(text), r"Token: $a \) b$")


if __name__ == "__main__":
    unittest.main()

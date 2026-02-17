import unittest

from src.ui.api_client import ImageAttachment, build_solve_payload, infer_image_media_type


class UIClientTestCase(unittest.TestCase):
    def test_build_payload_text_only(self) -> None:
        payload = build_solve_payload(problem="Calcule 2+2")
        self.assertEqual(payload["problem"], "Calcule 2+2")
        self.assertEqual(payload["ocr_mode"], "auto")
        self.assertNotIn("image_base64", payload)

    def test_build_payload_image_only(self) -> None:
        payload = build_solve_payload(problem="", image_bytes=b"abc", image_filename="problem.png")
        self.assertIn("images", payload)
        self.assertIn("image_base64", payload)
        self.assertEqual(payload["image_media_type"], "image/png")

    def test_build_payload_multi_image(self) -> None:
        payload = build_solve_payload(
            problem="",
            image_attachments=[
                ImageAttachment(image_bytes=b"aaa", image_filename="a.png"),
                ImageAttachment(image_bytes=b"bbb", image_filename="b.jpg"),
            ],
        )
        self.assertIn("images", payload)
        self.assertEqual(len(payload["images"]), 2)
        self.assertEqual(payload["images"][0]["image_media_type"], "image/png")
        self.assertEqual(payload["images"][1]["image_media_type"], "image/jpeg")

    def test_build_payload_analysis_only_flag(self) -> None:
        payload = build_solve_payload(problem="Analise", analysis_only=True)
        self.assertTrue(payload["analysis_only"])

    def test_build_payload_includes_ocr_fields(self) -> None:
        payload = build_solve_payload(problem="", ocr_mode="on", ocr_text="x^2 + 1 = 0", image_bytes=b"abc")
        self.assertEqual(payload["ocr_mode"], "on")
        self.assertEqual(payload["ocr_text"], "x^2 + 1 = 0")

    def test_build_payload_requires_input(self) -> None:
        with self.assertRaises(ValueError):
            build_solve_payload(problem="")

    def test_infer_media_type(self) -> None:
        self.assertEqual(infer_image_media_type("figure.jpg"), "image/jpeg")

    def test_build_payload_with_llm_overrides(self) -> None:
        payload = build_solve_payload(
            problem="Calcule a derivada de x^3",
            provider="maritaca",
            model_profile="sabia_4",
            temperature=0.7,
            max_tokens=4096,
        )
        self.assertEqual(payload["provider"], "maritaca")
        self.assertEqual(payload["model_profile"], "sabia_4")
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()

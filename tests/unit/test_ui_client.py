import unittest

from src.ui.api_client import build_solve_payload, infer_image_media_type


class UIClientTestCase(unittest.TestCase):
    def test_build_payload_text_only(self) -> None:
        payload = build_solve_payload(problem="Calcule 2+2")
        self.assertEqual(payload["problem"], "Calcule 2+2")
        self.assertNotIn("image_base64", payload)

    def test_build_payload_image_only(self) -> None:
        payload = build_solve_payload(problem="", image_bytes=b"abc", image_filename="problem.png")
        self.assertIn("image_base64", payload)
        self.assertEqual(payload["image_media_type"], "image/png")

    def test_build_payload_requires_input(self) -> None:
        with self.assertRaises(ValueError):
            build_solve_payload(problem="")

    def test_infer_media_type(self) -> None:
        self.assertEqual(infer_image_media_type("figure.jpg"), "image/jpeg")


if __name__ == "__main__":
    unittest.main()

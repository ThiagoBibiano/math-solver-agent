import unittest
from unittest.mock import patch

from src.tools.ocr import OCRProcessingError, OCRUnavailableError, extract_text_from_images


class _FakeRapidOCR:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image_ref):  # noqa: ANN001 - test double signature
        del image_ref
        self.calls += 1
        if self.calls == 1:
            return ([[None, "x^2 + 1 = 0", 0.9]], None)
        return ([[None, "x = i", 0.2]], None)


class OCRToolTestCase(unittest.TestCase):
    def test_raises_when_dependency_missing(self) -> None:
        with patch("src.tools.ocr.RapidOCR", None):
            with self.assertRaises(OCRUnavailableError):
                extract_text_from_images(images=[{"image_base64": "aGVsbG8=", "image_media_type": "image/png"}])

    def test_extracts_pages_and_warnings(self) -> None:
        with patch("src.tools.ocr.RapidOCR", _FakeRapidOCR):
            result = extract_text_from_images(
                images=[
                    {"image_base64": "aGVsbG8=", "image_media_type": "image/png"},
                    {"image_base64": "d29ybGQ=", "image_media_type": "image/png"},
                ],
                config={"min_confidence": 0.35, "merge_strategy": "lines"},
            )

        self.assertEqual(result["engine"], "rapidocr")
        self.assertEqual(len(result["pages"]), 2)
        self.assertIn("x^2 + 1 = 0", result["text"])
        self.assertEqual(result["pages"][1]["text"], "")
        self.assertGreaterEqual(len(result["warnings"]), 1)

    def test_requires_at_least_one_image(self) -> None:
        with patch("src.tools.ocr.RapidOCR", _FakeRapidOCR):
            with self.assertRaises(OCRProcessingError):
                extract_text_from_images(images=[])


if __name__ == "__main__":
    unittest.main()

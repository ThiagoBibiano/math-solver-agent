import unittest
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from src.llm.client import (
    _build_model_profile_registry,
    GenerativeMathClient,
    _resolve_model_profile,
    _slugify_model_alias,
    _build_human_content,
    _extract_json_dict,
    _manual_load_env_file,
    _resolve_api_key,
    _resolve_image_data_url,
)


class LLMClientTestCase(unittest.TestCase):
    def test_extract_json_from_fenced_block(self) -> None:
        content = """```json\n{\"domain\": \"calculo_i\", \"complexity_score\": 0.3}\n```"""
        parsed = _extract_json_dict(content, default={})
        self.assertEqual(parsed.get("domain"), "calculo_i")
        self.assertAlmostEqual(float(parsed.get("complexity_score")), 0.3, places=8)

    def test_client_disabled_by_config(self) -> None:
        client = GenerativeMathClient(config={"enabled": False})
        meta = client.describe()
        self.assertFalse(meta["available"])
        self.assertEqual(meta["reason"], "disabled_by_config")

    def test_resolve_image_data_url_from_base64(self) -> None:
        data_url = _resolve_image_data_url(
            image_input={"image_base64": "aGVsbG8=", "image_media_type": "image/png"},
            max_image_bytes=1024,
        )
        self.assertTrue(str(data_url).startswith("data:image/png;base64,"))

    def test_resolve_image_data_url_from_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "problem.png"
            image_path.write_bytes(b"fake-image")
            data_url = _resolve_image_data_url(
                image_input={"image_path": str(image_path)},
                max_image_bytes=1024,
            )
        self.assertTrue(str(data_url).startswith("data:image/png;base64,"))

    def test_build_human_content_includes_image_part(self) -> None:
        content = _build_human_content(
            user_prompt="Analise a imagem",
            image_input={"image_base64": "aGVsbG8=", "image_media_type": "image/png"},
            max_image_bytes=1024,
        )
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")

    def test_resolve_api_key_with_strip(self) -> None:
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "  test-key  "}, clear=False):
            value = _resolve_api_key(["NVIDIA_API_KEY"])
        self.assertEqual(value, "test-key")

    def test_manual_load_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("NVIDIA_API_KEY=abc123\n# comment\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                _manual_load_env_file(env_path)
                self.assertEqual(os.getenv("NVIDIA_API_KEY"), "abc123")

    def test_resolve_model_profile_from_alias(self) -> None:
        registry = _build_model_profile_registry(None)
        profile = _resolve_model_profile("glm5", "", registry)
        self.assertEqual(profile.model, "z-ai/glm5")
        self.assertTrue(profile.multimodal)

    def test_resolve_model_profile_from_explicit_model(self) -> None:
        registry = _build_model_profile_registry(None)
        profile = _resolve_model_profile("", "deepseek-ai/deepseek-v3.2", registry)
        self.assertEqual(profile.alias, "deepseek_v3_2")
        self.assertFalse(profile.multimodal)

    def test_slugify_model_alias(self) -> None:
        self.assertEqual(_slugify_model_alias("deepseek-ai/deepseek-v3.2"), "deepseek_ai_deepseek_v3_2")

    def test_config_disables_multimodal_for_non_multimodal_model(self) -> None:
        client = GenerativeMathClient(
            config={
                "enabled": False,
                "model_profile": "deepseek_v3_2",
                "multimodal_enabled": True,
            }
        )
        self.assertFalse(client.config.multimodal_enabled)


if __name__ == "__main__":
    unittest.main()

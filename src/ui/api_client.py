"""HTTP client helpers for the Streamlit frontend."""

from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SolveInput:
    """Represents minimal solve request input for UI interactions."""
    problem: str = ""
    session_id: Optional[str] = None
    resume: bool = False
    image_base64: Optional[str] = None
    image_media_type: str = "image/png"


def encode_image_bytes(image_bytes: bytes) -> str:
    """Encodes raw image bytes as base64 text.

    Args:
        image_bytes: Raw image binary data.

    Returns:
        Base64-encoded ASCII string.
    """
    return base64.b64encode(image_bytes).decode("ascii")


def infer_image_media_type(filename: str, fallback: str = "image/png") -> str:
    """Infers MIME type from filename extension.

    Args:
        filename: Original uploaded filename.
        fallback: Fallback media type when inference fails.

    Returns:
        Inferred image media type.
    """
    guessed, _ = mimetypes.guess_type(filename)
    if guessed and guessed.startswith("image/"):
        return guessed
    return fallback


def build_solve_payload(
    problem: str,
    session_id: Optional[str] = None,
    resume: bool = False,
    image_bytes: Optional[bytes] = None,
    image_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Builds `/v1/solve` payload from UI form data.

    Args:
        problem: Problem statement text.
        session_id: Optional session identifier.
        resume: Resume flag for checkpoint recovery.
        image_bytes: Optional raw image bytes.
        image_filename: Optional source filename for media-type inference.

    Returns:
        JSON-serializable payload dictionary.

    Raises:
        ValueError: If neither text nor image input is provided.
    """
    cleaned_problem = (problem or "").strip()
    payload: Dict[str, Any] = {
        "problem": cleaned_problem,
        "session_id": session_id,
        "resume": bool(resume),
    }

    if image_bytes:
        payload["image_base64"] = encode_image_bytes(image_bytes)
        payload["image_media_type"] = infer_image_media_type(image_filename or "")

    if not cleaned_problem and not resume and "image_base64" not in payload:
        raise ValueError("Informe texto, imagem, ou ambos.")

    return payload


def call_solve_api(base_url: str, payload: Dict[str, Any], timeout_seconds: int = 120) -> Dict[str, Any]:
    """Calls REST solve endpoint and returns parsed JSON response.

    Args:
        base_url: API base URL.
        payload: Solve request payload.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Response dictionary from API.

    Raises:
        RuntimeError: If HTTP/network errors occur.
    """
    endpoint = base_url.rstrip("/") + "/v1/solve"
    request = urllib.request.Request(
        url=endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError("API error {}: {}".format(exc.code, body)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Nao foi possivel conectar na API: {}".format(exc.reason)) from exc

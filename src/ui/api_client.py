"""HTTP client helpers for the Streamlit frontend."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from typing import AsyncIterator
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("math_solver_agent.ui.api_client")


@dataclass
class SolveInput:
    """Represents minimal solve request input for UI interactions."""
    problem: str = ""
    session_id: Optional[str] = None
    resume: bool = False
    image_base64: Optional[str] = None
    image_media_type: str = "image/png"


@dataclass
class ImageAttachment:
    """Represents one local image attachment collected by frontend UI."""

    image_bytes: bytes
    image_filename: Optional[str] = None
    image_media_type: Optional[str] = None


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
    analysis_only: bool = False,
    ocr_mode: str = "auto",
    ocr_text: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_filename: Optional[str] = None,
    image_attachments: Optional[list[Any]] = None,
    provider: Optional[str] = None,
    model_profile: Optional[str] = None,
    model: Optional[str] = None,
    api_key_env: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
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
        "analysis_only": bool(analysis_only),
        "ocr_mode": str(ocr_mode or "auto").strip().lower() or "auto",
    }
    cleaned_ocr_text = str(ocr_text or "").strip()
    if cleaned_ocr_text:
        payload["ocr_text"] = cleaned_ocr_text

    override_fields = {
        "provider": provider,
        "model_profile": model_profile,
        "model": model,
        "api_key_env": api_key_env,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    for key, value in override_fields.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        payload[key] = value

    images_payload: list[Dict[str, str]] = []
    if image_attachments:
        for attachment in image_attachments:
            if isinstance(attachment, ImageAttachment):
                raw_bytes = attachment.image_bytes
                raw_name = attachment.image_filename
                raw_mime = attachment.image_media_type
            elif isinstance(attachment, dict):
                raw_bytes = attachment.get("image_bytes")
                raw_name = attachment.get("image_filename")
                raw_mime = attachment.get("image_media_type")
            else:
                continue
            if not raw_bytes:
                continue
            media_type = str(raw_mime or "").strip() or infer_image_media_type(str(raw_name or ""))
            images_payload.append(
                {
                    "image_base64": encode_image_bytes(raw_bytes),
                    "image_media_type": media_type,
                }
            )

    if image_bytes:
        legacy_media_type = infer_image_media_type(image_filename or "")
        payload["image_base64"] = encode_image_bytes(image_bytes)
        payload["image_media_type"] = legacy_media_type
        images_payload.append(
            {
                "image_base64": payload["image_base64"],
                "image_media_type": legacy_media_type,
            }
        )

    if images_payload:
        payload["images"] = images_payload

    if not cleaned_problem and not resume and "images" not in payload and "image_base64" not in payload:
        raise ValueError("Informe texto, imagem, ou ambos.")

    return payload


def call_solve_api(
    base_url: str,
    payload: Dict[str, Any],
    timeout_seconds: int = 120,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
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
        headers={
            "Content-Type": "application/json",
            **({"X-Request-ID": request_id} if request_id else {}),
        },
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


async def call_solve_api_async(
    base_url: str,
    payload: Dict[str, Any],
    timeout_seconds: int = 120,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Calls REST solve endpoint asynchronously using httpx.

    Args:
        base_url: API base URL.
        payload: Solve request payload.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Response dictionary from API.

    Raises:
        RuntimeError: If HTTP/network errors occur.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("httpx is required for async API calls. Install dependencies from pyproject.toml.") from exc

    endpoint = base_url.rstrip("/") + "/v1/solve"
    started_at = time.perf_counter()
    logger.info("HTTP POST start endpoint=%s timeout=%ss", endpoint, timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            headers = {"X-Request-ID": request_id} if request_id else None
            response = await client.post(endpoint, json=payload, headers=headers)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            logger.info("HTTP POST done endpoint=%s status=%s elapsed_ms=%.1f", endpoint, response.status_code, elapsed_ms)
            if response.status_code >= 400:
                detail = _extract_error_detail(response.text)
                raise RuntimeError("API error {}: {}".format(response.status_code, detail))
            return response.json()
    except httpx.ConnectError as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP connect error endpoint=%s elapsed_ms=%.1f error=%s", endpoint, elapsed_ms, exc)
        raise RuntimeError("Nao foi possivel conectar na API: {}".format(exc)) from exc
    except httpx.TimeoutException as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP timeout endpoint=%s elapsed_ms=%.1f", endpoint, elapsed_ms)
        raise RuntimeError("Timeout ao consultar API apos {}s".format(timeout_seconds)) from exc


async def call_export_api_async(
    base_url: str,
    payload: Dict[str, Any],
    timeout_seconds: int = 120,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Calls REST export endpoint asynchronously using httpx."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("httpx is required for async API calls. Install dependencies from pyproject.toml.") from exc

    endpoint = base_url.rstrip("/") + "/v1/export"
    started_at = time.perf_counter()
    logger.info("HTTP POST start endpoint=%s timeout=%ss", endpoint, timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            headers = {"X-Request-ID": request_id} if request_id else None
            response = await client.post(endpoint, json=payload, headers=headers)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            logger.info("HTTP POST done endpoint=%s status=%s elapsed_ms=%.1f", endpoint, response.status_code, elapsed_ms)
            if response.status_code >= 400:
                detail = _extract_error_detail(response.text)
                raise RuntimeError("API error {}: {}".format(response.status_code, detail))
            return response.json()
    except httpx.ConnectError as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP connect error endpoint=%s elapsed_ms=%.1f error=%s", endpoint, elapsed_ms, exc)
        raise RuntimeError("Nao foi possivel conectar na API: {}".format(exc)) from exc
    except httpx.TimeoutException as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP timeout endpoint=%s elapsed_ms=%.1f", endpoint, elapsed_ms)
        raise RuntimeError("Timeout ao consultar API apos {}s".format(timeout_seconds)) from exc


async def call_sessions_api_async(
    base_url: str,
    limit: int = 10,
    timeout_seconds: int = 60,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Calls sessions listing endpoint asynchronously using httpx."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("httpx is required for async API calls. Install dependencies from pyproject.toml.") from exc

    endpoint = base_url.rstrip("/") + "/v1/sessions"
    started_at = time.perf_counter()
    params = {"limit": max(1, min(int(limit), 100))}
    logger.info("HTTP GET start endpoint=%s timeout=%ss", endpoint, timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            headers = {"X-Request-ID": request_id} if request_id else None
            response = await client.get(endpoint, params=params, headers=headers)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            logger.info("HTTP GET done endpoint=%s status=%s elapsed_ms=%.1f", endpoint, response.status_code, elapsed_ms)
            if response.status_code >= 400:
                detail = _extract_error_detail(response.text)
                raise RuntimeError("API error {}: {}".format(response.status_code, detail))
            return response.json()
    except httpx.ConnectError as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP connect error endpoint=%s elapsed_ms=%.1f error=%s", endpoint, elapsed_ms, exc)
        raise RuntimeError("Nao foi possivel conectar na API: {}".format(exc)) from exc
    except httpx.TimeoutException as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP timeout endpoint=%s elapsed_ms=%.1f", endpoint, elapsed_ms)
        raise RuntimeError("Timeout ao consultar API apos {}s".format(timeout_seconds)) from exc


async def call_runtime_status_api_async(
    base_url: str,
    timeout_seconds: int = 60,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Calls runtime status endpoint asynchronously using httpx."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("httpx is required for async API calls. Install dependencies from pyproject.toml.") from exc

    endpoint = base_url.rstrip("/") + "/v1/runtime/status"
    started_at = time.perf_counter()
    logger.info("HTTP GET start endpoint=%s timeout=%ss", endpoint, timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            headers = {"X-Request-ID": request_id} if request_id else None
            response = await client.get(endpoint, headers=headers)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            logger.info("HTTP GET done endpoint=%s status=%s elapsed_ms=%.1f", endpoint, response.status_code, elapsed_ms)
            if response.status_code >= 400:
                detail = _extract_error_detail(response.text)
                raise RuntimeError("API error {}: {}".format(response.status_code, detail))
            return response.json()
    except httpx.ConnectError as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP connect error endpoint=%s elapsed_ms=%.1f error=%s", endpoint, elapsed_ms, exc)
        raise RuntimeError("Nao foi possivel conectar na API: {}".format(exc)) from exc
    except httpx.TimeoutException as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP timeout endpoint=%s elapsed_ms=%.1f", endpoint, elapsed_ms)
        raise RuntimeError("Timeout ao consultar API apos {}s".format(timeout_seconds)) from exc


async def call_solve_job_submit_api_async(
    base_url: str,
    payload: Dict[str, Any],
    timeout_seconds: int = 60,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Submits a solve job asynchronously using httpx."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("httpx is required for async API calls. Install dependencies from pyproject.toml.") from exc

    endpoint = base_url.rstrip("/") + "/v1/jobs/solve"
    started_at = time.perf_counter()
    logger.info("HTTP POST start endpoint=%s timeout=%ss", endpoint, timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            headers = {"X-Request-ID": request_id} if request_id else None
            response = await client.post(endpoint, json=payload, headers=headers)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            logger.info("HTTP POST done endpoint=%s status=%s elapsed_ms=%.1f", endpoint, response.status_code, elapsed_ms)
            if response.status_code >= 400:
                detail = _extract_error_detail(response.text)
                raise RuntimeError("API error {}: {}".format(response.status_code, detail))
            return response.json()
    except httpx.ConnectError as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP connect error endpoint=%s elapsed_ms=%.1f error=%s", endpoint, elapsed_ms, exc)
        raise RuntimeError("Nao foi possivel conectar na API: {}".format(exc)) from exc
    except httpx.TimeoutException as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP timeout endpoint=%s elapsed_ms=%.1f", endpoint, elapsed_ms)
        raise RuntimeError("Timeout ao consultar API apos {}s".format(timeout_seconds)) from exc


async def call_solve_job_status_api_async(
    base_url: str,
    job_id: str,
    timeout_seconds: int = 60,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Polls solve job status asynchronously using httpx."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("httpx is required for async API calls. Install dependencies from pyproject.toml.") from exc

    endpoint = base_url.rstrip("/") + "/v1/jobs/{}".format(job_id)
    started_at = time.perf_counter()
    logger.info("HTTP GET start endpoint=%s timeout=%ss", endpoint, timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            headers = {"X-Request-ID": request_id} if request_id else None
            response = await client.get(endpoint, headers=headers)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            logger.info("HTTP GET done endpoint=%s status=%s elapsed_ms=%.1f", endpoint, response.status_code, elapsed_ms)
            if response.status_code >= 400:
                detail = _extract_error_detail(response.text)
                raise RuntimeError("API error {}: {}".format(response.status_code, detail))
            return response.json()
    except httpx.ConnectError as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP connect error endpoint=%s elapsed_ms=%.1f error=%s", endpoint, elapsed_ms, exc)
        raise RuntimeError("Nao foi possivel conectar na API: {}".format(exc)) from exc
    except httpx.TimeoutException as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP timeout endpoint=%s elapsed_ms=%.1f", endpoint, elapsed_ms)
        raise RuntimeError("Timeout ao consultar API apos {}s".format(timeout_seconds)) from exc


async def call_solve_job_cancel_api_async(
    base_url: str,
    job_id: str,
    timeout_seconds: int = 60,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Cancels solve job asynchronously using httpx."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("httpx is required for async API calls. Install dependencies from pyproject.toml.") from exc

    endpoint = base_url.rstrip("/") + "/v1/jobs/{}".format(job_id)
    started_at = time.perf_counter()
    logger.info("HTTP DELETE start endpoint=%s timeout=%ss", endpoint, timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            headers = {"X-Request-ID": request_id} if request_id else None
            response = await client.delete(endpoint, headers=headers)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            logger.info(
                "HTTP DELETE done endpoint=%s status=%s elapsed_ms=%.1f",
                endpoint,
                response.status_code,
                elapsed_ms,
            )
            if response.status_code >= 400:
                detail = _extract_error_detail(response.text)
                raise RuntimeError("API error {}: {}".format(response.status_code, detail))
            return response.json()
    except httpx.ConnectError as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP connect error endpoint=%s elapsed_ms=%.1f error=%s", endpoint, elapsed_ms, exc)
        raise RuntimeError("Nao foi possivel conectar na API: {}".format(exc)) from exc
    except httpx.TimeoutException as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP timeout endpoint=%s elapsed_ms=%.1f", endpoint, elapsed_ms)
        raise RuntimeError("Timeout ao consultar API apos {}s".format(timeout_seconds)) from exc


async def call_ocr_extract_api_async(
    base_url: str,
    payload: Dict[str, Any],
    timeout_seconds: int = 120,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Calls OCR extraction endpoint asynchronously using httpx."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("httpx is required for async API calls. Install dependencies from pyproject.toml.") from exc

    endpoint = base_url.rstrip("/") + "/v1/ocr/extract"
    started_at = time.perf_counter()
    logger.info("HTTP POST start endpoint=%s timeout=%ss", endpoint, timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            headers = {"X-Request-ID": request_id} if request_id else None
            response = await client.post(endpoint, json=payload, headers=headers)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            logger.info("HTTP POST done endpoint=%s status=%s elapsed_ms=%.1f", endpoint, response.status_code, elapsed_ms)
            if response.status_code >= 400:
                detail = _extract_error_detail(response.text)
                raise RuntimeError("API error {}: {}".format(response.status_code, detail))
            return response.json()
    except httpx.ConnectError as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP connect error endpoint=%s elapsed_ms=%.1f error=%s", endpoint, elapsed_ms, exc)
        raise RuntimeError("Nao foi possivel conectar na API: {}".format(exc)) from exc
    except httpx.TimeoutException as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("HTTP timeout endpoint=%s elapsed_ms=%.1f", endpoint, elapsed_ms)
        raise RuntimeError("Timeout ao consultar API apos {}s".format(timeout_seconds)) from exc


def _extract_error_detail(raw_body: str) -> str:
    """Extracts readable `detail` from API error payloads."""
    try:
        parsed = json.loads(raw_body)
        if isinstance(parsed, dict) and "detail" in parsed:
            return str(parsed["detail"])
    except json.JSONDecodeError:
        pass
    return raw_body.strip() or "Erro desconhecido"


def _to_ws_endpoint(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.startswith("https://"):
        return "wss://" + normalized[len("https://") :] + "/v1/solve/stream"
    if normalized.startswith("http://"):
        return "ws://" + normalized[len("http://") :] + "/v1/solve/stream"
    if normalized.startswith("wss://") or normalized.startswith("ws://"):
        return normalized + "/v1/solve/stream"
    return "ws://{}/v1/solve/stream".format(normalized)


async def stream_solve_api_async(
    base_url: str,
    payload: Dict[str, Any],
    timeout_seconds: int = 120,
    request_id: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Streams solve events from WS endpoint."""
    try:
        import websockets
        from websockets.exceptions import WebSocketException
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("websockets is required for streaming API calls. Install dependencies from pyproject.toml.") from exc

    endpoint = _to_ws_endpoint(base_url)
    started_at = time.perf_counter()
    logger.info("WS connect start endpoint=%s timeout=%ss", endpoint, timeout_seconds)
    headers = {"X-Request-ID": request_id} if request_id else None
    try:
        async with websockets.connect(
            endpoint,
            additional_headers=headers,
            open_timeout=timeout_seconds,
            close_timeout=5,
            max_size=20 * 1024 * 1024,
        ) as ws:
            await ws.send(json.dumps(payload, ensure_ascii=False))
            while True:
                raw = await ws.recv()
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    continue
                yield parsed
                if parsed.get("type") in {"done", "result", "error"} and parsed.get("type") != "result":
                    break
    except (OSError, WebSocketException) as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("WS failed endpoint=%s elapsed_ms=%.1f error=%s", endpoint, elapsed_ms, exc)
        raise RuntimeError("Nao foi possivel conectar no stream da API: {}".format(exc)) from exc

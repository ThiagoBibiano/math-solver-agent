"""OCR utilities backed by RapidOCR."""

from __future__ import annotations

import base64
import binascii
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

try:  # pragma: no cover - optional dependency
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover
    RapidOCR = None  # type: ignore[assignment]


class OCRUnavailableError(RuntimeError):
    """Raised when OCR dependency is unavailable."""


class OCRProcessingError(RuntimeError):
    """Raised when OCR processing fails."""


class OCRLine(TypedDict, total=False):
    text: str
    confidence: float


class OCRPage(TypedDict, total=False):
    index: int
    text: str
    confidence: float
    raw_confidence: float
    lines: List[OCRLine]


class OCRResult(TypedDict):
    text: str
    pages: List[OCRPage]
    engine: str
    warnings: List[str]


def extract_text_from_images(images: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> OCRResult:
    """Extracts OCR text from a list of image payloads.

    Args:
        images: List containing image descriptors (`image_path`, `image_url`, or `image_base64`).
        config: Optional OCR configuration mapping.

    Returns:
        OCRResult with merged text and per-page details.

    Raises:
        OCRUnavailableError: If RapidOCR dependency is unavailable.
        OCRProcessingError: If no image could be processed.
    """
    if RapidOCR is None:
        raise OCRUnavailableError(
            "RapidOCR is not available. Install dependency `rapidocr_onnxruntime` to enable dedicated OCR."
        )

    if not isinstance(images, list) or not images:
        raise OCRProcessingError("No image payload provided for OCR.")

    cfg = dict(config or {})
    min_confidence = _safe_float(cfg.get("min_confidence"), default=0.0)
    merge_strategy = str(cfg.get("merge_strategy", "lines")).strip().lower() or "lines"
    if merge_strategy not in {"lines", "paragraph"}:
        merge_strategy = "lines"

    ocr = RapidOCR()
    pages: List[OCRPage] = []
    warnings: List[str] = []

    temp_files: List[str] = []
    try:
        materialized = _materialize_images(images, temp_files=temp_files)
        if not materialized:
            raise OCRProcessingError("No valid image found for OCR.")

        for index, image_ref in enumerate(materialized, start=1):
            page = _run_ocr_page(
                ocr=ocr,
                image_ref=image_ref,
                index=index,
                min_confidence=min_confidence,
                merge_strategy=merge_strategy,
            )
            pages.append(page)
            confidence = page.get("raw_confidence", page.get("confidence"))
            if confidence is not None and confidence < min_confidence:
                warnings.append(
                    "Baixa confianca OCR na imagem {} ({:.2f} < {:.2f}).".format(index, confidence, min_confidence)
                )
    finally:
        for tmp in temp_files:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass

    merged_text = "\n\n".join(str(page.get("text", "")).strip() for page in pages if str(page.get("text", "")).strip())
    return OCRResult(
        text=merged_text,
        pages=pages,
        engine="rapidocr",
        warnings=warnings,
    )


def _run_ocr_page(
    ocr: Any,
    image_ref: str,
    index: int,
    min_confidence: float,
    merge_strategy: str,
) -> OCRPage:
    try:
        raw_result, _ = ocr(image_ref)
    except Exception as exc:
        raise OCRProcessingError("OCR failed on image {}: {}".format(index, exc)) from exc

    lines: List[OCRLine] = []
    confidences: List[float] = []
    raw_confidences: List[float] = []
    entries = raw_result if isinstance(raw_result, list) else []
    for item in entries:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = str(item[1] or "").strip()
        if not text:
            continue
        confidence = _safe_float(item[2], default=0.0)
        raw_confidences.append(confidence)
        if confidence < min_confidence:
            continue
        lines.append(OCRLine(text=text, confidence=confidence))
        confidences.append(confidence)

    if merge_strategy == "paragraph":
        page_text = " ".join(line["text"] for line in lines)
    else:
        page_text = "\n".join(line["text"] for line in lines)

    page: OCRPage = OCRPage(index=index, text=page_text, lines=lines)
    if confidences:
        page["confidence"] = round(sum(confidences) / len(confidences), 4)
    if raw_confidences:
        page["raw_confidence"] = round(sum(raw_confidences) / len(raw_confidences), 4)
    return page


def _materialize_images(images: List[Dict[str, Any]], temp_files: List[str]) -> List[str]:
    refs: List[str] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        media_type = str(image.get("image_media_type") or "image/png").strip() or "image/png"
        image_path = str(image.get("image_path") or "").strip()
        image_url = str(image.get("image_url") or "").strip()
        image_base64 = str(image.get("image_base64") or "").strip()

        if image_path:
            path = Path(image_path).expanduser()
            if path.exists() and path.is_file():
                refs.append(str(path))
                continue

        if image_url:
            payload = _download_bytes(image_url)
            if payload is not None:
                refs.append(_save_temp_image(payload, media_type=media_type, temp_files=temp_files))
                continue

        if image_base64:
            payload = _decode_base64_image(image_base64)
            if payload is not None:
                refs.append(_save_temp_image(payload, media_type=media_type, temp_files=temp_files))
                continue
    return refs


def _save_temp_image(payload: bytes, media_type: str, temp_files: List[str]) -> str:
    suffix = _media_type_to_suffix(media_type)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        temp_files.append(handle.name)
        return handle.name


def _download_bytes(url: str) -> Optional[bytes]:
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return response.read()
    except Exception:
        return None


def _decode_base64_image(data: str) -> Optional[bytes]:
    raw = data.strip()
    if raw.startswith("data:") and ";base64," in raw:
        raw = raw.split(";base64,", 1)[1]
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return None


def _media_type_to_suffix(media_type: str) -> str:
    normalized = media_type.lower()
    if "jpeg" in normalized or "jpg" in normalized:
        return ".jpg"
    if "webp" in normalized:
        return ".webp"
    if "bmp" in normalized:
        return ".bmp"
    return ".png"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

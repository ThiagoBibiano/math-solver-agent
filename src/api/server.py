"""REST and WebSocket interfaces for MathSolverAgent."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from src.agents import MathSolverAgent
from src.api.runtime import (
    AdmissionController,
    AdmissionRejectedError,
    InMemoryJobStore,
    JobNotFoundError,
    JobQueueFullError,
    RuntimeStats,
)
from src.tools.ocr import OCRProcessingError, OCRUnavailableError, extract_text_from_images
from src.utils.exporters import export_latex, export_notebook
from src.utils.logger import get_logger

try:  # pragma: no cover
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]
    Request = Any  # type: ignore[misc,assignment]
    WebSocket = Any  # type: ignore[misc,assignment]
    WebSocketDisconnect = Exception  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return None


logger = get_logger("math_solver_agent.api")


if FastAPI is not None:

    class ImageInput(BaseModel):
        image_path: Optional[str] = Field(default=None, description="Local path of image on server host")
        image_url: Optional[str] = Field(default=None, description="Public URL of an image containing the problem")
        image_base64: Optional[str] = Field(default=None, description="Base64 image payload")
        image_media_type: str = Field(default="image/png", description="Media type used for image payload")

    class SolveRequest(BaseModel):
        problem: str = Field(default="", description="Mathematical problem statement")
        session_id: Optional[str] = Field(default=None, description="Optional conversation/session id")
        resume: bool = Field(default=False, description="Resume from checkpoint when true")
        analysis_only: bool = Field(default=False, description="Run only analyzer stage and return normalized problem")
        ocr_mode: str = Field(default="auto", description="OCR policy mode: auto|on|off")
        ocr_text: Optional[str] = Field(default=None, description="Pre-extracted OCR text")
        image_path: Optional[str] = Field(default=None, description="Local path of image on server host")
        image_url: Optional[str] = Field(default=None, description="Public URL of an image containing the problem")
        image_base64: Optional[str] = Field(default=None, description="Base64 image payload")
        image_media_type: str = Field(default="image/png", description="Media type used for image_base64")
        images: list[ImageInput] = Field(default_factory=list, description="Optional list of image payloads")
        provider: Optional[str] = Field(default=None, description="Override LLM provider (e.g. nvidia, maritaca)")
        model_profile: Optional[str] = Field(default=None, description="Override configured model profile alias")
        model: Optional[str] = Field(default=None, description="Override provider model identifier")
        api_key_env: Optional[str] = Field(default=None, description="Override API key env variable name")
        temperature: Optional[float] = Field(default=None, description="Override generation temperature")
        top_p: Optional[float] = Field(default=None, description="Override nucleus sampling")
        max_tokens: Optional[int] = Field(default=None, description="Override max output tokens")


    class SolveResponse(BaseModel):
        session_id: str
        status: str
        normalized_problem: Optional[str]
        domain: Optional[str]
        strategy: Optional[str]
        llm: Dict[str, Any]
        has_visual_input: bool
        result: Optional[str]
        numeric_result: Optional[float]
        verification: Dict[str, Any]
        explanation: str
        metrics: Dict[str, float]
        decision_trace: list[Dict[str, Any]]
        artifacts: list[Dict[str, Any]]
        tool_call: Dict[str, Any]


    class ExportRequest(BaseModel):
        problem: str = Field(default="")
        format: str = Field(default="latex", pattern="^(latex|notebook|json)$")
        output_path: str = Field(default="artifacts/result.tex")
        session_id: Optional[str] = Field(default=None)
        resume: Optional[bool] = Field(default=None)
        ocr_mode: str = Field(default="auto")
        ocr_text: Optional[str] = Field(default=None)
        image_path: Optional[str] = Field(default=None)
        image_url: Optional[str] = Field(default=None)
        image_base64: Optional[str] = Field(default=None)
        image_media_type: str = Field(default="image/png")
        images: list[ImageInput] = Field(default_factory=list)
        provider: Optional[str] = Field(default=None)
        model_profile: Optional[str] = Field(default=None)
        model: Optional[str] = Field(default=None)
        api_key_env: Optional[str] = Field(default=None)
        temperature: Optional[float] = Field(default=None)
        top_p: Optional[float] = Field(default=None)
        max_tokens: Optional[int] = Field(default=None)


    class OCRExtractRequest(BaseModel):
        images: list[ImageInput] = Field(default_factory=list)
        language_hint: Optional[str] = Field(default=None)
        min_confidence: Optional[float] = Field(default=None)
        merge_strategy: Optional[str] = Field(default=None)


    class OCRExtractResponse(BaseModel):
        text: str
        pages: list[Dict[str, Any]]
        engine: str
        warnings: list[str]


    class RuntimeStatusResponse(BaseModel):
        busy: bool
        in_flight_total: int
        queue_depth: int
        queue_capacity: int
        in_flight_by_provider: Dict[str, int]
        timeouts_last_5m_by_provider: Dict[str, int]
        avg_latency_ms_last_5m: float
        suggested_retry_after_seconds: int


    class JobSubmitResponse(BaseModel):
        job_id: str
        status: str
        queue_position: Optional[int]
        submitted_at: str


    class JobStatusResponse(BaseModel):
        job_id: str
        status: str
        queue_position: Optional[int]
        submitted_at: Optional[str]
        started_at: Optional[str]
        finished_at: Optional[str]
        error: Optional[str]
        result: Optional[Dict[str, Any]]
else:

    class SolveRequest:  # type: ignore[too-many-ancestors]
        pass

    class SolveResponse:  # type: ignore[too-many-ancestors]
        pass

    class ExportRequest:  # type: ignore[too-many-ancestors]
        pass

    class OCRExtractRequest:  # type: ignore[too-many-ancestors]
        pass

    class OCRExtractResponse:  # type: ignore[too-many-ancestors]
        pass

    class RuntimeStatusResponse:  # type: ignore[too-many-ancestors]
        pass

    class JobSubmitResponse:  # type: ignore[too-many-ancestors]
        pass

    class JobStatusResponse:  # type: ignore[too-many-ancestors]
        pass


def create_app() -> Any:
    """Builds and configures the FastAPI application.

    Returns:
        Configured FastAPI app instance.

    Raises:
        RuntimeError: If FastAPI dependency is not installed.
    """
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install dependencies from pyproject.toml.")

    app = FastAPI(title="MathSolverAgent API", version="0.1.0")
    agent = MathSolverAgent()

    runtime_control = dict(getattr(agent.config, "runtime_control", {}) or {})
    max_inflight_global = int(runtime_control.get("max_inflight_global", 4))
    max_inflight_by_provider = dict(runtime_control.get("max_inflight_by_provider", {"nvidia": 2, "maritaca": 2}))
    max_queue_size = int(runtime_control.get("max_queue_size", 32))
    queue_wait_timeout_seconds = float(runtime_control.get("queue_wait_timeout_seconds", 2))
    job_retention_seconds = int(runtime_control.get("job_retention_seconds", 1800))
    solve_hard_timeout_seconds = float(runtime_control.get("solve_hard_timeout_seconds", 120))
    request_executor_workers = int(runtime_control.get("request_executor_max_workers", max(4, max_inflight_global * 2)))
    job_worker_count = int(runtime_control.get("job_worker_count", max(1, max_inflight_global)))

    admission = AdmissionController(
        max_inflight_global=max_inflight_global,
        max_inflight_by_provider=max_inflight_by_provider,
        max_queue_size=max_queue_size,
        queue_wait_timeout_seconds=queue_wait_timeout_seconds,
    )
    runtime_stats = RuntimeStats(window_seconds=300)
    job_store = InMemoryJobStore(max_queue_size=max_queue_size, retention_seconds=job_retention_seconds)
    request_executor = ThreadPoolExecutor(max_workers=max(2, request_executor_workers), thread_name_prefix="api-solve")
    job_workers: list[asyncio.Task[Any]] = []

    class _SolveHardTimeoutError(RuntimeError):
        pass

    class _ProviderTimeoutError(RuntimeError):
        pass

    def _is_provider_timeout_error(exc: Exception) -> bool:
        lowered = str(exc).lower()
        return "provider_timeout" in lowered or "request exceeded" in lowered and "timeout" in lowered

    def _payload_get(payload_like: Any, key: str, default: Any = None) -> Any:
        if isinstance(payload_like, dict):
            return payload_like.get(key, default)
        return getattr(payload_like, key, default)

    def _build_llm_overrides(payload_like: Any) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        for key in ("provider", "model_profile", "model", "api_key_env", "temperature", "top_p", "max_tokens"):
            value = _payload_get(payload_like, key, None)
            if value is not None:
                overrides[key] = value
        return overrides

    def _state_to_response(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "session_id": state.get("session_id"),
            "status": state.get("status"),
            "normalized_problem": state.get("normalized_problem"),
            "domain": state.get("domain"),
            "strategy": state.get("strategy"),
            "llm": state.get("llm", {}),
            "has_visual_input": bool(state.get("visual_input")),
            "result": state.get("symbolic_result"),
            "numeric_result": state.get("numeric_result"),
            "verification": state.get("verification", {}),
            "explanation": state.get("explanation", ""),
            "metrics": state.get("metrics", {}),
            "decision_trace": state.get("decision_trace", []),
            "artifacts": state.get("artifacts", []),
            "tool_call": state.get("tool_call", {}),
        }

    def _normalize_ocr_mode(value: Any) -> str:
        mode = str(value or "auto").strip().lower() or "auto"
        if mode not in {"auto", "on", "off"}:
            return "auto"
        return mode

    def _collect_images(payload_like: Any) -> list[Dict[str, Any]]:
        images: list[Dict[str, Any]] = []
        payload_images = _payload_get(payload_like, "images", None)
        if isinstance(payload_images, list):
            for item in payload_images:
                if isinstance(item, dict):
                    candidate = {
                        "image_path": item.get("image_path"),
                        "image_url": item.get("image_url"),
                        "image_base64": item.get("image_base64"),
                        "image_media_type": item.get("image_media_type", "image/png"),
                    }
                else:
                    candidate = {
                        "image_path": getattr(item, "image_path", None),
                        "image_url": getattr(item, "image_url", None),
                        "image_base64": getattr(item, "image_base64", None),
                        "image_media_type": getattr(item, "image_media_type", "image/png"),
                    }
                if any(candidate.get(key) for key in ("image_path", "image_url", "image_base64")):
                    images.append(candidate)
        return images

    def _build_ocr_config(payload_like: Any) -> Dict[str, Any]:
        base = dict(getattr(agent.config, "ocr", {}) or {})
        if _payload_get(payload_like, "language_hint", None):
            base["language_hint"] = _payload_get(payload_like, "language_hint")
        if _payload_get(payload_like, "min_confidence", None) is not None:
            base["min_confidence"] = _payload_get(payload_like, "min_confidence")
        if _payload_get(payload_like, "merge_strategy", None):
            base["merge_strategy"] = _payload_get(payload_like, "merge_strategy")
        return base

    def _to_busy_http_exception(exc: AdmissionRejectedError) -> HTTPException:
        return HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    async def _runtime_status_payload() -> Dict[str, Any]:
        admission_snapshot = await admission.snapshot()
        stats_snapshot = runtime_stats.snapshot()
        job_queue_depth = await job_store.queue_depth()
        queue_depth = int(admission_snapshot.get("queue_depth", 0)) + int(job_queue_depth)
        queue_capacity = int(admission_snapshot.get("queue_capacity", 0)) + int(job_store.max_queue_size)
        return {
            "busy": bool(admission_snapshot.get("busy", False)),
            "in_flight_total": int(admission_snapshot.get("in_flight_total", 0)),
            "queue_depth": queue_depth,
            "queue_capacity": queue_capacity,
            "in_flight_by_provider": dict(admission_snapshot.get("in_flight_by_provider", {})),
            "timeouts_last_5m_by_provider": dict(stats_snapshot.get("timeouts_last_5m_by_provider", {})),
            "avg_latency_ms_last_5m": float(stats_snapshot.get("avg_latency_ms_last_5m", 0.0)),
            "suggested_retry_after_seconds": int(admission_snapshot.get("suggested_retry_after_seconds", 1)),
        }

    def _classify_job_outcome(state: Dict[str, Any]) -> tuple[str, Optional[str]]:
        status = str(state.get("status", "")).strip().lower()
        if status == "timeout":
            return "timeout", "Solve hard timeout reached."
        if status == "failed_precondition":
            return "failed", "LLM provider unavailable in strict generative mode."
        if status == "ocr_required":
            errors = state.get("errors", [])
            if isinstance(errors, list) and errors:
                return "failed", str(errors[0])
            return "failed", "OCR is required for this request."
        if status == "resume_not_found":
            return "failed", "session checkpoint not found"
        if status.startswith("failed_"):
            errors = state.get("errors", [])
            if isinstance(errors, list) and errors:
                return "failed", str(errors[0])
            return "failed", "Solve pipeline failed."
        return "succeeded", None

    def _validate_solve_preconditions(
        *,
        problem: str,
        resume: bool,
        images: list[Dict[str, Any]],
        has_legacy_image: bool,
        supports_multimodal: bool,
        ocr_mode: str,
    ) -> None:
        if not problem and not resume and not (images or has_legacy_image):
            raise HTTPException(status_code=422, detail="problem or image input is required unless resume=true")
        if ocr_mode == "off" and not supports_multimodal and not problem and (images or has_legacy_image):
            raise HTTPException(
                status_code=422,
                detail="OCR mode is off but selected model does not support multimodal image input. Provide problem text or enable OCR.",
            )

    async def _run_solve_with_controls(
        *,
        provider: str,
        solve_kwargs: Dict[str, Any],
        request_id: str,
    ) -> Dict[str, Any]:
        ticket = await admission.acquire(provider=provider)
        started_at = time.perf_counter()
        release_later = False
        loop = asyncio.get_running_loop()
        solve_future: "asyncio.Future[Dict[str, Any]]" = loop.run_in_executor(
            request_executor,
            lambda: agent.solve(**solve_kwargs),
        )
        try:
            state = await asyncio.wait_for(solve_future, timeout=solve_hard_timeout_seconds)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            runtime_stats.record(provider=provider, latency_ms=elapsed_ms, timed_out=False)
            return state
        except asyncio.TimeoutError as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            runtime_stats.record(provider=provider, latency_ms=elapsed_ms, timed_out=True)
            logger.warning(
                "solve_hard_timeout request_id=%s provider=%s elapsed_ms=%.1f timeout_s=%.1f",
                request_id,
                provider,
                elapsed_ms,
                solve_hard_timeout_seconds,
            )

            async def _release_on_completion() -> None:
                try:
                    await solve_future
                except Exception:
                    pass
                await admission.release(ticket)

            release_later = True
            asyncio.create_task(_release_on_completion())
            raise _SolveHardTimeoutError(
                "Solve exceeded hard timeout of {:.1f}s".format(solve_hard_timeout_seconds)
            ) from exc
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if _is_provider_timeout_error(exc):
                runtime_stats.record(provider=provider, latency_ms=elapsed_ms, timed_out=True)
                raise _ProviderTimeoutError(str(exc)) from exc
            runtime_stats.record(provider=provider, latency_ms=elapsed_ms, timed_out=False)
            raise
        finally:
            if not release_later:
                await admission.release(ticket)

    async def _build_solve_artifacts(
        payload_like: Any,
        *,
        force_resume: Optional[bool] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any], list[Dict[str, Any]], bool, str, Dict[str, Any]]:
        images = _collect_images(payload_like)
        has_legacy_image = bool(
            _payload_get(payload_like, "image_path")
            or _payload_get(payload_like, "image_url")
            or _payload_get(payload_like, "image_base64")
        )
        llm_overrides = _build_llm_overrides(payload_like)
        llm_info = agent.describe_request_llm(llm_overrides)
        provider = str(llm_info.get("provider", "unknown") or "unknown").strip().lower() or "unknown"
        supports_multimodal = bool(
            llm_info.get("supports_multimodal", llm_info.get("multimodal_enabled", False))
        )
        ocr_mode = _normalize_ocr_mode(_payload_get(payload_like, "ocr_mode"))
        problem = str(_payload_get(payload_like, "problem", "") or "")
        resume_raw = _payload_get(payload_like, "resume", False)
        resume = bool(force_resume) if force_resume is not None else bool(resume_raw)

        _validate_solve_preconditions(
            problem=problem,
            resume=resume,
            images=images,
            has_legacy_image=has_legacy_image,
            supports_multimodal=supports_multimodal,
            ocr_mode=ocr_mode,
        )

        solve_kwargs = {
            "problem": problem,
            "session_id": _payload_get(payload_like, "session_id"),
            "resume": resume,
            "image_path": None if images else _payload_get(payload_like, "image_path"),
            "image_url": None if images else _payload_get(payload_like, "image_url"),
            "image_base64": None if images else _payload_get(payload_like, "image_base64"),
            "image_media_type": _payload_get(payload_like, "image_media_type", "image/png"),
            "images": images,
            "analysis_only": bool(_payload_get(payload_like, "analysis_only", False)),
            "ocr_mode": ocr_mode,
            "ocr_text": _payload_get(payload_like, "ocr_text"),
            "llm_overrides": llm_overrides,
        }
        return solve_kwargs, llm_info, images, has_legacy_image, provider, llm_overrides

    async def _job_worker(worker_id: int) -> None:
        while True:
            try:
                record = await job_store.pop_next()
                job_id = str(record.get("job_id"))
                payload_dict = dict(record.get("payload", {}))
                request_id = "job-{}-{}".format(worker_id, job_id[:8])
                try:
                    payload_model = SolveRequest(**payload_dict)
                    solve_kwargs, _, _, _, provider, _ = await _build_solve_artifacts(payload_model)
                    state = await _run_solve_with_controls(provider=provider, solve_kwargs=solve_kwargs, request_id=request_id)
                    job_status, error = _classify_job_outcome(state)
                    await job_store.set_terminal(
                        job_id=job_id,
                        status=job_status,
                        result=_state_to_response(state),
                        error=error,
                    )
                except _SolveHardTimeoutError as exc:
                    await job_store.set_terminal(job_id=job_id, status="timeout", error=str(exc))
                except _ProviderTimeoutError as exc:
                    logger.warning("job_worker_provider_timeout worker_id=%s job_id=%s error=%s", worker_id, job_id, exc)
                    await job_store.set_terminal(job_id=job_id, status="timeout", error=str(exc))
                except HTTPException as exc:
                    await job_store.set_terminal(job_id=job_id, status="failed", error=str(exc.detail))
                except AdmissionRejectedError as exc:
                    await job_store.set_terminal(job_id=job_id, status="failed", error=str(exc))
                except Exception as exc:
                    logger.exception("job_worker_failed worker_id=%s job_id=%s error=%s", worker_id, job_id, exc)
                    await job_store.set_terminal(job_id=job_id, status="failed", error=str(exc))
            except asyncio.CancelledError:  # pragma: no cover - shutdown path
                break

    @app.on_event("startup")
    async def on_startup() -> None:
        if job_workers:
            return
        for index in range(max(1, job_worker_count)):
            job_workers.append(asyncio.create_task(_job_worker(index)))

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        for task in job_workers:
            task.cancel()
        if job_workers:
            await asyncio.gather(*job_workers, return_exceptions=True)
            job_workers.clear()
        request_executor.shutdown(wait=False, cancel_futures=True)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/runtime/status", response_model=RuntimeStatusResponse)
    async def runtime_status() -> Dict[str, Any]:
        return await _runtime_status_payload()

    @app.get("/v1/sessions")
    async def sessions(limit: int = 10) -> Dict[str, Any]:
        safe_limit = max(1, min(int(limit), 100))
        return {"sessions": agent.checkpoints.list_recent(limit=safe_limit)}

    @app.post("/v1/ocr/extract", response_model=OCRExtractResponse)
    async def ocr_extract(payload: OCRExtractRequest) -> Dict[str, Any]:
        images = _collect_images(payload)
        if not images:
            raise HTTPException(status_code=422, detail="images payload is required")
        try:
            result = extract_text_from_images(images=images, config=_build_ocr_config(payload))
        except OCRUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OCRProcessingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="OCR extraction failed: {}".format(exc)) from exc
        return result

    @app.post("/v1/jobs/solve", response_model=JobSubmitResponse)
    async def jobs_solve(payload: SolveRequest, request: Request) -> Dict[str, Any]:
        request_id = request.headers.get("X-Request-ID", "-")
        try:
            _, _, _, _, provider, _ = await _build_solve_artifacts(payload)
            payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
            job = await job_store.submit(payload=payload_dict, provider=provider)
            logger.info("job_submitted request_id=%s job_id=%s provider=%s", request_id, job.get("job_id"), provider)
            return {
                "job_id": job.get("job_id"),
                "status": job.get("status"),
                "queue_position": job.get("queue_position"),
                "submitted_at": job.get("submitted_at"),
            }
        except JobQueueFullError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": str(max(1, int(queue_wait_timeout_seconds)))},
            ) from exc

    @app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
    async def jobs_status(job_id: str) -> Dict[str, Any]:
        try:
            return await job_store.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.delete("/v1/jobs/{job_id}", response_model=JobStatusResponse)
    async def jobs_cancel(job_id: str) -> Dict[str, Any]:
        try:
            return await job_store.cancel(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/v1/solve", response_model=SolveResponse)
    async def solve(payload: SolveRequest, request: Request) -> Dict[str, Any]:
        request_id = request.headers.get("X-Request-ID", "-")
        try:
            solve_kwargs, _, images, has_legacy_image, provider, _ = await _build_solve_artifacts(payload)
        except HTTPException:
            raise

        logger.info(
            "solve_start request_id=%s provider=%s model_profile=%s has_problem=%s has_image=%s session_id=%s",
            request_id,
            payload.provider,
            payload.model_profile,
            bool(payload.problem),
            bool(images or has_legacy_image),
            payload.session_id,
        )

        try:
            state = await _run_solve_with_controls(provider=provider, solve_kwargs=solve_kwargs, request_id=request_id)
        except AdmissionRejectedError as exc:
            raise _to_busy_http_exception(exc) from exc
        except _SolveHardTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except _ProviderTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("solve_failed request_id=%s error=%s", request_id, exc)
            raise HTTPException(status_code=502, detail="Provider call failed: {}".format(exc)) from exc

        if state.get("status") == "ocr_required":
            message = state.get("errors", ["OCR is required for this request."])[0]
            raise HTTPException(status_code=422, detail=message)
        if state.get("status") == "failed_precondition":
            raise HTTPException(status_code=503, detail="LLM provider unavailable in strict generative mode.")
        if state.get("status") == "timeout":
            raise HTTPException(status_code=504, detail="Solve pipeline timed out.")
        logger.info(
            "solve_done request_id=%s status=%s model=%s session_id=%s",
            request_id,
            state.get("status"),
            state.get("llm", {}).get("model"),
            state.get("session_id"),
        )
        return _state_to_response(state)

    @app.websocket("/v1/solve/stream")
    async def solve_stream(ws: WebSocket) -> None:
        await ws.accept()
        ticket = None
        provider = "unknown"
        started_at = time.perf_counter()
        try:
            payload = await ws.receive_json()
            problem = str(payload.get("problem", ""))
            session_id = payload.get("session_id")
            resume = bool(payload.get("resume", False))
            analysis_only = bool(payload.get("analysis_only", False))
            ocr_mode = _normalize_ocr_mode(payload.get("ocr_mode"))
            ocr_text = payload.get("ocr_text")
            image_path = payload.get("image_path")
            image_url = payload.get("image_url")
            image_base64 = payload.get("image_base64")
            image_media_type = payload.get("image_media_type", "image/png")
            images = payload.get("images", [])
            if not isinstance(images, list):
                images = []
            if not images and any([image_path, image_url, image_base64]):
                images.append(
                    {
                        "image_path": image_path,
                        "image_url": image_url,
                        "image_base64": image_base64,
                        "image_media_type": image_media_type,
                    }
                )
            llm_overrides = {
                key: payload[key]
                for key in ("provider", "model_profile", "model", "api_key_env", "temperature", "top_p", "max_tokens")
                if key in payload and payload[key] is not None
            }
            llm_info = agent.describe_request_llm(llm_overrides)
            provider = str(llm_info.get("provider", "unknown") or "unknown").strip().lower() or "unknown"
            supports_multimodal = bool(
                llm_info.get("supports_multimodal", llm_info.get("multimodal_enabled", False))
            )

            _validate_solve_preconditions(
                problem=problem,
                resume=resume,
                images=images,
                has_legacy_image=bool(image_path or image_url or image_base64),
                supports_multimodal=supports_multimodal,
                ocr_mode=ocr_mode,
            )

            ticket = await admission.acquire(provider=provider)
            queue_elapsed_ms = (ticket.acquired_at_monotonic - ticket.queued_at_monotonic) * 1000.0
            queue_snapshot = await _runtime_status_payload()
            await ws.send_json(
                {
                    "type": "queue_status",
                    "data": {
                        "queue_position": ticket.queue_position,
                        "queue_wait_ms": round(queue_elapsed_ms, 1),
                        "runtime": queue_snapshot,
                    },
                }
            )

            deadline = time.monotonic() + solve_hard_timeout_seconds
            event_stream = agent.solve_events(
                problem=problem,
                session_id=session_id,
                resume=resume,
                image_path=None if images else image_path,
                image_url=None if images else image_url,
                image_base64=None if images else image_base64,
                image_media_type=image_media_type,
                images=images,
                analysis_only=analysis_only,
                ocr_mode=ocr_mode,
                ocr_text=ocr_text,
                llm_overrides=llm_overrides,
            )

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _SolveHardTimeoutError(
                        "Solve stream exceeded hard timeout of {:.1f}s".format(solve_hard_timeout_seconds)
                    )
                try:
                    event = await asyncio.wait_for(event_stream.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                await ws.send_json(event)

            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            runtime_stats.record(provider=provider, latency_ms=elapsed_ms, timed_out=False)
            await ws.send_json({"type": "done"})
            await ws.close(code=1000)
        except AdmissionRejectedError as exc:
            await ws.send_json(
                {
                    "type": "error",
                    "message": str(exc),
                    "code": 429,
                    "retry_after": exc.retry_after_seconds,
                }
            )
            await ws.close(code=1013)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected by client")
        except _SolveHardTimeoutError as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            runtime_stats.record(provider=provider, latency_ms=elapsed_ms, timed_out=True)
            await ws.send_json({"type": "error", "message": str(exc), "code": 504})
            await ws.close(code=1011)
        except Exception as exc:  # pragma: no cover - runtime path
            if _is_provider_timeout_error(exc):
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                runtime_stats.record(provider=provider, latency_ms=elapsed_ms, timed_out=True)
                await ws.send_json({"type": "error", "message": str(exc), "code": 504})
                await ws.close(code=1011)
                return
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close(code=1011)
        finally:
            if ticket is not None:
                await admission.release(ticket)

    @app.post("/v1/export")
    async def export(payload: ExportRequest) -> Dict[str, Any]:
        images = _collect_images(payload)
        has_legacy_image = bool(payload.image_path or payload.image_url or payload.image_base64)
        auto_resume = bool(payload.session_id and not payload.problem and not images)
        resume = bool(payload.resume) if payload.resume is not None else auto_resume

        solve_kwargs, _, _, _, provider, _ = await _build_solve_artifacts(payload, force_resume=resume)
        request_id = "export-{}".format(str(payload.session_id or "new"))
        try:
            state = await _run_solve_with_controls(provider=provider, solve_kwargs=solve_kwargs, request_id=request_id)
        except AdmissionRejectedError as exc:
            raise _to_busy_http_exception(exc) from exc
        except _SolveHardTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except _ProviderTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc

        if state.get("status") == "ocr_required":
            message = state.get("errors", ["OCR is required for this request."])[0]
            raise HTTPException(status_code=422, detail=message)
        if state.get("status") == "failed_precondition":
            raise HTTPException(status_code=503, detail="LLM provider unavailable in strict generative mode.")
        if state.get("status") == "resume_not_found":
            raise HTTPException(status_code=404, detail="session checkpoint not found")
        if state.get("status") == "timeout":
            raise HTTPException(status_code=504, detail="Solve pipeline timed out.")
        fmt = payload.format.lower()
        if fmt == "latex":
            file_path = export_latex(state, payload.output_path)
        elif fmt == "notebook":
            file_path = export_notebook(state, payload.output_path)
        else:
            file_path = agent.export_result(state, payload.output_path)
        return {"session_id": state.get("session_id"), "format": fmt, "file_path": file_path}

    return app

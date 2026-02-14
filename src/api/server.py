"""REST and WebSocket interfaces for MathSolverAgent."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents import MathSolverAgent
from src.utils.exporters import export_latex, export_notebook
from src.utils.logger import get_logger

try:  # pragma: no cover
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]
    WebSocket = Any  # type: ignore[misc,assignment]
    WebSocketDisconnect = Exception  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return None


logger = get_logger("math_solver_agent.api")


if FastAPI is not None:

    class SolveRequest(BaseModel):
        problem: str = Field(default="", description="Mathematical problem statement")
        session_id: Optional[str] = Field(default=None, description="Optional conversation/session id")
        resume: bool = Field(default=False, description="Resume from checkpoint when true")
        image_path: Optional[str] = Field(default=None, description="Local path of image on server host")
        image_url: Optional[str] = Field(default=None, description="Public URL of an image containing the problem")
        image_base64: Optional[str] = Field(default=None, description="Base64 image payload")
        image_media_type: str = Field(default="image/png", description="Media type used for image_base64")
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
        domain: Optional[str]
        strategy: Optional[str]
        llm: Dict[str, Any]
        has_visual_input: bool
        result: Optional[str]
        numeric_result: Optional[float]
        verification: Dict[str, Any]
        explanation: str
        metrics: Dict[str, float]


    class ExportRequest(BaseModel):
        problem: str = Field(default="")
        format: str = Field(default="latex", pattern="^(latex|notebook|json)$")
        output_path: str = Field(default="artifacts/result.tex")
        session_id: Optional[str] = Field(default=None)
        image_path: Optional[str] = Field(default=None)
        image_url: Optional[str] = Field(default=None)
        image_base64: Optional[str] = Field(default=None)
        image_media_type: str = Field(default="image/png")
        provider: Optional[str] = Field(default=None)
        model_profile: Optional[str] = Field(default=None)
        model: Optional[str] = Field(default=None)
        api_key_env: Optional[str] = Field(default=None)
        temperature: Optional[float] = Field(default=None)
        top_p: Optional[float] = Field(default=None)
        max_tokens: Optional[int] = Field(default=None)
else:

    class SolveRequest:  # type: ignore[too-many-ancestors]
        pass

    class SolveResponse:  # type: ignore[too-many-ancestors]
        pass

    class ExportRequest:  # type: ignore[too-many-ancestors]
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

    def _build_llm_overrides(payload_like: Any) -> Dict[str, Any]:
        """Builds request-scoped LLM override dictionary.

        Args:
            payload_like: Request payload object with optional LLM override fields.

        Returns:
            Dictionary containing only defined LLM override values.
        """
        overrides: Dict[str, Any] = {}
        for key in ("provider", "model_profile", "model", "api_key_env", "temperature", "top_p", "max_tokens"):
            value = getattr(payload_like, key, None)
            if value is not None:
                overrides[key] = value
        return overrides

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/solve", response_model=SolveResponse)
    async def solve(payload: SolveRequest) -> Dict[str, Any]:
        if not payload.problem and not payload.resume and not any([payload.image_path, payload.image_url, payload.image_base64]):
            raise HTTPException(status_code=422, detail="problem or image input is required unless resume=true")
        state = agent.solve(
            problem=payload.problem,
            session_id=payload.session_id,
            resume=payload.resume,
            image_path=payload.image_path,
            image_url=payload.image_url,
            image_base64=payload.image_base64,
            image_media_type=payload.image_media_type,
            llm_overrides=_build_llm_overrides(payload),
        )
        if state.get("status") == "failed_precondition":
            raise HTTPException(status_code=503, detail="LLM provider unavailable in strict generative mode.")
        return {
            "session_id": state.get("session_id"),
            "status": state.get("status"),
            "domain": state.get("domain"),
            "strategy": state.get("strategy"),
            "llm": state.get("llm", {}),
            "has_visual_input": bool(state.get("visual_input")),
            "result": state.get("symbolic_result"),
            "numeric_result": state.get("numeric_result"),
            "verification": state.get("verification", {}),
            "explanation": state.get("explanation", ""),
            "metrics": state.get("metrics", {}),
        }

    @app.websocket("/v1/solve/stream")
    async def solve_stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            payload = await ws.receive_json()
            problem = str(payload.get("problem", ""))
            session_id = payload.get("session_id")
            resume = bool(payload.get("resume", False))
            image_path = payload.get("image_path")
            image_url = payload.get("image_url")
            image_base64 = payload.get("image_base64")
            image_media_type = payload.get("image_media_type", "image/png")
            llm_overrides = {
                key: payload[key]
                for key in ("provider", "model_profile", "model", "api_key_env", "temperature", "top_p", "max_tokens")
                if key in payload and payload[key] is not None
            }

            if not problem and not resume and not any([image_path, image_url, image_base64]):
                raise ValueError("problem or image input is required unless resume=true")

            state = agent.solve(
                problem=problem,
                session_id=session_id,
                resume=resume,
                image_path=image_path,
                image_url=image_url,
                image_base64=image_base64,
                image_media_type=image_media_type,
                llm_overrides=llm_overrides,
            )

            for event in state.get("decision_trace", []):
                await ws.send_json({"type": "trace", "data": event})

            await ws.send_json(
                {
                    "type": "result",
                    "data": {
                        "session_id": state.get("session_id"),
                        "status": state.get("status"),
                        "llm": state.get("llm", {}),
                        "has_visual_input": bool(state.get("visual_input")),
                        "result": state.get("symbolic_result"),
                        "verification": state.get("verification", {}),
                    },
                }
            )
            await ws.close(code=1000)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected by client")
        except Exception as exc:  # pragma: no cover - runtime path
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close(code=1011)

    @app.post("/v1/export")
    async def export(payload: ExportRequest) -> Dict[str, Any]:
        if not payload.problem and not any([payload.image_path, payload.image_url, payload.image_base64]):
            raise HTTPException(status_code=422, detail="problem or image input is required")
        state = agent.solve(
            problem=payload.problem,
            session_id=payload.session_id,
            resume=False,
            image_path=payload.image_path,
            image_url=payload.image_url,
            image_base64=payload.image_base64,
            image_media_type=payload.image_media_type,
            llm_overrides=_build_llm_overrides(payload),
        )
        if state.get("status") == "failed_precondition":
            raise HTTPException(status_code=503, detail="LLM provider unavailable in strict generative mode.")
        fmt = payload.format.lower()
        if fmt == "latex":
            file_path = export_latex(state, payload.output_path)
        elif fmt == "notebook":
            file_path = export_notebook(state, payload.output_path)
        else:
            file_path = agent.export_result(state, payload.output_path)
        return {"session_id": state.get("session_id"), "format": fmt, "file_path": file_path}

    return app

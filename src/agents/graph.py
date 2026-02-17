"""LangGraph orchestration for a fully generative MathSolverAgent pipeline."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterable, Iterator, Optional

from src.agents.state import AgentState, build_initial_state
from src.llm import GenerativeMathClient
from src.nodes import analyze_problem, convert_to_tool_syntax, solve_problem, verify_solution
from src.utils.config_loader import (
    GraphConfig,
    load_graph_config,
    load_prompts_config,
    load_prompts_registry,
    select_prompt_variant,
)
from src.utils.logger import get_logger

try:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    MemorySaver = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    END = "__end__"


class CheckpointStore:
    """File-based checkpoint persistence used for session recovery."""

    def __init__(self, checkpoint_dir: str) -> None:
        """Initializes a file-backed checkpoint store.

        Args:
            checkpoint_dir: Directory used to persist session checkpoints.
        """
        self.base = Path(checkpoint_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        """Builds checkpoint file path for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            Path for the checkpoint JSON file.
        """
        return self.base / "{}.json".format(session_id)

    def save(self, state: AgentState) -> None:
        """Persists state checkpoint on disk.

        Args:
            state: Agent state snapshot to persist.
        """
        session_id = state.get("session_id")
        if not session_id:
            return
        path = self._path(session_id)
        checkpoint_payload = _state_for_checkpoint(state)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(checkpoint_payload, handle, ensure_ascii=True, indent=2)

    def load(self, session_id: str) -> Optional[AgentState]:
        """Loads a state checkpoint from disk.

        Args:
            session_id: Session identifier to restore.

        Returns:
            Restored state when available, otherwise None.
        """
        path = self._path(session_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return AgentState(**payload)

    def list_recent(self, limit: int = 10) -> list[Dict[str, Any]]:
        """Lists recent checkpoint metadata sorted by modification time.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of summary dictionaries for recent sessions.
        """
        safe_limit = max(1, min(int(limit), 100))
        entries: list[Dict[str, Any]] = []

        paths = sorted(
            self.base.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue

            session_id = str(payload.get("session_id") or path.stem)
            problem = str(payload.get("problem") or "").strip()
            preview = " ".join(problem.split())
            updated_at = str(payload.get("updated_at") or "").strip()
            if not updated_at:
                updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")

            entries.append(
                {
                    "session_id": session_id,
                    "updated_at": updated_at,
                    "status": str(payload.get("status") or ""),
                    "domain": str(payload.get("domain") or ""),
                    "problem_preview": preview[:160],
                }
            )
            if len(entries) >= safe_limit:
                break
        return entries


def _state_for_checkpoint(state: AgentState) -> Dict[str, Any]:
    """Sanitizes state payload before writing checkpoint to disk.

    Args:
        state: Current in-memory state.

    Returns:
        Serializable state dictionary safe for persistence.
    """
    payload: Dict[str, Any] = deepcopy(dict(state))
    visual_input = payload.get("visual_input")
    if isinstance(visual_input, dict) and "image_base64" in visual_input:
        if visual_input.get("image_base64"):
            visual_input["image_base64_present"] = True
        visual_input.pop("image_base64", None)
    if isinstance(visual_input, dict):
        images = visual_input.get("images")
        if isinstance(images, list):
            sanitized_images = []
            for item in images:
                if not isinstance(item, dict):
                    continue
                cleaned = dict(item)
                if cleaned.get("image_base64"):
                    cleaned["image_base64_present"] = True
                cleaned.pop("image_base64", None)
                sanitized_images.append(cleaned)
            visual_input["images"] = sanitized_images
    return payload


def _canonicalize_images(
    image_path: Optional[str],
    image_url: Optional[str],
    image_base64: Optional[str],
    image_media_type: Optional[str],
    images: Optional[list[Dict[str, Any]]],
) -> list[Dict[str, Any]]:
    """Normalizes legacy and new image inputs into a canonical list."""
    canonical: list[Dict[str, Any]] = []

    if isinstance(images, list):
        for item in images:
            if not isinstance(item, dict):
                continue
            normalized = {
                "image_path": item.get("image_path"),
                "image_url": item.get("image_url"),
                "image_base64": item.get("image_base64"),
                "image_media_type": item.get("image_media_type") or "image/png",
            }
            if any(normalized.get(key) for key in ("image_path", "image_url", "image_base64")):
                canonical.append(normalized)

    legacy = {
        "image_path": image_path,
        "image_url": image_url,
        "image_base64": image_base64,
        "image_media_type": image_media_type or "image/png",
    }
    if any(legacy.get(key) for key in ("image_path", "image_url", "image_base64")):
        canonical.append(legacy)

    return canonical


def _normalize_ocr_mode(value: Optional[str]) -> str:
    mode = str(value or "auto").strip().lower() or "auto"
    if mode not in {"auto", "on", "off"}:
        return "auto"
    return mode


def should_use_ocr(
    ocr_mode: str,
    supports_multimodal: bool,
    has_image: bool,
    has_problem_text: bool,
    ocr_text_present: bool,
) -> bool:
    """Central OCR decision policy used across API/UI flows."""
    del has_problem_text  # currently retained for future policy nuances
    if not has_image:
        return False
    if ocr_text_present:
        return False
    mode = _normalize_ocr_mode(ocr_mode)
    if mode == "on":
        return True
    if mode == "off":
        return False
    return not supports_multimodal


class MathSolverAgent:
    """Enterprise-grade math solver agent with strict generative orchestration."""

    def __init__(
        self,
        graph_config_path: str = "configs/graph_config.yml",
        prompts_path: str = "configs/prompts.yml",
    ) -> None:
        """Initializes agent runtime and graph dependencies.

        Args:
            graph_config_path: Path to graph/runtime configuration file.
            prompts_path: Path to prompt registry configuration file.
        """
        self.logger = get_logger("math_solver_agent")
        self.config: GraphConfig = load_graph_config(graph_config_path)
        self.prompt_config = load_prompts_config(prompts_path)
        self.prompts = load_prompts_registry(prompts_path)
        self.llm_client = GenerativeMathClient(config=self._base_llm_config())
        self.checkpoints = CheckpointStore(self.config.runtime.checkpoint_dir)
        runtime_control = dict(getattr(self.config, "runtime_control", {}) or {})
        node_workers = int(runtime_control.get("node_executor_max_workers", 8))
        self._node_executor = ThreadPoolExecutor(max_workers=max(2, node_workers), thread_name_prefix="agent-node")

        self._graph = self._build_langgraph(self.llm_client) if StateGraph is not None else None

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        executor = getattr(self, "_node_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def _base_llm_config(self) -> Dict[str, Any]:
        merged = dict(self.config.llm)
        llm_timeouts = dict(getattr(self.config, "llm_timeouts", {}) or {})
        merged.update(llm_timeouts)
        return merged

    def _build_langgraph(self, llm_client: GenerativeMathClient) -> Optional[Any]:
        """Builds LangGraph runtime with typed node topology.

        Returns:
            Compiled graph instance when LangGraph is installed, else None.
        """

        if StateGraph is None:
            self.logger.info("LangGraph not installed; using fallback orchestrator.")
            return None

        graph = StateGraph(dict)
        graph.add_node(
            "analyzer",
            lambda st: analyze_problem(
                st,
                llm_client=llm_client,
                prompt_pack=self.prompts.get("analyzer", {}),
            ),
        )
        graph.add_node(
            "converter",
            lambda st: convert_to_tool_syntax(
                st,
                llm_client=llm_client,
                prompt_pack=self.prompts.get("converter", {}),
            ),
        )
        graph.add_node(
            "solver",
            lambda st: solve_problem(
                st,
                llm_client=llm_client,
                prompt_pack=self.prompts.get("solver", {}),
            ),
        )
        graph.add_node(
            "verifier",
            lambda st: verify_solution(
                st,
                llm_client=llm_client,
                prompt_pack=self.prompts.get("verifier", {}),
            ),
        )
        graph.set_entry_point("analyzer")
        graph.add_edge("analyzer", "converter")
        graph.add_edge("converter", "solver")
        graph.add_edge("solver", "verifier")
        graph.add_conditional_edges(
            "verifier",
            lambda st: "converter" if st.get("should_refine") else END,
            {"converter": "converter", END: END},
        )

        if MemorySaver is not None:
            return graph.compile(checkpointer=MemorySaver())
        return graph.compile()

    def solve(
        self,
        problem: str,
        session_id: Optional[str] = None,
        resume: bool = False,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_media_type: Optional[str] = None,
        images: Optional[list[Dict[str, Any]]] = None,
        analysis_only: bool = False,
        ocr_mode: str = "auto",
        ocr_text: Optional[str] = None,
        llm_overrides: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        """Runs a full solve cycle for one session.

        Args:
            problem: Textual math problem statement.
            session_id: Optional existing session id.
            resume: Whether to restore and continue from checkpoint.
            image_path: Optional local image path.
            image_url: Optional public image URL.
            image_base64: Optional inline image payload.
            image_media_type: Media type for inline image payload.
            images: Optional list of image payload dictionaries.
            analysis_only: Whether to run only analyzer stage.
            ocr_mode: OCR policy (`auto|on|off`).
            ocr_text: Pre-extracted OCR text.
            llm_overrides: Optional per-request LLM configuration overrides.

        Returns:
            Final agent state for the solve request.
        """
        state, request_llm_client, graph_runtime = self._prepare_execution(
            problem=problem,
            session_id=session_id,
            resume=resume,
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            image_media_type=image_media_type,
            images=images,
            ocr_mode=ocr_mode,
            ocr_text=ocr_text,
            llm_overrides=llm_overrides,
        )
        if state.get("status") in {"failed_precondition", "resume_not_found", "ocr_required"}:
            return state

        if analysis_only:
            return self._run_analysis_only(state, request_llm_client)

        if graph_runtime is not None:
            return self._run_langgraph(state, graph_runtime)
        return self._run_fallback(state, request_llm_client)

    async def solve_events(
        self,
        problem: str,
        session_id: Optional[str] = None,
        resume: bool = False,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_media_type: Optional[str] = None,
        images: Optional[list[Dict[str, Any]]] = None,
        analysis_only: bool = False,
        ocr_mode: str = "auto",
        ocr_text: Optional[str] = None,
        llm_overrides: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Runs solve pipeline and streams structured events in real time."""
        state, request_llm_client, graph_runtime = self._prepare_execution(
            problem=problem,
            session_id=session_id,
            resume=resume,
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            image_media_type=image_media_type,
            images=images,
            ocr_mode=ocr_mode,
            ocr_text=ocr_text,
            llm_overrides=llm_overrides,
        )
        if state.get("status") in {"failed_precondition", "resume_not_found", "ocr_required"}:
            yield {"type": "result", "data": self._state_to_public_payload(state)}
            return

        if analysis_only:
            for event in self._run_analysis_only_events(state, request_llm_client):
                yield event
            return

        if graph_runtime is not None:
            async for event in self._run_langgraph_events(state, graph_runtime):
                yield event
            return

        for event in self._run_fallback_events(state, request_llm_client):
            yield event

    def _prepare_execution(
        self,
        problem: str,
        session_id: Optional[str],
        resume: bool,
        image_path: Optional[str],
        image_url: Optional[str],
        image_base64: Optional[str],
        image_media_type: Optional[str],
        images: Optional[list[Dict[str, Any]]],
        ocr_mode: str,
        ocr_text: Optional[str],
        llm_overrides: Optional[Dict[str, Any]],
    ) -> tuple[AgentState, GenerativeMathClient, Optional[Any]]:
        """Builds request-scoped state and runtime dependencies."""
        sid = session_id or str(uuid.uuid4())

        if resume:
            resumed = self.checkpoints.load(sid)
            if resumed:
                state = resumed
                state["status"] = "resumed"
            else:
                state = self._new_state(problem, sid)
                state["status"] = "resume_not_found"
                state.setdefault("errors", []).append("No checkpoint found for session '{}'.".format(sid))
                self.checkpoints.save(state)
        else:
            state = self._new_state(problem, sid)

        request_llm_client = self._resolve_request_llm_client(llm_overrides)
        state["prompt_variant"] = select_prompt_variant(self.prompt_config, sid)
        state["llm"] = request_llm_client.describe()
        state.setdefault("artifacts", [])

        if state.get("status") == "resume_not_found":
            return state, request_llm_client, None

        require_llm = bool(self.config.llm.get("require_available", True))
        if require_llm and not request_llm_client.is_available:
            state["status"] = "failed_precondition"
            state.setdefault("errors", []).append(
                "Generative mode requires an available LLM provider. Check API key and provider dependencies."
            )
            self.checkpoints.save(state)
            return state, request_llm_client, None

        canonical_images = _canonicalize_images(
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            image_media_type=image_media_type,
            images=images,
        )
        normalized_mode = _normalize_ocr_mode(ocr_mode)
        ocr_text_clean = str(ocr_text or "").strip()
        supports_multimodal = bool(
            state.get("llm", {}).get(
                "supports_multimodal",
                state.get("llm", {}).get("multimodal_enabled", False),
            )
        )
        has_problem_text = bool(str(state.get("problem") or "").strip())
        ocr_required = should_use_ocr(
            ocr_mode=normalized_mode,
            supports_multimodal=supports_multimodal,
            has_image=bool(canonical_images),
            has_problem_text=has_problem_text,
            ocr_text_present=bool(ocr_text_clean),
        )
        state["ocr"] = {
            "mode": normalized_mode,
            "required": ocr_required,
            "provided_text": bool(ocr_text_clean),
            "supports_multimodal": supports_multimodal,
            "used": False,
        }

        if canonical_images:
            primary = canonical_images[0]
            state["visual_input"] = {
                "image_path": primary.get("image_path"),
                "image_url": primary.get("image_url"),
                "image_base64": primary.get("image_base64"),
                "image_media_type": primary.get("image_media_type"),
                "images": canonical_images,
            }
        else:
            state.setdefault("visual_input", {})

        if ocr_required:
            state["status"] = "ocr_required"
            state.setdefault("errors", []).append(
                "OCR is required for this request. Provide `ocr_text` or use `/v1/ocr/extract` before solve."
            )
            self.checkpoints.save(state)
            return state, request_llm_client, None

        if ocr_text_clean:
            merged_problem = str(state.get("problem") or "").strip()
            if merged_problem:
                merged_problem = "{}\n\n{}".format(merged_problem, ocr_text_clean)
            else:
                merged_problem = ocr_text_clean
            state["problem"] = merged_problem
            state["normalized_problem"] = merged_problem
            state.setdefault("ocr", {})["used"] = True

        if self._graph is not None and request_llm_client is self.llm_client:
            return state, request_llm_client, self._graph
        if self._graph is not None:
            request_graph = self._build_langgraph(request_llm_client)
            if request_graph is not None:
                return state, request_llm_client, request_graph
        return state, request_llm_client, None

    def _resolve_request_llm_client(self, llm_overrides: Optional[Dict[str, Any]]) -> GenerativeMathClient:
        """Resolves the LLM client for one request.

        Args:
            llm_overrides: Optional LLM settings passed at request time.

        Returns:
            Default client or a request-scoped client with merged settings.
        """
        if not llm_overrides:
            return self.llm_client

        effective_config = self._base_llm_config()
        provider_override = str(llm_overrides.get("provider", "")).strip().lower() if llm_overrides else ""
        has_provider_or_profile_override = bool(llm_overrides.get("provider") or llm_overrides.get("model_profile"))
        has_explicit_model_override = bool(llm_overrides.get("model"))
        for key, value in llm_overrides.items():
            if value is None:
                continue
            if key in {"model_profiles", "chat_template_kwargs"}:
                base_mapping = effective_config.get(key, {})
                if isinstance(base_mapping, dict) and isinstance(value, dict):
                    merged_mapping = dict(base_mapping)
                    merged_mapping.update(value)
                    effective_config[key] = merged_mapping
                    continue
            effective_config[key] = value
        if has_provider_or_profile_override and not has_explicit_model_override:
            # Avoid stale model inheritance from base config (e.g. nvidia model
            # kept while provider/profile switches to maritaca).
            effective_config.pop("model", None)
        if provider_override and "api_key_env" not in llm_overrides:
            effective_config["api_key_env"] = "MARITACA_API_KEY" if provider_override == "maritaca" else "NVIDIA_API_KEY"
        return GenerativeMathClient(config=effective_config)

    def describe_request_llm(self, llm_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Returns effective request-scoped LLM capabilities and metadata."""
        client = self._resolve_request_llm_client(llm_overrides)
        return client.describe()

    def _new_state(self, problem: str, session_id: str) -> AgentState:
        """Creates a fresh initial state for a session.

        Args:
            problem: Raw problem statement.
            session_id: Session identifier.

        Returns:
            Initialized agent state.
        """
        return build_initial_state(
            problem=problem,
            session_id=session_id,
            max_iterations=self.config.runtime.max_iterations,
            timeout_seconds=self.config.runtime.default_timeout_seconds,
        )

    def _state_to_public_payload(self, state: AgentState) -> Dict[str, Any]:
        """Builds API-friendly response payload from full agent state."""
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

    def _run_analysis_only(self, state: AgentState, llm_client: GenerativeMathClient) -> AgentState:
        """Runs only analyzer stage and persists checkpoint."""
        if state.get("status") not in {"initialized", "resumed"}:
            self.checkpoints.save(state)
            return state
        out = self._run_node(
            "analysis",
            lambda st: analyze_problem(
                st,
                llm_client=llm_client,
                prompt_pack=self.prompts.get("analyzer", {}),
            ),
            state,
        )
        self.checkpoints.save(out)
        return out

    def _run_analysis_only_events(self, state: AgentState, llm_client: GenerativeMathClient) -> Iterable[Dict[str, Any]]:
        """Streams analyzer-only execution events."""
        seen_trace_count = len(state.get("decision_trace", []))
        events: list[Dict[str, Any]] = [{"type": "node_status", "data": {"node": "analyzer", "status": "running"}}]

        out = self._run_analysis_only(state, llm_client)
        decision_trace = out.get("decision_trace", [])
        for trace_event in decision_trace[seen_trace_count:]:
            events.append({"type": "trace", "data": trace_event})

        status = "failed" if str(out.get("status", "")).startswith("failed_") else "done"
        events.append({"type": "node_status", "data": {"node": "analyzer", "status": status}})
        events.append({"type": "result", "data": self._state_to_public_payload(out)})
        return events

    @staticmethod
    def _stream_explanation_tokens(explanation: str) -> Iterator[str]:
        """Splits explanation text into lightweight UI-friendly token chunks."""
        text = str(explanation or "")
        if not text:
            return iter(())
        return (chunk + " " for chunk in text.split())

    def _run_langgraph(self, state: AgentState, graph_runtime: Any) -> AgentState:
        """Executes the compiled graph and persists final checkpoint.

        Args:
            state: Input state for graph invocation.
            graph_runtime: Compiled graph runtime instance.

        Returns:
            Final state returned by graph execution.
        """
        config = {"configurable": {"thread_id": state["session_id"]}}
        out = graph_runtime.invoke(state, config=config)
        final_state = AgentState(**out)
        final_state.setdefault("artifacts", [])
        self.checkpoints.save(final_state)
        return final_state

    async def _run_langgraph_events(self, state: AgentState, graph_runtime: Any) -> AsyncIterator[Dict[str, Any]]:
        """Streams node and trace updates from LangGraph runtime."""
        config = {"configurable": {"thread_id": state["session_id"]}}
        seen_trace_count = len(state.get("decision_trace", []))
        current_state = AgentState(**deepcopy(dict(state)))

        async for update in self._iter_langgraph_updates(state, graph_runtime, config):
            for node_name, payload in self._extract_node_payloads(update):
                yield {"type": "node_status", "data": {"node": node_name, "status": "running"}}
                merged_payload = deepcopy(dict(current_state))
                merged_payload.update(payload)
                current_state = AgentState(**merged_payload)
                current_state.setdefault("artifacts", [])

                decision_trace = current_state.get("decision_trace", [])
                for trace_event in decision_trace[seen_trace_count:]:
                    yield {"type": "trace", "data": trace_event}
                seen_trace_count = len(decision_trace)

                status = "failed" if str(current_state.get("status", "")).startswith("failed_") else "done"
                yield {"type": "node_status", "data": {"node": node_name, "status": status}}

        self.checkpoints.save(current_state)
        for token in self._stream_explanation_tokens(str(current_state.get("explanation", ""))):
            yield {"type": "token", "data": {"text": token}}
        yield {"type": "result", "data": self._state_to_public_payload(current_state)}

    async def _iter_langgraph_updates(
        self,
        state: AgentState,
        graph_runtime: Any,
        config: Dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Iterates LangGraph updates preferring async stream APIs."""
        if hasattr(graph_runtime, "astream"):
            try:
                async for item in graph_runtime.astream(state, config=config, stream_mode="updates"):
                    yield item
                return
            except TypeError:
                async for item in graph_runtime.astream(state, config=config):
                    yield item
                return

        if hasattr(graph_runtime, "stream"):
            try:
                for item in graph_runtime.stream(state, config=config, stream_mode="updates"):
                    yield item
                return
            except TypeError:
                for item in graph_runtime.stream(state, config=config):
                    yield item
                return

        out = graph_runtime.invoke(state, config=config)
        yield {"__final__": out}

    @staticmethod
    def _extract_node_payloads(update: Any) -> Iterable[tuple[str, Dict[str, Any]]]:
        """Extracts node/state payloads from LangGraph update chunks."""
        if not isinstance(update, dict):
            return []

        nodes = ("analyzer", "converter", "solver", "verifier")
        pairs: list[tuple[str, Dict[str, Any]]] = []

        for node in nodes:
            payload = update.get(node)
            if isinstance(payload, dict):
                pairs.append((node, payload))

        if pairs:
            return pairs

        if "session_id" in update and "status" in update:
            return [("verifier", update)]

        maybe_state = update.get("__final__")
        if isinstance(maybe_state, dict):
            return [("verifier", maybe_state)]
        return []

    def _run_fallback(self, state: AgentState, llm_client: GenerativeMathClient) -> AgentState:
        """Executes node sequence using threaded node runner fallback.

        Args:
            state: Initial state for fallback run.
            llm_client: Request-scoped LLM client.

        Returns:
            Final state produced by fallback execution.
        """

        if state.get("status") in {"initialized", "resumed"}:
            state = self._run_node(
                "analysis",
                lambda st: analyze_problem(
                    st,
                    llm_client=llm_client,
                    prompt_pack=self.prompts.get("analyzer", {}),
                ),
                state,
            )
            self.checkpoints.save(state)
            if state.get("status") in {"failed_analysis", "failed_precondition"}:
                return state

        while True:
            state = self._run_node(
                "planning",
                lambda st: convert_to_tool_syntax(
                    st,
                    llm_client=llm_client,
                    prompt_pack=self.prompts.get("converter", {}),
                ),
                state,
            )
            self.checkpoints.save(state)

            if state.get("status") in {"failed_planning", "failed_precondition"}:
                break

            state = self._run_node(
                "solving",
                lambda st: solve_problem(
                    st,
                    llm_client=llm_client,
                    prompt_pack=self.prompts.get("solver", {}),
                ),
                state,
            )
            self.checkpoints.save(state)
            if state.get("status") == "failed_solving":
                break

            state = self._run_node(
                "verification",
                lambda st: verify_solution(
                    st,
                    llm_client=llm_client,
                    prompt_pack=self.prompts.get("verifier", {}),
                ),
                state,
            )
            self.checkpoints.save(state)

            if not state.get("should_refine", False):
                break

        state.setdefault("artifacts", [])
        return state

    def _run_fallback_events(self, state: AgentState, llm_client: GenerativeMathClient) -> Iterable[Dict[str, Any]]:
        """Streams node and trace updates while executing fallback runtime."""
        seen_trace_count = len(state.get("decision_trace", []))

        if state.get("status") in {"initialized", "resumed"}:
            state, seen_trace_count, events = self._run_fallback_node_with_events(
                state=state,
                seen_trace_count=seen_trace_count,
                node_name="analyzer",
                timeout_key="analysis",
                fn=lambda st: analyze_problem(
                    st,
                    llm_client=llm_client,
                    prompt_pack=self.prompts.get("analyzer", {}),
                ),
            )
            for event in events:
                yield event
            if state.get("status") in {"failed_analysis", "failed_precondition"}:
                self.checkpoints.save(state)
                yield {"type": "result", "data": self._state_to_public_payload(state)}
                return

        while True:
            state, seen_trace_count, events = self._run_fallback_node_with_events(
                state=state,
                seen_trace_count=seen_trace_count,
                node_name="converter",
                timeout_key="planning",
                fn=lambda st: convert_to_tool_syntax(
                    st,
                    llm_client=llm_client,
                    prompt_pack=self.prompts.get("converter", {}),
                ),
            )
            for event in events:
                yield event
            if state.get("status") in {"failed_planning", "failed_precondition"}:
                break

            state, seen_trace_count, events = self._run_fallback_node_with_events(
                state=state,
                seen_trace_count=seen_trace_count,
                node_name="solver",
                timeout_key="solving",
                fn=lambda st: solve_problem(
                    st,
                    llm_client=llm_client,
                    prompt_pack=self.prompts.get("solver", {}),
                ),
            )
            for event in events:
                yield event
            if state.get("status") == "failed_solving":
                break

            state, seen_trace_count, events = self._run_fallback_node_with_events(
                state=state,
                seen_trace_count=seen_trace_count,
                node_name="verifier",
                timeout_key="verification",
                fn=lambda st: verify_solution(
                    st,
                    llm_client=llm_client,
                    prompt_pack=self.prompts.get("verifier", {}),
                ),
            )
            for event in events:
                yield event
            if not state.get("should_refine", False):
                break

        self.checkpoints.save(state)
        for token in self._stream_explanation_tokens(str(state.get("explanation", ""))):
            yield {"type": "token", "data": {"text": token}}
        yield {"type": "result", "data": self._state_to_public_payload(state)}

    def _run_fallback_node_with_events(
        self,
        state: AgentState,
        seen_trace_count: int,
        node_name: str,
        timeout_key: str,
        fn: Callable[[AgentState], AgentState],
    ) -> tuple[AgentState, int, list[Dict[str, Any]]]:
        """Runs one fallback node and returns corresponding stream events."""
        events: list[Dict[str, Any]] = [{"type": "node_status", "data": {"node": node_name, "status": "running"}}]

        out = self._run_node(timeout_key, fn, state)
        out.setdefault("artifacts", [])
        self.checkpoints.save(out)

        decision_trace = out.get("decision_trace", [])
        for trace_event in decision_trace[seen_trace_count:]:
            events.append({"type": "trace", "data": trace_event})
        seen_trace_count = len(decision_trace)

        status = "failed" if str(out.get("status", "")).startswith("failed_") else "done"
        events.append({"type": "node_status", "data": {"node": node_name, "status": status}})
        return out, seen_trace_count, events

    def _run_node(self, name: str, fn: Callable[[AgentState], AgentState], state: AgentState) -> AgentState:
        """Runs a node function with timeout isolation.

        Args:
            name: Logical node name for timeout/diagnostics.
            fn: Node callable.
            state: Current state snapshot.

        Returns:
            Updated state from node execution, or timeout-marked state.
        """
        timeout = self._resolve_timeout(name)
        candidate_state = AgentState(**deepcopy(dict(state)))
        future = self._node_executor.submit(fn, candidate_state)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            state.setdefault("warnings", []).append("Timeout at node '{}'".format(name))
            state.setdefault("errors", []).append("Execution timed out in node '{}'".format(name))
            state["status"] = "timeout"
            state["should_refine"] = False
            return state

    def _resolve_timeout(self, node_name: str) -> int:
        """Resolves timeout budget for a node name.

        Args:
            node_name: Logical node name.

        Returns:
            Timeout in seconds.
        """
        mapping = self.config.runtime.operation_timeouts
        return int(mapping.get(node_name, self.config.runtime.default_timeout_seconds))

    def export_result(self, state: AgentState, output_path: str) -> str:
        """Exports result payload to JSON for downstream integrations.

        Args:
            state: Final state to export.
            output_path: Destination JSON file path.

        Returns:
            Written file path as string.
        """

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "session_id": state.get("session_id"),
            "status": state.get("status"),
            "domain": state.get("domain"),
            "strategy": state.get("strategy"),
            "llm": state.get("llm", {}),
            "has_visual_input": bool(state.get("visual_input")),
            "result": state.get("symbolic_result"),
            "numeric_result": state.get("numeric_result"),
            "verification": state.get("verification"),
            "explanation": state.get("explanation"),
            "metrics": state.get("metrics"),
            "artifacts": state.get("artifacts", []),
        }
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        return str(output)

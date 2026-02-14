"""LangGraph orchestration for a fully generative MathSolverAgent pipeline."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
    return payload


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
        self.llm_client = GenerativeMathClient(config=self.config.llm)
        self.checkpoints = CheckpointStore(self.config.runtime.checkpoint_dir)

        self._graph = self._build_langgraph(self.llm_client) if StateGraph is not None else None

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
            llm_overrides: Optional per-request LLM configuration overrides.

        Returns:
            Final agent state for the solve request.
        """
        sid = session_id or str(uuid.uuid4())

        if resume:
            resumed = self.checkpoints.load(sid)
            if resumed:
                state = resumed
                state["status"] = "resumed"
            else:
                state = self._new_state(problem, sid)
        else:
            state = self._new_state(problem, sid)

        request_llm_client = self._resolve_request_llm_client(llm_overrides)
        state["prompt_variant"] = select_prompt_variant(self.prompt_config, sid)
        state["llm"] = request_llm_client.describe()
        require_llm = bool(self.config.llm.get("require_available", True))
        if require_llm and not request_llm_client.is_available:
            state["status"] = "failed_precondition"
            state.setdefault("errors", []).append(
                "Generative mode requires an available LLM provider. Check API key and provider dependencies."
            )
            self.checkpoints.save(state)
            return state
        if any([image_path, image_url, image_base64]):
            state["visual_input"] = {
                "image_path": image_path,
                "image_url": image_url,
                "image_base64": image_base64,
                "image_media_type": image_media_type,
            }
        else:
            state.setdefault("visual_input", {})

        if self._graph is not None and request_llm_client is self.llm_client:
            return self._run_langgraph(state, self._graph)
        if self._graph is not None:
            request_graph = self._build_langgraph(request_llm_client)
            if request_graph is not None:
                return self._run_langgraph(state, request_graph)
        return self._run_fallback(state, request_llm_client)

    def _resolve_request_llm_client(self, llm_overrides: Optional[Dict[str, Any]]) -> GenerativeMathClient:
        """Resolves the LLM client for one request.

        Args:
            llm_overrides: Optional LLM settings passed at request time.

        Returns:
            Default client or a request-scoped client with merged settings.
        """
        if not llm_overrides:
            return self.llm_client

        effective_config = dict(self.config.llm)
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
        self.checkpoints.save(final_state)
        return final_state

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

        return state

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
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn, candidate_state)
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
        }
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        return str(output)

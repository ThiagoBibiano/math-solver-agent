"""Generative AI client for analyzer/solver/explainer stages."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


try:  # pragma: no cover
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

try:  # pragma: no cover
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except ImportError:  # pragma: no cover
    ChatNVIDIA = None  # type: ignore[assignment]

try:  # pragma: no cover
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover
    HumanMessage = None  # type: ignore[assignment]
    SystemMessage = None  # type: ignore[assignment]


@dataclass
class LLMRuntimeConfig:
    """Runtime configuration for the generative LLM client.

    Attributes:
        enabled: Enables or disables the client bootstrap.
        provider: Provider name supported by this client facade.
        model: Provider model identifier.
        api_key_env: Preferred environment variable for the API key.
        temperature: Sampling temperature used by chat completions.
        top_p: Nucleus sampling parameter.
        max_completion_tokens: Maximum number of output tokens.
        thinking: Enables provider-specific reasoning mode when supported.
        multimodal_enabled: Enables image input in chat messages.
        max_image_bytes: Maximum image payload size accepted for local files.
    """

    enabled: bool = True
    provider: str = "nvidia"
    model: str = "moonshotai/kimi-k2.5"
    api_key_env: str = "NVIDIA_API_KEY"
    temperature: float = 0.2
    top_p: float = 1.0
    max_completion_tokens: int = 8192
    thinking: bool = True
    multimodal_enabled: bool = True
    max_image_bytes: int = 5242880


class GenerativeMathClient:
    """Facade for generative operations with provider/runtime safeguards.

    This class centralizes:
    - provider bootstrap and key resolution;
    - text and multimodal invocation;
    - JSON-oriented prompting helpers for pipeline nodes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Builds a client from runtime config and bootstraps provider access.

        Args:
            config: Optional runtime settings overriding defaults.
        """
        raw = config or {}
        self.config = LLMRuntimeConfig(
            enabled=bool(raw.get("enabled", True)),
            provider=str(raw.get("provider", "nvidia")),
            model=str(raw.get("model", "moonshotai/kimi-k2.5")),
            api_key_env=str(raw.get("api_key_env", "NVIDIA_API_KEY")),
            temperature=float(raw.get("temperature", 0.2)),
            top_p=float(raw.get("top_p", 1.0)),
            max_completion_tokens=int(raw.get("max_completion_tokens", 8192)),
            thinking=bool(raw.get("thinking", True)),
            multimodal_enabled=bool(raw.get("multimodal_enabled", True)),
            max_image_bytes=int(raw.get("max_image_bytes", 5242880)),
        )

        self._client: Optional[Any] = None
        self._unavailable_reason: Optional[str] = None
        self._bootstrap()

    @property
    def is_available(self) -> bool:
        """Indicates whether the provider client is ready for inference.

        Returns:
            True when the provider SDK is available and configured.
        """
        return self._client is not None

    def describe(self) -> Dict[str, Any]:
        """Returns diagnostics about provider and key availability.

        Returns:
            A serializable dictionary with provider/runtime metadata.
        """
        api_key_present = any(bool((os.getenv(name) or "").strip()) for name in self._key_candidates())
        return {
            "enabled": self.config.enabled,
            "provider": self.config.provider,
            "model": self.config.model,
            "available": self.is_available,
            "reason": self._unavailable_reason,
            "multimodal_enabled": self.config.multimodal_enabled,
            "api_key_env": self.config.api_key_env,
            "api_key_present": api_key_present,
        }

    def _bootstrap(self) -> None:
        """Initializes the provider client if runtime preconditions are met."""
        if not self.config.enabled:
            self._unavailable_reason = "disabled_by_config"
            return
        if self.config.provider != "nvidia":
            self._unavailable_reason = "unsupported_provider"
            return
        _load_environment_variables()
        if ChatNVIDIA is None:
            self._unavailable_reason = "missing_dependency_langchain_nvidia_ai_endpoints"
            return

        api_key = _resolve_api_key(self._key_candidates())

        if not api_key:
            self._unavailable_reason = "missing_api_key"
            return

        self._client = ChatNVIDIA(
            model=self.config.model,
            api_key=api_key,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_completion_tokens=self.config.max_completion_tokens,
        )

    def _key_candidates(self) -> List[str]:
        """Returns key environment names ordered by lookup preference.

        Returns:
            A list of environment variable names.
        """
        return [self.config.api_key_env, "NVIDIA_API_KEY", "nvidia_api_key"]

    def stream(self, messages: List[Dict[str, str]]) -> Iterable[Dict[str, str]]:
        """Streams chat chunks from the provider.

        Args:
            messages: Chat messages in role/content mapping format.

        Returns:
            An iterable of chunks with `reasoning` and `content`.
        """
        if not self.is_available:
            return iter(())

        assert self._client is not None
        return self._stream_chunks(messages)

    def _stream_chunks(self, messages: List[Dict[str, str]]) -> Iterable[Dict[str, str]]:
        """Internal chunk streamer used by :meth:`stream`.

        Args:
            messages: Provider-compatible message payload.

        Yields:
            Dictionaries with reasoning and textual token content.
        """
        assert self._client is not None
        for chunk in self._client.stream(messages, chat_template_kwargs={"thinking": self.config.thinking}):
            reasoning = ""
            if getattr(chunk, "additional_kwargs", None) and "reasoning_content" in chunk.additional_kwargs:
                reasoning = str(chunk.additional_kwargs["reasoning_content"])
            content = str(getattr(chunk, "content", ""))
            yield {"reasoning": reasoning, "content": content}

    def invoke_text(
        self,
        system_prompt: str,
        user_prompt: str,
        image_input: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Invokes the model and returns plain text.

        Args:
            system_prompt: System instruction.
            user_prompt: User message content.
            image_input: Optional multimodal image payload metadata.

        Returns:
            Assistant textual response, or empty string on unavailable client.
        """
        if not self.is_available:
            return ""

        assert self._client is not None
        messages = _build_langchain_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_input=image_input,
            multimodal_enabled=self.config.multimodal_enabled,
            max_image_bytes=self.config.max_image_bytes,
        )
        response = self._client.invoke(
            messages,
            chat_template_kwargs={"thinking": self.config.thinking},
        )
        return str(getattr(response, "content", "") or "")

    def invoke_json(
        self,
        system_prompt: str,
        user_prompt: str,
        default: Optional[Dict[str, Any]] = None,
        image_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invokes the model and parses a dictionary from response text.

        Args:
            system_prompt: System instruction.
            user_prompt: User message content.
            default: Default dictionary returned when parsing fails.
            image_input: Optional multimodal image payload metadata.

        Returns:
            Parsed dictionary payload or `default`.
        """
        default = default or {}
        text = self.invoke_text(system_prompt, user_prompt, image_input=image_input)
        if not text:
            return default
        return _extract_json_dict(text, default)

    def analyze_problem(
        self,
        problem: str,
        prompt_pack: Optional[Dict[str, str]] = None,
        prompt_variant: Optional[str] = None,
        image_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs generative analysis for domain, constraints, and normalization.

        Args:
            problem: Raw user problem statement.
            prompt_pack: Prompt templates for this node.
            prompt_variant: A/B prompt variant label.
            image_input: Optional multimodal input payload.

        Returns:
            A dictionary expected to include analysis keys for the analyzer node.
        """
        if not self.is_available:
            return {}

        has_image = bool(image_input and any(image_input.get(k) for k in ("image_url", "image_path", "image_base64")))
        system = (prompt_pack or {}).get(
            "system",
            "Você é um analisador matemático rigoroso. Responda apenas JSON válido.",
        )
        user = (
            "Analise o problema e retorne SOMENTE JSON no formato: "
            "{{\"domain\": string, \"constraints\": string[], \"complexity_score\": number entre 0 e 1, "
            "\"plan\": string[], \"normalized_problem\": string}}. "
            "Use os domínios: calculo_i, calculo_ii, calculo_iii, edo, algebra_linear. "
            "Se houver imagem anexada, extraia o enunciado matemático da imagem antes de classificar. "
            "Variant={} HasImage={} Problema: {}"
        ).format(prompt_variant or "default", has_image, problem)
        return self.invoke_json(system, user, default={}, image_input=image_input)

    def propose_strategy(
        self,
        problem: str,
        analysis: Dict[str, Any],
        iteration: int,
        prompt_pack: Optional[Dict[str, str]] = None,
        image_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generates strategy-oriented payload from problem and analysis context.

        Args:
            problem: Normalized or raw problem statement.
            analysis: Analyzer output summary.
            iteration: Current iteration number.
            prompt_pack: Prompt templates for this node.
            image_input: Optional multimodal input payload.

        Returns:
            Strategy dictionary emitted by the model.
        """
        if not self.is_available:
            return {}

        has_image = bool(image_input and any(image_input.get(k) for k in ("image_url", "image_path", "image_base64")))
        system = (prompt_pack or {}).get(
            "system",
            "Você é um planejador matemático. Responda apenas JSON válido.",
        )
        user = (
            "Com base no problema e análise, retorne SOMENTE JSON: "
            "{{\"strategy\": \"symbolic|numeric|hybrid\", \"expression\": string, \"steps\": string[], \"notes\": string}}. "
            "Se houver imagem anexada, considere o conteúdo visual na estratégia. "
            "Iteração={} HasImage={} Problema={} Analise={}"
        ).format(iteration, has_image, problem, json.dumps(analysis, ensure_ascii=True))
        return self.invoke_json(system, user, default={}, image_input=image_input)

    def plan_tool_execution(
        self,
        problem: str,
        analysis: Dict[str, Any],
        iteration: int,
        prompt_pack: Optional[Dict[str, str]] = None,
        image_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Builds a tool-execution plan in strict JSON format.

        Args:
            problem: Normalized or raw problem statement.
            analysis: Analyzer output summary.
            iteration: Current iteration number.
            prompt_pack: Prompt templates for this node.
            image_input: Optional multimodal input payload.

        Returns:
            A dictionary with `strategy`, `plan`, and `tool_call`.
        """
        if not self.is_available:
            return {}

        has_image = bool(image_input and any(image_input.get(k) for k in ("image_url", "image_path", "image_base64")))
        system = (prompt_pack or {}).get(
            "system",
            "Você é um conversor de linguagem natural para chamadas de ferramentas matemáticas.",
        )
        user = (
            "Retorne SOMENTE JSON no formato "
            "{{\"strategy\": \"symbolic|numeric|hybrid\", "
            "\"normalized_problem\": string, "
            "\"plan\": string[], "
            "\"tool_call\": {{\"name\": string, \"args\": object}}}}. "
            "Ferramentas permitidas: differentiate_expression, integrate_expression, solve_equation, "
            "simplify_expression, evaluate_expression, numerical_integration, solve_ode. "
            "Nao inclua texto fora do JSON. "
            "Iteracao={} HasImage={} Problem={} Analysis={}"
        ).format(iteration, has_image, problem, json.dumps(analysis, ensure_ascii=True))
        return self.invoke_json(system, user, default={}, image_input=image_input)

    def generate_explanation(
        self,
        state_summary: Dict[str, Any],
        prompt_pack: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generates pedagogical explanation text for final response.

        Args:
            state_summary: Compact state context for explanation generation.
            prompt_pack: Prompt templates for verifier/explainer.

        Returns:
            Generated explanation text or empty string if unavailable.
        """
        if not self.is_available:
            return ""

        system = (prompt_pack or {}).get(
            "system",
            "Você é um tutor de cálculo e deve explicar com rigor e clareza.",
        )
        user = (
            "Gere uma explicação pedagógica em PT-BR com passos, teoremas aplicados, alertas de erros comuns "
            "e interpretação intuitiva. Contexto JSON: {}"
        ).format(json.dumps(state_summary, ensure_ascii=True))
        return self.invoke_text(system, user).strip()


def _extract_json_dict(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts a JSON object from free-form model output.

    Args:
        text: Raw model output.
        default: Default dictionary used when extraction fails.

    Returns:
        Parsed dictionary when possible, otherwise `default`.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return default

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return default

    return default


def _build_langchain_messages(
    system_prompt: str,
    user_prompt: str,
    image_input: Optional[Dict[str, Any]],
    multimodal_enabled: bool,
    max_image_bytes: int,
) -> List[Any]:
    """Builds provider message payload with optional multimodal content.

    Args:
        system_prompt: System instruction.
        user_prompt: User text content.
        image_input: Optional image payload information.
        multimodal_enabled: Enables image attachment behavior.
        max_image_bytes: Maximum allowed image bytes for local file conversion.

    Returns:
        Provider-compatible message objects or role/content mappings.
    """
    human_content: Any = user_prompt
    if multimodal_enabled and image_input:
        human_content = _build_human_content(user_prompt, image_input, max_image_bytes=max_image_bytes)

    if SystemMessage is not None and HumanMessage is not None:
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_content},
    ]


def _build_human_content(user_prompt: str, image_input: Dict[str, Any], max_image_bytes: int) -> List[Dict[str, Any]]:
    """Builds a multimodal human content list for chat models.

    Args:
        user_prompt: User text content.
        image_input: Image payload metadata.
        max_image_bytes: Maximum allowed image bytes.

    Returns:
        Content blocks including text and optionally image_url block.
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    data_url = _resolve_image_data_url(image_input=image_input, max_image_bytes=max_image_bytes)
    if data_url:
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


def _resolve_image_data_url(image_input: Dict[str, Any], max_image_bytes: int) -> Optional[str]:
    """Resolves image payload to a model-consumable URL or data URL.

    Args:
        image_input: Image payload metadata containing url/path/base64.
        max_image_bytes: Maximum allowed image size for local files.

    Returns:
        URL/data URL when available and valid; otherwise None.
    """
    image_url = str(image_input.get("image_url") or "").strip()
    if image_url:
        return image_url

    media_type = str(image_input.get("image_media_type") or "").strip() or None

    image_base64 = str(image_input.get("image_base64") or "").strip()
    if image_base64:
        if image_base64.startswith("data:image/"):
            return image_base64
        resolved_media_type = media_type or "image/png"
        return "data:{};base64,{}".format(resolved_media_type, image_base64)

    image_path = str(image_input.get("image_path") or "").strip()
    if not image_path:
        return None

    path = Path(image_path).expanduser()
    if not path.exists() or not path.is_file():
        return None

    payload = path.read_bytes()
    if len(payload) > max_image_bytes:
        return None

    encoded = base64.b64encode(payload).decode("ascii")
    resolved_media_type = media_type or _guess_media_type(path) or "image/png"
    return "data:{};base64,{}".format(resolved_media_type, encoded)


def _guess_media_type(path: Path) -> Optional[str]:
    """Guesses image media type from a file path.

    Args:
        path: Path to local image file.

    Returns:
        MIME type when image-like; otherwise None.
    """
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("image/"):
        return guessed
    return None


def _load_environment_variables() -> None:
    """Loads environment variables from candidate `.env` files.

    Uses `python-dotenv` when installed; otherwise falls back to manual parsing.
    """
    env_candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]

    if load_dotenv is not None:
        for env_path in env_candidates:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
        return

    for env_path in env_candidates:
        if env_path.exists():
            _manual_load_env_file(env_path)


def _manual_load_env_file(path: Path) -> None:
    """Parses and loads key/value pairs from a `.env`-like file.

    Args:
        path: Path to environment file.
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_api_key(candidates: List[str]) -> Optional[str]:
    """Finds the first non-empty API key among candidate env vars.

    Args:
        candidates: Environment variable names ordered by preference.

    Returns:
        First non-empty key value, or None when no candidate is set.
    """
    for name in candidates:
        if not name:
            continue
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None

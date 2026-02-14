"""Generative AI client for analyzer/solver/explainer stages."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass, field
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
    from langchain_community.chat_models import ChatMaritalk
except ImportError:  # pragma: no cover
    ChatMaritalk = None  # type: ignore[assignment]

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
        model_profile: Model profile alias used to load defaults.
        api_key_env: Preferred environment variable for the API key.
        temperature: Sampling temperature used by chat completions.
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum number of output tokens.
        thinking: Enables provider-specific reasoning mode when supported.
        chat_template_kwargs: Runtime chat template options sent to provider.
        model_profiles: Optional custom/override model profile registry.
        multimodal_enabled: Enables image input in chat messages.
        max_image_bytes: Maximum image payload size accepted for local files.
    """

    enabled: bool = True
    provider: str = "nvidia"
    model: str = "moonshotai/kimi-k2.5"
    model_profile: str = "kimi_k2_5"
    api_key_env: str = "NVIDIA_API_KEY"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 8192
    thinking: bool = True
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    multimodal_enabled: bool = True
    max_image_bytes: int = 5242880


@dataclass
class ModelProfile:
    """Represents provider defaults and capabilities for one model family.

    Attributes:
        alias: Human-friendly alias used in config.
        provider: Backend provider name for this model.
        model: Provider model identifier.
        multimodal: Whether image inputs are supported.
        default_temperature: Default temperature for this model.
        default_top_p: Default top_p for this model.
        default_max_tokens: Default max token budget for this model.
        chat_template_kwargs: Provider-specific template kwargs.
    """

    alias: str
    provider: str
    model: str
    multimodal: bool
    default_temperature: float
    default_top_p: float
    default_max_tokens: int
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)


_DEFAULT_MODEL_PROFILES: Dict[str, ModelProfile] = {
    "kimi_k2_5": ModelProfile(
        alias="kimi_k2_5",
        provider="nvidia",
        model="moonshotai/kimi-k2.5",
        multimodal=True,
        default_temperature=1.0,
        default_top_p=1.0,
        default_max_tokens=16384,
        chat_template_kwargs={"thinking": True},
    ),
    "deepseek_v3_2": ModelProfile(
        alias="deepseek_v3_2",
        provider="nvidia",
        model="deepseek-ai/deepseek-v3.2",
        multimodal=False,
        default_temperature=1.0,
        default_top_p=0.95,
        default_max_tokens=8192,
        chat_template_kwargs={"thinking": True},
    ),
    "glm4_7": ModelProfile(
        alias="glm4_7",
        provider="nvidia",
        model="z-ai/glm4.7",
        multimodal=True,
        default_temperature=1.0,
        default_top_p=1.0,
        default_max_tokens=16384,
        chat_template_kwargs={"enable_thinking": True, "clear_thinking": False},
    ),
    "glm5": ModelProfile(
        alias="glm5",
        provider="nvidia",
        model="z-ai/glm5",
        multimodal=True,
        default_temperature=1.0,
        default_top_p=1.0,
        default_max_tokens=16384,
        chat_template_kwargs={"enable_thinking": True, "clear_thinking": False},
    ),
    "minimax_m2_1": ModelProfile(
        alias="minimax_m2_1",
        provider="nvidia",
        model="minimaxai/minimax-m2.1",
        multimodal=True,
        default_temperature=1.0,
        default_top_p=0.95,
        default_max_tokens=8192,
        chat_template_kwargs={},
    ),
    "sabia_4": ModelProfile(
        alias="sabia_4",
        provider="maritaca",
        model="sabia-4",
        multimodal=False,
        default_temperature=0.7,
        default_top_p=1.0,
        default_max_tokens=8192,
        chat_template_kwargs={},
    ),
    "sabiazinho_4": ModelProfile(
        alias="sabiazinho_4",
        provider="maritaca",
        model="sabiazinho-4",
        multimodal=False,
        default_temperature=0.7,
        default_top_p=1.0,
        default_max_tokens=8192,
        chat_template_kwargs={},
    ),
}


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
        requested_provider = str(raw.get("provider", "")).strip()
        profile_registry = _build_model_profile_registry(raw.get("model_profiles"))
        selected_profile = _resolve_model_profile(
            model_alias=str(raw.get("model_profile", "kimi_k2_5")),
            explicit_model=str(raw.get("model", "")),
            registry=profile_registry,
            provider_hint=requested_provider or None,
        )
        provider_value = requested_provider or selected_profile.provider

        temperature_value = float(raw.get("temperature", selected_profile.default_temperature))
        top_p_value = float(raw.get("top_p", selected_profile.default_top_p))
        max_tokens_raw = raw.get("max_tokens", raw.get("max_completion_tokens", selected_profile.default_max_tokens))
        max_tokens_value = int(max_tokens_raw)

        profile_chat_kwargs = dict(selected_profile.chat_template_kwargs)
        explicit_chat_kwargs = raw.get("chat_template_kwargs")
        if isinstance(explicit_chat_kwargs, dict):
            profile_chat_kwargs.update(explicit_chat_kwargs)
        elif bool(raw.get("thinking", True)) and not profile_chat_kwargs:
            profile_chat_kwargs = {"thinking": True}

        multimodal_enabled = bool(raw.get("multimodal_enabled", True)) and selected_profile.multimodal

        self.config = LLMRuntimeConfig(
            enabled=bool(raw.get("enabled", True)),
            provider=provider_value,
            model=selected_profile.model,
            model_profile=selected_profile.alias,
            api_key_env=str(raw.get("api_key_env", _default_api_key_env(provider_value))),
            temperature=temperature_value,
            top_p=top_p_value,
            max_tokens=max_tokens_value,
            thinking=bool(raw.get("thinking", True)),
            chat_template_kwargs=profile_chat_kwargs,
            model_profiles={},
            multimodal_enabled=multimodal_enabled,
            max_image_bytes=int(raw.get("max_image_bytes", 5242880)),
        )
        self.model_profile = selected_profile

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
            "model_profile": self.config.model_profile,
            "available": self.is_available,
            "reason": self._unavailable_reason,
            "multimodal_enabled": self.config.multimodal_enabled,
            "supports_multimodal": self.model_profile.multimodal,
            "chat_template_kwargs": self.config.chat_template_kwargs,
            "api_key_env": self.config.api_key_env,
            "api_key_present": api_key_present,
        }

    def _bootstrap(self) -> None:
        """Initializes the provider client if runtime preconditions are met."""
        if not self.config.enabled:
            self._unavailable_reason = "disabled_by_config"
            return
        if self.config.provider not in {"nvidia", "maritaca"}:
            self._unavailable_reason = "unsupported_provider"
            return
        _load_environment_variables()

        api_key = _resolve_api_key(self._key_candidates())

        if not api_key:
            self._unavailable_reason = "missing_api_key"
            return

        if self.config.provider == "nvidia":
            if ChatNVIDIA is None:
                self._unavailable_reason = "missing_dependency_langchain_nvidia_ai_endpoints"
                return
            self._client = ChatNVIDIA(
                model=self.config.model,
                api_key=api_key,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                extra_body=self._extra_body(),
            )
            return

        if ChatMaritalk is None:
            self._unavailable_reason = "missing_dependency_langchain_community"
            return
        self._client = ChatMaritalk(
            model=self.config.model,
            api_key=api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def _key_candidates(self) -> List[str]:
        """Returns key environment names ordered by lookup preference.

        Returns:
            A list of environment variable names.
        """
        if self.config.provider == "maritaca":
            return [self.config.api_key_env, "MARITACA_API_KEY", "maritaca_api_key"]
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
        for chunk in self._client.stream(messages, **self._invocation_kwargs()):
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
            **self._invocation_kwargs(),
        )
        return str(getattr(response, "content", "") or "")

    def _invocation_kwargs(self) -> Dict[str, Any]:
        """Builds provider invocation kwargs from effective runtime settings.

        Returns:
            Invocation kwargs with provider-specific template options.
        """
        chat_kwargs = self.config.chat_template_kwargs
        if self.config.provider != "nvidia":
            return {}
        if not chat_kwargs and self.config.thinking:
            chat_kwargs = {"thinking": True}
        if not chat_kwargs:
            return {}
        return {"extra_body": {"chat_template_kwargs": chat_kwargs}}

    def _extra_body(self) -> Dict[str, Any]:
        """Builds constructor extra_body for provider-specific options.

        Returns:
            Extra body dictionary used during client construction.
        """
        chat_kwargs = self.config.chat_template_kwargs
        if self.config.provider != "nvidia":
            return {}
        if not chat_kwargs and self.config.thinking:
            chat_kwargs = {"thinking": True}
        if not chat_kwargs:
            return {}
        return {"chat_template_kwargs": chat_kwargs}

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
        default_user = (
            "Analise o problema e retorne SOMENTE JSON no formato: "
            "{{\"domain\": string, \"constraints\": string[], \"complexity_score\": number entre 0 e 1, "
            "\"plan\": string[], \"normalized_problem\": string}}. "
            "Use os domínios: calculo_i, calculo_ii, calculo_iii, edo, algebra_linear. "
            "Se houver imagem anexada, extraia o enunciado matemático da imagem antes de classificar. "
            "Variant={{prompt_variant}} HasImage={{has_image}} Problema: {{problem}}"
        )
        user_template = (prompt_pack or {}).get("user", default_user)
        user = _render_prompt_template(
            user_template,
            {
                "problem": problem,
                "prompt_variant": prompt_variant or "default",
                "has_image": has_image,
            },
        )
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
        default_user = (
            "Com base no problema e análise, retorne SOMENTE JSON: "
            "{{\"strategy\": \"symbolic|numeric|hybrid\", \"expression\": string, \"steps\": string[], \"notes\": string}}. "
            "Se houver imagem anexada, considere o conteúdo visual na estratégia. "
            "Iteração={{iteration}} HasImage={{has_image}} Problema={{problem}} Analise={{analysis}}"
        )
        user_template = (prompt_pack or {}).get("user", default_user)
        user = _render_prompt_template(
            user_template,
            {
                "problem": problem,
                "analysis": analysis,
                "iteration": iteration,
                "has_image": has_image,
            },
        )
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
        default_user = (
            "Retorne SOMENTE JSON no formato "
            "{{\"strategy\": \"symbolic|numeric|hybrid\", "
            "\"normalized_problem\": string, "
            "\"plan\": string[], "
            "\"tool_call\": {{\"name\": string, \"args\": object}}}}. "
            "Ferramentas permitidas: differentiate_expression, integrate_expression, solve_equation, "
            "simplify_expression, evaluate_expression, numerical_integration, solve_ode. "
            "Nao inclua texto fora do JSON. "
            "Iteracao={{iteration}} HasImage={{has_image}} Problem={{problem}} Analysis={{analysis}}"
        )
        user_template = (prompt_pack or {}).get("user", default_user)
        user = _render_prompt_template(
            user_template,
            {
                "problem": problem,
                "analysis": analysis,
                "iteration": iteration,
                "has_image": has_image,
            },
        )
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
        default_user = (
            "Gere uma explicação pedagógica em PT-BR com passos, teoremas aplicados, alertas de erros comuns "
            "e interpretação intuitiva. Contexto JSON: {{state_summary}}"
        )
        user_template = (prompt_pack or {}).get("user", default_user)
        user = _render_prompt_template(user_template, {"state_summary": state_summary})
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


def _render_prompt_template(template: str, context: Dict[str, Any]) -> str:
    """Renders a simple `{{key}}` prompt template using context values.

    Args:
        template: Template string with placeholders in `{{key}}` form.
        context: Values available for placeholder replacement.

    Returns:
        Rendered string with available keys replaced.
    """
    rendered = template
    for key, value in context.items():
        rendered = rendered.replace("{{" + str(key) + "}}", _stringify_template_value(value))
    return rendered


def _stringify_template_value(value: Any) -> str:
    """Converts context values into template-safe strings.

    Args:
        value: Value to convert.

    Returns:
        String representation suitable for prompt templates.
    """
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


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


def _build_model_profile_registry(raw_profiles: Any) -> Dict[str, ModelProfile]:
    """Builds model profile registry from defaults and optional overrides.

    Args:
        raw_profiles: Optional dictionary containing profile overrides.

    Returns:
        Registry mapping aliases to `ModelProfile`.
    """
    registry: Dict[str, ModelProfile] = dict(_DEFAULT_MODEL_PROFILES)
    if not isinstance(raw_profiles, dict):
        return registry

    for alias, value in raw_profiles.items():
        if not isinstance(value, dict):
            continue
        profile = _profile_from_mapping(str(alias), value)
        if profile is not None:
            registry[str(alias)] = profile
    return registry


def _profile_from_mapping(alias: str, payload: Dict[str, Any]) -> Optional[ModelProfile]:
    """Creates a model profile from generic mapping data.

    Args:
        alias: Profile alias.
        payload: Profile settings dictionary.

    Returns:
        Parsed `ModelProfile` or None if payload is invalid.
    """
    model = str(payload.get("model", "")).strip()
    if not model:
        return None
    return ModelProfile(
        alias=alias,
        provider=str(payload.get("provider", "nvidia")).strip() or "nvidia",
        model=model,
        multimodal=bool(payload.get("multimodal", True)),
        default_temperature=float(payload.get("temperature", 1.0)),
        default_top_p=float(payload.get("top_p", 1.0)),
        default_max_tokens=int(payload.get("max_tokens", payload.get("max_completion_tokens", 8192))),
        chat_template_kwargs=dict(payload.get("chat_template_kwargs", {})),
    )


def _resolve_model_profile(
    model_alias: str,
    explicit_model: str,
    registry: Dict[str, ModelProfile],
    provider_hint: Optional[str] = None,
) -> ModelProfile:
    """Resolves effective model profile using alias and/or explicit model id.

    Args:
        model_alias: Preferred alias from config.
        explicit_model: Explicit model id from config.
        registry: Available model profile registry.
        provider_hint: Optional provider constraint (e.g., nvidia, maritaca).

    Returns:
        Resolved model profile.
    """
    alias = (model_alias or "").strip()
    if alias and alias in registry:
        profile = registry[alias]
        if provider_hint and profile.provider != provider_hint:
            profile = None
        if profile and explicit_model.strip():
            return ModelProfile(
                alias=profile.alias,
                provider=profile.provider,
                model=explicit_model.strip(),
                multimodal=profile.multimodal,
                default_temperature=profile.default_temperature,
                default_top_p=profile.default_top_p,
                default_max_tokens=profile.default_max_tokens,
                chat_template_kwargs=dict(profile.chat_template_kwargs),
            )
        if profile:
            return profile

    explicit = explicit_model.strip()
    if explicit:
        for profile in registry.values():
            if profile.model == explicit and (not provider_hint or profile.provider == provider_hint):
                return profile
        return ModelProfile(
            alias=_slugify_model_alias(explicit),
            provider=provider_hint or "nvidia",
            model=explicit,
            multimodal=True,
            default_temperature=1.0,
            default_top_p=1.0,
            default_max_tokens=8192,
            chat_template_kwargs={},
        )

    return _default_profile_for_provider(provider_hint, registry)


def _slugify_model_alias(model_name: str) -> str:
    """Converts model identifier into a stable alias.

    Args:
        model_name: Provider model identifier.

    Returns:
        Alias-safe slug string.
    """
    return re.sub(r"[^a-z0-9_]+", "_", model_name.lower()).strip("_")


def _default_profile_for_provider(provider_hint: Optional[str], registry: Dict[str, ModelProfile]) -> ModelProfile:
    """Returns default profile for a provider hint.

    Args:
        provider_hint: Optional provider name.
        registry: Available profile registry.

    Returns:
        A profile matching provider preference when possible.
    """
    if provider_hint:
        for profile in registry.values():
            if profile.provider == provider_hint:
                return profile
    return registry["kimi_k2_5"]


def _default_api_key_env(provider: str) -> str:
    """Returns default API key environment variable for provider.

    Args:
        provider: Provider name.

    Returns:
        Environment variable name.
    """
    if provider == "maritaca":
        return "MARITACA_API_KEY"
    return "NVIDIA_API_KEY"


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

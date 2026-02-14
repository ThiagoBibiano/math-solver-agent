"""Chainlit frontend for MathSolverAgent API."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import chainlit as cl
from chainlit.input_widget import Select, Slider, TextInput

try:
    from src.ui.api_client import build_solve_payload, call_solve_api_async, infer_image_media_type
    from src.utils import load_graph_config
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.ui.api_client import build_solve_payload, call_solve_api_async, infer_image_media_type
    from src.utils import load_graph_config


SESSION_SETTINGS_KEY = "settings"
SESSION_HISTORY_KEY = "history"
SESSION_ACTIVE_ID_KEY = "active_session_id"
LOG_LEVEL = os.getenv("MATH_SOLVER_UI_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("math_solver_agent.ui.chainlit")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)
logger.propagate = False


def _load_ui_defaults() -> Dict[str, Any]:
    """Loads default values from graph configuration."""
    cfg = load_graph_config()
    llm_cfg = dict(cfg.llm or {})
    profile_map = llm_cfg.get("model_profiles", {}) if isinstance(llm_cfg.get("model_profiles"), dict) else {}
    profile_limits: Dict[str, int] = {}
    provider_profiles: Dict[str, list[str]] = {}
    for alias, payload in profile_map.items():
        if not isinstance(payload, dict):
            continue
        provider = str(payload.get("provider", llm_cfg.get("provider", "nvidia"))).strip().lower() or "nvidia"
        provider_profiles.setdefault(provider, []).append(str(alias))
        try:
            profile_limits[str(alias)] = int(payload.get("max_tokens", payload.get("max_completion_tokens", 8192)))
        except (TypeError, ValueError):
            profile_limits[str(alias)] = 8192
    for provider, aliases in provider_profiles.items():
        provider_profiles[provider] = sorted(set(aliases))

    if not provider_profiles:
        provider_profiles = {"nvidia": [str(llm_cfg.get("model_profile", "kimi_k2_5"))]}

    providers = sorted(provider_profiles.keys())
    configured_provider = str(llm_cfg.get("provider", "")).strip().lower()
    if configured_provider not in provider_profiles:
        configured_provider = providers[0]

    configured_profile = str(llm_cfg.get("model_profile", "")).strip()
    eligible_profiles = provider_profiles.get(configured_provider, [])
    default_profile = configured_profile if configured_profile in eligible_profiles else (eligible_profiles[0] if eligible_profiles else None)

    return {
        "providers": providers,
        "provider_profiles": provider_profiles,
        "profile_limits": profile_limits,
        "all_model_profiles": sorted({alias for aliases in provider_profiles.values() for alias in aliases}),
        "default_provider": configured_provider,
        "default_model_profile": default_profile,
        "temperature": float(llm_cfg.get("temperature", 0.2)),
        "max_tokens": int(llm_cfg.get("max_tokens", 8192)),
        "api_url": os.getenv("MATH_SOLVER_API_URL", "http://localhost:8001"),
        "timeout_seconds": int(os.getenv("MATH_SOLVER_API_TIMEOUT", "300")),
    }


UI_DEFAULTS = _load_ui_defaults()


def _initial_index(values: Iterable[str], selected: str) -> int:
    normalized = [str(item) for item in values]
    try:
        return normalized.index(selected)
    except ValueError:
        return 0


def _profiles_for_provider(provider: str) -> list[str]:
    mapping = UI_DEFAULTS.get("provider_profiles", {})
    if isinstance(mapping, dict):
        values = mapping.get(str(provider or "").strip().lower())
        if isinstance(values, list) and values:
            return list(values)
    all_profiles = UI_DEFAULTS.get("all_model_profiles", [])
    if isinstance(all_profiles, list) and all_profiles:
        return list(all_profiles)
    return [str(UI_DEFAULTS.get("default_model_profile") or "kimi_k2_5")]


def _chat_settings_widgets(settings: Optional[Dict[str, Any]] = None) -> list[Any]:
    current = settings or {}
    providers = UI_DEFAULTS["providers"] or ["nvidia", "maritaca"]
    selected_provider = str(current.get("provider", UI_DEFAULTS["default_provider"])).strip().lower() or UI_DEFAULTS["default_provider"]
    profiles = UI_DEFAULTS.get("all_model_profiles", []) or _profiles_for_provider(selected_provider)
    selected_profile = str(current.get("model_profile", UI_DEFAULTS["default_model_profile"] or "")).strip()
    if selected_profile not in profiles and profiles:
        selected_profile = profiles[0]
    return [
        TextInput(
            id="api_url",
            label="API Base URL",
            initial=str(current.get("api_url", UI_DEFAULTS["api_url"])),
            placeholder="http://localhost:8001",
        ),
        Select(
            id="provider",
            label="Provider",
            values=providers,
            initial_index=_initial_index(providers, selected_provider),
        ),
        Select(
            id="model_profile",
            label="Model Profile",
            values=profiles,
            initial_index=_initial_index(profiles, selected_profile),
        ),
        Slider(
            id="temperature",
            label="Temperature",
            initial=min(max(float(current.get("temperature", UI_DEFAULTS["temperature"])), 0.0), 1.0),
            min=0.0,
            max=1.0,
            step=0.05,
        ),
        Slider(
            id="max_tokens",
            label="Max Tokens",
            initial=max(256, int(current.get("max_tokens", UI_DEFAULTS["max_tokens"]))),
            min=256,
            max=32768,
            step=256,
        ),
        Slider(
            id="timeout_seconds",
            label="Timeout (s)",
            initial=max(60, min(1000, int(current.get("timeout_seconds", UI_DEFAULTS["timeout_seconds"])))),
            min=60,
            max=1000,
            step=10,
        ),
        TextInput(
            id="session_id",
            label="Session ID (opcional)",
            initial=str(current.get("session_id", "") or ""),
            placeholder="Ex.: sessao-demo-001",
        ),
    ]


def _normalize_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = settings or {}
    api_url = str(data.get("api_url", UI_DEFAULTS["api_url"])).strip() or UI_DEFAULTS["api_url"]
    provider = str(data.get("provider", UI_DEFAULTS["default_provider"])).strip().lower() or UI_DEFAULTS["default_provider"]
    providers = UI_DEFAULTS.get("providers", [])
    if provider not in providers and providers:
        provider = providers[0]

    eligible_profiles = _profiles_for_provider(provider)
    model_profile = str(data.get("model_profile", UI_DEFAULTS["default_model_profile"] or "")).strip()
    if model_profile not in eligible_profiles and eligible_profiles:
        model_profile = eligible_profiles[0]

    try:
        temperature = float(data.get("temperature", UI_DEFAULTS["temperature"]))
    except (TypeError, ValueError):
        temperature = UI_DEFAULTS["temperature"]
    temperature = max(0.0, min(1.0, temperature))

    try:
        max_tokens = int(data.get("max_tokens", UI_DEFAULTS["max_tokens"]))
    except (TypeError, ValueError):
        max_tokens = UI_DEFAULTS["max_tokens"]
    max_tokens = max(256, min(32768, max_tokens))
    profile_limits = UI_DEFAULTS.get("profile_limits", {})
    if isinstance(profile_limits, dict):
        limit = profile_limits.get(model_profile)
        if isinstance(limit, int) and limit > 0:
            max_tokens = min(max_tokens, limit)

    try:
        timeout_seconds = int(data.get("timeout_seconds", UI_DEFAULTS["timeout_seconds"]))
    except (TypeError, ValueError):
        timeout_seconds = UI_DEFAULTS["timeout_seconds"]
    timeout_seconds = max(60, min(1000, timeout_seconds))

    session_id = str(data.get("session_id", "")).strip()

    return {
        "api_url": api_url,
        "provider": provider,
        "model_profile": model_profile or None,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "session_id": session_id or None,
    }


def _extract_image_attachment(message: cl.Message) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    for element in message.elements or []:
        element_path = getattr(element, "path", None)
        element_name = str(getattr(element, "name", "") or "")
        element_mime = str(getattr(element, "mime", "") or "")

        if not element_path:
            continue

        path = Path(str(element_path))
        if not path.exists():
            continue

        if element_mime and not element_mime.startswith("image/"):
            continue
        if not element_mime and infer_image_media_type(element_name or path.name, fallback="").strip() == "":
            continue

        with path.open("rb") as handle:
            return handle.read(), (element_name or path.name), (element_mime or None)

    return None, None, None


def _format_trace_content(event: Dict[str, Any]) -> str:
    summary = str(event.get("summary", "")).strip() or "Sem resumo fornecido."
    confidence = event.get("confidence")
    timestamp = str(event.get("timestamp", "")).strip()

    lines = [summary]
    if confidence is not None:
        try:
            lines.append("Confianca: {:.2f}".format(float(confidence)))
        except (TypeError, ValueError):
            lines.append("Confianca: {}".format(confidence))
    if timestamp:
        lines.append("Timestamp: `{}`".format(timestamp))
    return "\n\n".join(lines)

def _format_final_response(response: Dict[str, Any]) -> str:
    status = response.get("status", "-")
    session_id = response.get("session_id", "-")
    domain = response.get("domain", "-")
    strategy = response.get("strategy", "-")
    llm_info = response.get("llm", {}) if isinstance(response.get("llm"), dict) else {}
    result = response.get("result")
    numeric_result = response.get("numeric_result")
    explanation = str(response.get("explanation", "")).strip()
    verification = response.get("verification", {})

    blocks = [
        "### Resultado",
        "- **Session ID:** `{}`".format(session_id),
        "- **Status:** `{}`".format(status),
        "- **Dominio:** `{}`".format(domain),
        "- **Estrategia:** `{}`".format(strategy),
        "- **Provider:** `{}`".format(llm_info.get("provider", "-")),
        "- **Model:** `{}`".format(llm_info.get("model", "-")),
        "",
        "#### Resultado simbolico",
        "```text",
        "{}".format(result or "Sem resultado simbolico."),
        "```",
    ]

    if numeric_result is not None:
        blocks.extend(["", "#### Resultado numerico", "`{}`".format(numeric_result)])

    if isinstance(verification, dict) and verification:
        blocks.extend(["", "#### Verificacao", "```json", json.dumps(verification, ensure_ascii=False, indent=2), "```"])

    blocks.extend(["", "#### Explicacao"])
    blocks.append(explanation or "_Sem explicacao retornada._")

    return "\n".join(blocks)


def _friendly_error_message(raw_error: str) -> str:
    text = raw_error.strip()
    if "API error 503" in text:
        return "A API retornou 503: LLM indisponivel no modo estrito. Verifique API key e provider."
    if "API error 422" in text:
        return "A API rejeitou a requisicao (422). Envie texto, imagem, ou use `/resume` com session_id valido."
    if "API error 405" in text:
        return "A URL configurada nao aponta para a API do MathSolverAgent. Confira `API Base URL` (ex.: http://localhost:8001)."
    if "Nao foi possivel conectar na API" in text:
        return "Nao foi possivel conectar na API. Confira a URL da API e se o servidor esta em execucao."
    return text


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = await cl.ChatSettings(_chat_settings_widgets()).send()
    normalized = _normalize_settings(settings)
    cl.user_session.set(SESSION_SETTINGS_KEY, normalized)
    cl.user_session.set(SESSION_HISTORY_KEY, [])
    cl.user_session.set(SESSION_ACTIVE_ID_KEY, normalized.get("session_id"))
    logger.info(
        "chat_start api_url=%s provider=%s model_profile=%s timeout=%ss",
        normalized.get("api_url"),
        normalized.get("provider"),
        normalized.get("model_profile"),
        normalized.get("timeout_seconds"),
    )

    await cl.Message(
        content=(
            "MathSolverAgent pronto.\n\n"
            "- Escreva o enunciado em Markdown/LaTeX.\n"
            "- Anexe imagem pelo clipe na caixa de mensagem.\n"
            "- Use `/resume` para retomar checkpoint da `session_id` configurada."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: Dict[str, Any]) -> None:
    normalized = _normalize_settings(settings)
    cl.user_session.set(SESSION_SETTINGS_KEY, normalized)
    if normalized.get("session_id"):
        cl.user_session.set(SESSION_ACTIVE_ID_KEY, normalized.get("session_id"))
    logger.info(
        "settings_update api_url=%s provider=%s model_profile=%s temperature=%s max_tokens=%s timeout=%ss session_id=%s",
        normalized.get("api_url"),
        normalized.get("provider"),
        normalized.get("model_profile"),
        normalized.get("temperature"),
        normalized.get("max_tokens"),
        normalized.get("timeout_seconds"),
        normalized.get("session_id"),
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    request_id = str(uuid.uuid4())[:8]
    started_at = time.perf_counter()
    settings = _normalize_settings(cl.user_session.get(SESSION_SETTINGS_KEY))
    active_session_id = cl.user_session.get(SESSION_ACTIVE_ID_KEY)
    session_id = settings.get("session_id") or active_session_id

    raw_content = (message.content or "").strip()
    resume = raw_content.lower() in {"/resume", "resume"}
    problem = "" if resume else raw_content

    image_bytes, image_filename, image_mime = _extract_image_attachment(message)
    logger.info(
        "request_received request_id=%s chars=%s resume=%s has_image=%s image_name=%s provider=%s model_profile=%s api_url=%s",
        request_id,
        len(problem),
        resume,
        bool(image_bytes),
        image_filename or "-",
        settings.get("provider"),
        settings.get("model_profile"),
        settings.get("api_url"),
    )

    try:
        payload = build_solve_payload(
            problem=problem,
            session_id=session_id,
            resume=resume,
            image_bytes=image_bytes,
            image_filename=image_filename,
            provider=settings.get("provider"),
            model_profile=settings.get("model_profile"),
            temperature=settings.get("temperature"),
            max_tokens=settings.get("max_tokens"),
        )
        if image_mime:
            payload["image_media_type"] = image_mime
        logger.info(
            "payload_ready request_id=%s keys=%s session_id=%s",
            request_id,
            sorted(payload.keys()),
            payload.get("session_id"),
        )
    except ValueError as exc:
        logger.warning("payload_invalid request_id=%s error=%s", request_id, exc)
        await cl.Message(content=str(exc)).send()
        return

    progress = cl.Message(content="Enviando requisicao para API...")
    await progress.send()

    try:
        async with cl.Step(name="api_call", type="tool") as step:
            step.output = "POST {}/v1/solve (request_id={})".format(settings["api_url"].rstrip("/"), request_id)
            response = await call_solve_api_async(
                base_url=settings["api_url"],
                payload=payload,
                timeout_seconds=settings["timeout_seconds"],
                request_id=request_id,
            )
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            step.output = "Sucesso em {:.1f} ms (request_id={})".format(elapsed_ms, request_id)
            logger.info(
                "request_success request_id=%s elapsed_ms=%.1f status=%s response_keys=%s",
                request_id,
                elapsed_ms,
                response.get("status"),
                sorted(response.keys()) if isinstance(response, dict) else [],
            )
    except Exception as exc:  # pragma: no cover - runtime path
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("request_failed request_id=%s elapsed_ms=%.1f error=%s", request_id, elapsed_ms, exc)
        progress.content = "Falha ao consultar API (request_id={}).".format(request_id)
        await progress.update()
        await cl.Message(content=_friendly_error_message(str(exc))).send()
        return

    progress.content = "Resposta recebida da API (request_id={}).".format(request_id)
    await progress.update()

    resolved_session = response.get("session_id") or session_id
    cl.user_session.set(SESSION_ACTIVE_ID_KEY, resolved_session)

    history = list(cl.user_session.get(SESSION_HISTORY_KEY) or [])
    history.append({"request": payload, "response": response})
    cl.user_session.set(SESSION_HISTORY_KEY, history)

    trace = response.get("decision_trace", [])
    if isinstance(trace, list):
        for index, event in enumerate(trace, start=1):
            if not isinstance(event, dict):
                continue
            node = str(event.get("node", "step")).strip() or "step"
            async with cl.Step(name="{} - {}".format(index, node), type="tool") as step:
                step.output = _format_trace_content(event)

    await cl.Message(content=_format_final_response(response)).send()

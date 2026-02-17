"""Chainlit frontend for MathSolverAgent API."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import chainlit as cl
from chainlit.input_widget import Select, Slider, TextInput

try:
    from src.ui.api_client import (
        ImageAttachment,
        build_solve_payload,
        call_runtime_status_api_async,
        call_solve_job_cancel_api_async,
        call_solve_job_status_api_async,
        call_solve_job_submit_api_async,
        call_ocr_extract_api_async,
        call_export_api_async,
        call_sessions_api_async,
        call_solve_api_async,
        infer_image_media_type,
        stream_solve_api_async,
    )
    from src.utils import load_graph_config
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.ui.api_client import (
        ImageAttachment,
        build_solve_payload,
        call_runtime_status_api_async,
        call_solve_job_cancel_api_async,
        call_solve_job_status_api_async,
        call_solve_job_submit_api_async,
        call_ocr_extract_api_async,
        call_export_api_async,
        call_sessions_api_async,
        call_solve_api_async,
        infer_image_media_type,
        stream_solve_api_async,
    )
    from src.utils import load_graph_config


SESSION_SETTINGS_KEY = "settings"
SESSION_HISTORY_KEY = "history"
SESSION_ACTIVE_ID_KEY = "active_session_id"
SESSION_PENDING_OCR_KEY = "pending_ocr"
SESSION_LAST_RESPONSE_KEY = "last_response"

ACTION_CONFIRM_OCR = "confirm_ocr"
ACTION_CANCEL_OCR = "cancel_ocr"
ACTION_RESUME_SESSION = "resume_session"
ACTION_MORE_DETAILS = "more_details"
ACTION_NEXT_STEP = "next_step"
ACTION_GENERATE_PLOT = "generate_plot"

NODE_LABELS = {
    "analyzer": "A interpretar enunciado...",
    "converter": "A planear calculos...",
    "solver": "A executar calculos...",
    "verifier": "A verificar resultado...",
}

STATUS_LABELS = {
    "running": "em execucao",
    "done": "concluido",
    "failed": "falhou",
    "pending": "pendente",
}

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
    ocr_cfg = dict(getattr(cfg, "ocr", {}) or {})
    profile_map = llm_cfg.get("model_profiles", {}) if isinstance(llm_cfg.get("model_profiles"), dict) else {}
    profile_limits: Dict[str, int] = {}
    profile_multimodal: Dict[str, bool] = {}
    provider_profiles: Dict[str, list[str]] = {}
    for alias, payload in profile_map.items():
        if not isinstance(payload, dict):
            continue
        provider = str(payload.get("provider", llm_cfg.get("provider", "nvidia"))).strip().lower() or "nvidia"
        provider_profiles.setdefault(provider, []).append(str(alias))
        profile_multimodal[str(alias)] = bool(payload.get("multimodal", llm_cfg.get("multimodal_enabled", True)))
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
        "profile_multimodal": profile_multimodal,
        "all_model_profiles": sorted({alias for aliases in provider_profiles.values() for alias in aliases}),
        "default_provider": configured_provider,
        "default_model_profile": default_profile,
        "temperature": float(llm_cfg.get("temperature", 0.2)),
        "max_tokens": int(llm_cfg.get("max_tokens", 8192)),
        "api_url": os.getenv("MATH_SOLVER_API_URL", "http://localhost:8001"),
        "timeout_seconds": int(os.getenv("MATH_SOLVER_API_TIMEOUT", "600")),
        "default_ocr_mode": str(ocr_cfg.get("default_mode", "auto")).strip().lower() or "auto",
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


def _profile_supports_multimodal(model_profile: Optional[str]) -> bool:
    mapping = UI_DEFAULTS.get("profile_multimodal", {})
    profile = str(model_profile or "").strip()
    if isinstance(mapping, dict) and profile in mapping:
        return bool(mapping.get(profile))
    return True


def _normalize_ocr_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower() or "auto"
    if mode not in {"auto", "on", "off"}:
        return "auto"
    return mode


def _chat_settings_widgets(settings: Optional[Dict[str, Any]] = None) -> list[Any]:
    current = settings or {}
    providers = UI_DEFAULTS["providers"] or ["nvidia", "maritaca"]
    selected_provider = str(current.get("provider", UI_DEFAULTS["default_provider"])).strip().lower() or UI_DEFAULTS["default_provider"]
    profiles = UI_DEFAULTS.get("all_model_profiles", []) or _profiles_for_provider(selected_provider)
    selected_profile = str(current.get("model_profile", UI_DEFAULTS["default_model_profile"] or "")).strip()
    if selected_profile not in profiles and profiles:
        selected_profile = profiles[0]
    ocr_modes = ["auto", "on", "off"]
    selected_ocr_mode = _normalize_ocr_mode(current.get("ocr_mode", UI_DEFAULTS.get("default_ocr_mode", "auto")))
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
            initial=max(60, min(1000, int(current.get("timeout_seconds", UI_DEFAULTS["timeout_seconds"]))),),
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
        Select(
            id="ocr_mode",
            label="OCR Mode",
            values=ocr_modes,
            initial_index=_initial_index(ocr_modes, selected_ocr_mode),
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
    ocr_mode = _normalize_ocr_mode(data.get("ocr_mode", UI_DEFAULTS.get("default_ocr_mode", "auto")))

    return {
        "api_url": api_url,
        "provider": provider,
        "model_profile": model_profile or None,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "session_id": session_id or None,
        "ocr_mode": ocr_mode,
    }


def _extract_image_attachments(message: cl.Message) -> list[ImageAttachment]:
    attachments: list[ImageAttachment] = []
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
            attachments.append(
                ImageAttachment(
                    image_bytes=handle.read(),
                    image_filename=(element_name or path.name),
                    image_media_type=(element_mime or None),
                )
            )

    return attachments


def _friendly_node_name(node: str) -> str:
    return NODE_LABELS.get(str(node).strip().lower(), "A executar etapa...")


def _friendly_status_name(status: str) -> str:
    return STATUS_LABELS.get(str(status).strip().lower(), str(status).strip() or "-" )


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


def _resolve_chainlit_thread_id() -> Optional[str]:
    try:
        context_var = getattr(getattr(cl, "context", None), "context_var", None)
        if context_var is None:
            return None
        current = context_var.get()
        session = getattr(current, "session", None)
        thread_id = getattr(session, "thread_id", None)
        if thread_id is not None:
            return str(thread_id)
    except Exception:  # pragma: no cover - runtime fallback
        return None
    return None


def _format_final_response(response: Dict[str, Any], include_explanation: bool = True) -> str:
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
        "{}".format(result or "_Sem resultado simbolico._"),
    ]

    if numeric_result is not None:
        blocks.extend(["", "#### Resultado numerico", "`{}`".format(numeric_result)])

    if isinstance(verification, dict) and verification:
        blocks.extend(["", "#### Verificacao", "```json", json.dumps(verification, ensure_ascii=False, indent=2), "```"])

    if include_explanation:
        blocks.extend(["", "#### Explicacao"])
        blocks.append(explanation or "_Sem explicacao retornada._")
    else:
        blocks.extend(["", "_Explicacao exibida em streaming acima._"])

    return "\n".join(blocks)


def _friendly_error_message(raw_error: str) -> str:
    text = raw_error.strip()
    lowered = text.lower()

    if "ocr is required for this request" in lowered or "provide `ocr_text`" in lowered:
        return (
            "Esta requisicao precisa de OCR antes do calculo. "
            "Use `OCR Mode = on`/`auto` e confirme o texto extraido, ou envie `ocr_text` explicitamente."
        )
    if "ocr mode is off but selected model does not support multimodal image input" in lowered:
        return (
            "O modelo atual nao processa imagem sem OCR e `OCR Mode` esta em `off`. "
            "Ative `OCR Mode` (`auto` ou `on`) ou envie o enunciado em texto."
        )
    if "rapidocr is not available" in lowered:
        return (
            "OCR dedicado indisponivel no servidor. "
            "Instale `rapidocr_onnxruntime` no ambiente da API ou troque para modelo multimodal com `OCR Mode=off`."
        )
    if "api error 503" in lowered:
        return "A API retornou 503: LLM indisponivel no modo estrito. Verifique API key e provider."
    if "api error 504" in lowered or "hard timeout" in lowered:
        return (
            "A requisicao excedeu o timeout do servidor. "
            "Tente novamente, reduza complexidade, ou selecione um modelo/perfil com menor fila."
        )
    if "provider_timeout" in lowered:
        return (
            "O provider excedeu o tempo limite de resposta. "
            "Tente novamente em alguns instantes ou selecione um profile/modelo menos congestionado."
        )
    if "api error 429" in lowered or "queue is full" in lowered or "queue wait timeout" in lowered:
        return (
            "Servidor ocupado no momento (fila/concurrency no limite). "
            "Aguarde alguns segundos e tente novamente; se possivel, use um modelo menos congestionado."
        )
    if "api error 422" in lowered:
        return "A API rejeitou a requisicao (422). Envie texto, imagem, ou use `/resume` com session_id valido."
    if "api error 405" in lowered:
        return "A URL configurada nao aponta para a API do MathSolverAgent. Confira `API Base URL` (ex.: http://localhost:8001)."
    if "stream da api" in lowered:
        return "Nao foi possivel conectar no streaming da API. Confira URL/porta e se o endpoint WebSocket esta ativo."
    if "nao foi possivel conectar na api" in lowered:
        return "Nao foi possivel conectar na API. Confira a URL da API e se o servidor esta em execucao."

    if "sympy is required for symbolic" in lowered or "sympy is required" in lowered:
        return (
            "A estrategia simbolica falhou por indisponibilidade do SymPy. "
            "Tente reformular para abordagem numerica (ex.: solicitar aproximacao decimal)."
        )
    if "could not parse" in lowered or "invalid syntax" in lowered:
        return (
            "Nao consegui interpretar a expressao matematica. "
            "Reescreva com operadores explicitos (ex.: `*`, `/`, `^`) e sem texto misturado na formula."
        )
    if "expression contains unsupported characters" in lowered:
        return (
            "A expressao tem caracteres nao suportados. "
            "Use apenas notacao matematica basica e LaTeX simples, evitando simbolos especiais fora de formula."
        )
    if "invalid tool args" in lowered:
        return (
            "A etapa de calculo recebeu parametros incompletos. "
            "Tente simplificar o pedido e informar claramente variavel, intervalo ou equacao alvo."
        )
    return text


def _parse_export_command(content: str) -> tuple[str, Optional[str]]:
    parts = [token.strip() for token in content.split() if token.strip()]
    if not parts:
        return "latex", None

    fmt = "latex"
    filename: Optional[str] = None
    if len(parts) >= 2 and parts[1].lower() in {"latex", "notebook", "json"}:
        fmt = parts[1].lower()
        if len(parts) >= 3:
            filename = parts[2]
    elif len(parts) >= 2:
        filename = parts[1]
    return fmt, filename


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", name.strip())
    return cleaned or "result"


def _extract_plot_expression(response: Dict[str, Any]) -> Optional[str]:
    tool_call = response.get("tool_call") if isinstance(response.get("tool_call"), dict) else {}
    args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
    expression = str(args.get("expression", "")).strip()
    if expression:
        return expression

    result = str(response.get("result", "")).strip()
    if result and re.match(r"^[a-zA-Z0-9_+\-*/^().\s]+$", result):
        return result
    return None


def _build_final_actions(response: Dict[str, Any]) -> list[Any]:
    actions = [
        cl.Action(name=ACTION_MORE_DETAILS, label="Mais Detalhes", payload={}),
        cl.Action(name=ACTION_NEXT_STEP, label="Passo Seguinte", payload={}),
    ]
    expression = _extract_plot_expression(response)
    if expression:
        actions.append(cl.Action(name=ACTION_GENERATE_PLOT, label="Gerar Grafico", payload={"expression": expression}))
    return actions


def _build_follow_up_prompt(base_prompt: str, last_response: Optional[Dict[str, Any]]) -> str:
    prompt = str(base_prompt or "").strip()
    if not isinstance(last_response, dict):
        return prompt

    normalized_problem = str(last_response.get("normalized_problem", "")).strip()
    result = str(last_response.get("result", "")).strip()
    numeric = last_response.get("numeric_result")
    explanation = str(last_response.get("explanation", "")).strip()
    tool_call = last_response.get("tool_call") if isinstance(last_response.get("tool_call"), dict) else {}
    tool_name = str(tool_call.get("name", "")).strip()

    blocks = [
        "Use obrigatoriamente o contexto da resposta anterior abaixo.",
        "Problema anterior: {}".format(normalized_problem or "-"),
        "Resultado simbolico anterior: {}".format(result or "-"),
        "Resultado numerico anterior: {}".format(numeric if numeric is not None else "-"),
    ]
    if tool_name:
        blocks.append("Ferramenta usada anteriormente: {}".format(tool_name))
    if explanation:
        blocks.append("Explicacao anterior (resumo): {}".format(explanation[:1200]))
    blocks.append("Pedido atual: {}".format(prompt))
    return "\n\n".join(blocks)


def _build_session_actions(sessions: list[Dict[str, Any]]) -> list[Any]:
    actions: list[Any] = []
    for item in sessions[:8]:
        if not isinstance(item, dict):
            continue
        session_id = str(item.get("session_id", "")).strip()
        if not session_id:
            continue
        status = str(item.get("status", "-")).strip() or "-"
        label = "{} ({})".format(session_id[:10], status)
        actions.append(cl.Action(name=ACTION_RESUME_SESSION, label=label, payload={"session_id": session_id}))
    return actions


async def _send_recent_sessions(settings: Dict[str, Any], request_id: str) -> None:
    try:
        payload = await call_sessions_api_async(
            base_url=settings["api_url"],
            limit=8,
            timeout_seconds=min(120, settings["timeout_seconds"]),
            request_id=request_id,
        )
    except Exception as exc:  # pragma: no cover - runtime path
        logger.warning("sessions_list_failed request_id=%s error=%s", request_id, exc)
        return

    sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
    if not isinstance(sessions, list) or not sessions:
        return

    lines = ["### Sessoes recentes"]
    for item in sessions[:8]:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- `{}` | status=`{}` | dominio=`{}`".format(
                item.get("session_id", "-"),
                item.get("status", "-"),
                item.get("domain", "-"),
            )
        )

    await cl.Message(content="\n".join(lines), actions=_build_session_actions(sessions)).send()


async def _auto_resume_from_thread(settings: Dict[str, Any], request_id: str) -> Optional[Dict[str, Any]]:
    """Attempts checkpoint resume based on Chainlit thread id."""
    thread_id = _resolve_chainlit_thread_id()
    if not thread_id:
        return None

    payload = build_solve_payload(
        problem="",
        session_id=thread_id,
        resume=True,
        provider=settings.get("provider"),
        model_profile=settings.get("model_profile"),
        temperature=settings.get("temperature"),
        max_tokens=settings.get("max_tokens"),
    )
    response: Optional[Dict[str, Any]] = None
    async for event in stream_solve_api_async(
        base_url=settings["api_url"],
        payload=payload,
        timeout_seconds=settings["timeout_seconds"],
        request_id=request_id,
    ):
        if event.get("type") == "result" and isinstance(event.get("data"), dict):
            response = event["data"]
        if event.get("type") == "error":
            raise RuntimeError(str(event.get("message", "Falha no stream de retomada.")))
    return response


async def _render_artifacts(response: Dict[str, Any]) -> None:
    artifacts = response.get("artifacts", [])
    if not isinstance(artifacts, list):
        return

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if str(artifact.get("type", "")).strip().lower() != "image":
            continue
        path_str = str(artifact.get("path", "")).strip()
        if not path_str:
            continue

        path = Path(path_str)
        if not path.exists() and not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            continue

        label = str(artifact.get("label", "Grafico gerado")).strip() or "Grafico gerado"
        image = cl.Image(name=path.name, path=str(path), display="inline")
        await cl.Message(content="{}: `{}`".format(label, path.name), elements=[image]).send()


async def _execute_solve_payload(settings: Dict[str, Any], payload: Dict[str, Any], request_id: str) -> Optional[Dict[str, Any]]:
    started_at = time.perf_counter()
    progress = cl.Message(content="Consultando estado operacional da API...")
    await progress.send()

    try:
        runtime = await call_runtime_status_api_async(
            base_url=settings["api_url"],
            timeout_seconds=min(60, settings["timeout_seconds"]),
            request_id=request_id,
        )
        async with cl.Step(name="runtime_status", type="tool") as step:
            step.output = (
                "busy=`{}` in_flight=`{}` queue=`{}/{}` retry_after=`{}s`".format(
                    runtime.get("busy", False),
                    runtime.get("in_flight_total", 0),
                    runtime.get("queue_depth", 0),
                    runtime.get("queue_capacity", 0),
                    runtime.get("suggested_retry_after_seconds", 1),
                )
            )
    except Exception as exc:  # pragma: no cover - runtime path
        logger.warning("runtime_status_failed request_id=%s error=%s", request_id, exc)

    try:
        submit = await call_solve_job_submit_api_async(
            base_url=settings["api_url"],
            payload=payload,
            timeout_seconds=settings["timeout_seconds"],
            request_id=request_id,
        )
        job_id = str(submit.get("job_id", "")).strip()
        if not job_id:
            raise RuntimeError("API nao retornou job_id para processamento assíncrono.")

        queue_position = submit.get("queue_position")
        progress.content = "Job `{}` submetido. Fila atual: {}.".format(job_id[:10], queue_position if queue_position else "-")
        await progress.update()
        logger.info("job_submitted request_id=%s job_id=%s queue_position=%s", request_id, job_id, queue_position)

        response: Dict[str, Any] = {}
        last_status = ""
        while True:
            status_payload = await call_solve_job_status_api_async(
                base_url=settings["api_url"],
                job_id=job_id,
                timeout_seconds=settings["timeout_seconds"],
                request_id=request_id,
            )
            job_status = str(status_payload.get("status", "")).strip().lower()
            queue_position = status_payload.get("queue_position")
            if job_status != last_status or job_status in {"queued", "running"}:
                progress.content = "Job `{}`: status=`{}` queue_position=`{}`".format(
                    job_id[:10],
                    job_status or "-",
                    queue_position if queue_position is not None else "-",
                )
                await progress.update()
                last_status = job_status

            if job_status in {"queued", "running"}:
                await asyncio.sleep(1.0)
                continue

            if job_status == "succeeded":
                result = status_payload.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("Job concluido sem resultado estruturado.")
                response = result
                break

            error_message = str(status_payload.get("error", "")).strip() or "Falha ao processar job."
            raise RuntimeError(error_message)
    except Exception as exc:  # pragma: no cover - runtime path
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.error("job_flow_failed request_id=%s elapsed_ms=%.1f error=%s", request_id, elapsed_ms, exc)
        progress.content = "Falha no processamento assíncrono da API (request_id={}).".format(request_id)
        await progress.update()
        await cl.Message(content=_friendly_error_message(str(exc))).send()
        return None

    if not response:
        progress.content = "A API nao retornou resultado final (request_id={}).".format(request_id)
        await progress.update()
        await cl.Message(content="Nao foi possivel concluir a resposta da API.").send()
        return None

    progress.content = "Resposta recebida da API (request_id={}).".format(request_id)
    await progress.update()

    resolved_session = response.get("session_id") or payload.get("session_id")
    cl.user_session.set(SESSION_ACTIVE_ID_KEY, resolved_session)
    cl.user_session.set(SESSION_LAST_RESPONSE_KEY, response)

    if str(response.get("status", "")).strip().lower() == "resume_not_found":
        await cl.Message(
            content=(
                "Nao existe checkpoint para esta sessao.\n\n"
                "Defina um `session_id` valido ou envie um novo enunciado para iniciar uma sessao."
            )
        ).send()
        return response

    history = list(cl.user_session.get(SESSION_HISTORY_KEY) or [])
    history.append({"request": payload, "response": response})
    cl.user_session.set(SESSION_HISTORY_KEY, history)

    trace = response.get("decision_trace", [])
    if isinstance(trace, list):
        for index, event in enumerate(trace, start=1):
            if not isinstance(event, dict):
                continue
            node = str(event.get("node", "step")).strip() or "step"
            async with cl.Step(name="{} - {}".format(index, _friendly_node_name(node)), type="tool") as trace_step:
                trace_step.output = _format_trace_content(event)

    await _render_artifacts(response)
    await cl.Message(
        content=_format_final_response(response, include_explanation=True),
        actions=_build_final_actions(response),
    ).send()
    return response


async def _run_follow_up_request(prompt: str) -> None:
    settings = _normalize_settings(cl.user_session.get(SESSION_SETTINGS_KEY))
    thread_id = _resolve_chainlit_thread_id()
    active_session_id = cl.user_session.get(SESSION_ACTIVE_ID_KEY)
    session_id = settings.get("session_id") or active_session_id or thread_id
    request_id = str(uuid.uuid4())[:8]
    last_response = cl.user_session.get(SESSION_LAST_RESPONSE_KEY)
    follow_up_prompt = _build_follow_up_prompt(prompt, last_response if isinstance(last_response, dict) else None)
    use_resume = bool(session_id)

    payload = build_solve_payload(
        problem=follow_up_prompt,
        session_id=session_id,
        resume=use_resume,
        ocr_mode=settings.get("ocr_mode", "auto"),
        provider=settings.get("provider"),
        model_profile=settings.get("model_profile"),
        temperature=settings.get("temperature"),
        max_tokens=settings.get("max_tokens"),
    )
    await _execute_solve_payload(settings=settings, payload=payload, request_id=request_id)


async def _resume_session(session_id: str) -> None:
    settings = _normalize_settings(cl.user_session.get(SESSION_SETTINGS_KEY))
    request_id = str(uuid.uuid4())[:8]
    payload = build_solve_payload(
        problem="",
        session_id=session_id,
        resume=True,
        ocr_mode=settings.get("ocr_mode", "auto"),
        provider=settings.get("provider"),
        model_profile=settings.get("model_profile"),
        temperature=settings.get("temperature"),
        max_tokens=settings.get("max_tokens"),
    )
    await _execute_solve_payload(settings=settings, payload=payload, request_id=request_id)


def _attachments_to_ocr_images(attachments: list[ImageAttachment]) -> list[Dict[str, Any]]:
    images: list[Dict[str, Any]] = []
    for item in attachments:
        if not isinstance(item, ImageAttachment):
            continue
        if not item.image_bytes:
            continue
        media_type = str(item.image_media_type or "").strip() or infer_image_media_type(item.image_filename or "")
        images.append(
            {
                "image_base64": base64.b64encode(item.image_bytes).decode("ascii"),
                "image_media_type": media_type,
            }
        )
    return images


def _should_require_ocr(ocr_mode: str, supports_multimodal: bool, has_images: bool) -> bool:
    if not has_images:
        return False
    mode = _normalize_ocr_mode(ocr_mode)
    if mode == "on":
        return True
    if mode == "off":
        return False
    return not supports_multimodal


async def _start_ocr_confirmation(
    settings: Dict[str, Any],
    request_id: str,
    solve_payload: Dict[str, Any],
    attachments: list[ImageAttachment],
) -> bool:
    ocr_images = _attachments_to_ocr_images(attachments)
    if not ocr_images:
        await cl.Message(content="Nao foi possivel preparar imagens para OCR.").send()
        return False
    ocr_payload = {
        "images": ocr_images,
    }

    try:
        ocr_response = await call_ocr_extract_api_async(
            base_url=settings["api_url"],
            payload=ocr_payload,
            timeout_seconds=settings["timeout_seconds"],
            request_id=request_id,
        )
    except Exception as exc:  # pragma: no cover - runtime path
        await cl.Message(content=_friendly_error_message(str(exc))).send()
        return False

    extracted_text = str(ocr_response.get("text", "")).strip()
    if not extracted_text:
        await cl.Message(
            content=(
                "Nao foi possivel extrair texto da imagem com confianca suficiente. "
                "Ajuste a imagem ou mude `OCR Mode` para `off` ao usar modelo multimodal."
            )
        ).send()
        return False

    final_payload = dict(solve_payload)
    final_payload["ocr_text"] = extracted_text

    cl.user_session.set(
        SESSION_PENDING_OCR_KEY,
        {
            "payload": final_payload,
            "ocr_text": extracted_text,
            "attachments_count": len(attachments),
        },
    )

    await cl.Message(
        content=(
            "### Validacao de OCR\n"
            "Texto extraido para calculo:\n\n"
            "```text\n{}\n```\n\n"
            "Confirme para continuar os calculos com este enunciado."
        ).format(extracted_text or "(vazio)"),
        actions=[
            cl.Action(name=ACTION_CONFIRM_OCR, label="Confirmar OCR", payload={}),
            cl.Action(name=ACTION_CANCEL_OCR, label="Cancelar OCR", payload={}),
        ],
    ).send()
    return True


@cl.action_callback(ACTION_CONFIRM_OCR)
async def on_confirm_ocr(action: Any) -> None:  # pragma: no cover - runtime path
    del action
    pending = cl.user_session.get(SESSION_PENDING_OCR_KEY)
    if not isinstance(pending, dict):
        await cl.Message(content="Nao ha validacao de OCR pendente.").send()
        return

    payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else None
    cl.user_session.set(SESSION_PENDING_OCR_KEY, None)
    if not payload:
        await cl.Message(content="Nao foi possivel recuperar os dados pendentes do OCR.").send()
        return

    settings = _normalize_settings(cl.user_session.get(SESSION_SETTINGS_KEY))
    request_id = str(uuid.uuid4())[:8]
    await cl.Message(content="OCR confirmado. A iniciar calculos...").send()
    await _execute_solve_payload(settings=settings, payload=payload, request_id=request_id)


@cl.action_callback(ACTION_CANCEL_OCR)
async def on_cancel_ocr(action: Any) -> None:  # pragma: no cover - runtime path
    del action
    cl.user_session.set(SESSION_PENDING_OCR_KEY, None)
    await cl.Message(content="OCR cancelado. Reenvie o enunciado ajustado para continuar.").send()


@cl.action_callback(ACTION_RESUME_SESSION)
async def on_resume_session_action(action: Any) -> None:  # pragma: no cover - runtime path
    payload = getattr(action, "payload", {}) if action is not None else {}
    session_id = str((payload or {}).get("session_id", "")).strip()
    if not session_id:
        await cl.Message(content="Session ID invalido para retomada.").send()
        return
    cl.user_session.set(SESSION_ACTIVE_ID_KEY, session_id)
    await cl.Message(content="Retomando sessao `{}`...".format(session_id)).send()
    await _resume_session(session_id)


@cl.action_callback(ACTION_MORE_DETAILS)
async def on_more_details_action(action: Any) -> None:  # pragma: no cover - runtime path
    del action
    await _run_follow_up_request("Forneca mais detalhes pedagogicos da ultima solucao, com exemplos intermediarios.")


@cl.action_callback(ACTION_NEXT_STEP)
async def on_next_step_action(action: Any) -> None:  # pragma: no cover - runtime path
    del action
    await _run_follow_up_request("Qual e o proximo passo natural a partir da ultima resposta?")


@cl.action_callback(ACTION_GENERATE_PLOT)
async def on_generate_plot_action(action: Any) -> None:  # pragma: no cover - runtime path
    payload = getattr(action, "payload", {}) if action is not None else {}
    expression = str((payload or {}).get("expression", "")).strip()
    if not expression:
        await cl.Message(content="Nao foi possivel identificar uma expressao nao ambigua para gerar grafico.").send()
        return
    await _run_follow_up_request(
        (
            "Gere um grafico 2D para a expressao solicitada.\n"
            "[PLOT_REQUEST]\n"
            "expression: {}\n"
            "[/PLOT_REQUEST]"
        ).format(expression)
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    request_id = str(uuid.uuid4())[:8]
    settings = await cl.ChatSettings(_chat_settings_widgets()).send()
    normalized = _normalize_settings(settings)
    cl.user_session.set(SESSION_SETTINGS_KEY, normalized)
    cl.user_session.set(SESSION_HISTORY_KEY, [])
    cl.user_session.set(SESSION_PENDING_OCR_KEY, None)
    thread_id = _resolve_chainlit_thread_id()
    cl.user_session.set(SESSION_ACTIVE_ID_KEY, thread_id or normalized.get("session_id"))
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
            "- Anexe uma ou mais imagens pelo clipe na caixa de mensagem.\n"
            "- Use `/resume` para retomar checkpoint da `session_id` configurada.\n"
            "- Use `/export [latex|notebook|json] [nome_opcional]` para exportar."
        )
    ).send()

    await _send_recent_sessions(normalized, request_id=request_id)

    if thread_id:
        try:
            resumed_response = await _auto_resume_from_thread(normalized, request_id=request_id)
        except Exception as exc:  # pragma: no cover - runtime path
            logger.warning("auto_resume_failed request_id=%s error=%s", request_id, exc)
            return

        if not isinstance(resumed_response, dict):
            return
        status = str(resumed_response.get("status", ""))
        cl.user_session.set(SESSION_ACTIVE_ID_KEY, resumed_response.get("session_id") or thread_id)
        if status == "resume_not_found":
            logger.info("auto_resume_not_found thread_id=%s", thread_id)
            return
        await cl.Message(content="Contexto retomado automaticamente da sessao anterior desta conversa.").send()


@cl.on_settings_update
async def on_settings_update(settings: Dict[str, Any]) -> None:
    normalized = _normalize_settings(settings)
    cl.user_session.set(SESSION_SETTINGS_KEY, normalized)
    if normalized.get("session_id"):
        cl.user_session.set(SESSION_ACTIVE_ID_KEY, normalized.get("session_id"))
    logger.info(
        "settings_update api_url=%s provider=%s model_profile=%s temperature=%s max_tokens=%s timeout=%ss session_id=%s ocr_mode=%s",
        normalized.get("api_url"),
        normalized.get("provider"),
        normalized.get("model_profile"),
        normalized.get("temperature"),
        normalized.get("max_tokens"),
        normalized.get("timeout_seconds"),
        normalized.get("session_id"),
        normalized.get("ocr_mode"),
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    request_id = str(uuid.uuid4())[:8]
    settings = _normalize_settings(cl.user_session.get(SESSION_SETTINGS_KEY))
    thread_id = _resolve_chainlit_thread_id()
    active_session_id = cl.user_session.get(SESSION_ACTIVE_ID_KEY)
    session_id = settings.get("session_id") or active_session_id or thread_id

    raw_content = (message.content or "").strip()
    lowered = raw_content.lower()

    if lowered.startswith("/export"):
        fmt, filename = _parse_export_command(raw_content)
        target_session = str(session_id or "").strip()
        if not target_session:
            await cl.Message(content="Defina ou retome uma sessao antes de exportar.").send()
            return

        extension = {"latex": "tex", "notebook": "ipynb", "json": "json"}.get(fmt, "txt")
        base_name = _sanitize_filename(filename or "{}_result".format(target_session))
        output_path = "artifacts/{}.{}".format(base_name, extension)
        export_payload = {
            "problem": "",
            "session_id": target_session,
            "resume": True,
            "ocr_mode": settings.get("ocr_mode", "auto"),
            "format": fmt,
            "output_path": output_path,
            "provider": settings.get("provider"),
            "model_profile": settings.get("model_profile"),
            "temperature": settings.get("temperature"),
            "max_tokens": settings.get("max_tokens"),
        }
        try:
            response = await call_export_api_async(
                base_url=settings["api_url"],
                payload=export_payload,
                timeout_seconds=settings["timeout_seconds"],
                request_id=request_id,
            )
        except Exception as exc:  # pragma: no cover - runtime path
            await cl.Message(content=_friendly_error_message(str(exc))).send()
            return

        file_path = str(response.get("file_path", "")).strip()
        if not file_path:
            await cl.Message(content="Exportacao concluida sem caminho de ficheiro retornado.").send()
            return

        path = Path(file_path)
        if not path.exists() and not path.is_absolute():
            path = Path.cwd() / path

        if path.exists():
            await cl.Message(
                content="Exportacao concluida em `{}`.".format(path),
                elements=[cl.File(name=path.name, path=str(path), display="inline")],
            ).send()
        else:
            await cl.Message(content="Exportacao concluida. Caminho retornado: `{}`".format(file_path)).send()
        return

    resume = lowered in {"/resume", "resume"}
    problem = "" if resume else raw_content

    attachments = _extract_image_attachments(message)
    ocr_mode = _normalize_ocr_mode(settings.get("ocr_mode", "auto"))
    supports_multimodal = _profile_supports_multimodal(settings.get("model_profile"))
    requires_ocr = _should_require_ocr(
        ocr_mode=ocr_mode,
        supports_multimodal=supports_multimodal,
        has_images=bool(attachments),
    )
    logger.info(
        "request_received request_id=%s chars=%s resume=%s has_images=%s provider=%s model_profile=%s ocr_mode=%s requires_ocr=%s api_url=%s",
        request_id,
        len(problem),
        resume,
        len(attachments),
        settings.get("provider"),
        settings.get("model_profile"),
        ocr_mode,
        requires_ocr,
        settings.get("api_url"),
    )

    if (
        not resume
        and attachments
        and ocr_mode == "off"
        and not supports_multimodal
        and not str(problem or "").strip()
    ):
        await cl.Message(
            content=(
                "O modelo selecionado nao aceita imagem diretamente e `OCR Mode` esta em `off`.\n\n"
                "Forneca texto no enunciado ou altere `OCR Mode` para `auto`/`on`."
            )
        ).send()
        return

    try:
        payload = build_solve_payload(
            problem=problem,
            session_id=session_id,
            resume=resume,
            ocr_mode=ocr_mode,
            image_attachments=attachments,
            provider=settings.get("provider"),
            model_profile=settings.get("model_profile"),
            temperature=settings.get("temperature"),
            max_tokens=settings.get("max_tokens"),
        )
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

    if attachments and not resume and requires_ocr:
        started = await _start_ocr_confirmation(
            settings=settings,
            request_id=request_id,
            solve_payload=payload,
            attachments=attachments,
        )
        if started:
            return
        # OCR is mandatory for this request path; if extraction/confirmation
        # could not start, do not continue to solve to avoid predictable 422.
        return

    await _execute_solve_payload(settings=settings, payload=payload, request_id=request_id)

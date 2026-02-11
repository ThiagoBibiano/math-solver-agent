"""Streamlit frontend for MathSolverAgent API."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

try:
    from src.ui.api_client import build_solve_payload, call_solve_api
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.ui.api_client import build_solve_payload, call_solve_api


PAGE_TITLE = "MathSolverAgent UI"


def _render_intro() -> None:
    """Renders intro text and usage hints on the main page."""
    st.title(PAGE_TITLE)
    st.markdown(
        """
Interface para consumir o agente via FastAPI.

Voce pode enviar:
- somente texto;
- somente imagem;
- texto + imagem.

A caixa de texto aceita escrita livre e voce pode descrever o problema em **Markdown/LaTeX**.
        """.strip()
    )


def _build_request_summary(payload: Dict[str, Any]) -> str:
    """Builds a human-readable summary for the current input mode.

    Args:
        payload: Solve payload that will be sent to the API.

    Returns:
        Markdown string describing text/image mode.
    """
    has_text = bool(payload.get("problem"))
    has_image = bool(payload.get("image_base64"))

    if has_text and has_image:
        mode = "texto + imagem"
    elif has_image:
        mode = "somente imagem"
    else:
        mode = "somente texto"

    return "Modo enviado: **{}**".format(mode)


def _render_response(response: Dict[str, Any]) -> None:
    """Renders solve response details and explanation sections.

    Args:
        response: API response dictionary.
    """
    st.subheader("Resposta")
    st.markdown(
        "\n".join(
            [
                "- **Session ID:** `{}`".format(response.get("session_id", "-")),
                "- **Status:** `{}`".format(response.get("status", "-")),
                "- **Dominio:** `{}`".format(response.get("domain", "-")),
                "- **Estrategia:** `{}`".format(response.get("strategy", "-")),
                "- **LLM disponivel:** `{}`".format(response.get("llm", {}).get("available", False)),
                "- **Entrada visual:** `{}`".format(response.get("has_visual_input", False)),
            ]
        )
    )

    st.markdown("### Resultado principal")
    st.markdown("```text\n{}\n```".format(response.get("result", "")))

    st.markdown("### Explicacao (Markdown)")
    explanation = response.get("explanation", "")
    if explanation:
        st.markdown(explanation)
    else:
        st.markdown("_Sem explicacao retornada._")

    with st.expander("Resposta JSON completa"):
        st.code(json.dumps(response, ensure_ascii=False, indent=2), language="json")


def main() -> None:
    """Runs the Streamlit app lifecycle."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="∫", layout="wide")
    _render_intro()

    with st.sidebar:
        st.header("Configuracao")
        api_url = st.text_input("Base URL da API", value="http://localhost:8000")
        timeout_seconds = st.number_input("Timeout (s)", value=120, min_value=10, max_value=1000, step=10)
        session_id = st.text_input("Session ID (opcional)", value="")

    problem = st.text_area(
        "Enunciado (opcional se enviar imagem)",
        placeholder="Ex.: Calcule a derivada de x^3. Voce pode escrever em Markdown e incluir LaTeX como $x^2$.",
        height=180,
    )
    uploaded = st.file_uploader(
        "Imagem do problema (opcional)",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        st.image(uploaded, caption="Imagem carregada", use_container_width=True)

    solve_clicked = st.button("Resolver", type="primary", use_container_width=True)

    if solve_clicked:
        image_bytes = uploaded.getvalue() if uploaded is not None else None
        image_filename = uploaded.name if uploaded is not None else None

        try:
            payload = build_solve_payload(
                problem=problem,
                session_id=session_id or None,
                resume=False,
                image_bytes=image_bytes,
                image_filename=image_filename,
            )
            st.markdown(_build_request_summary(payload))

            with st.spinner("Consultando o agente..."):
                response = call_solve_api(api_url, payload, timeout_seconds=int(timeout_seconds))

            _render_response(response)
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()

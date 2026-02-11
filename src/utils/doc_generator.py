"""Generate human-readable documentation from YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.utils.config_loader import GraphConfig


def generate_config_docs(graph_config: GraphConfig, prompts_registry: Dict[str, Dict[str, str]], output_path: str) -> str:
    lines = [
        "# Configuração do MathSolverAgent",
        "",
        "## Runtime",
        "",
        "- max_iterations: {}".format(graph_config.runtime.max_iterations),
        "- divergence_patience: {}".format(graph_config.runtime.divergence_patience),
        "- checkpoint_dir: {}".format(graph_config.runtime.checkpoint_dir),
        "- default_timeout_seconds: {}".format(graph_config.runtime.default_timeout_seconds),
        "",
        "## LLM",
        "",
        "- enabled: {}".format(graph_config.llm.get("enabled", False)),
        "- require_available: {}".format(graph_config.llm.get("require_available", True)),
        "- provider: {}".format(graph_config.llm.get("provider", "n/a")),
        "- model: {}".format(graph_config.llm.get("model", "n/a")),
        "- multimodal_enabled: {}".format(graph_config.llm.get("multimodal_enabled", False)),
        "- max_image_bytes: {}".format(graph_config.llm.get("max_image_bytes", "n/a")),
        "",
        "## Prompts Registrados",
        "",
    ]

    for name, payload in sorted(prompts_registry.items()):
        lines.append("### {}".format(name))
        lines.append("")
        lines.append("- system: {} chars".format(len(payload.get("system", ""))))
        lines.append("- user: {} chars".format(len(payload.get("user", ""))))
        lines.append("- tool: {} chars".format(len(payload.get("tool", ""))))
        lines.append("")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return str(output)

"""Configuration loaders for YAML-based runtime settings and prompt registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


try:
    import yaml
except ImportError:  # pragma: no cover - dependency is declared in pyproject
    yaml = None  # type: ignore[assignment]


@dataclass
class RuntimeSettings:
    max_iterations: int = 4
    divergence_patience: int = 2
    checkpoint_dir: str = ".checkpoints"
    default_timeout_seconds: int = 20
    operation_timeouts: Dict[str, int] = field(default_factory=dict)


@dataclass
class GraphConfig:
    version: str = "1.0.0"
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    llm: Dict[str, Any] = field(default_factory=dict)
    precision: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    fallback: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)


class ConfigError(RuntimeError):
    """Raised when configuration files cannot be loaded or validated."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ConfigError("PyYAML is required. Install dependencies from pyproject.toml.")
    if not path.exists():
        raise ConfigError("Configuration file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigError("Configuration root must be a mapping in {}".format(path))
    return loaded


def load_graph_config(path: str = "configs/graph_config.yml") -> GraphConfig:
    data = _load_yaml(Path(path))
    runtime_data = data.get("runtime", {})

    runtime = RuntimeSettings(
        max_iterations=int(runtime_data.get("max_iterations", 4)),
        divergence_patience=int(runtime_data.get("divergence_patience", 2)),
        checkpoint_dir=str(runtime_data.get("checkpoint_dir", ".checkpoints")),
        default_timeout_seconds=int(runtime_data.get("default_timeout_seconds", 20)),
        operation_timeouts=dict(runtime_data.get("operation_timeouts", {})),
    )

    return GraphConfig(
        version=str(data.get("version", "1.0.0")),
        runtime=runtime,
        llm=dict(data.get("llm", {})),
        precision=dict(data.get("precision", {})),
        resources=dict(data.get("resources", {})),
        fallback=dict(data.get("fallback", {})),
        security=dict(data.get("security", {})),
    )


def _resolve_prompt(name: str, prompts: Dict[str, Any], seen: Optional[set] = None) -> Dict[str, str]:
    seen = seen or set()
    if name in seen:
        raise ConfigError("Cyclic prompt inheritance detected at '{}'".format(name))
    seen.add(name)

    registry = prompts.get("registry", {})
    node = registry.get(name)
    if not isinstance(node, dict):
        raise ConfigError("Prompt '{}' not found in registry".format(name))

    base: Dict[str, str] = {}
    parent = node.get("extends")
    if parent:
        base = _resolve_prompt(str(parent), prompts, seen)

    merged = dict(base)
    for key in ("system", "user", "tool"):
        if key in node:
            merged[key] = str(node[key])
    return merged


def load_prompts_registry(path: str = "configs/prompts.yml") -> Dict[str, Dict[str, str]]:
    data = _load_yaml(Path(path))
    registry = data.get("registry", {})
    if not isinstance(registry, dict):
        raise ConfigError("'registry' must be a mapping in prompts configuration")

    resolved: Dict[str, Dict[str, str]] = {}
    for name in registry:
        resolved[name] = _resolve_prompt(str(name), data)
    return resolved


def load_prompts_config(path: str = "configs/prompts.yml") -> Dict[str, Any]:
    return _load_yaml(Path(path))


def select_prompt_variant(prompts_config: Dict[str, Any], session_id: str) -> Optional[str]:
    experiments = prompts_config.get("experiments", {})
    ab = experiments.get("ab_testing", {})
    if not ab.get("enabled", False):
        return None

    variants = ab.get("variants", {})
    if not isinstance(variants, dict) or not variants:
        return None

    ordered = sorted(str(k) for k in variants.keys())
    index = abs(hash(session_id)) % len(ordered)
    return ordered[index]

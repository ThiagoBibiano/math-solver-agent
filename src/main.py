"""CLI/API entrypoint for MathSolverAgent."""

from __future__ import annotations

import argparse
import json
from typing import Optional

from src.agents import MathSolverAgent
from src.api import create_app
from src.utils import generate_config_docs, load_graph_config, load_prompts_registry
from src.utils.logger import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for app entrypoints.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="MathSolverAgent")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli")
    parser.add_argument("--problem", type=str, default="", help="Problem statement (required in cli mode)")
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume an existing checkpoint")
    parser.add_argument("--image-path", type=str, default=None, help="Local path of problem image for multimodal models")
    parser.add_argument("--image-url", type=str, default=None, help="Public image URL of the problem statement")
    parser.add_argument("--image-base64", type=str, default=None, help="Base64-encoded image payload")
    parser.add_argument("--image-media-type", type=str, default="image/png", help="Media type used with image-base64")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--generate-docs", action="store_true", help="Generate markdown docs from YAML configs")
    parser.add_argument("--docs-output", type=str, default="docs/CONFIGURATION.md")
    return parser


def run_cli(
    problem: str,
    session_id: Optional[str],
    resume: bool,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    image_media_type: str = "image/png",
) -> int:
    """Executes one solve request using CLI parameters.

    Args:
        problem: Problem statement text.
        session_id: Optional session identifier.
        resume: Whether to resume from checkpoint.
        image_path: Optional local image path.
        image_url: Optional image URL.
        image_base64: Optional inline image payload.
        image_media_type: MIME type for inline image payload.

    Returns:
        Process exit code.

    Raises:
        ValueError: If no valid input is provided.
    """
    if not problem and not resume and not any([image_path, image_url, image_base64]):
        raise ValueError("--problem or --image-* is required in cli mode unless --resume is used")

    agent = MathSolverAgent()
    state = agent.solve(
        problem=problem,
        session_id=session_id,
        resume=resume,
        image_path=image_path,
        image_url=image_url,
        image_base64=image_base64,
        image_media_type=image_media_type,
    )

    print(
        json.dumps(
            {
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
            },
            ensure_ascii=True,
            indent=2,
        )
    )

    return 0


def run_api(host: str, port: int) -> int:
    """Runs FastAPI server using Uvicorn.

    Args:
        host: Bind host.
        port: Bind port.

    Returns:
        Process exit code.

    Raises:
        RuntimeError: If Uvicorn dependency is unavailable.
    """
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is required for API mode") from exc

    app = create_app()
    uvicorn.run(app, host=host, port=port)
    return 0


def main() -> int:
    """Application entrypoint for CLI and API modes.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.generate_docs:
        graph = load_graph_config()
        prompts = load_prompts_registry()
        output = generate_config_docs(graph, prompts, args.docs_output)
        print("Configuration docs generated at {}".format(output))
        return 0

    if args.mode == "api":
        return run_api(args.host, args.port)
    return run_cli(
        problem=args.problem,
        session_id=args.session_id,
        resume=args.resume,
        image_path=args.image_path,
        image_url=args.image_url,
        image_base64=args.image_base64,
        image_media_type=args.image_media_type,
    )


if __name__ == "__main__":
    raise SystemExit(main())

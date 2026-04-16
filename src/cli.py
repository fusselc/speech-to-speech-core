"""
cli.py — Typer-based command-line interface.
"""

import os
import sys
from typing import Annotated

import typer
from loguru import logger

# ---------------------------------------------------------------------------
# Path setup for direct module execution and package entrypoint compatibility.
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logging_config import configure_logging

cli = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Speech-to-speech pipeline CLI.",
)


@cli.callback()
def _root() -> None:
    """Root CLI command group."""


def _run_app(loop_mode: bool, max_turns: int) -> None:
    """Import app lazily to avoid hardware imports during CLI bootstrap."""
    from app import run_app

    run_app(loop_mode=loop_mode, max_turns=max_turns)


@cli.command()
def run(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Whisper model size (e.g. base, small)."),
    ] = config.WHISPER_MODEL,
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="Device: auto, cpu, or cuda."),
    ] = config.WHISPER_DEVICE,
    streaming: Annotated[
        bool,
        typer.Option(
            "--streaming/--no-streaming", help="Enable streaming capture mode."
        ),
    ] = config.USE_STREAMING,
    loop: Annotated[
        bool,
        typer.Option("--loop/--no-loop", help="Repeat turns until interrupted."),
    ] = config.LOOP_MODE,
    vad_sensitivity: Annotated[
        float,
        typer.Option(
            "--vad-sensitivity",
            min=0.0,
            max=1.0,
            help="Silero speech threshold sensitivity.",
        ),
    ] = config.SILENCE_THRESHOLD,
    debug: Annotated[
        bool,
        typer.Option("--debug/--no-debug", help="Enable debug-level logs."),
    ] = False,
) -> None:
    """Run the speech-to-speech pipeline."""
    device_normalized = device.lower().strip()
    if device_normalized not in {"auto", "cpu", "cuda"}:
        raise typer.BadParameter("--device must be one of: auto, cpu, cuda")

    config.WHISPER_MODEL = model
    config.WHISPER_DEVICE = device_normalized
    config.USE_STREAMING = streaming
    config.LOOP_MODE = loop
    config.SILENCE_THRESHOLD = vad_sensitivity

    configure_logging(debug=debug)
    logger.info(
        "Starting pipeline with model='{}', device='{}', streaming={}, loop={}, vad_sensitivity={:.2f}",
        config.WHISPER_MODEL,
        config.WHISPER_DEVICE,
        config.USE_STREAMING,
        config.LOOP_MODE,
        config.SILENCE_THRESHOLD,
    )
    _run_app(loop_mode=config.LOOP_MODE, max_turns=config.MAX_TURNS)


def main() -> None:
    """Console script entrypoint."""
    cli()


if __name__ == "__main__":
    main()

"""
app.py — Orchestration entry point for the speech-to-speech pipeline.
"""

import os
import sys

from loguru import logger

# ---------------------------------------------------------------------------
# Path setup — allow `python src/app.py` from the project root.
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from audio_input import build_recording_filepath, record_audio, save_wav
from latency_logger import LatencyLogger, LatencyTracker
from logging_config import configure_logging
from responder import generate_response
from synthesize import speak_text
from transcribe import transcribe_file
from turn_controller import TurnController


def _safe_total_ms(latency: LatencyLogger) -> float:
    """Compute best-effort total from recorded stage timings."""
    return latency.total_stages_ms()


def run_pipeline(latency_tracker: LatencyTracker | None = None) -> None:
    """Execute one full speech-to-speech turn."""
    logger.info("=" * 50)
    logger.info("  Speech-to-Speech Core")
    logger.info("=" * 50)

    latency = LatencyLogger()
    try:
        samples = latency.measure("recording_ms", record_audio)
        wav_path = latency.measure(
            "save_ms", save_wav, samples, build_recording_filepath()
        )
        transcript = latency.measure("transcription_ms", transcribe_file, wav_path)
    except KeyboardInterrupt:
        raise
    except Exception:
        logger.exception("Pipeline stopped due to input/transcription error.")
        logger.error("Skipping turn safely.")
        return

    if not transcript.strip():
        logger.warning("Empty transcript detected. Skipping response and synthesis.")
        latency.print_summary()
        if latency_tracker is not None:
            latency_tracker.record_turn(_safe_total_ms(latency))
            latency_tracker.print_rolling_summary()
        logger.info("=" * 50)
        logger.info("  Done.")
        logger.info("=" * 50)
        return
    logger.info("Transcript: {}", transcript)

    try:
        response = latency.measure("response_ms", generate_response, transcript)
        latency.measure("synthesis_ms", speak_text, response)
    except KeyboardInterrupt:
        raise
    except Exception:
        logger.exception("Response/TTS stage failed.")
        logger.error("Could not complete synthesis for this turn.")
        return

    latency.print_summary()
    if latency_tracker is not None:
        total_ms = _safe_total_ms(latency)
        latency_tracker.record_turn(total_ms)
        latency_tracker.print_rolling_summary()

    logger.info("=" * 50)
    logger.info("  Done.")
    logger.info("=" * 50)


def run_app(loop_mode: bool | None = None, max_turns: int | None = None) -> None:
    """Run the pipeline, looping when enabled."""
    resolved_loop_mode = config.LOOP_MODE if loop_mode is None else loop_mode
    resolved_max_turns = config.MAX_TURNS if max_turns is None else max_turns
    controller = TurnController(
        loop_mode=resolved_loop_mode,
        max_turns=resolved_max_turns,
    )
    latency_tracker = LatencyTracker()
    if resolved_loop_mode:
        logger.info("Loop mode active. Press Ctrl+C to exit.")
    try:
        while controller.should_continue():
            if controller.turn_count > 0:
                logger.info("-" * 50)
            run_pipeline(latency_tracker=latency_tracker)
            controller.mark_turn_completed()
    except KeyboardInterrupt:
        logger.info("Exiting gracefully. Goodbye!")


if __name__ == "__main__":
    configure_logging()
    run_app()

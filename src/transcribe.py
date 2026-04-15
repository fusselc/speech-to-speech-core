"""
transcribe.py — Speech-to-text transcription using faster-whisper.

The model is loaded once and reused across calls to avoid repeated model
initialization overhead during a session.
"""

import time

import torch
from faster_whisper import WhisperModel
from loguru import logger

from config import LANGUAGE, WHISPER_MODEL

# ---------------------------------------------------------------------------
# Module-level model cache — loaded on first use.
# ---------------------------------------------------------------------------
_model: WhisperModel | None = None


def _resolve_device_and_compute_type() -> tuple[str, str]:
    """Pick device + compute type defaults based on CUDA availability."""
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "float32"


def _get_model() -> WhisperModel:
    """Load and cache the faster-whisper model (lazy initialization).

    Returns:
        The loaded Whisper model instance.
    """
    global _model
    if _model is None:
        device, compute_type = _resolve_device_and_compute_type()
        logger.info(
            "Loading faster-whisper model '{}' on {} ({})...",
            WHISPER_MODEL,
            device,
            compute_type,
        )
        _model = WhisperModel(
            WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
        )
        logger.info(
            "faster-whisper model '{}' ready on {}.",
            WHISPER_MODEL,
            device,
        )
    return _model


def transcribe_file(file_path: str) -> str:
    """Transcribe a WAV file and return the text.

    Args:
        file_path: Path to a WAV audio file.

    Returns:
        Transcribed text string (stripped of leading/trailing whitespace).
    """
    model = _get_model()
    start = time.perf_counter()
    segments, _ = model.transcribe(
        str(file_path),
        beam_size=5,
        language=LANGUAGE,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info("Transcription completed in {:.2f} ms.", elapsed_ms)
    logger.debug("Transcript: {!r}", text)
    return text

"""
transcribe.py — Speech-to-text transcription using faster-whisper.

The model is loaded once and reused across calls to avoid repeated model
initialization overhead during a session.
"""

import time

import torch
from faster_whisper import WhisperModel
from loguru import logger

import config

# ---------------------------------------------------------------------------
# Module-level model cache — loaded on first use.
# ---------------------------------------------------------------------------
_model: WhisperModel | None = None


def _resolve_device_and_compute_type() -> tuple[str, str]:
    """Pick device + compute type defaults based on CUDA availability."""
    preferred = str(config.WHISPER_DEVICE or "auto").lower()
    if preferred == "cuda":
        if torch.cuda.is_available():
            return "cuda", "float16"
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu", "float32"
    if preferred == "cpu":
        return "cpu", "float32"
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "float32"


def _build_model(device: str, compute_type: str) -> WhisperModel:
    """Create a faster-whisper model instance."""
    return WhisperModel(
        config.WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
    )


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
            config.WHISPER_MODEL,
            device,
            compute_type,
        )
        try:
            _model = _build_model(device=device, compute_type=compute_type)
            logger.info(
                "faster-whisper model '{}' ready on {}.",
                config.WHISPER_MODEL,
                device,
            )
        except Exception:
            if device == "cuda":
                logger.warning(
                    "Failed loading model on CUDA; falling back to CPU (float32)."
                )
                _model = _build_model(device="cpu", compute_type="float32")
                logger.info(
                    "faster-whisper model '{}' ready on cpu.",
                    config.WHISPER_MODEL,
                )
            else:
                logger.exception(
                    "Model loading failed. Check internet connection or model files."
                )
                raise
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
    try:
        segments, _ = model.transcribe(
            str(file_path),
            beam_size=5,
            language=config.LANGUAGE,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info("Transcription completed in {:.2f} ms.", elapsed_ms)
        logger.debug("Transcript: {!r}", text)
        return text
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.warning("CUDA out of memory during transcription; retrying on CPU.")
            global _model
            _model = _build_model(device="cpu", compute_type="float32")
            segments, _ = _model.transcribe(
                str(file_path),
                beam_size=5,
                language=config.LANGUAGE,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "CPU fallback transcription completed in {:.2f} ms.", elapsed_ms
            )
            logger.debug("Transcript: {!r}", text)
            return text
        logger.exception("Transcription failed unexpectedly.")
        raise
    except Exception:
        logger.exception(
            "Transcription failed. Verify the audio file, model download, and runtime."
        )
        raise

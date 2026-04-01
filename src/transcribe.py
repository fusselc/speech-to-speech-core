"""
transcribe.py — Speech-to-text transcription using faster-whisper.

The model is loaded once and reused across calls to avoid repeated
initialisation overhead during a session.

Future streaming integration point:
  - Swap ``transcribe_file`` for a function that accepts raw audio bytes/chunks
    and yields partial transcripts using a compatible streaming API.
"""

from typing import Mapping, cast

from faster_whisper import WhisperModel

from config import WHISPER_LANGUAGE, WHISPER_MODEL

# ---------------------------------------------------------------------------
# Module-level model cache — loaded on first use
# ---------------------------------------------------------------------------
_model: WhisperModel | None = None


def _resolve_model_name(model_name: str) -> str:
    """Map legacy Whisper names to faster-whisper model names."""
    if model_name == "large":
        return "large-v3"
    return model_name


def _get_model() -> WhisperModel:
    """Load and cache the faster-whisper model (lazy initialisation).

    Returns:
        The loaded Whisper model instance.
    """
    global _model
    if _model is None:
        print(f"[transcribe] Loading Whisper model '{WHISPER_MODEL}'…")
        _model = WhisperModel(_resolve_model_name(WHISPER_MODEL))
        print("[transcribe] Model loaded.")
    return _model


def transcribe_file(wav_path: str) -> str:
    """Transcribe a WAV file and return the text.

    Args:
        wav_path: Path to a WAV audio file.

    Returns:
        Transcribed text string (stripped of leading/trailing whitespace).
    """
    model = _get_model()
    options: dict = {}
    if WHISPER_LANGUAGE:
        options["language"] = WHISPER_LANGUAGE
    result = model.transcribe(wav_path, **options)
    if isinstance(result, tuple) and len(result) == 2:
        segments, _ = result
        text = "".join(segment.text for segment in segments).strip()
    else:
        # Compatibility fallback for dict-shaped mocked results and
        # potential backend swaps with the same external function contract.
        text = cast(Mapping[str, str], result)["text"].strip()
    print(f"[transcribe] Transcript: {text!r}")
    return text

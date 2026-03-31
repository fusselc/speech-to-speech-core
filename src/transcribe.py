"""
transcribe.py — Speech-to-text transcription using OpenAI Whisper.

Whisper is loaded once and reused across calls to avoid repeated model
initialisation overhead during a session.

Future streaming integration point:
  - Swap ``transcribe_file`` for a function that accepts raw audio bytes/chunks
    and yields partial transcripts using ``whisper.DecodingTask`` or a
    compatible streaming API.
"""

import whisper

from config import WHISPER_LANGUAGE, WHISPER_MODEL

# ---------------------------------------------------------------------------
# Module-level model cache — loaded on first use
# ---------------------------------------------------------------------------
_model: whisper.Whisper | None = None


def _get_model() -> whisper.Whisper:
    """Load and cache the Whisper model (lazy initialisation).

    Returns:
        The loaded Whisper model instance.
    """
    global _model
    if _model is None:
        print(f"[transcribe] Loading Whisper model '{WHISPER_MODEL}'…")
        _model = whisper.load_model(WHISPER_MODEL)
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
    text: str = result["text"].strip()
    print(f"[transcribe] Transcript: {text!r}")
    return text

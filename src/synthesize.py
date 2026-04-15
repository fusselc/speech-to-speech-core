"""
synthesize.py — Text-to-speech synthesis and audio playback.

Next recommended upgrade: Piper TTS or XTTS-v2 (both drop-in compatible with
the current ``speak_text`` / ``save_speech`` interface). OpenVoice example is
already prepared below.

Phase 1 uses pyttsx3 for fully offline, zero-latency TTS.
pyttsx3 drives the OS-native speech engine (espeak on Linux,
NSSpeechSynthesizer on macOS, SAPI5 on Windows) so no internet
connection or API key is needed.

OpenVoice integration point:
  - When you are ready to upgrade synthesis quality, replace the body of
    ``speak_text`` (and optionally ``save_speech``) with calls to the
    OpenVoice ToneColorConverter pipeline.
  - The function signatures intentionally stay the same so app.py does
    not need to change.

Future streaming integration point:
  - Yield audio chunks from ``speak_text`` instead of blocking, and pipe
    them directly to the playback device for lower perceived latency.
"""

import os

import pyttsx3
from loguru import logger

from config import OUTPUTS_DIR, TTS_RATE
from utils import ensure_dir, timestamped_filename

# ---------------------------------------------------------------------------
# Module-level engine cache — created on first use
# ---------------------------------------------------------------------------
_engine: pyttsx3.Engine | None = None


def _get_engine() -> pyttsx3.Engine:
    """Initialise and cache the pyttsx3 TTS engine (lazy initialisation).

    Returns:
        A configured pyttsx3 Engine instance.
    """
    global _engine
    if _engine is None:
        try:
            engine = pyttsx3.init()
            if engine is None:
                raise RuntimeError("pyttsx3 returned no engine instance.")
            engine.setProperty("rate", TTS_RATE)
            _engine = engine
            logger.info("TTS engine initialized (rate={}).", TTS_RATE)
        except Exception:
            logger.exception(
                "Text-to-speech engine failed to initialize. "
                "Check system TTS dependencies and audio permissions."
            )
            raise
    return _engine


def speak_text(text: str) -> None:
    """Speak *text* aloud using the local TTS engine.

    This call blocks until playback is complete.

    Args:
        text: The string to synthesise and play.

    # OpenVoice integration point — replace below with OpenVoice synthesis:
    #   from openvoice import se_extractor
    #   from openvoice.api import ToneColorConverter
    #   converter.convert(audio_src_path, src_se, tgt_se, output_path)
    #   then play output_path with sounddevice or pygame.
    """
    try:
        engine = _get_engine()
        logger.info("Speaking text.")
        engine.say(text)
        engine.runAndWait()
    except Exception:
        logger.exception(
            "TTS playback failed. Please check your audio device and TTS backend."
        )
        raise


def save_speech(text: str) -> str:
    """Synthesise *text* and save it to a WAV file in OUTPUTS_DIR.

    Useful for debugging or recording sessions without live playback.

    Args:
        text: The string to synthesise.

    Returns:
        Absolute path to the saved audio file.
    """
    try:
        engine = _get_engine()
        filename = timestamped_filename("response", "wav")
        filepath = os.path.join(ensure_dir(OUTPUTS_DIR), filename)
        engine.save_to_file(text, filepath)
        engine.runAndWait()
        logger.info("Saved response audio to: {}", filepath)
        return filepath
    except Exception:
        logger.exception("Failed to save TTS audio output.")
        raise

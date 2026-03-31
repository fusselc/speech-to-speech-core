"""
config.py — App-wide configuration for the speech-to-speech pipeline.

All tunable settings live here so each module stays clean and easy to swap out.
"""

import os

# ---------------------------------------------------------------------------
# Audio recording settings
# ---------------------------------------------------------------------------

# Microphone sample rate (Hz). 16 kHz is standard for speech models.
SAMPLE_RATE: int = 16_000

# Number of audio channels (1 = mono, which Whisper expects)
CHANNELS: int = 1

# How many seconds to record per utterance
RECORD_DURATION: float = 5.0

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

# Project root is two levels up from this file (src/config.py → repo root)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory where raw microphone recordings are saved
RECORDINGS_DIR: str = os.path.join(_ROOT, "recordings")

# Directory where synthesised audio responses are saved
OUTPUTS_DIR: str = os.path.join(_ROOT, "outputs")

# ---------------------------------------------------------------------------
# Whisper settings
# ---------------------------------------------------------------------------

# Whisper model size. Options: "tiny", "base", "small", "medium", "large"
# "base" is a good balance of speed and accuracy for local use.
WHISPER_MODEL: str = "base"

# Language hint (None = auto-detect, or e.g. "en" to force English)
WHISPER_LANGUAGE: str | None = None

# ---------------------------------------------------------------------------
# Text-to-speech settings
# ---------------------------------------------------------------------------

# Speech rate for pyttsx3 (words per minute).
# NOTE: This setting is ignored when OpenVoice replaces pyttsx3.
TTS_RATE: int = 180

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

# Maximum number of seconds to record per utterance (streaming capture may end
# earlier when voice activity detection sees trailing silence).
RECORD_DURATION: float = 5.0

# Streaming chunk duration in seconds. Smaller chunks improve VAD responsiveness
# but increase per-chunk overhead.
STREAM_CHUNK_SECONDS: float = 0.2

# Voice activity threshold on raw int16 amplitude.
VAD_AMPLITUDE_THRESHOLD: int = 500

# Stop recording once this many seconds of silence have followed detected speech.
VAD_SILENCE_SECONDS: float = 1.0

# Require this many voiced chunks before silence can end the recording.
VAD_MIN_VOICE_CHUNKS: int = 1

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

# ---------------------------------------------------------------------------
# Conversational loop settings
# ---------------------------------------------------------------------------

# When True, the pipeline repeats after each turn until Ctrl+C is pressed or
# MAX_TURNS is reached. When False, the pipeline runs once and exits.
LOOP_MODE: bool = True

# Maximum number of turns to run in loop mode. Set to 0 for unlimited turns.
MAX_TURNS: int = 0

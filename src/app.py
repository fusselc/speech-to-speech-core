"""
app.py — Entry point for the Phase 1 speech-to-speech pipeline.

Run with:
    python src/app.py

Pipeline steps:
  1. Record a short utterance from the microphone.
  2. Save it as a WAV file in recordings/.
  3. Transcribe the WAV with Whisper.
  4. Print the transcript.
  5. Generate a simple text response.
  6. Speak the response aloud via local TTS.

Each step maps to exactly one module so the pipeline is easy to extend
(e.g. add streaming, swap the TTS engine, plug in an LLM responder).
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup — allow `python src/app.py` from the project root
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from audio_input import record_to_file
from transcribe import transcribe_file
from responder import generate_response
from synthesize import speak_text


def run_pipeline() -> None:
    """Execute one full speech-to-speech turn."""
    print("=" * 50)
    print("  Speech-to-Speech Core  |  Phase 1")
    print("=" * 50)

    # Step 1 & 2 — Record microphone audio and save WAV
    wav_path = record_to_file()

    # Step 3 & 4 — Transcribe and print
    transcript = transcribe_file(wav_path)
    print(f"\nTranscript: {transcript}\n")

    # Step 5 — Generate response
    response = generate_response(transcript)

    # Step 6 — Speak response
    speak_text(response)

    print("=" * 50)
    print("  Done.")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()

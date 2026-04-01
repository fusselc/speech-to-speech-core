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

from audio_input import build_recording_filepath, record_audio, save_wav
from latency_logger import LatencyLogger
from transcribe import transcribe_file
from responder import generate_response
from synthesize import speak_text


def run_pipeline() -> None:
    """Execute one full speech-to-speech turn."""

    print("=" * 50)
    print("  Speech-to-Speech Core  |  Phase 1")
    print("=" * 50)

    latency = LatencyLogger()
    # Step 1 & 2 — Record microphone audio and save WAV
    samples = latency.measure("recording_ms", record_audio)
    wav_path = latency.measure("save_ms", save_wav, samples, build_recording_filepath())

    # Step 3 & 4 — Transcribe and print
    transcript = latency.measure("transcription_ms", transcribe_file, wav_path)
    print(f"\nTranscript: {transcript}\n")

    # Step 5 — Generate response
    response = latency.measure("response_ms", generate_response, transcript)

    # Step 6 — Speak response
    latency.measure("synthesis_ms", speak_text, response)
    latency.print_summary()

    print("=" * 50)
    print("  Done.")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()

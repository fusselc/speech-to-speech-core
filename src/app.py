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

Loop mode:
  Set LOOP_MODE = True in config.py to repeat the pipeline after each turn.
  Press Ctrl+C to exit gracefully. MAX_TURNS limits the number of turns
  (0 = unlimited).
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
from config import LOOP_MODE, MAX_TURNS
from latency_logger import LatencyLogger, LatencyTracker
from transcribe import transcribe_file
from responder import generate_response
from synthesize import speak_text
from turn_controller import TurnController


def run_pipeline(latency_tracker: LatencyTracker | None = None) -> None:
    """Execute one full speech-to-speech turn."""

    print("=" * 50)
    print("  Speech-to-Speech Core  |  Phase 1")
    print("=" * 50)

    latency = LatencyLogger()
    # Step 1 & 2 — Record microphone audio and save WAV
    try:
        samples = latency.measure("recording_ms", record_audio)
    except Exception as exc:  # pragma: no cover - exercised by tests with mocks
        print(f"[audio_input] Recording failed: {exc}")
        print("[app] Skipping turn safely.")
        return
    wav_path = latency.measure("save_ms", save_wav, samples, build_recording_filepath())

    # Step 3 & 4 — Transcribe and print
    transcript = latency.measure("transcription_ms", transcribe_file, wav_path)
    if not transcript.strip():
        print("[app] Empty transcript detected. Skipping response and synthesis.")
        latency.print_summary()
        if latency_tracker is not None:
            latency_tracker.record_turn((latency._stages_ms.get("recording_ms", 0.0)
                                         + latency._stages_ms.get("save_ms", 0.0)
                                         + latency._stages_ms.get("transcription_ms", 0.0)))
            latency_tracker.print_rolling_summary()
        print("=" * 50)
        print("  Done.")
        print("=" * 50)
        return
    print(f"\nTranscript: {transcript}\n")

    # Step 5 — Generate response
    response = latency.measure("response_ms", generate_response, transcript)

    # Step 6 — Speak response
    latency.measure("synthesis_ms", speak_text, response)
    latency.print_summary()
    if latency_tracker is not None:
        total_ms = (
            latency._stages_ms.get("recording_ms", 0.0)
            + latency._stages_ms.get("save_ms", 0.0)
            + latency._stages_ms.get("transcription_ms", 0.0)
            + latency._stages_ms.get("response_ms", 0.0)
            + latency._stages_ms.get("synthesis_ms", 0.0)
        )
        latency_tracker.record_turn(total_ms)
        latency_tracker.print_rolling_summary()

    print("=" * 50)
    print("  Done.")
    print("=" * 50)


def run_app() -> None:
    """Run the pipeline, looping if LOOP_MODE is enabled.

    In loop mode the pipeline repeats until Ctrl+C is pressed or MAX_TURNS
    is reached (MAX_TURNS == 0 means unlimited). A separator is printed
    between turns so output from consecutive runs is easy to distinguish.
    """
    controller = TurnController(loop_mode=LOOP_MODE, max_turns=MAX_TURNS)
    latency_tracker = LatencyTracker()
    if LOOP_MODE:
        print("Loop mode active. Press Ctrl+C to exit.\n")
    try:
        while controller.should_continue():
            if controller.turn_count > 0:
                print("\n" + "-" * 50 + "\n")
            run_pipeline(latency_tracker=latency_tracker)
            controller.mark_turn_completed()
    except KeyboardInterrupt:
        print("\n\nExiting. Goodbye!")


if __name__ == "__main__":
    run_app()

"""
tests/test_app.py — Integration tests for src/app.py.

All pipeline components are mocked so the full run_pipeline flow can be
verified without audio hardware, Whisper model weights, or a TTS backend.
"""

import os
import sys
from unittest.mock import call, patch, MagicMock

import pytest

# Make src/ importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Stub sounddevice before app.py (via audio_input) imports it at module load.
# PortAudio is not available in headless / CI environments.
# ---------------------------------------------------------------------------
sys.modules.setdefault("sounddevice", MagicMock())


class TestRunPipeline:
    """Tests for app.run_pipeline."""

    def _run_with_mocks(self, transcript: str = "hello world"):
        """Helper: run the pipeline with all external calls mocked."""
        with patch("app.record_audio", return_value=[1, 2, 3]) as mock_record, \
             patch("app.build_recording_filepath", return_value="/tmp/fake.wav"), \
             patch("app.save_wav", return_value="/tmp/fake.wav"), \
             patch("app.transcribe_file", return_value=transcript) as mock_transcribe, \
             patch("app.generate_response", return_value=f"I heard: {transcript}") as mock_respond, \
             patch("app.speak_text") as mock_speak:
            from app import run_pipeline
            run_pipeline()
            return mock_record, mock_transcribe, mock_respond, mock_speak

    def test_record_audio_is_called(self):
        mock_record, _, _, _ = self._run_with_mocks()
        mock_record.assert_called_once()

    def test_transcribe_file_receives_wav_path(self):
        _, mock_transcribe, _, _ = self._run_with_mocks()
        mock_transcribe.assert_called_once_with("/tmp/fake.wav")

    def test_generate_response_receives_transcript(self):
        _, _, mock_respond, _ = self._run_with_mocks(transcript="test phrase")
        mock_respond.assert_called_once_with("test phrase")

    def test_speak_text_receives_response(self):
        _, _, _, mock_speak = self._run_with_mocks(transcript="hello")
        mock_speak.assert_called_once_with("I heard: hello")

    def test_pipeline_completes_without_error(self):
        # Ensure no exceptions are raised during a normal run
        self._run_with_mocks()

    def test_all_steps_called_in_order(self):
        """Verify the four pipeline steps fire in the correct sequence."""
        call_order = []

        def fake_record():
            call_order.append("record")
            return [1, 2, 3]

        def fake_save(samples, path):
            call_order.append("save")
            return "/tmp/fake.wav"

        def fake_transcribe(path):
            call_order.append("transcribe")
            return "hi"

        def fake_respond(text):
            call_order.append("respond")
            return "I heard: hi"

        def fake_speak(text):
            call_order.append("speak")

        with patch("app.record_audio", side_effect=fake_record), \
             patch("app.build_recording_filepath", return_value="/tmp/fake.wav"), \
             patch("app.save_wav", side_effect=fake_save), \
             patch("app.transcribe_file", side_effect=fake_transcribe), \
             patch("app.generate_response", side_effect=fake_respond), \
             patch("app.speak_text", side_effect=fake_speak):
            from app import run_pipeline
            run_pipeline()

        assert call_order == ["record", "save", "transcribe", "respond", "speak"]

    def test_latency_summary_is_printed(self, capsys):
        self._run_with_mocks()
        out = capsys.readouterr().out
        for key in (
            "recording_ms",
            "save_ms",
            "transcription_ms",
            "response_ms",
            "synthesis_ms",
            "total_ms",
        ):
            assert key in out

    def test_each_pipeline_stage_executes_once(self):
        stage_counts = {
            "record": 0,
            "save": 0,
            "transcribe": 0,
            "respond": 0,
            "speak": 0,
        }

        def fake_record():
            stage_counts["record"] += 1
            return [1, 2, 3]

        def fake_save(samples, path):
            stage_counts["save"] += 1
            return "/tmp/fake.wav"

        def fake_transcribe(path):
            stage_counts["transcribe"] += 1
            return "hello"

        def fake_respond(text):
            stage_counts["respond"] += 1
            return "I heard: hello"

        def fake_speak(text):
            stage_counts["speak"] += 1

        with patch("app.record_audio", side_effect=fake_record), \
             patch("app.build_recording_filepath", return_value="/tmp/fake.wav"), \
             patch("app.save_wav", side_effect=fake_save), \
             patch("app.transcribe_file", side_effect=fake_transcribe), \
             patch("app.generate_response", side_effect=fake_respond), \
             patch("app.speak_text", side_effect=fake_speak):
            from app import run_pipeline
            run_pipeline()

        assert stage_counts == {
            "record": 1,
            "save": 1,
            "transcribe": 1,
            "respond": 1,
            "speak": 1,
        }

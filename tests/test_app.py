"""
tests/test_app.py — Integration tests for src/app.py.

All pipeline components are mocked so the full run_pipeline flow can be
verified without audio hardware, Whisper model weights, or a TTS backend.
"""

import os
import sys
from unittest.mock import MagicMock, patch

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
        with (
            patch("app.record_audio", return_value=[1, 2, 3]) as mock_record,
            patch("app.build_recording_filepath", return_value="/tmp/fake.wav"),
            patch("app.save_wav", return_value="/tmp/fake.wav"),
            patch("app.transcribe_file", return_value=transcript) as mock_transcribe,
            patch(
                "app.generate_response", return_value=f"I heard: {transcript}"
            ) as mock_respond,
            patch("app.speak_text") as mock_speak,
        ):
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

    def test_pipeline_handles_microphone_failure_without_crash(self, capsys):
        with (
            patch("app.record_audio", side_effect=RuntimeError("mic error")),
            patch("app.build_recording_filepath"),
            patch("app.save_wav"),
            patch("app.transcribe_file"),
            patch("app.generate_response"),
            patch("app.speak_text"),
        ):
            from app import run_pipeline

            run_pipeline()

        out = capsys.readouterr().out
        assert "Recording failed" in out
        assert "Skipping turn safely" in out

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

        with (
            patch("app.record_audio", side_effect=fake_record),
            patch("app.build_recording_filepath", return_value="/tmp/fake.wav"),
            patch("app.save_wav", side_effect=fake_save),
            patch("app.transcribe_file", side_effect=fake_transcribe),
            patch("app.generate_response", side_effect=fake_respond),
            patch("app.speak_text", side_effect=fake_speak),
        ):
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

        with (
            patch("app.record_audio", side_effect=fake_record),
            patch("app.build_recording_filepath", return_value="/tmp/fake.wav"),
            patch("app.save_wav", side_effect=fake_save),
            patch("app.transcribe_file", side_effect=fake_transcribe),
            patch("app.generate_response", side_effect=fake_respond),
            patch("app.speak_text", side_effect=fake_speak),
        ):
            from app import run_pipeline

            run_pipeline()

        assert stage_counts == {
            "record": 1,
            "save": 1,
            "transcribe": 1,
            "respond": 1,
            "speak": 1,
        }

    def test_whitespace_transcript_skips_response_and_synthesis(self, capsys):
        with (
            patch("app.record_audio", return_value=[1, 2, 3]),
            patch("app.build_recording_filepath", return_value="/tmp/fake.wav"),
            patch("app.save_wav", return_value="/tmp/fake.wav"),
            patch("app.transcribe_file", return_value="   "),
            patch("app.generate_response") as mock_respond,
            patch("app.speak_text") as mock_speak,
        ):
            from app import run_pipeline

            run_pipeline()

        mock_respond.assert_not_called()
        mock_speak.assert_not_called()
        out = capsys.readouterr().out
        assert "Empty transcript detected" in out


class TestRunApp:
    """Tests for app.run_app loop behaviour."""

    def _mocked_pipeline(self, call_counter: dict):
        """Patch all pipeline stages and count run_pipeline invocations."""

        def fake_record():
            return [1, 2, 3]

        def fake_save(samples, path):
            return "/tmp/fake.wav"

        def fake_transcribe(path):
            return "hello"

        def fake_respond(text):
            return "I heard: hello"

        def fake_speak(text):
            call_counter["turns"] = call_counter.get("turns", 0) + 1

        return (
            patch("app.record_audio", side_effect=fake_record),
            patch("app.build_recording_filepath", return_value="/tmp/fake.wav"),
            patch("app.save_wav", side_effect=fake_save),
            patch("app.transcribe_file", side_effect=fake_transcribe),
            patch("app.generate_response", side_effect=fake_respond),
            patch("app.speak_text", side_effect=fake_speak),
        )

    def test_run_app_single_turn_when_loop_mode_false(self):
        """run_app runs exactly one turn when LOOP_MODE is False."""
        counter = {}
        patches = self._mocked_pipeline(counter)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patch("app.LOOP_MODE", False),
        ):
            from app import run_app

            run_app()
        assert counter.get("turns", 0) == 1

    def test_run_app_loop_respects_max_turns(self):
        """run_app stops after MAX_TURNS turns when LOOP_MODE is True."""
        counter = {}
        patches = self._mocked_pipeline(counter)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patch("app.LOOP_MODE", True),
            patch("app.MAX_TURNS", 3),
        ):
            from app import run_app

            run_app()
        assert counter.get("turns", 0) == 3

    def test_run_app_exits_gracefully_on_keyboard_interrupt(self, capsys):
        """run_app catches KeyboardInterrupt and prints a goodbye message."""
        call_count = [0]

        def fake_speak(text):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt

        with (
            patch("app.record_audio", return_value=[1, 2, 3]),
            patch("app.build_recording_filepath", return_value="/tmp/fake.wav"),
            patch("app.save_wav", return_value="/tmp/fake.wav"),
            patch("app.transcribe_file", return_value="hello"),
            patch("app.generate_response", return_value="I heard: hello"),
            patch("app.speak_text", side_effect=fake_speak),
            patch("app.LOOP_MODE", True),
            patch("app.MAX_TURNS", 0),
        ):
            from app import run_app

            run_app()  # must not propagate KeyboardInterrupt

        out = capsys.readouterr().out
        assert "Goodbye" in out

    def test_run_app_prints_separator_between_turns(self, capsys):
        """A turn separator is printed before every turn after the first."""
        counter = [0]

        def fake_speak(text):
            counter[0] += 1

        with (
            patch("app.record_audio", return_value=[1, 2, 3]),
            patch("app.build_recording_filepath", return_value="/tmp/fake.wav"),
            patch("app.save_wav", return_value="/tmp/fake.wav"),
            patch("app.transcribe_file", return_value="hello"),
            patch("app.generate_response", return_value="I heard: hello"),
            patch("app.speak_text", side_effect=fake_speak),
            patch("app.LOOP_MODE", True),
            patch("app.MAX_TURNS", 2),
        ):
            from app import run_app

            run_app()

        out = capsys.readouterr().out
        assert "-" * 50 in out

    def test_run_app_latency_logged_each_turn(self, capsys):
        """Latency summary is printed once per turn in loop mode."""
        with (
            patch("app.record_audio", return_value=[1, 2, 3]),
            patch("app.build_recording_filepath", return_value="/tmp/fake.wav"),
            patch("app.save_wav", return_value="/tmp/fake.wav"),
            patch("app.transcribe_file", return_value="hello"),
            patch("app.generate_response", return_value="I heard: hello"),
            patch("app.speak_text"),
            patch("app.LOOP_MODE", True),
            patch("app.MAX_TURNS", 2),
        ):
            from app import run_app

            run_app()

        out = capsys.readouterr().out
        # total_ms should appear twice (once per turn)
        assert out.count("total_ms") == 2

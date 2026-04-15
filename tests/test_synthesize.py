"""
tests/test_synthesize.py — Unit tests for src/synthesize.py.

The pyttsx3 engine is mocked so these tests run in headless/CI environments
without any TTS backend installed.
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Make src/ importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestSpeakText:
    """Tests for synthesize.speak_text."""

    def test_calls_say_with_text(self):
        mock_engine = MagicMock()
        with patch("synthesize._get_engine", return_value=mock_engine):
            import synthesize

            synthesize.speak_text("hello world")

        mock_engine.say.assert_called_once_with("hello world")

    def test_calls_run_and_wait(self):
        mock_engine = MagicMock()
        with patch("synthesize._get_engine", return_value=mock_engine):
            import synthesize

            synthesize.speak_text("test")

        mock_engine.runAndWait.assert_called_once()

    def test_returns_none(self):
        mock_engine = MagicMock()
        with patch("synthesize._get_engine", return_value=mock_engine):
            import synthesize

            result = synthesize.speak_text("anything")

        assert result is None


class TestSaveSpeech:
    """Tests for synthesize.save_speech."""

    def test_calls_save_to_file(self, tmp_path):
        mock_engine = MagicMock()
        with (
            patch("synthesize._get_engine", return_value=mock_engine),
            patch("synthesize.OUTPUTS_DIR", str(tmp_path)),
        ):
            import synthesize

            synthesize.save_speech("save this")

        mock_engine.save_to_file.assert_called_once()
        call_text = mock_engine.save_to_file.call_args[0][0]
        assert call_text == "save this"

    def test_returns_filepath_string(self, tmp_path):
        mock_engine = MagicMock()
        with (
            patch("synthesize._get_engine", return_value=mock_engine),
            patch("synthesize.OUTPUTS_DIR", str(tmp_path)),
        ):
            import synthesize

            result = synthesize.save_speech("hello")

        assert isinstance(result, str)
        assert result.endswith(".wav")

    def test_saved_file_in_outputs_dir(self, tmp_path):
        mock_engine = MagicMock()
        with (
            patch("synthesize._get_engine", return_value=mock_engine),
            patch("synthesize.OUTPUTS_DIR", str(tmp_path)),
        ):
            import synthesize

            result = synthesize.save_speech("hello")

        assert result.startswith(str(tmp_path))

    def test_calls_run_and_wait_after_save(self, tmp_path):
        mock_engine = MagicMock()
        with (
            patch("synthesize._get_engine", return_value=mock_engine),
            patch("synthesize.OUTPUTS_DIR", str(tmp_path)),
        ):
            import synthesize

            synthesize.save_speech("text")

        mock_engine.runAndWait.assert_called_once()

"""
tests/test_transcribe.py — Unit tests for src/transcribe.py.

The Whisper model is mocked so these tests run without GPU, model weights,
or any audio file on disk.
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Make src/ importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestTranscribeFile:
    """Tests for transcribe.transcribe_file."""

    def _make_mock_model(self, text: str) -> MagicMock:
        """Return a mock that behaves like a loaded Whisper model."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": text}
        return mock_model

    def test_returns_stripped_text(self):
        mock_model = self._make_mock_model("  hello world  ")
        with (
            patch("transcribe._get_model", return_value=mock_model),
            patch("transcribe.WHISPER_LANGUAGE", None),
        ):
            # Re-import to pick up the patch cleanly
            import transcribe

            result = transcribe.transcribe_file("fake.wav")

        assert result == "hello world"

    def test_calls_transcribe_with_filepath(self):
        mock_model = self._make_mock_model("test")
        with (
            patch("transcribe._get_model", return_value=mock_model),
            patch("transcribe.WHISPER_LANGUAGE", None),
        ):
            import transcribe

            transcribe.transcribe_file("/path/to/audio.wav")

        mock_model.transcribe.assert_called_once()
        args, kwargs = mock_model.transcribe.call_args
        assert args[0] == "/path/to/audio.wav"

    def test_language_option_passed_when_set(self):
        mock_model = self._make_mock_model("bonjour")
        with (
            patch("transcribe._get_model", return_value=mock_model),
            patch("transcribe.WHISPER_LANGUAGE", "fr"),
        ):
            import transcribe

            transcribe.transcribe_file("audio.wav")

        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("language") == "fr"

    def test_no_language_option_when_none(self):
        mock_model = self._make_mock_model("hello")
        with (
            patch("transcribe._get_model", return_value=mock_model),
            patch("transcribe.WHISPER_LANGUAGE", None),
        ):
            import transcribe

            transcribe.transcribe_file("audio.wav")

        _, kwargs = mock_model.transcribe.call_args
        assert "language" not in kwargs

    def test_returns_string(self):
        mock_model = self._make_mock_model("anything")
        with (
            patch("transcribe._get_model", return_value=mock_model),
            patch("transcribe.WHISPER_LANGUAGE", None),
        ):
            import transcribe

            result = transcribe.transcribe_file("audio.wav")

        assert isinstance(result, str)

"""
tests/test_transcribe.py — Unit tests for src/transcribe.py.

The Whisper model is mocked so these tests run without GPU, model weights,
or any audio file on disk.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Make src/ importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import transcribe


class TestTranscribeFile:
    """Tests for transcribe.transcribe_file."""

    def _make_mock_model(self, text: str) -> MagicMock:
        """Return a mock that behaves like a loaded Whisper model."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": text}
        return mock_model

    def test_returns_stripped_text(self):
        mock_model = self._make_mock_model("  hello world  ")
        with patch("transcribe._get_model", return_value=mock_model), \
             patch("transcribe.WHISPER_LANGUAGE", None):
            result = transcribe.transcribe_file("fake.wav")

        assert result == "hello world"

    def test_calls_transcribe_with_filepath(self):
        mock_model = self._make_mock_model("test")
        with patch("transcribe._get_model", return_value=mock_model), \
             patch("transcribe.WHISPER_LANGUAGE", None):
            transcribe.transcribe_file("/path/to/audio.wav")

        mock_model.transcribe.assert_called_once()
        args, kwargs = mock_model.transcribe.call_args
        assert args[0] == "/path/to/audio.wav"

    def test_language_option_passed_when_set(self):
        mock_model = self._make_mock_model("bonjour")
        with patch("transcribe._get_model", return_value=mock_model), \
             patch("transcribe.WHISPER_LANGUAGE", "fr"):
            transcribe.transcribe_file("audio.wav")

        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("language") == "fr"

    def test_no_language_option_when_none(self):
        mock_model = self._make_mock_model("hello")
        with patch("transcribe._get_model", return_value=mock_model), \
             patch("transcribe.WHISPER_LANGUAGE", None):
            transcribe.transcribe_file("audio.wav")

        _, kwargs = mock_model.transcribe.call_args
        assert "language" not in kwargs

    def test_returns_string(self):
        mock_model = self._make_mock_model("anything")
        with patch("transcribe._get_model", return_value=mock_model), \
             patch("transcribe.WHISPER_LANGUAGE", None):
            result = transcribe.transcribe_file("audio.wav")

        assert isinstance(result, str)


class TestResolveModelName:
    """Tests for transcribe._resolve_model_name."""

    def test_maps_large_to_large_v3(self):
        assert transcribe._resolve_model_name("large") == "large-v3"

    @pytest.mark.parametrize("name", ["tiny", "base", "small", "medium", "large-v3"])
    def test_passthrough_for_other_names(self, name: str):
        assert transcribe._resolve_model_name(name) == name

    def test_get_model_uses_resolved_model_name(self):
        with patch("transcribe.WHISPER_MODEL", "large"), \
             patch("transcribe.WhisperModel") as mock_whisper_model:
            transcribe._model = None
            transcribe._get_model()

        mock_whisper_model.assert_called_once_with("large-v3")
        transcribe._model = None

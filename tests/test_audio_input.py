"""
tests/test_audio_input.py — Unit tests for src/audio_input.py.

All hardware interactions (sounddevice, scipy WAV write) are mocked so
these tests run in any CI environment without a microphone or audio device.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Make src/ importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Stub sounddevice before audio_input imports it — PortAudio is not available
# in headless / CI environments.
# ---------------------------------------------------------------------------
_sd_stub = MagicMock()
sys.modules.setdefault("sounddevice", _sd_stub)


class TestRecordAudio:
    """Tests for audio_input.record_audio."""

    def test_calls_sd_rec_with_correct_params(self):
        import config

        mock_samples = np.zeros((config.SAMPLE_RATE * 5, 1), dtype="int16")
        with patch("sounddevice.rec", return_value=mock_samples) as mock_rec, \
             patch("sounddevice.wait"):
            from audio_input import record_audio
            record_audio(duration=5.0)

        mock_rec.assert_called_once_with(
            frames=int(5.0 * config.SAMPLE_RATE),
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="int16",
        )

    def test_returns_squeezed_1d_array(self):
        # sd.rec returns shape (frames, channels); result should be 1-D
        mock_2d = np.zeros((1000, 1), dtype="int16")
        with patch("sounddevice.rec", return_value=mock_2d), \
             patch("sounddevice.wait"):
            from audio_input import record_audio
            result = record_audio(duration=0.1)

        assert result.ndim == 1
        assert len(result) == 1000

    def test_wait_is_called(self):
        mock_samples = np.zeros((100, 1), dtype="int16")
        with patch("sounddevice.rec", return_value=mock_samples), \
             patch("sounddevice.wait") as mock_wait:
            from audio_input import record_audio
            record_audio(duration=0.1)

        mock_wait.assert_called_once()


class TestSaveWav:
    """Tests for audio_input.save_wav."""

    def test_creates_wav_file(self, tmp_path):
        filepath = str(tmp_path / "test.wav")
        samples = np.zeros(100, dtype="int16")

        import config
        with patch("audio_input.wav_write") as mock_write:
            from audio_input import save_wav
            result = save_wav(samples, filepath)

        mock_write.assert_called_once()
        args = mock_write.call_args[0]
        assert args[0] == filepath
        assert args[1] == config.SAMPLE_RATE
        np.testing.assert_array_equal(args[2], samples)

    def test_returns_filepath(self, tmp_path):
        filepath = str(tmp_path / "out.wav")
        samples = np.zeros(100, dtype="int16")

        with patch("scipy.io.wavfile.write"):
            from audio_input import save_wav
            result = save_wav(samples, filepath)

        assert result == filepath

    def test_creates_parent_directory(self, tmp_path):
        nested = str(tmp_path / "sub" / "dir" / "out.wav")
        samples = np.zeros(100, dtype="int16")

        with patch("scipy.io.wavfile.write"):
            from audio_input import save_wav
            save_wav(samples, nested)

        assert os.path.isdir(os.path.dirname(nested))


class TestRecordToFile:
    """Tests for audio_input.record_to_file (integration of record + save)."""

    def test_returns_filepath_string(self, tmp_path):
        mock_samples = np.zeros(100, dtype="int16")
        with patch("audio_input.record_audio", return_value=mock_samples), \
             patch("audio_input.save_wav", return_value=str(tmp_path / "rec.wav")) as mock_save:
            from audio_input import record_to_file
            result = record_to_file()

        assert isinstance(result, str)

    def test_filename_uses_wav_extension(self, tmp_path):
        mock_samples = np.zeros(100, dtype="int16")
        captured_path = []

        def fake_save(samples, filepath):
            captured_path.append(filepath)
            return filepath

        with patch("audio_input.record_audio", return_value=mock_samples), \
             patch("audio_input.save_wav", side_effect=fake_save), \
             patch("audio_input.RECORDINGS_DIR", str(tmp_path)):
            from audio_input import record_to_file
            record_to_file()

        assert captured_path[0].endswith(".wav")

    def test_filename_starts_with_recording(self, tmp_path):
        mock_samples = np.zeros(100, dtype="int16")
        captured_path = []

        def fake_save(samples, filepath):
            captured_path.append(filepath)
            return filepath

        with patch("audio_input.record_audio", return_value=mock_samples), \
             patch("audio_input.save_wav", side_effect=fake_save), \
             patch("audio_input.RECORDINGS_DIR", str(tmp_path)):
            from audio_input import record_to_file
            record_to_file()

        basename = os.path.basename(captured_path[0])
        assert basename.startswith("recording_")

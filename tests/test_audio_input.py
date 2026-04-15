"""
tests/test_audio_input.py — Unit tests for src/audio_input.py.

All hardware interactions (sounddevice, scipy WAV write) are mocked so
these tests run in any CI environment without a microphone or audio device.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np

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

    def _fake_stream(self, chunks):
        stream = MagicMock()
        stream.read = MagicMock(side_effect=[(chunk, False) for chunk in chunks])
        stream.__enter__.return_value = stream
        stream.__exit__.return_value = None
        return stream

    def test_stream_opened_with_correct_params(self):
        import config

        chunks = [np.zeros((100, 1), dtype="int16")]
        stream = self._fake_stream(chunks)
        with patch("sounddevice.InputStream", return_value=stream) as mock_stream:
            from audio_input import record_audio

            record_audio(duration=0.2)

        mock_stream.assert_called_once_with(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="int16",
        )

    def test_returns_1d_concatenated_int16_array(self):
        chunks = [
            np.array([[1], [2], [3]], dtype="int16"),
            np.array([[4], [5]], dtype="int16"),
        ]
        stream = self._fake_stream(chunks)
        with (
            patch("sounddevice.InputStream", return_value=stream),
            patch("audio_input._chunk_has_voice", side_effect=[True, False]),
        ):
            from audio_input import record_audio

            result = record_audio(duration=0.4)

        assert result.ndim == 1
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5], dtype="int16"))

    def test_stops_after_trailing_silence_once_voice_detected(self):
        voice = np.array([[1000], [1000]], dtype="int16")
        silence = np.array([[0], [0]], dtype="int16")
        chunks = [voice, silence, silence, voice]
        stream = self._fake_stream(chunks)

        with (
            patch("sounddevice.InputStream", return_value=stream),
            patch("audio_input.STREAM_CHUNK_SECONDS", 0.2),
            patch("audio_input.VAD_SILENCE_SECONDS", 0.4),
        ):
            from audio_input import record_audio

            result = record_audio(duration=1.0)

        # First voice + two silent chunks; fourth chunk should not be consumed.
        assert stream.read.call_count == 3
        np.testing.assert_array_equal(
            result, np.array([1000, 1000, 0, 0, 0, 0], dtype="int16")
        )

    def test_does_not_stop_early_on_initial_silence(self):
        silence = np.array([[0], [0]], dtype="int16")
        chunks = [silence, silence]
        stream = self._fake_stream(chunks)

        with (
            patch("sounddevice.InputStream", return_value=stream),
            patch("audio_input.STREAM_CHUNK_SECONDS", 0.2),
            patch("audio_input.VAD_SILENCE_SECONDS", 0.2),
        ):
            from audio_input import record_audio

            record_audio(duration=0.4)

        # Reads full max duration because voice was never detected.
        assert stream.read.call_count == 2

    def test_chunk_has_voice_uses_peak_amplitude_threshold(self):
        from audio_input import _chunk_has_voice

        assert (
            _chunk_has_voice(np.array([0, 100, -499], dtype="int16"), threshold=500)
            is False
        )
        assert (
            _chunk_has_voice(np.array([0, 500], dtype="int16"), threshold=500) is True
        )
        assert (
            _chunk_has_voice(np.array([0, -500], dtype="int16"), threshold=500) is True
        )

    def test_chunk_has_voice_returns_false_for_empty_chunk(self):
        from audio_input import _chunk_has_voice

        assert _chunk_has_voice(np.array([], dtype="int16")) is False

    def test_should_finalize_for_silence_requires_prior_voice(self):
        from audio_input import _should_finalize_for_silence

        assert _should_finalize_for_silence(0, 5, 3, min_voice_chunks=1) is False
        assert _should_finalize_for_silence(1, 2, 3, min_voice_chunks=1) is False
        assert _should_finalize_for_silence(1, 3, 3, min_voice_chunks=1) is True


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
        with (
            patch("audio_input.record_audio", return_value=mock_samples),
            patch(
                "audio_input.save_wav", return_value=str(tmp_path / "rec.wav")
            ) as mock_save,
        ):
            from audio_input import record_to_file

            result = record_to_file()

        assert isinstance(result, str)

    def test_filename_uses_wav_extension(self, tmp_path):
        mock_samples = np.zeros(100, dtype="int16")
        captured_path = []

        def fake_save(samples, filepath):
            captured_path.append(filepath)
            return filepath

        with (
            patch("audio_input.record_audio", return_value=mock_samples),
            patch("audio_input.save_wav", side_effect=fake_save),
            patch("audio_input.RECORDINGS_DIR", str(tmp_path)),
        ):
            from audio_input import record_to_file

            record_to_file()

        assert captured_path[0].endswith(".wav")

    def test_filename_starts_with_recording(self, tmp_path):
        mock_samples = np.zeros(100, dtype="int16")
        captured_path = []

        def fake_save(samples, filepath):
            captured_path.append(filepath)
            return filepath

        with (
            patch("audio_input.record_audio", return_value=mock_samples),
            patch("audio_input.save_wav", side_effect=fake_save),
            patch("audio_input.RECORDINGS_DIR", str(tmp_path)),
        ):
            from audio_input import record_to_file

            record_to_file()

        basename = os.path.basename(captured_path[0])
        assert basename.startswith("recording_")

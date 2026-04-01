"""
audio_input.py — Microphone recording and WAV file saving.

This module is intentionally thin: it captures raw PCM audio from the default
input device and writes it to disk.  All audio parameters come from config.py
so they can be adjusted in one place.

Future streaming integration point:
  - Replace ``record_to_file`` with an async generator that yields audio chunks
    to a real-time transcription consumer.
"""

import os
from collections.abc import Iterator

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

from config import (
    CHANNELS,
    RECORD_DURATION,
    RECORDINGS_DIR,
    SAMPLE_RATE,
    STREAM_CHUNK_SECONDS,
    VAD_AMPLITUDE_THRESHOLD,
    VAD_MIN_VOICE_CHUNKS,
    VAD_SILENCE_SECONDS,
)
from utils import ensure_dir, timestamped_filename


def _chunk_has_voice(chunk: np.ndarray, threshold: int = VAD_AMPLITUDE_THRESHOLD) -> bool:
    """Return True when *chunk* contains speech-like amplitude."""
    if chunk.size == 0:
        return False
    peak = int(np.max(np.abs(chunk.astype(np.int32))))
    return peak >= threshold


def _stream_microphone_chunks(
    duration: float = RECORD_DURATION,
    chunk_seconds: float = STREAM_CHUNK_SECONDS,
) -> Iterator[np.ndarray]:
    """Yield microphone audio chunks from an input stream up to *duration*."""
    frames_per_chunk = max(1, int(chunk_seconds * SAMPLE_RATE))
    max_chunks = max(1, int(np.ceil(duration / chunk_seconds)))
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
        for _ in range(max_chunks):
            chunk, overflowed = stream.read(frames_per_chunk)
            if overflowed:
                print("[audio_input] Warning: input overflow detected during capture.")
            yield np.asarray(chunk).squeeze()


def record_audio(duration: float = RECORD_DURATION) -> np.ndarray:
    """Capture microphone audio using streaming chunks plus simple VAD.

    Args:
        duration: Maximum recording length in seconds.

    Returns:
        A 1-D NumPy array of int16 PCM samples.
    """
    print(
        f"[audio_input] Streaming capture started "
        f"(max {duration:.1f}s, silence stop {VAD_SILENCE_SECONDS:.1f}s)… speak now."
    )
    silence_chunk_limit = max(1, int(np.ceil(VAD_SILENCE_SECONDS / STREAM_CHUNK_SECONDS)))
    chunks: list[np.ndarray] = []
    voiced_chunks = 0
    trailing_silence_chunks = 0

    for chunk in _stream_microphone_chunks(duration=duration, chunk_seconds=STREAM_CHUNK_SECONDS):
        if chunk.ndim > 1:
            chunk = chunk.squeeze()
        chunks.append(chunk)

        if _chunk_has_voice(chunk):
            voiced_chunks += 1
            trailing_silence_chunks = 0
        else:
            trailing_silence_chunks += 1

        if voiced_chunks >= VAD_MIN_VOICE_CHUNKS and trailing_silence_chunks >= silence_chunk_limit:
            break

    if not chunks:
        return np.array([], dtype=np.int16)

    print("[audio_input] Recording complete.")
    return np.concatenate(chunks).astype(np.int16, copy=False)


def save_wav(samples: np.ndarray, filepath: str) -> str:
    """Write *samples* to *filepath* as a WAV file.

    Args:
        samples: 1-D int16 PCM array.
        filepath: Destination path (directories are created if needed).

    Returns:
        The same *filepath* that was passed in.
    """
    ensure_dir(os.path.dirname(filepath))
    wav_write(filepath, SAMPLE_RATE, samples)
    print(f"[audio_input] Saved recording to: {filepath}")
    return filepath


def build_recording_filepath() -> str:
    """Create a timestamped destination path in ``RECORDINGS_DIR``."""
    filename = timestamped_filename("recording", "wav")
    return os.path.join(ensure_dir(RECORDINGS_DIR), filename)


def record_to_file(duration: float = RECORD_DURATION) -> str:
    """Convenience wrapper: record audio and immediately save it to disk.

    The file is placed in ``RECORDINGS_DIR`` with a timestamped name so
    recordings never overwrite each other.

    Args:
        duration: Recording length in seconds.

    Returns:
        Absolute path to the saved WAV file.
    """
    samples = record_audio(duration)
    return save_wav(samples, build_recording_filepath())

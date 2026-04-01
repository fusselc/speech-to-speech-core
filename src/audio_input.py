"""
audio_input.py — Microphone recording and WAV file saving.

This module is intentionally thin: it captures raw PCM audio from the default
input device and writes it to disk.  All audio parameters come from config.py
so they can be adjusted in one place.

Streaming extension points
--------------------------
* ``record_to_file`` is the current pipeline entry point.  It blocks until the
  full utterance is captured and returns a path to a saved WAV file.

* ``record_audio_chunks`` is the foundation for future streaming transcription.
  It yields fixed-size PCM chunks in real time using an ``sd.InputStream`` so a
  consumer can process audio incrementally instead of waiting for the whole
  recording to finish.

  Future work (Phase 2):
    - Wire each yielded chunk into a partial-transcription consumer
      (e.g. faster-whisper streaming API or a WebSocket endpoint).
    - Tune ``CHUNK_DURATION`` in config.py to balance latency and accuracy.
    - Replace ``record_to_file`` in app.py with an async loop over
      ``record_audio_chunks`` once streaming transcription is available.
"""

import os
from typing import Generator

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

from config import CHANNELS, CHUNK_DURATION, RECORD_DURATION, RECORDINGS_DIR, SAMPLE_RATE
from utils import ensure_dir, timestamped_filename


def record_audio(duration: float = RECORD_DURATION) -> np.ndarray:
    """Capture ``duration`` seconds of audio from the default microphone.

    Args:
        duration: Recording length in seconds.

    Returns:
        A 1-D NumPy array of int16 PCM samples.
    """
    print(f"[audio_input] Recording for {duration:.1f} second(s)… speak now.")
    samples = sd.rec(
        frames=int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()  # Block until recording is complete
    print("[audio_input] Recording complete.")
    # sd.rec returns shape (frames, channels); squeeze to 1-D for mono
    return samples.squeeze()


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


def record_audio_chunks(
    total_duration: float = RECORD_DURATION,
    chunk_duration: float = CHUNK_DURATION,
) -> Generator[np.ndarray, None, None]:
    """Capture microphone audio and yield it as a series of fixed-size chunks.

    This generator is designed as the foundation for streaming transcription.
    Rather than blocking until the entire utterance is recorded, it opens a
    continuous ``sd.InputStream`` and yields each chunk as soon as it is
    available.  A future consumer can begin processing audio incrementally
    while the speaker is still talking.

    Current behaviour:  chunks are yielded but not yet fed to a transcription
    model — the full streaming pipeline will be wired up in Phase 2.

    # Future streaming extension point:
    #   Pass each yielded chunk to a partial-transcription consumer, e.g.:
    #
    #       for chunk in record_audio_chunks():
    #           partial_text = streaming_transcriber.feed(chunk)
    #           if partial_text:
    #               handle_partial(partial_text)
    #
    #   Tune CHUNK_DURATION in config.py to balance latency and accuracy.

    Args:
        total_duration: Total recording length in seconds.
        chunk_duration: Length of each yielded chunk in seconds.

    Yields:
        1-D int16 NumPy arrays, each containing ``chunk_duration`` seconds of
        mono PCM audio at ``SAMPLE_RATE``.
    """
    chunk_frames = int(chunk_duration * SAMPLE_RATE)
    num_chunks = max(1, round(total_duration / chunk_duration))

    print(
        f"[audio_input] Recording {num_chunks} chunk(s) "
        f"of {chunk_duration:.2f}s each… speak now."
    )
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
        for _ in range(num_chunks):
            chunk, _overflowed = stream.read(chunk_frames)
            yield chunk.squeeze()
    print("[audio_input] Chunk recording complete.")

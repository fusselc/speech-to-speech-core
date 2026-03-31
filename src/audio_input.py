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

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

from config import CHANNELS, RECORD_DURATION, RECORDINGS_DIR, SAMPLE_RATE
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
    filename = timestamped_filename("recording", "wav")
    filepath = os.path.join(ensure_dir(RECORDINGS_DIR), filename)
    return save_wav(samples, filepath)

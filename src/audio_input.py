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
import torch
from loguru import logger
from scipy.io.wavfile import write as wav_write
from silero_vad import get_speech_timestamps, load_silero_vad

import config
from utils import ensure_dir, timestamped_filename

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

# ---------------------------------------------------------------------------
# Module-level Silero VAD cache — loaded on first use.
# ---------------------------------------------------------------------------
_silero_vad_model = None


def _get_silero_vad_model():
    """Load and cache the Silero VAD ONNX model."""
    global _silero_vad_model
    if _silero_vad_model is None:
        logger.info("Loading Silero VAD ONNX model...")
        _silero_vad_model = load_silero_vad(onnx=True)
        logger.info("Silero VAD model loaded.")
    return _silero_vad_model


def _resolve_vad_chunk_seconds() -> float:
    """Resolve chunk duration used by VAD and streaming-compatible capture."""
    base = (
        config.STREAMING_CHUNK_DURATION
        if config.USE_STREAMING
        else config.VAD_CHUNK_SECONDS
    )
    return min(max(base, 0.5), 1.0)


def _resolve_vad_device() -> str:
    """Resolve VAD runtime device preference for logging/fallback messages."""
    preferred = config.WHISPER_DEVICE.lower()
    if preferred == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested for VAD but unavailable; falling back to CPU.")
        return "cpu"
    if preferred == "auto" and torch.cuda.is_available():
        logger.info("Silero VAD ONNX is running on CPU (CUDA not used for VAD)")
    return "cpu"


def is_speech_chunk(audio_chunk: np.ndarray) -> bool:
    """Return True when a chunk contains speech according to Silero VAD."""
    if audio_chunk.size == 0:
        return False
    model = _get_silero_vad_model()
    audio_float = audio_chunk.astype(np.float32, copy=False) / 32768.0
    try:
        vad_device = _resolve_vad_device()
        logger.debug("VAD runtime device: {}", vad_device)
        audio_tensor = torch.from_numpy(audio_float)
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            threshold=config.SILENCE_THRESHOLD,
            sampling_rate=config.SAMPLE_RATE,
        )
    except Exception:
        logger.exception(
            "Silence detection failed; treating chunk as silence for resilience."
        )
        return False
    has_speech = len(speech_timestamps) > 0
    logger.debug(
        "VAD decision: speech={} chunk_samples={} threshold={}",
        has_speech,
        int(audio_chunk.size),
        config.SILENCE_THRESHOLD,
    )
    return has_speech


def _stream_microphone_chunks(
    duration: float = config.RECORD_DURATION,
    chunk_seconds: float | None = None,
) -> Iterator[np.ndarray]:
    """Yield microphone audio chunks from an input stream up to *duration*."""
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if chunk_seconds is None:
        chunk_seconds = _resolve_vad_chunk_seconds()
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if sd is None:
        logger.error(
            "No microphone backend found (sounddevice/PortAudio missing). "
            "Install PortAudio and grant microphone permissions."
        )
        raise RuntimeError("Microphone backend unavailable.")
    frames_per_chunk = max(1, int(chunk_seconds * config.SAMPLE_RATE))
    max_chunks = max(1, int(np.ceil(duration / chunk_seconds)))
    try:
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="int16",
        ) as stream:
            for _ in range(max_chunks):
                chunk, overflowed = stream.read(frames_per_chunk)
                if overflowed:
                    logger.warning("Input overflow detected during capture.")
                yield np.asarray(chunk).squeeze()
    except Exception:
        logger.exception(
            "No microphone found or permission denied while opening audio input."
        )
        raise


def _should_finalize_for_silence(
    voiced_chunks: int,
    trailing_silence_chunks: int,
    silence_chunk_limit: int,
    min_voice_chunks: int = config.VAD_MIN_VOICE_CHUNKS,
) -> bool:
    """Return True when recording should stop due to sustained silence."""
    return (
        voiced_chunks >= min_voice_chunks
        and trailing_silence_chunks >= silence_chunk_limit
    )


def record_until_silence(duration: float = config.RECORD_DURATION) -> np.ndarray:
    """Capture microphone audio using Silero VAD until trailing silence.

    Args:
        duration: Maximum recording length in seconds.

    Returns:
        A 1-D NumPy array of int16 PCM samples.
    """
    chunk_seconds = _resolve_vad_chunk_seconds()
    logger.info(
        "Streaming capture started (max {:.1f}s, silence stop {:.1f}s, chunk {:.2f}s)... speak now.",
        duration,
        config.VAD_SILENCE_SECONDS,
        chunk_seconds,
    )
    silence_chunk_limit = max(
        1,
        int(np.ceil(config.VAD_SILENCE_SECONDS / chunk_seconds)),
    )
    chunks: list[np.ndarray] = []
    voiced_chunks = 0
    trailing_silence_chunks = 0

    for chunk in _stream_microphone_chunks(
        duration=duration, chunk_seconds=chunk_seconds
    ):
        chunk_pcm = np.asarray(chunk, dtype=np.int16).reshape(-1)
        chunks.append(chunk_pcm)

        if is_speech_chunk(chunk_pcm):
            voiced_chunks += 1
            trailing_silence_chunks = 0
        else:
            trailing_silence_chunks += 1

        if _should_finalize_for_silence(
            voiced_chunks=voiced_chunks,
            trailing_silence_chunks=trailing_silence_chunks,
            silence_chunk_limit=silence_chunk_limit,
            min_voice_chunks=config.VAD_MIN_VOICE_CHUNKS,
        ):
            break

    if not chunks:
        return np.array([], dtype=np.int16)

    logger.info("Recording complete.")
    return np.concatenate(chunks).astype(np.int16, copy=False)


def record_audio(duration: float = config.RECORD_DURATION) -> np.ndarray:
    """Backward-compatible wrapper for microphone recording with VAD auto-stop."""
    return record_until_silence(duration=duration)


def save_wav(samples: np.ndarray, filepath: str) -> str:
    """Write *samples* to *filepath* as a WAV file.

    Args:
        samples: 1-D int16 PCM array.
        filepath: Destination path (directories are created if needed).

    Returns:
        The same *filepath* that was passed in.
    """
    ensure_dir(os.path.dirname(filepath))
    wav_write(filepath, config.SAMPLE_RATE, samples)
    logger.info("Saved recording to: {}", filepath)
    return filepath


def build_recording_filepath() -> str:
    """Create a timestamped destination path in ``RECORDINGS_DIR``."""
    filename = timestamped_filename("recording", "wav")
    return os.path.join(ensure_dir(config.RECORDINGS_DIR), filename)


def record_to_file(duration: float = config.RECORD_DURATION) -> str:
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

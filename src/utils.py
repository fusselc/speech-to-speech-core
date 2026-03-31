"""
utils.py — Shared helper utilities for the speech-to-speech pipeline.

Functions here are intentionally small and dependency-free so that every
other module can import them without pulling in heavy libraries.
"""

import os
from datetime import datetime, timezone


def ensure_dir(path: str) -> str:
    """Create *path* (and any missing parents) if it does not already exist.

    Returns the same *path* so callers can write::

        filepath = os.path.join(ensure_dir(RECORDINGS_DIR), filename)

    Args:
        path: Absolute or relative directory path.

    Returns:
        The same *path* that was passed in.
    """
    os.makedirs(path, exist_ok=True)
    return path


def timestamped_filename(prefix: str, extension: str) -> str:
    """Return a filename that includes a UTC timestamp to avoid collisions.

    Example output: ``recording_20240101_153045.wav``

    Args:
        prefix: Short label for the file type, e.g. ``"recording"`` or ``"response"``.
        extension: File extension *without* a leading dot, e.g. ``"wav"``.

    Returns:
        A filename string (no directory component).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{extension}"

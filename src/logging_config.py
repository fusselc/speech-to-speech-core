"""
logging_config.py — Centralized Loguru setup.
"""

import sys

from loguru import logger

_configured = False


def configure_logging(debug: bool = False) -> None:
    """Configure colored, timestamped console logging once per process."""
    global _configured
    if _configured:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if debug else "INFO",
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    _configured = True

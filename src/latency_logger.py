"""
latency_logger.py — Lightweight pipeline latency instrumentation helpers.
"""

import time
from dataclasses import dataclass, field


@dataclass
class LatencyLogger:
    """Collect and print per-stage latency metrics in milliseconds."""

    _start: float = field(default_factory=time.perf_counter)
    _stages_ms: dict[str, float] = field(default_factory=dict)

    def measure(self, stage_name: str, func, *args, **kwargs):
        """Run *func* and record elapsed time for *stage_name*."""
        started = time.perf_counter()
        result = func(*args, **kwargs)
        self._stages_ms[stage_name] = (time.perf_counter() - started) * 1000.0
        return result

    def print_summary(self) -> None:
        """Print a clear one-run latency summary."""
        total_ms = (time.perf_counter() - self._start) * 1000.0
        print("[latency] summary:")
        print(f"[latency] recording_ms={self._stages_ms.get('recording_ms', 0.0):.2f}")
        print(f"[latency] save_ms={self._stages_ms.get('save_ms', 0.0):.2f}")
        print(
            f"[latency] transcription_ms="
            f"{self._stages_ms.get('transcription_ms', 0.0):.2f}"
        )
        print(f"[latency] response_ms={self._stages_ms.get('response_ms', 0.0):.2f}")
        print(f"[latency] synthesis_ms={self._stages_ms.get('synthesis_ms', 0.0):.2f}")
        print(f"[latency] total_ms={total_ms:.2f}")

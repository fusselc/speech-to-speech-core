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

    def total_stages_ms(self) -> float:
        """Return sum of measured stage latencies."""
        return sum(self._stages_ms.values())

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


@dataclass
class LatencyTracker:
    """Track rolling and average latency across multiple turns.

    Uses constant-memory counters instead of an unbounded history list so
    that long-running loop mode does not grow memory without bound.
    """

    _count: int = 0
    _running_sum: float = 0.0
    _latest: float = 0.0

    def record_turn(self, total_ms: float) -> None:
        """Record total latency for one completed turn."""
        self._count += 1
        self._running_sum += total_ms
        self._latest = total_ms

    @property
    def turn_count(self) -> int:
        """Return number of recorded turns."""
        return self._count

    @property
    def latest_total_ms(self) -> float:
        """Return total latency of the most recent turn (0.0 if none recorded)."""
        return self._latest

    def average_total_ms(self) -> float:
        """Return average total latency across recorded turns."""
        if self._count == 0:
            return 0.0
        return self._running_sum / self._count

    def metrics(self) -> dict[str, float | int]:
        """Return structured latency metrics for easy testing/integration."""
        return {
            "latest_total_ms": self.latest_total_ms,
            "average_total_ms": self.average_total_ms(),
            "turn_count": self.turn_count,
        }

    def print_rolling_summary(self) -> None:
        """Print rolling latency summary after each turn."""
        if self._count == 0:
            return
        print("[latency] rolling_summary:")
        print(f"[latency] latest_turn_ms={self._latest:.2f}")
        print(f"[latency] avg_turn_ms={self.average_total_ms():.2f}")
        print(f"[latency] turns={self.turn_count}")

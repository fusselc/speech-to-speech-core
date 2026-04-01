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


@dataclass
class LatencyTracker:
    """Track rolling and average latency across multiple turns."""

    _total_ms_history: list[float] = field(default_factory=list)

    def record_turn(self, total_ms: float) -> None:
        """Record total latency for one completed turn."""
        self._total_ms_history.append(total_ms)

    @property
    def turn_count(self) -> int:
        """Return number of recorded turns."""
        return len(self._total_ms_history)

    def average_total_ms(self) -> float:
        """Return average total latency across recorded turns."""
        if not self._total_ms_history:
            return 0.0
        return sum(self._total_ms_history) / len(self._total_ms_history)

    def print_rolling_summary(self) -> None:
        """Print rolling latency summary after each turn."""
        if not self._total_ms_history:
            return
        latest_ms = self._total_ms_history[-1]
        print("[latency] rolling_summary:")
        print(f"[latency] latest_total_ms={latest_ms:.2f}")
        print(f"[latency] average_total_ms={self.average_total_ms():.2f}")
        print(f"[latency] turns={self.turn_count}")

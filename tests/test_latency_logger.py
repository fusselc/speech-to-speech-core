"""
tests/test_latency_logger.py — Unit tests for src/latency_logger.py.
"""

import os
import sys
from unittest.mock import patch

# Make src/ importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_measure_records_elapsed_time_ms():
    from latency_logger import LatencyLogger

    logger = LatencyLogger(_start=10.0)
    with patch("latency_logger.time.perf_counter", side_effect=[10.2, 11.0]):
        result = logger.measure("recording_ms", lambda: "ok")

    assert result == "ok"
    assert abs(logger._stages_ms["recording_ms"] - 800.0) < 1e-9


def test_print_summary_includes_required_keys(capsys):
    from latency_logger import LatencyLogger

    with patch("latency_logger.time.perf_counter", return_value=2.0):
        logger = LatencyLogger()
    logger._stages_ms = {
        "recording_ms": 1.0,
        "save_ms": 2.0,
        "transcription_ms": 3.0,
        "response_ms": 4.0,
        "synthesis_ms": 5.0,
    }
    with patch("latency_logger.time.perf_counter", return_value=2.5):
        logger.print_summary()

    out = capsys.readouterr().out
    for key in (
        "recording_ms",
        "save_ms",
        "transcription_ms",
        "response_ms",
        "synthesis_ms",
        "total_ms",
    ):
        assert key in out


def test_latency_tracker_prints_average_and_latest(capsys):
    from latency_logger import LatencyTracker

    tracker = LatencyTracker()
    tracker.record_turn(10.0)
    tracker.record_turn(30.0)
    tracker.print_rolling_summary()
    out = capsys.readouterr().out

    assert "rolling_summary" in out
    assert "latest_turn_ms=30.00" in out
    assert "avg_turn_ms=20.00" in out
    assert "turns=2" in out


def test_latency_tracker_constant_memory(capsys):
    """LatencyTracker must not grow with the number of turns recorded."""
    from latency_logger import LatencyTracker

    tracker = LatencyTracker()
    for i in range(1000):
        tracker.record_turn(float(i))

    # Internal state must only be scalar fields — no list or deque.
    assert not hasattr(tracker, "_total_ms_history")
    assert tracker.turn_count == 1000
    # Average of 0..999 = 499.5
    assert abs(tracker.average_total_ms() - 499.5) < 1e-6
    # Verify the latest value via the rolling summary output.
    tracker.print_rolling_summary()
    out = capsys.readouterr().out
    assert "latest_turn_ms=999.00" in out


def test_latency_tracker_empty_returns_zero():
    from latency_logger import LatencyTracker

    tracker = LatencyTracker()
    assert tracker.turn_count == 0
    assert tracker.average_total_ms() == 0.0

"""
run_latency_benchmark.py — Simple local latency benchmark runner.

Run with:
    python -m benchmarks.run_latency_benchmark
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from audio_input import build_recording_filepath, record_audio, save_wav
from latency_logger import LatencyLogger
from logging_config import configure_logging
from responder import generate_response
from synthesize import speak_text
from transcribe import transcribe_file
from turn_controller import TurnController

CSV_HEADERS = [
    "timestamp",
    "turn",
    "recording_ms",
    "save_ms",
    "transcription_ms",
    "response_ms",
    "synthesis_ms",
    "total_ms",
    "status",
]


@dataclass
class BenchmarkRow:
    timestamp: str
    turn: int
    recording_ms: float
    save_ms: float
    transcription_ms: float
    response_ms: float
    synthesis_ms: float
    total_ms: float
    status: str

    def as_csv_row(self) -> list[str | int | float]:
        return [
            self.timestamp,
            self.turn,
            self.recording_ms,
            self.save_ms,
            self.transcription_ms,
            self.response_ms,
            self.synthesis_ms,
            self.total_ms,
            self.status,
        ]


def _run_single_turn(turn_number: int) -> BenchmarkRow:
    latency = LatencyLogger()
    timestamp = datetime.now(timezone.utc).isoformat()
    status = "ok"
    try:
        samples = latency.measure("recording_ms", record_audio)
        wav_path = latency.measure(
            "save_ms", save_wav, samples, build_recording_filepath()
        )
        transcript = latency.measure("transcription_ms", transcribe_file, wav_path)
        response = latency.measure("response_ms", generate_response, transcript)
        latency.measure("synthesis_ms", speak_text, response)
    except Exception as exc:
        status = f"error:{type(exc).__name__}"

    stage_map = latency._stages_ms
    return BenchmarkRow(
        timestamp=timestamp,
        turn=turn_number,
        recording_ms=stage_map.get("recording_ms", 0.0),
        save_ms=stage_map.get("save_ms", 0.0),
        transcription_ms=stage_map.get("transcription_ms", 0.0),
        response_ms=stage_map.get("response_ms", 0.0),
        synthesis_ms=stage_map.get("synthesis_ms", 0.0),
        total_ms=latency.total_stages_ms(),
        status=status,
    )


def run_benchmark(turns: int = 10, csv_path: Path | None = None) -> list[BenchmarkRow]:
    """Run benchmark turns and persist CSV rows."""
    output_path = csv_path or (_ROOT / "benchmarks" / "latency_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    controller = TurnController(loop_mode=True, max_turns=turns)
    rows: list[BenchmarkRow] = []
    while controller.should_continue():
        row = _run_single_turn(controller.turn_count + 1)
        rows.append(row)
        controller.mark_turn_completed()

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADERS)
        for row in rows:
            writer.writerow(row.as_csv_row())

    return rows


def print_summary(rows: list[BenchmarkRow]) -> None:
    """Render benchmark summary with Rich."""
    console = Console()
    table = Table(title="Latency Benchmark Summary")
    table.add_column("Turns", justify="right")
    table.add_column("Avg total (ms)", justify="right")
    table.add_column("P95 total (ms)", justify="right")
    table.add_column("Min total (ms)", justify="right")
    table.add_column("Max total (ms)", justify="right")
    table.add_column("Errors", justify="right")

    totals = sorted(row.total_ms for row in rows)
    errors = sum(1 for row in rows if row.status != "ok")
    count = len(totals)
    avg = sum(totals) / count if count else 0.0
    p95_index = min(count - 1, int(0.95 * (count - 1))) if count else 0
    p95 = totals[p95_index] if totals else 0.0
    minimum = totals[0] if totals else 0.0
    maximum = totals[-1] if totals else 0.0

    table.add_row(
        str(count),
        f"{avg:.2f}",
        f"{p95:.2f}",
        f"{minimum:.2f}",
        f"{maximum:.2f}",
        str(errors),
    )
    console.print(table)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local speech pipeline latency benchmark."
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Number of turns to run (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()
    rows = run_benchmark(turns=args.turns)
    print_summary(rows)


if __name__ == "__main__":
    main()

# speech-to-speech-core

Phase 1 speech-to-speech AI core in Python: microphone recording, faster-whisper transcription, and text-to-speech playback for a modular low-latency voice pipeline with modern packaging and CI tooling.

---

## Overview

This repository implements a single speech-to-speech turn:

```text
Microphone → WAV file → faster-whisper → transcript → responder → TTS → speaker
```

Each stage lives in its own module so components can be swapped or extended later—for example, streaming transcription, LLM-backed responses, or higher-quality speech synthesis—without changing the rest of the pipeline.

---

## Design Goals

* Keep each stage isolated in its own module
* Keep functions small and readable
* Centralize configuration in `src/config.py`
* Make the pipeline runnable first, then extensible
* Measure latency so performance is visible and improvable

---

## Module Responsibilities

* `audio_input.py` — microphone capture and WAV persistence
* `transcribe.py` — faster-whisper speech-to-text
* `responder.py` — simple text response generation
* `synthesize.py` — local text-to-speech playback
* `latency_logger.py` — per-stage and total latency timing
* `logging_config.py` — centralized loguru formatting/configuration
* `cli.py` — Typer CLI entry point
* `utils.py` — shared helpers
* `app.py` — orchestration of the full speech-to-speech turn

---

## Repository Structure

```text
speech-to-speech-core/
├── src/
│   ├── app.py
│   ├── config.py
│   ├── audio_input.py
│   ├── transcribe.py
│   ├── responder.py
│   ├── synthesize.py
│   ├── latency_logger.py
│   ├── logging_config.py
│   ├── cli.py
│   └── utils.py
├── tests/
│   ├── test_app.py
│   ├── test_audio_input.py
│   ├── test_latency_logger.py
│   ├── test_responder.py
│   ├── test_synthesize.py
│   ├── test_transcribe.py
│   └── test_utils.py
├── recordings/
├── outputs/
├── pyproject.toml
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
└── README.md
```

---

## Requirements

* Python 3.10 or newer
* A working microphone
* On Linux, `espeak` must be installed for `pyttsx3`

```bash
sudo apt-get install espeak
```

---

## Setup

### Option A (recommended): uv

```bash
# 1. Clone the repository
git clone https://github.com/fusselc/speech-to-speech-core.git
cd speech-to-speech-core

# 2. Create and activate a virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install runtime + dev dependencies
uv sync --group dev
```

### Option B: pip

```bash
# 1. Clone the repository
git clone https://github.com/fusselc/speech-to-speech-core.git
cd speech-to-speech-core

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install runtime + dev dependencies
pip install -e ".[dev]"
```

> **Note:** `faster-whisper` downloads model weights on first run (model-size dependent). Internet is only required for that initial download.

---

## Run

### CLI (recommended)

After `uv sync` or `pip install -e ".[dev]"`, run:

```bash
speech-to-speech run --model small --device auto --streaming --loop
```

Flags:

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--model`, `-m` | `str` | `base` | faster-whisper model size |
| `--device`, `-d` | `auto\|cpu\|cuda` | `auto` | STT device preference |
| `--streaming/--no-streaming` | `bool` | `--streaming` | Enable streaming-oriented chunk handling |
| `--loop/--no-loop` | `bool` | `--loop` | Keep listening after each response |
| `--vad-sensitivity` | `float` | `0.5` | Silero VAD speech threshold |
| `--debug/--no-debug` | `bool` | `--no-debug` | Enable debug-level logs |

### Backward-compatible Python entry points

```bash
python src/app.py
python -m src.app
```

When the program starts:

1. Audio is recorded from microphone
2. WAV file is saved
3. faster-whisper transcribes speech
4. A response is generated
5. TTS speaks the response
6. Latency is logged
7. Rolling latency stats (latest turn + running average) are logged each turn

By default the pipeline runs in loop mode.

Press **Ctrl+C** to stop.

To disable looping quickly from CLI:

```bash
speech-to-speech run --no-loop
```

---

## Configuration

All settings live in `src/config.py`.

| Setting            | Default  | Description                   |
| ------------------ | -------- | ----------------------------- |
| `RECORD_DURATION`  | `5.0`    | Maximum recording duration in seconds |
| `USE_STREAMING` | `True` | Enable streaming-oriented chunking foundation |
| `STREAMING_CHUNK_DURATION` | `0.5` | Streaming chunk duration (seconds) |
| `VAD_CHUNK_SECONDS` | `0.5` | Silero VAD chunk duration (seconds) |
| `SILENCE_THRESHOLD` | `0.5` | Silero speech confidence threshold |
| `VAD_SILENCE_SECONDS` | `1.0` | Trailing silence required to auto-stop |
| `VAD_MIN_VOICE_CHUNKS` | `1` | Minimum voiced chunks before silence stop is allowed |
| `WHISPER_MODEL`    | `"base"` | faster-whisper model size     |
| `WHISPER_DEVICE`   | `"auto"` | Preferred device (`auto`, `cpu`, `cuda`) |
| `LANGUAGE`         | `None`   | Auto-detect language          |
| `TTS_RATE`         | `180`    | Speech speed                  |
| `LOOP_MODE`        | `True`   | Repeat turns                  |
| `MAX_TURNS`        | `0`      | Unlimited turns               |

---

## Logging Output Example

```text
2026-04-15 21:45:01.102 | INFO     | app:run_pipeline:32 - ==================================================
2026-04-15 21:45:01.103 | INFO     | audio_input:record_until_silence:138 - Streaming capture started (max 5.0s, silence stop 1.0s, chunk 0.50s)... speak now.
2026-04-15 21:45:06.182 | INFO     | transcribe:transcribe_file:104 - Transcription completed in 842.17 ms.
2026-04-15 21:45:06.182 | INFO     | responder:generate:29 - Response generated: 'I heard: what time is it?'
2026-04-15 21:45:06.184 | INFO     | latency_logger:print_summary:31 - [latency] summary:
2026-04-15 21:45:06.184 | INFO     | latency_logger:print_summary:49 - [latency] total_ms=5974.88
```

---

## Latency Metrics

Latency is tracked per stage:

* `recording_ms`
* `save_ms`
* `transcription_ms`
* `response_ms`
* `synthesis_ms`
* `total_ms`
* `latest_turn_ms` (rolling summary)
* `avg_turn_ms` (rolling summary)

This makes optimization measurable.

Typical local expectations (hardware-dependent):

* `small` model on CPU: transcription may be around ~1–3s per short utterance
* `small` model on CUDA: transcription often drops significantly below CPU latency
* VAD chunking at `0.5s` improves responsiveness for future streaming upgrades

---

## Tests

Run lint + type checks:

```bash
ruff check .
black --check .
isort --check-only .
mypy src tests
pyright
```

Run tests:

```bash
pytest tests/
```

Set up local pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

Tests are written to remain CI-friendly and avoid hardware dependency through mocking.

---

## OpenVoice Integration (Future)

`src/synthesize.py` already contains the correct integration point for replacing current TTS with OpenVoice later.

---

## Extending the Pipeline

| Goal                 | File             |
| -------------------- | ---------------- |
| Stream audio         | `audio_input.py` |
| Faster transcription | `transcribe.py`  |
| LLM responses        | `responder.py`   |
| Better TTS           | `synthesize.py`  |
| Config changes       | `config.py`      |

---

## Troubleshooting

* Verify microphone permissions if recording fails
* Install `espeak` if Linux TTS fails
* faster-whisper first run requires model download
* Use smaller faster-whisper models if slow

---

## Roadmap

Planned next steps:

* Faster transcription backend
* Streaming audio input
* Advanced response generation
* Higher quality speech synthesis
* Interrupt handling
* OpenVoice integration

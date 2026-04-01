# speech-to-speech-core

Phase 1 speech-to-speech AI core in Python: microphone recording, Whisper transcription, and text-to-speech playback for a modular low-latency voice pipeline.

---

## Architecture Overview

This repository implements a single-turn speech-to-speech pipeline:

```text
Microphone → WAV file → Whisper → transcript → responder → TTS → speaker
```

### Design Goals

- Keep each stage isolated in its own module
- Keep functions small and readable
- Centralize settings in `src/config.py`
- Make the pipeline runnable first, then extensible

### Module Responsibilities

- `audio_input.py` — microphone capture + WAV persistence
- `transcribe.py` — Whisper speech-to-text
- `responder.py` — simple text response generation
- `synthesize.py` — local text-to-speech playback
- `latency_logger.py` — per-stage and total latency timing
- `app.py` — orchestration of the full turn

---

## Repository Structure

```text
speech-to-speech-core/
├── .github/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── recordings/            # Auto-populated with captured WAVs (gitignored)
├── outputs/               # Auto-populated with synthesized output (gitignored)
├── src/
│   ├── app.py             # Entry point; runs the full pipeline
│   ├── config.py          # Centralized configuration
│   ├── audio_input.py     # Microphone recording + WAV save
│   ├── transcribe.py      # Whisper speech-to-text
│   ├── responder.py       # Simple response logic ("I heard: ...")
│   ├── synthesize.py      # Local TTS playback
│   ├── latency_logger.py  # Stage timing + summary output
│   └── utils.py           # Shared helpers
└── tests/
    ├── test_app.py
    ├── test_audio_input.py
    ├── test_latency_logger.py
    ├── test_responder.py
    ├── test_synthesize.py
    ├── test_transcribe.py
    └── test_utils.py
```

---

## Setup Instructions

### Requirements

- Python 3.10 or newer
- Working microphone
- On Linux, `espeak` for `pyttsx3`:

```bash
sudo apt-get install espeak
```

### Install

```bash
# 1) Clone
git clone https://github.com/fusselc/speech-to-speech-core.git
cd speech-to-speech-core

# 2) Create venv
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 3) Install dependencies
pip install -r requirements.txt
```

> **Note:** `openai-whisper` downloads model weights on first run (~150 MB for the
> default `base` model). An internet connection is required only for that initial download.

---

## Run Instructions

```bash
python src/app.py
```

Typical flow:

1. App prints a header and prompts for recording
2. Microphone audio is captured for the configured duration
3. WAV file is written to `recordings/`
4. Whisper generates transcript, which is printed
5. Responder generates a simple reply
6. TTS speaks reply locally
7. Latency summary is printed

All settings live in `src/config.py`. Key options:

| Setting | Default | Description |
|---|---|---|
| `RECORD_DURATION` | `5.0` | Recording length in seconds |
| `WHISPER_MODEL` | `"base"` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `WHISPER_LANGUAGE` | `None` | Language code (`None` = auto-detect, `"en"` = English) |
| `TTS_RATE` | `180` | Speech rate in words per minute |

---

## Expected Output Example

```text
==================================================
  Speech-to-Speech Core  |  Phase 1
==================================================
Recording for 5.0 second(s)… speak now.

Transcript: What time is it?

Response: I heard: What time is it?
[latency] summary:
[latency] recording_ms=5000.00
[latency] save_ms=3.12
[latency] transcription_ms=850.00
[latency] response_ms=2.00
[latency] synthesis_ms=120.00
[latency] total_ms=5975.12
==================================================
  Done.
==================================================
```

---

## Latency Metrics Explanation

The pipeline logs per-stage timing in milliseconds after each turn.

- **`recording_ms`** — time spent capturing microphone input (typically near `RECORD_DURATION`)
- **`save_ms`** — time to write captured audio to a WAV file on disk
- **`transcription_ms`** — time Whisper spends converting WAV to text; depends on model size and hardware
- **`response_ms`** — time to generate response text in `responder.py` (minimal in Phase 1)
- **`synthesis_ms`** — time for the local TTS engine to synthesize and play the response
- **`total_ms`** — end-to-end wall-clock time for the full turn

Use `transcription_ms` to compare model sizes (`tiny` vs `base` etc.) and `total_ms` to evaluate user-perceived latency.

---

## Future Roadmap

### Phase 1 (current)

- [x] Microphone recording
- [x] WAV saving
- [x] Whisper transcription
- [x] Simple text response
- [x] Local TTS playback
- [x] Per-stage latency logging
- [x] Unit tests
- [x] Setup/run documentation

### Next phases (planned)

- Expanded test coverage for audio/transcription edge cases (stubs/mocks already in place)
- Optional model/runtime tuning paths in configuration
- Extension hooks for advanced synthesizers (including OpenVoice)
- Benchmark scripts for repeatable latency tracking

### Out of current scope

- Streaming partial transcripts
- Translation
- Voice cloning
- Emotional speech transfer
- Memory
- Gaming integrations
- Database storage
- Cloud deployment

---

## Tests

```bash
pytest tests/
```

Tests are CI-friendly and stub/mock `sounddevice`, `pyttsx3`, and Whisper to avoid requiring local audio hardware.

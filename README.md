# speech-to-speech-core

Phase 1 speech-to-speech AI core in Python: microphone recording, Whisper transcription, and text-to-speech playback for a modular low-latency voice pipeline.

---

## Overview

This repository implements a single speech-to-speech turn:

```text
Microphone → WAV file → Whisper → transcript → responder → TTS → speaker
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
* `transcribe.py` — Whisper speech-to-text
* `responder.py` — simple text response generation
* `synthesize.py` — local text-to-speech playback
* `latency_logger.py` — per-stage and total latency timing
* `utils.py` — shared helpers
* `app.py` — orchestration of the full speech-to-speech turn

---

## Repository Structure

```text
speech-to-speech-core/
├── src/
│   ├── app.py              # Entry point; runs the full pipeline
│   ├── config.py           # Centralized configuration
│   ├── audio_input.py      # Microphone recording + WAV save
│   ├── transcribe.py       # Whisper speech-to-text
│   ├── responder.py        # Simple response logic ("I heard: ...")
│   ├── synthesize.py       # Local TTS playback
│   ├── latency_logger.py   # Stage timing + summary output
│   └── utils.py            # Shared helpers
├── tests/
│   ├── test_app.py
│   ├── test_audio_input.py
│   ├── test_responder.py
│   ├── test_synthesize.py
│   ├── test_transcribe.py
│   └── test_utils.py
├── recordings/             # Auto-populated with captured WAVs (gitignored)
├── outputs/                # Auto-populated with synthesized output (gitignored)
├── requirements.txt
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

```bash
# 1. Clone the repository
git clone https://github.com/fusselc/speech-to-speech-core.git
cd speech-to-speech-core

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** `openai-whisper` downloads model weights on first run (~150 MB for the default `base` model). An internet connection is required only for that initial download.

---

## Run

```bash
python src/app.py
```

When the program starts, it records a short utterance from your microphone. After recording finishes:

1. Audio is captured and saved as a WAV file
2. Whisper transcribes it
3. A simple text response is generated
4. The response is spoken aloud via local TTS
5. Latency metrics are printed

At the end of each run, latency is reported for:

* `recording_ms`
* `save_ms`
* `transcription_ms`
* `response_ms`
* `synthesis_ms`
* `total_ms`

---

## Configuration

All settings are in `src/config.py`.

| Setting            | Default  | Description                                                     |
| ------------------ | -------- | --------------------------------------------------------------- |
| `RECORD_DURATION`  | `5.0`    | Recording length in seconds                                     |
| `WHISPER_MODEL`    | `"base"` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `WHISPER_LANGUAGE` | `None`   | Language code (`None` = auto-detect, `"en"` = English)          |
| `TTS_RATE`         | `180`    | Speech rate in words per minute                                 |

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

## Latency Metrics

Latency is logged per stage in milliseconds (`ms`):

* `recording_ms` — microphone capture time
* `save_ms` — WAV write time
* `transcription_ms` — Whisper inference time
* `response_ms` — response generation time
* `synthesis_ms` — text-to-speech playback time
* `total_ms` — full pipeline duration

These metrics make performance trade-offs measurable.

---

## Tests

```bash
pip install pytest
pytest tests/
```

Tests are written to remain CI-friendly and avoid hard dependency on local audio hardware by mocking external interfaces where needed.

---

## OpenVoice Integration (Future)

`src/synthesize.py` contains a clearly marked hook showing where OpenVoice can later replace the current TTS engine without changing the rest of the pipeline.

---

## Extending the Pipeline

| Goal                        | File to change   |
| --------------------------- | ---------------- |
| Stream audio in real time   | `audio_input.py` |
| Upgrade transcription       | `transcribe.py`  |
| Add LLM-backed replies      | `responder.py`   |
| Upgrade TTS / voice cloning | `synthesize.py`  |
| Change file paths or model  | `config.py`      |

---

## Troubleshooting

* If microphone capture fails, verify microphone permissions and default input device
* If `pyttsx3` fails on Linux, confirm `espeak` is installed
* If Whisper fails initially, ensure model download completed successfully
* If transcription is slow, try a smaller Whisper model

---

## Roadmap

Planned next steps:

* conversational loop support
* faster transcription backend
* streaming audio input
* advanced response generation
* higher-quality speech synthesis

# speech-to-speech-core

Phase 1 speech-to-speech AI core in Python: microphone recording, Whisper transcription, and text-to-speech playback for a modular low-latency voice pipeline.

---

## Overview

The pipeline runs one full speech-to-speech turn:

```
Microphone → WAV file → Whisper → transcript → responder → TTS → speaker
```

Each step lives in its own module so any component can be swapped out
(e.g. a streaming transcriber, an LLM responder, or OpenVoice for synthesis)
without touching the others.

---

## Project structure

```
speech-to-speech-core/
├── src/
│   ├── app.py          # Entry point — runs the full pipeline
│   ├── config.py       # Centralised settings (sample rate, model, paths…)
│   ├── audio_input.py  # Microphone recording + WAV save
│   ├── transcribe.py   # Whisper speech-to-text
│   ├── responder.py    # Response generation ("I heard: …")
│   ├── synthesize.py   # Local TTS via pyttsx3 (OpenVoice hook included)
│   ├── latency_logger.py # Per-stage pipeline latency instrumentation
│   └── utils.py        # Shared helpers (ensure_dir, timestamped_filename)
├── tests/
│   ├── test_app.py
│   ├── test_audio_input.py
│   ├── test_responder.py
│   ├── test_synthesize.py
│   ├── test_transcribe.py
│   ├── test_latency_logger.py
│   └── test_utils.py
├── recordings/         # Auto-populated with captured WAVs (gitignored)
├── outputs/            # Auto-populated with TTS output WAVs (gitignored)
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10 or newer
- A working microphone
- On Linux, `espeak` must be installed for pyttsx3:

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

> **Note:** `openai-whisper` downloads model weights on first run
> (~150 MB for the default `base` model). An internet connection is required
> only for that initial download.

---

## Run

```bash
python src/app.py
```

When you see `Recording for 5.0 second(s)… speak now.` — say something into
your microphone. After the recording finishes, Whisper will transcribe it,
the transcript will be printed, and the TTS engine will speak the response.
At the end of each run, a latency summary is printed with:
`recording_ms`, `save_ms`, `transcription_ms`, `response_ms`,
`synthesis_ms`, and `total_ms`.

### Configuration

All settings are in `src/config.py`. Key options:

| Setting | Default | Description |
|---|---|---|
| `RECORD_DURATION` | `5.0` | Recording length in seconds |
| `WHISPER_MODEL` | `"base"` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `WHISPER_LANGUAGE` | `None` | Language code (`None` = auto-detect, `"en"` = English) |
| `TTS_RATE` | `180` | Speech rate in words per minute |

---

## Tests

```bash
pip install pytest
pytest tests/
```

---

## OpenVoice integration (future)

`src/synthesize.py` contains a clearly marked comment block showing exactly
where to plug in the [OpenVoice](https://github.com/myshell-ai/OpenVoice)
`ToneColorConverter` once you are ready to upgrade synthesis quality.
No other files need to change.

---

## Extending the pipeline

| Goal | File to change |
|---|---|
| Stream audio in real time | `audio_input.py` |
| Upgrade transcription | `transcribe.py` |
| Add LLM-backed replies | `responder.py` |
| Upgrade TTS / voice cloning | `synthesize.py` |
| Change file paths or model | `config.py` |

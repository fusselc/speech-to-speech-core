# Repository instructions for Copilot

This repository is a Python speech-to-speech prototype.

## Goal
Build a clean Phase 1 foundation:
microphone input -> WAV file -> Whisper transcription -> simple response -> local text-to-speech playback.

## Architecture rules
- Keep code modular.
- Put each stage in its own module.
- Avoid mixing audio capture, transcription, response generation, and synthesis in one file.
- Prefer small, readable functions.
- Use explicit configuration in `src/config.py`.
- Add comments only where they help future maintenance.

## Phase 1 scope
Include only:
- microphone recording
- WAV saving
- Whisper transcription
- simple text response
- local TTS playback
- unit tests
- README setup/run instructions

Do not add yet:
- streaming partial transcripts
- translation
- voice cloning
- emotional speech transfer
- memory
- gaming integrations
- database storage
- cloud deployment

## Testing rules
- Add or update tests for every new module.
- Keep tests runnable in CI without special hardware when possible.
- Stub microphone or audio dependencies in tests if needed.

## Documentation rules
- Keep README accurate and synchronized with the repo structure.
- Update dependency notes when packages change.
- Include setup, run, and troubleshooting steps.

## Quality rules
- Prefer clarity over cleverness.
- Make the first version runnable.
- If a dependency choice is uncertain, choose the simplest working option first.
- Leave clear extension points for OpenVoice later.

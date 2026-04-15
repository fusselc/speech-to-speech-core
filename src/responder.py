"""
responder.py — Generate a text response from a transcript.

The default behavior remains deterministic and simple while exposing a
pluggable strategy interface so an LLM-backed responder can be introduced
later without changing app-level call sites.
"""

from typing import Protocol


class ResponseGenerator(Protocol):
    """Interface for response generation backends."""

    def generate(self, transcript: str) -> str:
        """Return a response for a transcript."""


class EchoResponseGenerator:
    """Default deterministic response generator."""

    def generate(self, transcript: str) -> str:
        cleaned = transcript.strip()
        if not cleaned:
            return "Sorry, I didn't catch that. Could you please repeat?"
        response = f"I heard: {cleaned}"
        print(f"[responder] Response: {response!r}")
        return response


def generate_response(
    transcript: str, generator: ResponseGenerator | None = None
) -> str:
    """Build a reply string from a transcript.

    Args:
        transcript: The text that was transcribed from the user's speech.
        generator: Optional response backend for dependency injection.
            When None, the default deterministic echo generator is used.

    Returns:
        A response string ready to be passed to the TTS synthesiser.
    """
    active_generator = generator if generator is not None else EchoResponseGenerator()
    return active_generator.generate(transcript)

"""
responder.py — Generate a text response from a transcript.

Phase 1 keeps this deliberately simple: the response is just a formatted
echo of what was heard.  The function signature is designed so that a
more sophisticated LLM-backed responder can be dropped in later without
changing any call sites.

Future integration points:
  - Replace ``generate_response`` body with an LLM call (e.g. OpenAI Chat
    Completions, a local llama.cpp model, etc.).
  - Add conversation history / agent memory as a parameter.
"""


def generate_response(transcript: str) -> str:
    """Build a reply string from a transcript.

    Args:
        transcript: The text that was transcribed from the user's speech.

    Returns:
        A response string ready to be passed to the TTS synthesiser.
    """
    if not transcript:
        return "Sorry, I didn't catch that. Could you please repeat?"

    response = f"I heard: {transcript}"
    print(f"[responder] Response: {response!r}")
    return response

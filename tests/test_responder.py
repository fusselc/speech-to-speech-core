"""
tests/test_responder.py — Unit tests for src/responder.py.

These tests are intentionally focused on logic only and do not require
audio hardware, Whisper, or TTS to be installed.
"""

import sys
import os

# Make src/ importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from responder import generate_response


class TestGenerateResponse:
    def test_normal_transcript_returns_i_heard(self):
        result = generate_response("hello world")
        assert result == "I heard: hello world"

    def test_response_contains_transcript(self):
        transcript = "the quick brown fox"
        result = generate_response(transcript)
        assert transcript in result

    def test_empty_transcript_returns_fallback(self):
        result = generate_response("")
        assert "didn't catch" in result.lower() or "sorry" in result.lower()

    def test_whitespace_only_is_truthy_so_returns_i_heard(self):
        # A string of spaces is truthy in Python; responder echoes it as-is
        result = generate_response("   ")
        assert result.startswith("I heard:")

    def test_response_is_string(self):
        result = generate_response("test")
        assert isinstance(result, str)

    def test_punctuation_preserved(self):
        result = generate_response("How are you?")
        assert "How are you?" in result

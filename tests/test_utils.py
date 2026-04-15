"""
tests/test_utils.py — Unit tests for src/utils.py.

Tests cover the two public helpers without requiring any audio hardware
or heavy ML libraries.
"""

import os
import re
import sys

# Make src/ importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import ensure_dir, timestamped_filename


class TestEnsureDir:
    def test_creates_new_directory(self, tmp_path):
        new_dir = str(tmp_path / "new_folder")
        assert not os.path.exists(new_dir)
        result = ensure_dir(new_dir)
        assert os.path.isdir(new_dir)

    def test_returns_path(self, tmp_path):
        target = str(tmp_path / "return_check")
        result = ensure_dir(target)
        assert result == target

    def test_existing_directory_does_not_raise(self, tmp_path):
        existing = str(tmp_path)
        # Should not raise even though the directory already exists
        ensure_dir(existing)

    def test_creates_nested_directories(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c")
        ensure_dir(nested)
        assert os.path.isdir(nested)


class TestTimestampedFilename:
    def test_contains_prefix(self):
        name = timestamped_filename("recording", "wav")
        assert name.startswith("recording_")

    def test_contains_extension(self):
        name = timestamped_filename("recording", "wav")
        assert name.endswith(".wav")

    def test_matches_expected_pattern(self):
        name = timestamped_filename("response", "mp3")
        # Expected pattern: response_YYYYMMDD_HHMMSS.mp3
        pattern = r"^response_\d{8}_\d{6}\.mp3$"
        assert re.match(pattern, name), f"Unexpected filename: {name}"

    def test_two_calls_differ(self):
        import time

        name1 = timestamped_filename("test", "wav")
        time.sleep(1.1)  # Ensure the second is different
        name2 = timestamped_filename("test", "wav")
        assert name1 != name2

    def test_no_directory_component(self):
        name = timestamped_filename("rec", "wav")
        assert os.sep not in name

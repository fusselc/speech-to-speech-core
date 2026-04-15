"""
tests/test_cli.py — Unit tests for Typer CLI.
"""

import os
import sys
from unittest.mock import patch

from typer.testing import CliRunner

# Make src/ importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

runner = CliRunner()


def test_run_command_applies_runtime_options():
    import cli

    with (
        patch("cli._run_app") as mock_run_app,
        patch("cli.configure_logging"),
    ):
        result = runner.invoke(
            cli.cli,
            [
                "run",
                "--model",
                "small",
                "--device",
                "cpu",
                "--streaming",
                "--loop",
                "--vad-sensitivity",
                "0.7",
            ],
        )

    assert result.exit_code == 0
    mock_run_app.assert_called_once()


def test_run_command_rejects_invalid_device():
    import cli

    result = runner.invoke(cli.cli, ["run", "--device", "tpu"])
    assert result.exit_code != 0
    assert "auto, cpu, cuda" in (result.stdout + result.stderr)

"""
tests/test_turn_controller.py — Unit tests for src/turn_controller.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_single_turn_when_loop_mode_false():
    from turn_controller import TurnController

    controller = TurnController(loop_mode=False, max_turns=10)
    assert controller.should_continue() is True
    controller.mark_turn_completed()
    assert controller.should_continue() is False


def test_loop_mode_respects_max_turns():
    from turn_controller import TurnController

    controller = TurnController(loop_mode=True, max_turns=2)
    assert controller.should_continue() is True
    controller.mark_turn_completed()
    assert controller.should_continue() is True
    controller.mark_turn_completed()
    assert controller.should_continue() is False


def test_unlimited_loop_when_max_turns_zero():
    from turn_controller import TurnController

    controller = TurnController(loop_mode=True, max_turns=0)
    for _ in range(5):
        assert controller.should_continue() is True
        controller.mark_turn_completed()


def test_negative_max_turns_rejected():
    from turn_controller import TurnController

    with pytest.raises(ValueError):
        TurnController(loop_mode=True, max_turns=-1)

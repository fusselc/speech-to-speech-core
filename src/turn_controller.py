"""
turn_controller.py — Conversation loop progression and turn counting.
"""

from dataclasses import dataclass


@dataclass
class TurnController:
    """Control loop continuation and track completed turns."""

    loop_mode: bool
    max_turns: int = 0
    turn_count: int = 0

    def __post_init__(self) -> None:
        if self.max_turns < 0:
            raise ValueError("max_turns must be >= 0")

    def should_continue(self) -> bool:
        """Return True when another turn should run."""
        if not self.loop_mode:
            return self.turn_count == 0
        return self.max_turns == 0 or self.turn_count < self.max_turns

    def mark_turn_completed(self) -> None:
        """Increment completed turn count."""
        self.turn_count += 1

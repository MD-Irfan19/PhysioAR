"""Tests for Phase 8 — Visual Overlay System.

Focuses on safe rendering without crashing under various states
(missing data, edge cases). Does not test OpenCV pixel accuracy.

Phase 8 — Visual Overlay System.
"""

import numpy as np
import pytest

from src.overlay import (
    draw_angle,
    draw_rom_progress,
    draw_feedback,
    draw_rep_info,
    draw_hud,
)


@pytest.fixture
def blank_frame():
    """Returns a black 640x480 BGR frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ================================================================
# Test 1 — Angle Display
# ================================================================

class TestDrawAngle:
    """Tests for live angle display."""

    def test_valid_angle(self, blank_frame):
        # Should not crash.
        draw_angle(blank_frame, 67.4, "left")

    def test_none_angle(self, blank_frame):
        # Missing pose/angle.
        draw_angle(blank_frame, None, "right")

    def test_empty_side(self, blank_frame):
        # Should gracefully handle empty string.
        draw_angle(blank_frame, 45.0, "")

    def test_zero_angle(self, blank_frame):
        draw_angle(blank_frame, 0.0, "left")


# ================================================================
# Test 2 — ROM Progress
# ================================================================

class TestDrawRomProgress:
    """Tests for the ROM progress bar."""

    def test_half_progress(self, blank_frame):
        draw_rom_progress(blank_frame, 45.0, 90.0, "RISING")

    def test_zero_progress(self, blank_frame):
        draw_rom_progress(blank_frame, 0.0, 90.0, "WAIT_FOR_START")

    def test_full_progress(self, blank_frame):
        draw_rom_progress(blank_frame, 90.0, 90.0, "TOP")

    def test_overshoot_progress(self, blank_frame):
        draw_rom_progress(blank_frame, 110.0, 90.0, "TOP")

    def test_none_angle(self, blank_frame):
        draw_rom_progress(blank_frame, None, 90.0, "WAIT_FOR_START")

    def test_zero_target(self, blank_frame):
        draw_rom_progress(blank_frame, 45.0, 0.0, "RISING")


# ================================================================
# Test 3 — Rep Info
# ================================================================

class TestDrawRepInfo:
    """Tests for repetition count display."""

    def test_zero_reps(self, blank_frame):
        draw_rep_info(blank_frame, 0)

    def test_multiple_reps(self, blank_frame):
        draw_rep_info(blank_frame, 5)


# ================================================================
# Test 4 — Feedback Display
# ================================================================

class TestDrawFeedback:
    """Tests for corrective feedback display."""

    def test_no_feedback(self, blank_frame):
        draw_feedback(blank_frame, [])

    def test_one_feedback(self, blank_frame):
        draw_feedback(blank_frame, [
            "Torso Lean = 11° — reduce torso lean to maintain shoulder isolation."
        ])

    def test_multiple_feedback(self, blank_frame):
        draw_feedback(blank_frame, [
            "Torso Lean = 11° — reduce torso lean",
            "Neck Tilt = 5° — keep neck straight",
        ])

    def test_long_feedback(self, blank_frame):
        long_msg = "x" * 200
        draw_feedback(blank_frame, [long_msg])


# ================================================================
# Test 5 — Combined HUD
# ================================================================

class TestDrawHUD:
    """Tests for the unified HUD function."""

    def test_all_valid_data(self, blank_frame):
        draw_hud(
            blank_frame,
            angle=45.0,
            target_rom=90.0,
            rep_state="RISING",
            rep_count=3,
            feedback_messages=["Torso Lean = 11° — warning"],
            side="left",
        )

    def test_all_missing_data(self, blank_frame):
        draw_hud(
            blank_frame,
            angle=None,
            target_rom=90.0,
            rep_state="WAIT_FOR_START",
            rep_count=0,
            feedback_messages=[],
            side="",
        )

    def test_zero_target_rom(self, blank_frame):
        draw_hud(
            blank_frame,
            angle=10.0,
            target_rom=0.0,
            rep_state="RISING",
            rep_count=1,
            feedback_messages=[],
            side="right",
        )

"""Tests for Phase 5 — Rep Detection (Segmentation).

All tests use deterministic synthetic angle sequences. No webcam,
OpenCV, MediaPipe inference, GUI, or random data.

Tests cover:
  - Basic counting (1, 2, 5 reps)
  - Threshold boundaries (exact start/end)
  - Partial movements
  - Slow movements
  - Fast movements (threshold jumping)
  - Initial state behavior
  - Missing data (None angles)
  - Reset behavior
  - Event/boundary structure
  - Reusable exercise thresholds

Phase 5 — Rep Detection (Segmentation).
"""

from dataclasses import dataclass

import pytest

from src.exercises.base import CameraOrientation, ExerciseDefinition
from src.rep_detection import RepDetector, RepEvent, RepState


# ================================================================
# Helper: mock exercise definition
# ================================================================


def _make_exercise(start: float = 15.0, end: float = 80.0) -> ExerciseDefinition:
    """Create a minimal ExerciseDefinition with specified thresholds."""
    return ExerciseDefinition(
        name="Test Exercise",
        camera_orientation=CameraOrientation.FRONT,
        target_joint="test",
        rom_target=90.0,
        rep_start_angle=start,
        rep_end_angle=end,
        angle_calculator=lambda lm, side: 0.0,
    )


def _run_sequence(detector: RepDetector, angles: list) -> list[RepEvent]:
    """Feed a sequence of angles and collect all events."""
    events = []
    for i, angle in enumerate(angles):
        event = detector.update(angle, frame_index=i)
        if event is not None:
            events.append(event)
    return events


# ================================================================
# 1-5. BASIC COUNTING
# ================================================================


class TestBasicCounting:
    """Basic repetition counting."""

    def test_one_complete_rep(self):
        """down → up → down = 1 rep."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 5])
        assert d.rep_count == 1
        assert len(events) == 1

    def test_five_complete_reps(self):
        """Five complete repetitions count as 5."""
        d = RepDetector(_make_exercise())
        angles = []
        for _ in range(5):
            angles.extend([5, 85, 5])
        events = _run_sequence(d, angles)
        assert d.rep_count == 5
        assert len(events) == 5

    def test_two_complete_reps(self):
        """Two complete repetitions count as 2."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 5, 85, 5])
        assert d.rep_count == 2

    def test_holding_at_bottom_no_extra_count(self):
        """Holding at the bottom does not increment count."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 5, 5, 5, 5, 5])
        assert d.rep_count == 1

    def test_holding_at_top_no_extra_count(self):
        """Holding at the top does not increment count."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 85, 85, 85, 5])
        assert d.rep_count == 1


# ================================================================
# 6-10. THRESHOLD BOUNDARIES
# ================================================================


class TestThresholdBoundaries:
    """Exact boundary behavior with <= and >=."""

    def test_exact_start_angle_is_down(self):
        """angle == rep_start_angle → treated as DOWN."""
        d = RepDetector(_make_exercise(start=15.0, end=80.0))
        events = _run_sequence(d, [15.0])
        assert d.state == RepState.DOWN

    def test_exact_end_angle_is_up(self):
        """angle == rep_end_angle → treated as UP."""
        d = RepDetector(_make_exercise(start=15.0, end=80.0))
        events = _run_sequence(d, [15.0, 80.0])
        assert d.state == RepState.UP

    def test_return_to_exact_start_completes_rep(self):
        """Returning exactly to rep_start_angle completes the rep."""
        d = RepDetector(_make_exercise(start=15.0, end=80.0))
        events = _run_sequence(d, [15.0, 80.0, 15.0])
        assert d.rep_count == 1
        assert len(events) == 1

    def test_just_below_end_not_up(self):
        """angle = rep_end_angle - 0.1 does NOT reach UP."""
        d = RepDetector(_make_exercise(start=15.0, end=80.0))
        events = _run_sequence(d, [15.0, 79.9, 15.0])
        assert d.rep_count == 0

    def test_just_above_start_leaves_down(self):
        """angle = rep_start_angle + 0.1 leaves DOWN."""
        d = RepDetector(_make_exercise(start=15.0, end=80.0))
        _run_sequence(d, [15.0, 15.1])
        assert d.state == RepState.RISING


# ================================================================
# 11-15. PARTIAL MOVEMENTS
# ================================================================


class TestPartialMovements:
    """Partial movements do not count."""

    def test_halfway_up_no_rep(self):
        """down → halfway → down = no rep."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 30, 5])
        assert d.rep_count == 0

    def test_halfway_repeated_no_rep(self):
        """down → halfway → halfway → down = no rep."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 40, 50, 5])
        assert d.rep_count == 0

    def test_top_then_halfway_no_completed_rep(self):
        """down → top → halfway = no completed rep yet."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 50])
        assert d.rep_count == 0
        assert d.state == RepState.FALLING

    def test_top_halfway_top_down_counts_one(self):
        """down → top → halfway → top → down = 1 rep."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 50, 85, 5])
        assert d.rep_count == 1

    def test_partial_does_not_corrupt_next_rep(self):
        """A partial movement does not break the next complete rep."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 30, 5, 85, 5])
        assert d.rep_count == 1


# ================================================================
# 16-18. SLOW MOVEMENTS
# ================================================================


class TestSlowMovements:
    """Slow multi-frame movements."""

    def test_slow_rep_counts_once(self):
        """A slow multi-frame repetition counts once."""
        d = RepDetector(_make_exercise(start=10.0, end=80.0))
        angles = [5, 8, 12, 20, 35, 50, 65, 80, 85,
                  80, 65, 50, 30, 15, 10, 8, 5]
        events = _run_sequence(d, angles)
        assert d.rep_count == 1
        assert len(events) == 1

    def test_holding_at_top_no_multiple_reps(self):
        """Holding at the top does not create multiple reps."""
        d = RepDetector(_make_exercise())
        angles = [5, 85, 85, 85, 85, 85, 85, 5]
        events = _run_sequence(d, angles)
        assert d.rep_count == 1

    def test_holding_at_bottom_after_completion_no_extra(self):
        """Holding at the bottom after completion doesn't add reps."""
        d = RepDetector(_make_exercise())
        angles = [5, 85, 5, 5, 5, 5, 5, 5]
        events = _run_sequence(d, angles)
        assert d.rep_count == 1


# ================================================================
# 19-21. FAST MOVEMENTS
# ================================================================


class TestFastMovements:
    """Fast threshold-jumping samples."""

    def test_direct_jump_start_to_end(self):
        """5° → 95° handled (fast rise across both thresholds)."""
        d = RepDetector(_make_exercise(start=10.0, end=80.0))
        events = _run_sequence(d, [5, 95, 5])
        assert d.rep_count == 1

    def test_jump_top_to_below_start_completes(self):
        """Direct jump from top to below start completes if top was reached."""
        d = RepDetector(_make_exercise(start=10.0, end=80.0))
        events = _run_sequence(d, [5, 85, 5])
        assert d.rep_count == 1

    def test_five_fast_reps(self):
        """Five fast repetitions count exactly 5."""
        d = RepDetector(_make_exercise(start=10.0, end=80.0))
        angles = [5, 95, 5, 95, 5, 95, 5, 95, 5, 95, 5]
        events = _run_sequence(d, angles)
        assert d.rep_count == 5
        assert len(events) == 5


# ================================================================
# 22-25. INITIAL STATE
# ================================================================


class TestInitialState:
    """Start-position behavior."""

    def test_starting_down_allows_rep(self):
        """Starting at down position, complete rep is counted."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 5])
        assert d.rep_count == 1

    def test_starting_halfway_no_rep(self):
        """Starting at 40° → WAITING_FOR_START, no rep on return."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [40, 85, 40])
        assert d.rep_count == 0

    def test_starting_at_top_no_rep(self):
        """Starting at 90° → WAITING_FOR_START, no rep."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [90])
        assert d.state == RepState.WAITING_FOR_START
        assert d.rep_count == 0

    def test_returning_down_after_invalid_start_no_count(self):
        """Start at 90° → come down → no false rep."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [90, 50, 5])
        assert d.rep_count == 0
        # But now in DOWN, so next complete rep works.
        assert d.state == RepState.DOWN
        events = _run_sequence(d, [85, 5])
        # Continuing sequence: frame_index doesn't matter here.
        assert d.rep_count == 1


# ================================================================
# 26-32. MISSING DATA
# ================================================================


class TestMissingData:
    """None angle handling."""

    def test_none_not_counted_as_down(self):
        """None does not transition to DOWN."""
        d = RepDetector(_make_exercise())
        _run_sequence(d, [None, None, None])
        assert d.state == RepState.WAITING_FOR_START
        assert d.rep_count == 0

    def test_none_while_down_preserves_down(self):
        d = RepDetector(_make_exercise())
        _run_sequence(d, [5])
        assert d.state == RepState.DOWN
        _run_sequence(d, [None, None])
        assert d.state == RepState.DOWN

    def test_none_while_rising_preserves_rising(self):
        d = RepDetector(_make_exercise())
        _run_sequence(d, [5, 50])
        assert d.state == RepState.RISING
        _run_sequence(d, [None, None])
        assert d.state == RepState.RISING

    def test_none_while_up_preserves_up(self):
        d = RepDetector(_make_exercise())
        _run_sequence(d, [5, 85])
        assert d.state == RepState.UP
        _run_sequence(d, [None, None])
        assert d.state == RepState.UP

    def test_none_while_falling_preserves_falling(self):
        d = RepDetector(_make_exercise())
        _run_sequence(d, [5, 85, 50])
        assert d.state == RepState.FALLING
        _run_sequence(d, [None, None])
        assert d.state == RepState.FALLING

    def test_none_does_not_create_fake_boundaries(self):
        """None frames should not create events."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, None, None, None, 85, None, 5])
        assert d.rep_count == 1
        assert len(events) == 1

    def test_valid_rep_with_missing_frames_still_counts(self):
        """A valid rep with interspersed None still counts once."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, None, 50, None, 85, None, 50, None, 5])
        assert d.rep_count == 1


# ================================================================
# 33-36. RESET
# ================================================================


class TestReset:
    """Reset behavior."""

    def test_reset_clears_count(self):
        d = RepDetector(_make_exercise())
        _run_sequence(d, [5, 85, 5])
        assert d.rep_count == 1
        d.reset()
        assert d.rep_count == 0

    def test_reset_returns_to_initial_state(self):
        d = RepDetector(_make_exercise())
        _run_sequence(d, [5, 85])
        assert d.state == RepState.UP
        d.reset()
        assert d.state == RepState.WAITING_FOR_START

    def test_reset_clears_pending_boundaries(self):
        """After reset, no stale boundaries leak into next rep."""
        d = RepDetector(_make_exercise())
        _run_sequence(d, [5, 85])  # In UP state, has start/top boundaries.
        d.reset()
        # Start fresh. Next rep should have clean boundaries.
        events = []
        for i, angle in enumerate([5, 85, 5]):
            event = d.update(angle, frame_index=100 + i)
            if event:
                events.append(event)
        assert len(events) == 1
        assert events[0].start_frame == 100  # Not a stale frame from before reset.

    def test_new_rep_after_reset(self):
        """A new repetition can be counted after reset."""
        d = RepDetector(_make_exercise())
        _run_sequence(d, [5, 85, 5])
        assert d.rep_count == 1
        d.reset()
        _run_sequence(d, [5, 85, 5])
        assert d.rep_count == 1  # Fresh count, not 2.


# ================================================================
# 37-43. BOUNDARIES / EVENTS
# ================================================================


class TestBoundariesEvents:
    """RepEvent structure and boundary correctness."""

    def test_completed_rep_emits_exactly_one_event(self):
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 5])
        assert len(events) == 1

    def test_event_has_correct_rep_number(self):
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 5, 85, 5])
        assert events[0].rep_number == 1
        assert events[1].rep_number == 2

    def test_event_has_start_boundary(self):
        d = RepDetector(_make_exercise())
        events = []
        for i, angle in enumerate([5, 85, 5]):
            event = d.update(angle, frame_index=i)
            if event:
                events.append(event)
        assert events[0].start_frame is not None

    def test_event_has_top_boundary(self):
        d = RepDetector(_make_exercise())
        events = []
        for i, angle in enumerate([5, 85, 5]):
            event = d.update(angle, frame_index=i)
            if event:
                events.append(event)
        assert events[0].top_frame is not None

    def test_event_has_end_boundary(self):
        d = RepDetector(_make_exercise())
        events = []
        for i, angle in enumerate([5, 85, 5]):
            event = d.update(angle, frame_index=i)
            if event:
                events.append(event)
        assert events[0].end_frame is not None

    def test_no_event_on_incomplete_movement(self):
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 30, 5])
        assert len(events) == 0

    def test_boundaries_ordered_correctly(self):
        """start < top < end."""
        d = RepDetector(_make_exercise())
        events = []
        for i, angle in enumerate([5, 50, 85, 50, 5]):
            event = d.update(angle, frame_index=i)
            if event:
                events.append(event)
        assert len(events) == 1
        e = events[0]
        assert e.start_frame < e.top_frame < e.end_frame


# ================================================================
# 44-46. REUSABLE EXERCISE THRESHOLDS
# ================================================================


class TestReusableThresholds:
    """Detector uses exercise thresholds, not hardcoded values."""

    def test_custom_start_angle(self):
        """Changing rep_start_angle changes start behavior."""
        d = RepDetector(_make_exercise(start=30.0, end=80.0))
        # 20° is below new start threshold → DOWN.
        _run_sequence(d, [20])
        assert d.state == RepState.DOWN
        # With default start=15, angle 20 would be RISING.

    def test_custom_end_angle(self):
        """Changing rep_end_angle changes top behavior."""
        d = RepDetector(_make_exercise(start=15.0, end=60.0))
        # 65° is above new end threshold → UP.
        _run_sequence(d, [10, 65])
        assert d.state == RepState.UP
        # With default end=80, angle 65 would be RISING.

    def test_no_hardcoded_shoulder_values(self):
        """Detector works with non-shoulder thresholds."""
        d = RepDetector(_make_exercise(start=5.0, end=150.0))
        events = _run_sequence(d, [3, 160, 3])
        assert d.rep_count == 1


# ================================================================
# ADDITIONAL EDGE CASES
# ================================================================


class TestEdgeCases:
    """Additional edge case coverage."""

    def test_initial_state_is_waiting(self):
        d = RepDetector(_make_exercise())
        assert d.state == RepState.WAITING_FOR_START
        assert d.rep_count == 0

    def test_down_top_down_top_down_is_two_reps(self):
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 5, 85, 5])
        assert d.rep_count == 2
        assert len(events) == 2

    def test_falling_reversal_back_to_up(self):
        """FALLING → angle >= end → back to UP."""
        d = RepDetector(_make_exercise())
        _run_sequence(d, [5, 85, 50])
        assert d.state == RepState.FALLING
        _run_sequence(d, [85])
        assert d.state == RepState.UP

    def test_falling_reversal_then_complete(self):
        """FALLING → UP → FALLING → DOWN = 1 rep."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 50, 85, 5])
        assert d.rep_count == 1

    def test_rep_event_is_frozen_dataclass(self):
        """RepEvent should be immutable."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 85, 5])
        with pytest.raises(AttributeError):
            events[0].rep_number = 999

    def test_timestamp_stored_when_provided(self):
        """Timestamps are stored in the event when provided."""
        d = RepDetector(_make_exercise())
        events = []
        timestamps = [0, 100, 200]
        for i, (angle, ts) in enumerate(zip([5, 85, 5], timestamps)):
            event = d.update(angle, frame_index=i, timestamp_ms=ts)
            if event:
                events.append(event)
        assert len(events) == 1
        assert events[0].start_timestamp_ms is not None
        assert events[0].top_timestamp_ms is not None
        assert events[0].end_timestamp_ms is not None

    def test_uses_real_shoulder_abduction_definition(self):
        """Works with the actual SHOULDER_ABDUCTION definition."""
        from src.exercises.shoulder_abduction import SHOULDER_ABDUCTION
        d = RepDetector(SHOULDER_ABDUCTION)
        # SHOULDER_ABDUCTION: start=15, end=80
        events = _run_sequence(d, [10, 85, 10])
        assert d.rep_count == 1
        assert len(events) == 1

    def test_multiple_partials_then_complete(self):
        """Multiple partials followed by a complete rep."""
        d = RepDetector(_make_exercise())
        events = _run_sequence(d, [5, 30, 5, 40, 5, 50, 5, 85, 5])
        assert d.rep_count == 1
        assert len(events) == 1

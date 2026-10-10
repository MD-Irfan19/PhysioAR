"""Repetition detection state machine for PhysioAR.

Identifies complete exercise repetitions using a deterministic
state machine over the exercise's tracked joint angle.

A complete repetition requires:

    DOWN → RISING → UP → FALLING → DOWN (rep complete)

The thresholds are read from the active ExerciseDefinition:

    exercise.rep_start_angle  — defines DOWN (angle <= threshold)
    exercise.rep_end_angle    — defines UP   (angle >= threshold)

This module is independent of OpenCV, MediaPipe, camera capture,
and drawing code. It consumes only an angle value per frame.

Phase 5 — Rep Detection (Segmentation).

============================================================
STATE MACHINE
============================================================

    WAITING_FOR_START
        Initial state. Waits for a valid start/down position.
        Prevents counting a rep from a mid-air starting pose.

    DOWN
        Angle is at or below rep_start_angle.
        A rep may begin from here.

    RISING
        Angle has left the start position but not yet reached
        the end threshold.

    UP
        Angle has reached or exceeded rep_end_angle.
        The user is at the top of the movement.

    FALLING
        Angle has left the top position and is returning
        toward the start position.

============================================================
BOUNDARY CONVENTIONS
============================================================

    angle <= rep_start_angle   → DOWN threshold reached
    angle >= rep_end_angle     → UP threshold reached

A repetition is counted exactly once when FALLING → DOWN
transition occurs (angle returns to start after reaching top).

============================================================
MISSING-ANGLE HANDLING
============================================================

When angle is None:
    - State is held (no transition)
    - No rep is counted
    - No fake boundaries are created
    - None is NOT treated as 0

============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.exercises.base import ExerciseDefinition, RepState

@dataclass(frozen=True)
class RepEvent:
    """Record of a completed repetition.

    Retains boundary information for future quality-scoring modules.

    Attributes:
        rep_number: The 1-based repetition number.
        start_frame: Frame index when the rep started (DOWN entered),
            or None if frame indices are not tracked.
        top_frame: Frame index when the top was reached (UP entered),
            or None.
        end_frame: Frame index when the rep completed (returned to
            DOWN), or None.
        start_timestamp_ms: Timestamp in milliseconds when the rep
            started, or None if timestamps are not tracked.
        top_timestamp_ms: Timestamp when the top was reached, or None.
        end_timestamp_ms: Timestamp when the rep completed, or None.
    """

    rep_number: int
    start_frame: Optional[int]
    top_frame: Optional[int]
    end_frame: Optional[int]
    start_timestamp_ms: Optional[int]
    top_timestamp_ms: Optional[int]
    end_timestamp_ms: Optional[int]


class RepDetector:
    """Deterministic repetition detection state machine.

    Consumes one angle value per frame and detects complete
    exercise repetitions using thresholds from the active
    ExerciseDefinition.

    A complete repetition requires:
        DOWN → (angle rises above start) → RISING
        RISING → (angle reaches end) → UP
        UP → (angle falls below end) → FALLING
        FALLING → (angle returns to start) → DOWN + RepEvent

    Usage::

        detector = RepDetector(exercise_definition)

        for each frame:
            event = detector.update(angle, frame_index, timestamp_ms)
            if event is not None:
                print(f"Rep {event.rep_number} complete!")

    Attributes:
        state: Current RepState (read-only for UI/tests).
        rep_count: Total completed repetitions (read-only).
    """

    def __init__(self, exercise: ExerciseDefinition) -> None:
        """Initialize the detector with exercise thresholds.

        Args:
            exercise: The active ExerciseDefinition. Thresholds
                are read from exercise.rep_start_angle and
                exercise.rep_end_angle.
        """
        self._start_threshold = exercise.rep_start_angle
        self._end_threshold = exercise.rep_end_angle
        self._state = RepState.WAITING_FOR_START
        self._rep_count = 0

        # Active rep boundary tracking.
        self._rep_start_frame: Optional[int] = None
        self._rep_start_ts: Optional[int] = None
        self._rep_top_frame: Optional[int] = None
        self._rep_top_ts: Optional[int] = None

    @property
    def state(self) -> RepState:
        """Current state of the state machine."""
        return self._state

    @property
    def rep_count(self) -> int:
        """Total number of completed repetitions."""
        return self._rep_count

    def reset(self) -> None:
        """Reset the detector to its initial state.

        Clears:
            - current state (→ WAITING_FOR_START)
            - repetition count (→ 0)
            - active repetition boundaries

        After reset, the detector behaves like a newly created
        detector with the same exercise thresholds.
        """
        self._state = RepState.WAITING_FOR_START
        self._rep_count = 0
        self._rep_start_frame = None
        self._rep_start_ts = None
        self._rep_top_frame = None
        self._rep_top_ts = None

    def update(
        self,
        angle: Optional[float],
        frame_index: Optional[int] = None,
        timestamp_ms: Optional[int] = None,
    ) -> Optional[RepEvent]:
        """Process one frame's angle and return a RepEvent if a rep completes.

        Args:
            angle: The current exercise angle in degrees, or None
                if the angle is unavailable (e.g., landmarks missing).
            frame_index: Optional frame counter for boundary tracking.
            timestamp_ms: Optional timestamp in milliseconds.

        Returns:
            A RepEvent if a repetition was completed on this frame,
            or None otherwise. A RepEvent is emitted exactly once
            per completed repetition.
        """
        # Missing angle: hold current state, no transitions.
        if angle is None:
            return None

        if self._state == RepState.WAITING_FOR_START:
            return self._handle_waiting(angle, frame_index, timestamp_ms)
        elif self._state == RepState.DOWN:
            return self._handle_down(angle, frame_index, timestamp_ms)
        elif self._state == RepState.RISING:
            return self._handle_rising(angle, frame_index, timestamp_ms)
        elif self._state == RepState.UP:
            return self._handle_up(angle, frame_index, timestamp_ms)
        elif self._state == RepState.FALLING:
            return self._handle_falling(angle, frame_index, timestamp_ms)

        return None  # Should never reach here.

    # ============================================================
    # State handlers
    # ============================================================

    def _handle_waiting(
        self, angle: float, frame_index: Optional[int],
        timestamp_ms: Optional[int],
    ) -> None:
        """WAITING_FOR_START: wait for a valid start/down position.

        angle <= start_threshold → DOWN
        angle > start_threshold  → remain WAITING_FOR_START
        """
        if angle <= self._start_threshold:
            self._state = RepState.DOWN
            self._rep_start_frame = frame_index
            self._rep_start_ts = timestamp_ms
        return None

    def _handle_down(
        self, angle: float, frame_index: Optional[int],
        timestamp_ms: Optional[int],
    ) -> None:
        """DOWN: arm is at the start position.

        angle <= start_threshold → remain DOWN (update start boundary)
        angle > start_threshold and angle < end_threshold → RISING
        angle >= end_threshold → UP (fast jump over RISING)
        """
        if angle <= self._start_threshold:
            # Update start boundary to the latest DOWN frame.
            self._rep_start_frame = frame_index
            self._rep_start_ts = timestamp_ms
            return None

        if angle >= self._end_threshold:
            # Fast jump: skip RISING, go directly to UP.
            self._rep_top_frame = frame_index
            self._rep_top_ts = timestamp_ms
            self._state = RepState.UP
            return None

        # angle > start_threshold and < end_threshold → RISING.
        self._state = RepState.RISING
        return None

    def _handle_rising(
        self, angle: float, frame_index: Optional[int],
        timestamp_ms: Optional[int],
    ) -> None:
        """RISING: moving from start toward top.

        angle >= end_threshold → UP
        angle <= start_threshold → DOWN (partial movement, abort)
        otherwise → remain RISING
        """
        if angle >= self._end_threshold:
            self._rep_top_frame = frame_index
            self._rep_top_ts = timestamp_ms
            self._state = RepState.UP
        elif angle <= self._start_threshold:
            # Partial movement: returned to start without reaching top.
            self._state = RepState.DOWN
            self._rep_start_frame = frame_index
            self._rep_start_ts = timestamp_ms
            self._rep_top_frame = None
            self._rep_top_ts = None
        return None

    def _handle_up(
        self, angle: float, frame_index: Optional[int],
        timestamp_ms: Optional[int],
    ) -> Optional[RepEvent]:
        """UP: arm has reached the top position.

        angle >= end_threshold → remain UP
        angle < end_threshold and angle <= start_threshold → fast fall,
            complete rep immediately (skip FALLING)
        angle < end_threshold and angle > start_threshold → FALLING
        """
        if angle >= self._end_threshold:
            return None

        # Angle fell below end threshold.
        if angle <= self._start_threshold:
            # Fast fall: jumped from UP directly to below start.
            # Complete the rep immediately (skip FALLING state).
            self._rep_count += 1
            event = RepEvent(
                rep_number=self._rep_count,
                start_frame=self._rep_start_frame,
                top_frame=self._rep_top_frame,
                end_frame=frame_index,
                start_timestamp_ms=self._rep_start_ts,
                top_timestamp_ms=self._rep_top_ts,
                end_timestamp_ms=timestamp_ms,
            )
            # Transition to DOWN for the next rep.
            self._state = RepState.DOWN
            self._rep_start_frame = frame_index
            self._rep_start_ts = timestamp_ms
            self._rep_top_frame = None
            self._rep_top_ts = None
            return event

        # Normal transition to FALLING.
        self._state = RepState.FALLING
        return None

    def _handle_falling(
        self, angle: float, frame_index: Optional[int],
        timestamp_ms: Optional[int],
    ) -> Optional[RepEvent]:
        """FALLING: returning from top toward start.

        angle <= start_threshold → REP COMPLETE → DOWN
        angle >= end_threshold → back to UP (reversal)
        otherwise → remain FALLING
        """
        if angle <= self._start_threshold:
            # Repetition complete!
            self._rep_count += 1
            event = RepEvent(
                rep_number=self._rep_count,
                start_frame=self._rep_start_frame,
                top_frame=self._rep_top_frame,
                end_frame=frame_index,
                start_timestamp_ms=self._rep_start_ts,
                top_timestamp_ms=self._rep_top_ts,
                end_timestamp_ms=timestamp_ms,
            )
            # Transition to DOWN for the next rep.
            self._state = RepState.DOWN
            self._rep_start_frame = frame_index
            self._rep_start_ts = timestamp_ms
            self._rep_top_frame = None
            self._rep_top_ts = None
            return event

        if angle >= self._end_threshold:
            # Reversal: went back up before reaching start.
            self._state = RepState.UP
        return None

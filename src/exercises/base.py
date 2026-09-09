"""Exercise definition base module for PhysioAR.

Provides the reusable exercise-definition architecture:

- ``CameraOrientation``: Enum specifying required camera view (FRONT, SIDE).
- ``ExerciseDefinition``: Generic dataclass containing exercise configuration
  and angle-calculation strategy.

The design allows new exercises to be added by creating new
ExerciseDefinition instances without modifying the main application
pipeline.

Phase 3 — Exercise Definition Structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class CameraOrientation(Enum):
    """Required camera orientation for an exercise.

    Attributes:
        FRONT: Camera facing the user head-on (e.g., shoulder abduction).
        SIDE: Camera viewing the user from the side (e.g., bicep curl).
    """

    FRONT = "front"
    SIDE = "side"


@dataclass
class ExerciseDefinition:
    """Generic exercise configuration and calculation strategy.

    Contains all metadata, landmark requirements, and the angle
    calculation callable for a single exercise type.

    The ``angle_calculator`` field holds a callable with the signature::

        angle_calculator(smoothed_landmarks: list, side: str) -> Optional[float]

    where:
      - ``smoothed_landmarks`` is the list of SmoothedLandmark objects
        from PoseResult.smoothed_landmarks.
      - ``side`` is ``"left"`` or ``"right"`` indicating which body
        side to calculate for.
      - Returns the computed angle in degrees, or None if required
        landmarks are unavailable or geometry is degenerate.

    This allows the main application to call::

        angle = exercise.angle_calculator(smoothed_landmarks, side)

    without knowing the mathematical details of any specific exercise.

    Attributes:
        name: Human-readable exercise name (e.g., "Shoulder Abduction").
        camera_orientation: Required camera view for this exercise.
        target_joint: Primary joint being exercised (e.g., "shoulder").
        rom_target: Target range-of-motion in degrees.
        rep_start_angle: Angle threshold defining the start of a rep.
        rep_end_angle: Angle threshold defining the end of a rep.
        angle_calculator: Callable that computes the exercise angle
            from smoothed landmarks and a side selector.
        compensation_checks: List of compensation check identifiers
            (populated in future phases).
        feedback_templates: Dict of feedback template strings
            (populated in future phases).
        expected_landmarks: List of landmark names required by this
            exercise (e.g., ["left_shoulder", "left_elbow", "left_wrist"]).
    """

    name: str
    camera_orientation: CameraOrientation
    target_joint: str
    rom_target: float
    rep_start_angle: float
    rep_end_angle: float
    angle_calculator: Callable
    compensation_checks: list[str] = field(default_factory=list)
    feedback_templates: dict = field(default_factory=dict)
    expected_landmarks: list[str] = field(default_factory=list)

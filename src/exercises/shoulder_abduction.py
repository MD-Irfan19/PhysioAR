"""Shoulder Abduction exercise definition for PhysioAR.

Provides the concrete ExerciseDefinition for shoulder abduction and
the angle-calculation function.

Shoulder abduction angle is measured at the shoulder joint between
the arm direction (shoulder → elbow) and a vertical reference
(shoulder → straight down).

In MediaPipe image coordinates:
    x: 0 → 1, left → right
    y: 0 → 1, top → bottom

Therefore "straight down" from the shoulder is (shoulder_x, shoulder_y + 1).

The angle is computed using geometry.calculate_angle() with:
    point_a = elbow       (arm direction)
    point_b = shoulder    (vertex)
    point_c = vertical_ref (downward reference)

This produces:
    arm hanging down  ≈  0°
    arm at 45°        ≈ 45°
    arm horizontal    ≈ 90°

Phase 3 — Shoulder Abduction Implementation.
"""

from __future__ import annotations

from typing import Optional

from src.config import LANDMARK_VISIBILITY_THRESHOLD
from src.exercises.base import CameraOrientation, ExerciseDefinition
from src.utils.geometry import calculate_angle


# ============================================================
# MediaPipe landmark indices
# ============================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# Mapping from side string to (shoulder_idx, elbow_idx, wrist_idx).
_SIDE_LANDMARKS = {
    "left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}


def calculate_shoulder_abduction_angle(
    smoothed_landmarks: list,
    side: str = "left",
    visibility_threshold: float | None = None,
) -> Optional[float]:
    """Calculate the shoulder abduction angle for the specified arm.

    Uses the smoothed landmark coordinates from the Phase 1.5 EMA
    pipeline. Does NOT perform additional smoothing.

    The angle is computed at the shoulder joint between:
      - The arm vector: shoulder → elbow
      - The vertical reference: shoulder → (shoulder_x, shoulder_y + 1)
        which points straight down in MediaPipe image coordinates.

    This produces approximately:
      - 0° when the arm hangs vertically downward
      - 45° at mid-range abduction
      - 90° when the arm is extended horizontally

    Visibility gating uses the raw MediaPipe visibility value
    (SmoothedLandmark.visibility, NOT smoothed by EMA) to ensure
    that only reliable landmarks are used. If any required landmark
    is missing or below the visibility threshold, returns None.

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects from
            PoseResult.smoothed_landmarks.
        side: ``"left"`` or ``"right"`` indicating which arm.
        visibility_threshold: Minimum raw visibility for a landmark
            to be considered reliable. Defaults to
            LANDMARK_VISIBILITY_THRESHOLD from config.

    Returns:
        The shoulder abduction angle in degrees [0, 180], or None
        if required landmarks are unavailable, below visibility
        threshold, or geometry is degenerate.

    Raises:
        ValueError: If ``side`` is not ``"left"`` or ``"right"``.
    """
    if side not in _SIDE_LANDMARKS:
        raise ValueError(
            f"Invalid side '{side}': must be 'left' or 'right'."
        )

    if visibility_threshold is None:
        visibility_threshold = LANDMARK_VISIBILITY_THRESHOLD

    shoulder_idx, elbow_idx, _wrist_idx = _SIDE_LANDMARKS[side]
    landmark_count = len(smoothed_landmarks)

    # Check that required landmarks exist.
    if shoulder_idx >= landmark_count or elbow_idx >= landmark_count:
        return None

    shoulder = smoothed_landmarks[shoulder_idx]
    elbow = smoothed_landmarks[elbow_idx]

    # Visibility gating: use raw MediaPipe visibility (NOT smoothed).
    if shoulder.visibility < visibility_threshold:
        return None
    if elbow.visibility < visibility_threshold:
        return None

    # Construct points using smoothed coordinates.
    shoulder_xy = (shoulder.x, shoulder.y)
    elbow_xy = (elbow.x, elbow.y)

    # Vertical reference: straight down from shoulder.
    # In MediaPipe image coords, y increases downward,
    # so "down" = (shoulder_x, shoulder_y + 1).
    vertical_ref = (shoulder.x, shoulder.y + 1.0)

    # Calculate angle at the shoulder between the arm and vertical down.
    # calculate_angle(point_a, point_b, point_c) measures angle at point_b.
    try:
        angle = calculate_angle(elbow_xy, shoulder_xy, vertical_ref)
    except ValueError:
        # Degenerate geometry (e.g., elbow coincident with shoulder).
        return None

    return angle


# ============================================================
# Exercise definition instance
# ============================================================

SHOULDER_ABDUCTION = ExerciseDefinition(
    name="Shoulder Abduction",
    camera_orientation=CameraOrientation.FRONT,
    target_joint="shoulder",
    rom_target=90.0,
    rep_start_angle=15.0,
    rep_end_angle=80.0,
    angle_calculator=calculate_shoulder_abduction_angle,
    compensation_checks=[],
    feedback_templates={},
    expected_landmarks=[
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
    ],
)

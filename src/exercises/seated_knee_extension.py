"""Seated Knee Extension exercise definition for PhysioAR.

Provides the concrete ExerciseDefinition for seated knee extension and
the angle-calculation function.

Seated knee extension angle is measured at the knee joint between
the thigh (hip → knee) and the lower leg (knee → ankle).

In MediaPipe image coordinates, using geometry.calculate_angle():
    point_a = hip
    point_b = knee      (vertex)
    point_c = ankle

This produces:
    knee flexed (seated)     ≈  90°
    knee fully extended      ≈ 180°

Phase 10 — Generalize to a Second Exercise.
"""

from __future__ import annotations

from typing import Optional

from src.config import LANDMARK_VISIBILITY_THRESHOLD
from src.exercises.base import CameraOrientation, ExerciseDefinition
from src.utils.geometry import calculate_angle


# ============================================================
# MediaPipe landmark indices
# ============================================================

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Mapping from side string to (hip_idx, knee_idx, ankle_idx).
_SIDE_LANDMARKS = {
    "left": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "right": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
}


def calculate_seated_knee_extension_angle(
    smoothed_landmarks: list,
    side: str = "left",
    visibility_threshold: float | None = None,
) -> Optional[float]:
    """Calculate the knee extension angle for the specified leg.

    Uses the smoothed landmark coordinates from the Phase 1.5 EMA
    pipeline. Does NOT perform additional smoothing.

    The angle is computed at the knee joint between:
      - The thigh vector: knee → hip
      - The lower leg vector: knee → ankle

    This produces approximately:
      - 90° when the knee is flexed in a seated position.
      - 180° when the leg is fully extended horizontally.

    Visibility gating uses the raw MediaPipe visibility value
    (SmoothedLandmark.visibility, NOT smoothed by EMA) to ensure
    that only reliable landmarks are used. If any required landmark
    is missing or below the visibility threshold, returns None.

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects from
            PoseResult.smoothed_landmarks.
        side: ``"left"`` or ``"right"`` indicating which leg.
        visibility_threshold: Minimum raw visibility for a landmark
            to be considered reliable. Defaults to
            LANDMARK_VISIBILITY_THRESHOLD from config.

    Returns:
        The knee angle in degrees [0, 180], or None
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

    hip_idx, knee_idx, ankle_idx = _SIDE_LANDMARKS[side]
    landmark_count = len(smoothed_landmarks)

    # Check that required landmarks exist.
    if (hip_idx >= landmark_count or 
        knee_idx >= landmark_count or 
        ankle_idx >= landmark_count):
        return None

    hip = smoothed_landmarks[hip_idx]
    knee = smoothed_landmarks[knee_idx]
    ankle = smoothed_landmarks[ankle_idx]

    # Visibility gating: use raw MediaPipe visibility (NOT smoothed).
    if hip.visibility < visibility_threshold:
        return None
    if knee.visibility < visibility_threshold:
        return None
    if ankle.visibility < visibility_threshold:
        return None

    # Construct points using smoothed coordinates.
    hip_xy = (hip.x, hip.y)
    knee_xy = (knee.x, knee.y)
    ankle_xy = (ankle.x, ankle.y)

    # Calculate angle at the knee between hip and ankle.
    # calculate_angle(point_a, point_b, point_c) measures angle at point_b.
    try:
        angle = calculate_angle(hip_xy, knee_xy, ankle_xy)
    except ValueError:
        # Degenerate geometry (e.g., knee coincident with hip or ankle).
        return None

    return angle


# ============================================================
# Exercise definition instance
# ============================================================

SEATED_KNEE_EXTENSION = ExerciseDefinition(
    name="Seated Knee Extension",
    camera_orientation=CameraOrientation.SIDE,
    target_joint="knee",
    rom_target=150.0,
    rep_start_angle=90.0,
    rep_end_angle=140.0,
    angle_calculator=calculate_seated_knee_extension_angle,
    compensation_checks=["torso_lean", "neck_tilt", "hip_rotation", "lateral_trunk_lean"],
    feedback_templates={
        "torso_lean": (
            "Torso Lean = {value}° — keep your trunk stable while"
            " extending the knee."
        ),
        "neck_tilt": (
            "Neck Tilt = {value}° — keep your head aligned with"
            " your torso while seated."
        ),
        "hip_rotation": (
            "Hip Rotation = {value}° — keep your hips aligned and avoid rotating your pelvis during knee extension."
        ),
        "lateral_trunk_lean": (
            "Lateral Trunk Lean = {value}° — keep your torso upright and avoid leaning to the side during knee extension."
        ),
    },
    expected_landmarks=[
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
    ],
)

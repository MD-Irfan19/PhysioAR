"""Phase 13 — Sit to Stand Exercise Definition.

Defines the Sit to Stand exercise for the PhysioAR generic pipeline.
"""

from __future__ import annotations

from typing import Optional
from src.exercises.base import ExerciseDefinition, CameraOrientation
from src.utils.geometry import calculate_angle

# MediaPipe Pose Landmark Indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def calculate_knee_flexion_angle(
    smoothed_landmarks: list,
    side: str = "right",
    visibility_threshold: float = 0.5,
) -> Optional[float]:
    """Calculate the interior knee angle from smoothed landmarks.

    Uses the project's standard 3-point angle calculation at the knee:
    (hip, knee, ankle).

    A standing posture has an interior angle near 180°.
    A seated posture has an interior angle near 90°.

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects.
        side: "left" or "right" leg.
        visibility_threshold: Minimum required confidence.

    Returns:
        The knee angle in degrees, or None if landmarks are missing,
        low-confidence, or geometry is degenerate.
    """
    if side == "left":
        hip_idx = LEFT_HIP
        knee_idx = LEFT_KNEE
        ankle_idx = LEFT_ANKLE
    elif side == "right":
        hip_idx = RIGHT_HIP
        knee_idx = RIGHT_KNEE
        ankle_idx = RIGHT_ANKLE
    else:
        return None

    if max(hip_idx, knee_idx, ankle_idx) >= len(smoothed_landmarks):
        return None

    hp = smoothed_landmarks[hip_idx]
    kn = smoothed_landmarks[knee_idx]
    an = smoothed_landmarks[ankle_idx]

    if (
        hp.visibility < visibility_threshold
        or kn.visibility < visibility_threshold
        or an.visibility < visibility_threshold
    ):
        return None

    hp_xy = (hp.x, hp.y)
    kn_xy = (kn.x, kn.y)
    an_xy = (an.x, an.y)

    try:
        interior_angle = calculate_angle(hp_xy, kn_xy, an_xy)
        return interior_angle
    except ValueError:
        # Degenerate geometry
        return None


SIT_TO_STAND = ExerciseDefinition(
    name="Sit to Stand",
    camera_orientation=CameraOrientation.SIDE,
    target_joint="knee",
    angle_calculator=calculate_knee_flexion_angle,
    rep_start_angle=105.0,  # <= 105 is seated (start)
    rep_end_angle=155.0,    # >= 155 is standing (end)
    rom_target=90.0,        # Excursion from ~90 to ~180 is 90 degrees
    expected_landmarks=[
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_hip",
        "right_knee",
        "right_ankle",
    ],
    compensation_checks=[],
    feedback_templates={},
)

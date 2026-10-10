"""Phase 12 — Elbow Flexion-Extension Exercise Definition.

Defines the Elbow Flexion-Extension exercise for the PhysioAR generic pipeline.
"""

from __future__ import annotations

from typing import Optional
from src.exercises.base import ExerciseDefinition, CameraOrientation
from src.utils.geometry import calculate_angle

# MediaPipe Pose Landmark Indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


def calculate_elbow_flexion_extension_angle(
    smoothed_landmarks: list,
    side: str = "right",
    visibility_threshold: float = 0.5,
) -> Optional[float]:
    """Calculate the elbow flexion angle from smoothed landmarks.

    Uses the project's standard 3-point angle calculation at the elbow:
    (shoulder, elbow, wrist).

    Applies the transformed angle convention to suit the generic rep detector:
        flexion_angle = 180 - interior_angle

    This means:
        - Straight arm (interior 180°) -> 0°
        - Right angle (interior 90°) -> 90°
        - Full flexion (interior ~45°) -> 135°

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects.
        side: "left" or "right" arm.
        visibility_threshold: Minimum required confidence.

    Returns:
        The flexion angle in degrees (0 = straight, >0 = flexed),
        or None if landmarks are missing, low-confidence, or geometry
        is degenerate.
    """
    if side == "left":
        sh_idx = LEFT_SHOULDER
        el_idx = LEFT_ELBOW
        wr_idx = LEFT_WRIST
    elif side == "right":
        sh_idx = RIGHT_SHOULDER
        el_idx = RIGHT_ELBOW
        wr_idx = RIGHT_WRIST
    else:
        return None

    if max(sh_idx, el_idx, wr_idx) >= len(smoothed_landmarks):
        return None

    sh = smoothed_landmarks[sh_idx]
    el = smoothed_landmarks[el_idx]
    wr = smoothed_landmarks[wr_idx]

    if (
        sh.visibility < visibility_threshold
        or el.visibility < visibility_threshold
        or wr.visibility < visibility_threshold
    ):
        return None

    sh_xy = (sh.x, sh.y)
    el_xy = (el.x, el.y)
    wr_xy = (wr.x, wr.y)

    try:
        interior_angle = calculate_angle(sh_xy, el_xy, wr_xy)
        # Transformed convention:
        flexion_angle = 180.0 - interior_angle
        # Clamp near 0 to avoid -0.0 artifacts
        return max(0.0, flexion_angle)
    except ValueError:
        # Degenerate geometry (e.g. overlapping points)
        return None


ELBOW_FLEXION_EXTENSION = ExerciseDefinition(
    name="Elbow Flexion-Extension",
    camera_orientation=CameraOrientation.SIDE,
    target_joint="elbow",
    angle_calculator=calculate_elbow_flexion_extension_angle,
    rep_start_angle=15.0,
    rep_end_angle=120.0,
    rom_target=130.0,
    expected_landmarks=[
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ],
    compensation_checks=[
        "torso_lean",
        "shoulder_hike",
        "shoulder_substitution",
    ],
    feedback_templates={
        "torso_lean": "Trunk Lean = {value}° — keep your torso steady and avoid leaning forward or backward to complete the movement.",
        "shoulder_hike": "Shoulder Hike = {value} — keep your shoulder relaxed while bending and straightening your elbow.",
        "shoulder_substitution": "Shoulder Substitution = {value}° — keep your upper arm steady and bend your elbow without swinging your shoulder.",
    },
)

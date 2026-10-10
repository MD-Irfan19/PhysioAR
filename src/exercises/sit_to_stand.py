"""Phase 13 — Sit to Stand Exercise Definition.

Defines the Sit to Stand exercise for the PhysioAR generic pipeline.
"""

from __future__ import annotations

from typing import Optional
from src.exercises.base import ExerciseDefinition, CameraOrientation
from src.utils.geometry import calculate_angle

# MediaPipe Pose Landmark Indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
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


def compute_forward_trunk_lean(
    smoothed_landmarks: list,
    side: str,
    visibility_threshold: float,
) -> Optional[float]:
    """Calculate forward trunk inclination relative to vertical.

    Measures the angle between the vertical axis and the trunk
    (hip to shoulder) using the side-facing camera view.
    0 degrees = perfectly upright.
    Increasing values = leaning forward.

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects.
        side: "left" or "right".
        visibility_threshold: Minimum required confidence.

    Returns:
        The forward inclination angle in degrees, or None if invalid.
    """
    if side == "left":
        shoulder_idx = LEFT_SHOULDER
        hip_idx = LEFT_HIP
    elif side == "right":
        shoulder_idx = RIGHT_SHOULDER
        hip_idx = RIGHT_HIP
    else:
        return None

    if max(shoulder_idx, hip_idx) >= len(smoothed_landmarks):
        return None

    sh = smoothed_landmarks[shoulder_idx]
    hp = smoothed_landmarks[hip_idx]

    if sh.visibility < visibility_threshold or hp.visibility < visibility_threshold:
        return None

    # Vertex is at the hip.
    # Vertical reference point is directly above the hip (y is smaller in image coordinates).
    vertical_pt = (hp.x, hp.y - 1.0)
    hip_pt = (hp.x, hp.y)
    shoulder_pt = (sh.x, sh.y)

    try:
        # Angle between vertical (straight up) and the trunk (hip to shoulder)
        angle = calculate_angle(vertical_pt, hip_pt, shoulder_pt)
        return angle
    except ValueError:
        return None


from src.exercises.base import GenericCompensationCheck, CheckCondition, RepState

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
        "left_shoulder",
        "right_shoulder",
    ],
    compensation_checks=[],
    generic_checks=[
        GenericCompensationCheck(
            id="insufficient_forward_trunk_lean",
            condition=CheckCondition.BELOW,
            lower_threshold=25.0,  # Provisional engineering/MVP threshold for insufficient lean
            allowed_phases={RepState.RISING},
            metric_calculator=compute_forward_trunk_lean,
        )
    ],
    feedback_templates={
        "insufficient_forward_trunk_lean": "Lean forward more during the rising phase (current lean: {value})."
    },
)

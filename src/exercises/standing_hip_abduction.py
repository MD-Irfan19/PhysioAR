"""Standing Hip Abduction exercise definition.

Phase 11 — Add Third Exercise.
"""

from typing import Optional

from src.exercises.base import ExerciseDefinition, CameraOrientation
from src.utils.geometry import calculate_angle
from src.config import LANDMARK_VISIBILITY_THRESHOLD

# MediaPipe landmark indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


def calculate_hip_abduction_angle(
    landmarks: list,
    side: str = "left",
) -> Optional[float]:
    """Calculate the hip abduction angle for the given side.
    
    The angle is the lateral deviation of the thigh from the body's
    vertical axis. Leg hanging vertically ≈ 0°.
    
    Args:
        landmarks: List of SmoothedLandmark objects.
        side: "left" or "right".
        
    Returns:
        Angle in degrees, or None if landmarks are unavailable.
    """
    if side == "left":
        hip_idx = LEFT_HIP
        knee_idx = LEFT_KNEE
    elif side == "right":
        hip_idx = RIGHT_HIP
        knee_idx = RIGHT_KNEE
    else:
        return None

    if len(landmarks) <= max(hip_idx, knee_idx):
        return None

    hip = landmarks[hip_idx]
    knee = landmarks[knee_idx]

    if hip.visibility < LANDMARK_VISIBILITY_THRESHOLD or knee.visibility < LANDMARK_VISIBILITY_THRESHOLD:
        return None

    hip_xy = (hip.x, hip.y)
    knee_xy = (knee.x, knee.y)

    if hip_xy == knee_xy:
        return None

    # Vertical reference is always vertex + (0, -1)
    vertical_ref = (hip.x, hip.y - 1.0)

    try:
        angle_from_up = calculate_angle(vertical_ref, hip_xy, knee_xy)
        # Leg hanging down -> angle_from_up is 180. We want 0.
        abduction_angle = abs(180.0 - angle_from_up)
        return float(abduction_angle)
    except ValueError:
        return None


STANDING_HIP_ABDUCTION = ExerciseDefinition(
    name="Standing Hip Abduction",
    camera_orientation=CameraOrientation.FRONT,
    target_joint="hip",
    rom_target=45.0,  # Expected abduction ROM
    rep_start_angle=10.0,
    rep_end_angle=35.0,
    angle_calculator=calculate_hip_abduction_angle,
    compensation_checks=["trunk_lean", "hip_hike"],
    feedback_templates={
        "trunk_lean": (
            "Trunk Lean = {value}° — keep your torso upright and avoid leaning while abducting the leg."
        ),
        "hip_hike": (
            "Hip Hike = {value} — keep your hips level and avoid lifting one side of your pelvis during abduction."
        ),
    },
    expected_landmarks=[
        "left_hip", "left_knee", "right_hip", "right_knee"
    ],
)

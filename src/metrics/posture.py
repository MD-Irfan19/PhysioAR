"""Posture metrics computation for PhysioAR.

Computes three raw posture metrics from smoothed MediaPipe pose
landmarks:

1. **Torso lean** — deviation of the torso line from vertical (degrees).
2. **Shoulder height difference** — vertical displacement between
   left and right shoulders (normalized image units).
3. **Neck tilt** — deviation of the head/neck line from vertical (degrees).

This module is COMPUTE ONLY.
It does NOT perform compensation detection, thresholding, flagging,
quality scoring, feedback, or clinical interpretation.

Each metric is independently computed. A missing or unreliable
landmark for one metric does not prevent the other metrics from
being computed.

All functions use SMOOTHED landmark coordinates from the Phase 1.5
EMA pipeline. No additional smoothing is performed.

Visibility gating uses the raw MediaPipe visibility value
(SmoothedLandmark.visibility, NOT smoothed by EMA) against
LANDMARK_VISIBILITY_THRESHOLD from config.

Phase 4A — Posture Metrics (Compute Only).

============================================================
METRIC DEFINITIONS
============================================================

TORSO LEAN:
    Line: hip_midpoint → shoulder_midpoint
    Vertex: hip_midpoint
    Vertical reference: (hip_mid_x, hip_mid_y - 1)
    Result: angle in degrees (0° = upright, larger = more lean)

    Uses the same formula as calibration.compute_spine_angle().
    Delegates to that function to avoid duplicating math.

SHOULDER HEIGHT DIFFERENCE:
    Value: abs(left_shoulder.y - right_shoulder.y)
    Units: normalized image coordinates (0→1)
    Result: 0 = level, larger = more asymmetry

    Uses the same formula as calibration.compute_shoulder_height_difference().

NECK TILT:
    Line: shoulder_midpoint → nose
    Vertex: shoulder_midpoint
    Vertical reference: (sh_mid_x, sh_mid_y - 1)
    Result: angle in degrees (0° = upright, larger = more tilt)

    Uses the same formula as calibration.compute_neck_tilt().

============================================================
COORDINATE CONVENTION
============================================================

MediaPipe image coordinates:
    x: 0 → 1, left → right
    y: 0 → 1, top → bottom

"Up" in image space = (0, -1).

============================================================
LANDMARK INDICES
============================================================

    NOSE            = 0
    LEFT_SHOULDER   = 11
    RIGHT_SHOULDER  = 12
    LEFT_HIP        = 23
    RIGHT_HIP       = 24
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.calibration import (
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    compute_spine_angle,
    compute_shoulder_height_difference,
    compute_neck_tilt,
)
from src.config import LANDMARK_VISIBILITY_THRESHOLD


@dataclass
class PostureMetrics:
    """Frame-level posture metrics result.

    Each field is the computed raw metric value, or None if the
    required landmarks were unavailable or unreliable.

    A None value means the metric could not be computed for this
    frame. It does NOT mean the metric is zero or neutral.

    Attributes:
        torso_lean: Torso lean angle in degrees (0° = upright),
            or None if unavailable.
        shoulder_height_difference: Absolute vertical difference
            between left and right shoulders in normalized image
            units, or None if unavailable.
        neck_tilt: Neck tilt angle in degrees (0° = upright),
            or None if unavailable.
    """

    torso_lean: Optional[float]
    shoulder_height_difference: Optional[float]
    neck_tilt: Optional[float]


def _get_landmark_xy(
    smoothed_landmarks: list,
    index: int,
    visibility_threshold: float,
) -> Optional[tuple[float, float]]:
    """Extract smoothed (x, y) for a landmark if it passes visibility.

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects.
        index: MediaPipe landmark index.
        visibility_threshold: Minimum raw visibility required.

    Returns:
        (x, y) tuple of smoothed coordinates if the landmark exists
        and passes the visibility threshold, or None otherwise.
    """
    if index >= len(smoothed_landmarks):
        return None

    lm = smoothed_landmarks[index]

    if lm.visibility < visibility_threshold:
        return None

    return (lm.x, lm.y)


def compute_torso_lean(
    smoothed_landmarks: list,
    visibility_threshold: float | None = None,
) -> Optional[float]:
    """Compute torso lean from smoothed landmarks.

    Torso lean is the angle of the hip_midpoint → shoulder_midpoint
    line relative to vertical. 0° = upright, larger = more lean.

    Required landmarks: LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP,
    RIGHT_HIP.

    If any required landmark is missing or below the visibility
    threshold, returns None.

    Uses the same mathematical formula as
    calibration.compute_spine_angle().

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects from
            PoseResult.smoothed_landmarks.
        visibility_threshold: Minimum raw visibility. Defaults to
            LANDMARK_VISIBILITY_THRESHOLD from config.

    Returns:
        Torso lean angle in degrees, or None if unavailable.
    """
    if visibility_threshold is None:
        visibility_threshold = LANDMARK_VISIBILITY_THRESHOLD

    l_shoulder = _get_landmark_xy(
        smoothed_landmarks, LEFT_SHOULDER, visibility_threshold,
    )
    r_shoulder = _get_landmark_xy(
        smoothed_landmarks, RIGHT_SHOULDER, visibility_threshold,
    )
    l_hip = _get_landmark_xy(
        smoothed_landmarks, LEFT_HIP, visibility_threshold,
    )
    r_hip = _get_landmark_xy(
        smoothed_landmarks, RIGHT_HIP, visibility_threshold,
    )

    if any(p is None for p in (l_shoulder, r_shoulder, l_hip, r_hip)):
        return None

    try:
        return compute_spine_angle(l_shoulder, r_shoulder, l_hip, r_hip)
    except ValueError:
        # Degenerate geometry (e.g., midpoints coincide).
        return None


def compute_shoulder_height_diff(
    smoothed_landmarks: list,
    visibility_threshold: float | None = None,
) -> Optional[float]:
    """Compute shoulder height difference from smoothed landmarks.

    Returns the absolute vertical difference between left and right
    shoulders in normalized image coordinates. 0 = level.

    Required landmarks: LEFT_SHOULDER, RIGHT_SHOULDER.

    Does NOT depend on arm/elbow/wrist/hip landmarks.

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects from
            PoseResult.smoothed_landmarks.
        visibility_threshold: Minimum raw visibility. Defaults to
            LANDMARK_VISIBILITY_THRESHOLD from config.

    Returns:
        Absolute y-difference in normalized units, or None if
        unavailable.
    """
    if visibility_threshold is None:
        visibility_threshold = LANDMARK_VISIBILITY_THRESHOLD

    l_shoulder = _get_landmark_xy(
        smoothed_landmarks, LEFT_SHOULDER, visibility_threshold,
    )
    r_shoulder = _get_landmark_xy(
        smoothed_landmarks, RIGHT_SHOULDER, visibility_threshold,
    )

    if l_shoulder is None or r_shoulder is None:
        return None

    return compute_shoulder_height_difference(l_shoulder, r_shoulder)


def compute_neck_tilt_metric(
    smoothed_landmarks: list,
    visibility_threshold: float | None = None,
) -> Optional[float]:
    """Compute neck tilt from smoothed landmarks.

    Neck tilt is the angle of the shoulder_midpoint → nose line
    relative to vertical. 0° = upright, larger = more tilt.

    Required landmarks: NOSE, LEFT_SHOULDER, RIGHT_SHOULDER.

    Does NOT depend on hip/elbow/wrist landmarks.

    Uses the same mathematical formula as calibration.compute_neck_tilt().

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects from
            PoseResult.smoothed_landmarks.
        visibility_threshold: Minimum raw visibility. Defaults to
            LANDMARK_VISIBILITY_THRESHOLD from config.

    Returns:
        Neck tilt angle in degrees, or None if unavailable.
    """
    if visibility_threshold is None:
        visibility_threshold = LANDMARK_VISIBILITY_THRESHOLD

    nose = _get_landmark_xy(
        smoothed_landmarks, NOSE, visibility_threshold,
    )
    l_shoulder = _get_landmark_xy(
        smoothed_landmarks, LEFT_SHOULDER, visibility_threshold,
    )
    r_shoulder = _get_landmark_xy(
        smoothed_landmarks, RIGHT_SHOULDER, visibility_threshold,
    )

    if any(p is None for p in (nose, l_shoulder, r_shoulder)):
        return None

    try:
        return compute_neck_tilt(l_shoulder, r_shoulder, nose)
    except ValueError:
        # Degenerate geometry (e.g., nose coincides with shoulder midpoint).
        return None


def compute_posture_metrics(
    smoothed_landmarks: list,
    visibility_threshold: float | None = None,
) -> PostureMetrics:
    """Compute all three posture metrics for a single frame.

    Each metric is computed independently. If one metric is
    unavailable (e.g., hips not visible), the other metrics may
    still be valid.

    All coordinates come from smoothed_landmarks (Phase 1.5 EMA).
    No additional smoothing is performed.

    Visibility gating uses raw MediaPipe visibility (NOT smoothed).

    Args:
        smoothed_landmarks: List of SmoothedLandmark objects from
            PoseResult.smoothed_landmarks.
        visibility_threshold: Minimum raw visibility. Defaults to
            LANDMARK_VISIBILITY_THRESHOLD from config.

    Returns:
        A PostureMetrics dataclass with the three raw metric values.
        Any value may be None if the required landmarks are
        unavailable.
    """
    return PostureMetrics(
        torso_lean=compute_torso_lean(
            smoothed_landmarks, visibility_threshold,
        ),
        shoulder_height_difference=compute_shoulder_height_diff(
            smoothed_landmarks, visibility_threshold,
        ),
        neck_tilt=compute_neck_tilt_metric(
            smoothed_landmarks, visibility_threshold,
        ),
    )

"""Tests for Phase 11 — Standing Hip Abduction.

Validates the definition and angle calculation for the Standing Hip Abduction
exercise, following the same architectural patterns as Shoulder Abduction.

Phase 11 — Add Third Exercise.
"""

from __future__ import annotations

import math
import pytest

from src.exercises.base import CameraOrientation
from src.exercises.standing_hip_abduction import (
    STANDING_HIP_ABDUCTION,
    calculate_hip_abduction_angle,
    LEFT_HIP,
    LEFT_KNEE,
    RIGHT_HIP,
    RIGHT_KNEE,
)
from src.exercises import EXERCISE_REGISTRY


# ================================================================
# Mock Landmarks
# ================================================================

class MockLandmark:
    """Mock for SmoothedLandmark."""
    def __init__(self, x: float, y: float, visibility: float = 0.9):
        self.x = x
        self.y = y
        self.visibility = visibility


def create_mock_landmarks(
    hip_xy=(0.5, 0.5),
    knee_xy=(0.5, 0.7),
    side="left",
    visibility=0.9,
    length=33,
) -> list[MockLandmark]:
    """Create a list of 33 mock landmarks for testing."""
    landmarks = [MockLandmark(0, 0, 0) for _ in range(length)]
    
    if side == "left":
        hip_idx, knee_idx = LEFT_HIP, LEFT_KNEE
    else:
        hip_idx, knee_idx = RIGHT_HIP, RIGHT_KNEE

    if hip_idx < length:
        landmarks[hip_idx] = MockLandmark(hip_xy[0], hip_xy[1], visibility)
    if knee_idx < length:
        landmarks[knee_idx] = MockLandmark(knee_xy[0], knee_xy[1], visibility)

    return landmarks


# ================================================================
# Test 1 — Exercise Definition
# ================================================================

class TestStandingHipAbductionDefinition:
    """Validate the ExerciseDefinition structure."""

    def test_name(self):
        assert STANDING_HIP_ABDUCTION.name == "Standing Hip Abduction"

    def test_camera_orientation(self):
        assert STANDING_HIP_ABDUCTION.camera_orientation == CameraOrientation.FRONT

    def test_target_joint(self):
        assert STANDING_HIP_ABDUCTION.target_joint == "hip"

    def test_rom_target(self):
        assert STANDING_HIP_ABDUCTION.rom_target == 45.0

    def test_rep_start_end_angles(self):
        assert STANDING_HIP_ABDUCTION.rep_start_angle == 10.0
        assert STANDING_HIP_ABDUCTION.rep_end_angle == 35.0

    def test_angle_calculator(self):
        assert STANDING_HIP_ABDUCTION.angle_calculator is calculate_hip_abduction_angle

    def test_expected_landmarks(self):
        expected = STANDING_HIP_ABDUCTION.expected_landmarks
        assert "left_hip" in expected
        assert "left_knee" in expected
        assert "right_hip" in expected
        assert "right_knee" in expected

    def test_compensation_checks(self):
        checks = STANDING_HIP_ABDUCTION.compensation_checks
        assert "trunk_lean" in checks
        assert "hip_hike" in checks

    def test_feedback_templates_populated(self):
        templates = STANDING_HIP_ABDUCTION.feedback_templates
        assert "trunk_lean" in templates
        assert "hip_hike" in templates
        for template in templates.values():
            assert "{value}" in template


# ================================================================
# Test 2 — Hip Abduction Angle Calculation (Geometry)
# ================================================================

class TestHipAbductionAngleCalculation:
    """Test geometric edge cases for hip abduction."""

    def test_starting_position(self):
        # Hip at (0.5, 0.5)
        # Knee at (0.5, 0.7) -> Leg is vertical downwards
        # Angle should be 0 degrees.
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5),
            knee_xy=(0.5, 0.7),
            side="left"
        )
        angle = calculate_hip_abduction_angle(landmarks, side="left")
        assert angle is not None
        assert math.isclose(angle, 0.0, abs_tol=1e-5)

    def test_moderate_abduction(self):
        # Left leg moves laterally (to the left in user's body, which might be -x depending on facing)
        # Hip at (0.5, 0.5)
        # Knee at (0.3, 0.7) -> dx=-0.2, dy=0.2
        # Angle from straight down (dy=0.2) and lateral shift (dx=-0.2) is 45 degrees.
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5),
            knee_xy=(0.3, 0.7),
            side="left"
        )
        angle = calculate_hip_abduction_angle(landmarks, side="left")
        assert angle is not None
        assert math.isclose(angle, 45.0, abs_tol=1e-5)

    def test_greater_abduction(self):
        # Hip at (0.5, 0.5)
        # Knee at (0.2, 0.7) -> dx=-0.3, dy=0.2
        # Angle should be greater than moderate.
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5),
            knee_xy=(0.2, 0.7),
            side="left"
        )
        angle = calculate_hip_abduction_angle(landmarks, side="left")
        assert angle is not None
        # atan(0.3 / 0.2) = atan(1.5) = 56.3 degrees
        assert math.isclose(angle, 56.30993247, abs_tol=1e-4)

    def test_right_leg_abduction(self):
        # Hip at (0.5, 0.5)
        # Knee at (0.7, 0.7) -> dx=0.2, dy=0.2
        # Angle should be 45 degrees.
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5),
            knee_xy=(0.7, 0.7),
            side="right"
        )
        angle = calculate_hip_abduction_angle(landmarks, side="right")
        assert angle is not None
        assert math.isclose(angle, 45.0, abs_tol=1e-5)

    def test_missing_hip(self):
        landmarks = create_mock_landmarks(side="left")
        landmarks[LEFT_HIP].visibility = 0.1
        angle = calculate_hip_abduction_angle(landmarks, side="left")
        assert angle is None

    def test_missing_knee(self):
        landmarks = create_mock_landmarks(side="left")
        landmarks[LEFT_KNEE].visibility = 0.1
        angle = calculate_hip_abduction_angle(landmarks, side="left")
        assert angle is None

    def test_degenerate_geometry(self):
        # Hip and knee at same point
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5),
            knee_xy=(0.5, 0.5),
            side="left"
        )
        angle = calculate_hip_abduction_angle(landmarks, side="left")
        assert angle is None

    def test_invalid_side(self):
        landmarks = create_mock_landmarks(side="left")
        angle = calculate_hip_abduction_angle(landmarks, side="middle")
        assert angle is None

# ================================================================
# Test 3 — Registry
# ================================================================

class TestExerciseRegistry:
    """Verify registry integration."""

    def test_registry_contains_exercise(self):
        assert STANDING_HIP_ABDUCTION in EXERCISE_REGISTRY

"""Tests for Phase 10 — Seated Knee Extension.

Validates the definition and angle calculation for the Seated Knee Extension
exercise, following the same architectural patterns as Shoulder Abduction.

Phase 10 — Generalize to a Second Exercise.
"""

from __future__ import annotations

import math
import pytest

from src.exercises.base import CameraOrientation
from src.exercises.seated_knee_extension import (
    SEATED_KNEE_EXTENSION,
    calculate_seated_knee_extension_angle,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_ANKLE,
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
    ankle_xy=(0.5, 0.9),
    side="left",
    visibility=0.9,
    length=33,
) -> list[MockLandmark]:
    """Create a list of 33 mock landmarks for testing."""
    landmarks = [MockLandmark(0, 0, 0) for _ in range(length)]
    
    if side == "left":
        hip_idx, knee_idx, ankle_idx = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
    else:
        hip_idx, knee_idx, ankle_idx = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE

    if hip_idx < length:
        landmarks[hip_idx] = MockLandmark(hip_xy[0], hip_xy[1], visibility)
    if knee_idx < length:
        landmarks[knee_idx] = MockLandmark(knee_xy[0], knee_xy[1], visibility)
    if ankle_idx < length:
        landmarks[ankle_idx] = MockLandmark(ankle_xy[0], ankle_xy[1], visibility)

    return landmarks


# ================================================================
# Test 1 — Exercise Definition
# ================================================================

class TestSeatedKneeExtensionDefinition:
    """Validate the ExerciseDefinition structure."""

    def test_name(self):
        assert SEATED_KNEE_EXTENSION.name == "Seated Knee Extension"

    def test_camera_orientation(self):
        assert SEATED_KNEE_EXTENSION.camera_orientation == CameraOrientation.SIDE

    def test_target_joint(self):
        assert SEATED_KNEE_EXTENSION.target_joint == "knee"

    def test_rom_target(self):
        assert SEATED_KNEE_EXTENSION.rom_target == 160.0

    def test_rep_start_end_angles(self):
        assert SEATED_KNEE_EXTENSION.rep_start_angle == 90.0
        assert SEATED_KNEE_EXTENSION.rep_end_angle == 150.0

    def test_angle_calculator(self):
        assert SEATED_KNEE_EXTENSION.angle_calculator is calculate_seated_knee_extension_angle

    def test_expected_landmarks(self):
        expected = SEATED_KNEE_EXTENSION.expected_landmarks
        assert "left_hip" in expected
        assert "left_knee" in expected
        assert "left_ankle" in expected
        assert "right_hip" in expected
        assert "right_knee" in expected
        assert "right_ankle" in expected

    def test_compensation_checks_empty(self):
        assert SEATED_KNEE_EXTENSION.compensation_checks == []

    def test_feedback_templates_populated(self):
        templates = SEATED_KNEE_EXTENSION.feedback_templates
        assert "torso_lean" in templates
        assert "neck_tilt" in templates
        for template in templates.values():
            assert "{value}" in template


# ================================================================
# Test 2 — Knee Angle Calculation (Geometry)
# ================================================================

class TestKneeAngleCalculation:
    """Test geometric edge cases for knee extension."""

    def test_bent_knee(self):
        # Hip at (0.5, 0.5)
        # Knee at (0.7, 0.5) -> Thigh is horizontal
        # Ankle at (0.7, 0.7) -> Lower leg is vertical downwards
        # Angle should be 90 degrees.
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5),
            knee_xy=(0.7, 0.5),
            ankle_xy=(0.7, 0.7),
            side="left"
        )
        angle = calculate_seated_knee_extension_angle(landmarks, side="left")
        assert angle is not None
        assert math.isclose(angle, 90.0, abs_tol=1e-5)

    def test_extended_knee(self):
        # Hip at (0.5, 0.5)
        # Knee at (0.7, 0.5)
        # Ankle at (0.9, 0.5)
        # All in a line -> 180 degrees.
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5),
            knee_xy=(0.7, 0.5),
            ankle_xy=(0.9, 0.5),
            side="left"
        )
        angle = calculate_seated_knee_extension_angle(landmarks, side="left")
        assert angle is not None
        assert math.isclose(angle, 180.0, abs_tol=1e-5)

    def test_degenerate_geometry(self):
        # Knee coincident with ankle
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5),
            knee_xy=(0.7, 0.7),
            ankle_xy=(0.7, 0.7),
            side="left"
        )
        angle = calculate_seated_knee_extension_angle(landmarks, side="left")
        assert angle is None


# ================================================================
# Test 3 — Left/Right Side Handling
# ================================================================

class TestSideHandling:
    """Test left and right side independence."""

    def test_left_side(self):
        landmarks = create_mock_landmarks(side="left")
        # Give right side garbage visibility
        landmarks[RIGHT_HIP] = MockLandmark(0, 0, 0.0)
        landmarks[RIGHT_KNEE] = MockLandmark(0, 0, 0.0)
        landmarks[RIGHT_ANKLE] = MockLandmark(0, 0, 0.0)
        
        angle = calculate_seated_knee_extension_angle(landmarks, side="left")
        assert angle is not None

    def test_right_side(self):
        landmarks = create_mock_landmarks(side="right")
        # Give left side garbage visibility
        landmarks[LEFT_HIP] = MockLandmark(0, 0, 0.0)
        landmarks[LEFT_KNEE] = MockLandmark(0, 0, 0.0)
        landmarks[LEFT_ANKLE] = MockLandmark(0, 0, 0.0)
        
        angle = calculate_seated_knee_extension_angle(landmarks, side="right")
        assert angle is not None

    def test_invalid_side(self):
        landmarks = create_mock_landmarks()
        with pytest.raises(ValueError, match="Invalid side"):
            calculate_seated_knee_extension_angle(landmarks, side="middle")


# ================================================================
# Test 4 — Missing Landmarks & Visibility
# ================================================================

class TestMissingLandmarks:
    """Test safe handling of missing or low-visibility landmarks."""

    def test_low_visibility_hip(self):
        landmarks = create_mock_landmarks()
        landmarks[LEFT_HIP].visibility = 0.1
        angle = calculate_seated_knee_extension_angle(
            landmarks, side="left", visibility_threshold=0.5
        )
        assert angle is None

    def test_low_visibility_knee(self):
        landmarks = create_mock_landmarks()
        landmarks[LEFT_KNEE].visibility = 0.1
        angle = calculate_seated_knee_extension_angle(
            landmarks, side="left", visibility_threshold=0.5
        )
        assert angle is None

    def test_low_visibility_ankle(self):
        landmarks = create_mock_landmarks()
        landmarks[LEFT_ANKLE].visibility = 0.1
        angle = calculate_seated_knee_extension_angle(
            landmarks, side="left", visibility_threshold=0.5
        )
        assert angle is None

    def test_missing_landmark_index_out_of_bounds(self):
        # Create a list shorter than required indices
        landmarks = create_mock_landmarks(length=10)
        angle = calculate_seated_knee_extension_angle(landmarks, side="left")
        assert angle is None


# ================================================================
# Test 5 & 6 — Registry and Integration
# ================================================================

class TestRegistryIntegration:
    """Test that the exercise is properly registered."""

    def test_exercise_in_registry(self):
        assert SEATED_KNEE_EXTENSION in EXERCISE_REGISTRY
        
    def test_shoulder_abduction_still_works(self):
        from src.exercises.shoulder_abduction import SHOULDER_ABDUCTION
        assert SHOULDER_ABDUCTION in EXERCISE_REGISTRY

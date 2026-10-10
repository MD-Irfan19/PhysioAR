"""Tests for Phase 12 — Elbow Flexion-Extension.

Validates the definition and angle calculation for the Elbow Flexion-Extension
exercise, ensuring generic pipeline compatibility.
"""

from __future__ import annotations

import math
import pytest

from src.exercises.base import CameraOrientation
from src.exercises.elbow_flexion_extension import (
    ELBOW_FLEXION_EXTENSION,
    calculate_elbow_flexion_extension_angle,
    LEFT_SHOULDER,
    LEFT_ELBOW,
    LEFT_WRIST,
    RIGHT_SHOULDER,
    RIGHT_ELBOW,
    RIGHT_WRIST,
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
    sh_xy=(0.5, 0.2),
    el_xy=(0.5, 0.5),
    wr_xy=(0.5, 0.8),
    side="right",
    visibility=0.9,
    length=33,
) -> list[MockLandmark]:
    """Create a list of mock landmarks for elbow testing."""
    landmarks = [MockLandmark(0, 0, 0) for _ in range(length)]
    
    if side == "left":
        sh_idx, el_idx, wr_idx = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
    else:
        sh_idx, el_idx, wr_idx = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST

    if max(sh_idx, el_idx, wr_idx) < length:
        landmarks[sh_idx] = MockLandmark(sh_xy[0], sh_xy[1], visibility)
        landmarks[el_idx] = MockLandmark(el_xy[0], el_xy[1], visibility)
        landmarks[wr_idx] = MockLandmark(wr_xy[0], wr_xy[1], visibility)

    return landmarks


# ================================================================
# Test 1 — Exercise Definition
# ================================================================

class TestElbowFlexionExtensionDefinition:
    """Validate the ExerciseDefinition structure."""

    def test_name(self):
        assert ELBOW_FLEXION_EXTENSION.name == "Elbow Flexion-Extension"

    def test_camera_orientation(self):
        assert ELBOW_FLEXION_EXTENSION.camera_orientation == CameraOrientation.SIDE

    def test_target_joint(self):
        assert ELBOW_FLEXION_EXTENSION.target_joint == "elbow"

    def test_rom_target(self):
        assert ELBOW_FLEXION_EXTENSION.rom_target == 130.0

    def test_rep_start_end_angles(self):
        assert ELBOW_FLEXION_EXTENSION.rep_start_angle == 15.0
        assert ELBOW_FLEXION_EXTENSION.rep_end_angle == 120.0

    def test_angle_calculator(self):
        assert ELBOW_FLEXION_EXTENSION.angle_calculator is calculate_elbow_flexion_extension_angle

    def test_expected_landmarks(self):
        expected = ELBOW_FLEXION_EXTENSION.expected_landmarks
        assert "left_shoulder" in expected
        assert "left_elbow" in expected
        assert "left_wrist" in expected
        assert "right_shoulder" in expected
        assert "right_elbow" in expected
        assert "right_wrist" in expected

    def test_compensation_checks(self):
        checks = ELBOW_FLEXION_EXTENSION.compensation_checks
        assert "torso_lean" in checks
        assert "shoulder_hike" in checks
        assert "shoulder_substitution" in checks
        assert len(checks) == 3

    def test_feedback_templates_populated(self):
        templates = ELBOW_FLEXION_EXTENSION.feedback_templates
        assert "torso_lean" in templates
        assert "shoulder_hike" in templates
        for template in templates.values():
            assert "{value}" in template


# ================================================================
# Test 2 — Elbow Angle Calculation (Geometry)
# ================================================================

class TestElbowAngleCalculation:
    """Test geometric edge cases for elbow flexion."""

    def test_straight_elbow(self):
        # Interior angle 180 -> flexion 0
        landmarks = create_mock_landmarks(
            sh_xy=(0.5, 0.2),
            el_xy=(0.5, 0.5),
            wr_xy=(0.5, 0.8),
            side="right"
        )
        angle = calculate_elbow_flexion_extension_angle(landmarks, side="right")
        assert angle is not None
        assert math.isclose(angle, 0.0, abs_tol=1e-5)

    def test_flexed_90_degrees(self):
        # Interior angle 90 -> flexion 90
        # Shoulder at (0.5, 0.2), Elbow at (0.5, 0.5) -> arm points down
        # Wrist at (0.8, 0.5) -> forearm points right
        landmarks = create_mock_landmarks(
            sh_xy=(0.5, 0.2),
            el_xy=(0.5, 0.5),
            wr_xy=(0.8, 0.5),
            side="right"
        )
        angle = calculate_elbow_flexion_extension_angle(landmarks, side="right")
        assert angle is not None
        assert math.isclose(angle, 90.0, abs_tol=1e-5)

    def test_flexed_135_degrees(self):
        # Greater flexion produces greater tracked angle (180 - 45 = 135)
        # Forearm points up-right
        landmarks = create_mock_landmarks(
            sh_xy=(0.5, 0.2),
            el_xy=(0.5, 0.5),
            wr_xy=(0.8, 0.2),
            side="right"
        )
        angle = calculate_elbow_flexion_extension_angle(landmarks, side="right")
        assert angle is not None
        assert math.isclose(angle, 135.0, abs_tol=1e-5)

    def test_left_arm_equivalence(self):
        landmarks_r = create_mock_landmarks(
            sh_xy=(0.5, 0.2),
            el_xy=(0.5, 0.5),
            wr_xy=(0.8, 0.2),
            side="right"
        )
        landmarks_l = create_mock_landmarks(
            sh_xy=(0.5, 0.2),
            el_xy=(0.5, 0.5),
            wr_xy=(0.8, 0.2),
            side="left"
        )
        angle_r = calculate_elbow_flexion_extension_angle(landmarks_r, side="right")
        angle_l = calculate_elbow_flexion_extension_angle(landmarks_l, side="left")
        assert angle_r == angle_l
        assert math.isclose(angle_l, 135.0, abs_tol=1e-5)

    def test_missing_shoulder(self):
        landmarks = create_mock_landmarks(side="right")
        landmarks[RIGHT_SHOULDER].visibility = 0.1
        angle = calculate_elbow_flexion_extension_angle(landmarks, side="right")
        assert angle is None

    def test_missing_elbow(self):
        landmarks = create_mock_landmarks(side="right")
        landmarks[RIGHT_ELBOW].visibility = 0.1
        angle = calculate_elbow_flexion_extension_angle(landmarks, side="right")
        assert angle is None

    def test_missing_wrist(self):
        landmarks = create_mock_landmarks(side="right")
        landmarks[RIGHT_WRIST].visibility = 0.1
        angle = calculate_elbow_flexion_extension_angle(landmarks, side="right")
        assert angle is None

    def test_degenerate_geometry(self):
        # Elbow and wrist at same point
        landmarks = create_mock_landmarks(
            el_xy=(0.5, 0.5),
            wr_xy=(0.5, 0.5),
            side="right"
        )
        angle = calculate_elbow_flexion_extension_angle(landmarks, side="right")
        assert angle is None

    def test_invalid_side(self):
        landmarks = create_mock_landmarks(side="right")
        angle = calculate_elbow_flexion_extension_angle(landmarks, side="middle")
        assert angle is None

# ================================================================
# Test 3 — Registry
# ================================================================

class TestExerciseRegistry:
    """Verify registry integration."""

    def test_registry_contains_exercise(self):
        assert ELBOW_FLEXION_EXTENSION in EXERCISE_REGISTRY

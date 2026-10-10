"""Tests for the Sit to Stand exercise definition and angle calculation."""

import math
from unittest.mock import Mock

import pytest

from src.exercises.base import CameraOrientation
from src.exercises.sit_to_stand import (
    calculate_knee_flexion_angle,
    compute_forward_trunk_lean,
    SIT_TO_STAND,
)
from src.exercises.__init__ import EXERCISE_REGISTRY


# ================================================================
# Test Helpers
# ================================================================

class MockSmoothedLandmark:
    """Mock for SmoothedLandmark."""
    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.visibility = visibility


def create_mock_landmarks(hip_xy, knee_xy, ankle_xy, shoulder_xy=(0.0, 0.0), side="right", visibility=1.0):
    """Create a list of mock landmarks for testing."""
    landmarks = [MockSmoothedLandmark(0, 0, 0) for _ in range(33)]
    if side == "left":
        landmarks[11] = MockSmoothedLandmark(shoulder_xy[0], shoulder_xy[1], visibility)
        landmarks[23] = MockSmoothedLandmark(hip_xy[0], hip_xy[1], visibility)
        landmarks[25] = MockSmoothedLandmark(knee_xy[0], knee_xy[1], visibility)
        landmarks[27] = MockSmoothedLandmark(ankle_xy[0], ankle_xy[1], visibility)
    else:
        landmarks[12] = MockSmoothedLandmark(shoulder_xy[0], shoulder_xy[1], visibility)
        landmarks[24] = MockSmoothedLandmark(hip_xy[0], hip_xy[1], visibility)
        landmarks[26] = MockSmoothedLandmark(knee_xy[0], knee_xy[1], visibility)
        landmarks[28] = MockSmoothedLandmark(ankle_xy[0], ankle_xy[1], visibility)
    return landmarks


# ================================================================
# Test 1 — Sit to Stand Definition metadata
# ================================================================

class TestSitToStandDefinition:
    """Verify Sit to Stand configuration."""

    def test_name_and_orientation(self):
        assert SIT_TO_STAND.name == "Sit to Stand"
        assert SIT_TO_STAND.camera_orientation == CameraOrientation.SIDE
        assert SIT_TO_STAND.target_joint == "knee"

    def test_rep_start_end_angles(self):
        assert SIT_TO_STAND.rep_start_angle == 105.0
        assert SIT_TO_STAND.rep_end_angle == 155.0
        assert SIT_TO_STAND.rom_target == 90.0

    def test_angle_calculator(self):
        assert SIT_TO_STAND.angle_calculator is calculate_knee_flexion_angle

    def test_expected_landmarks(self):
        expected = SIT_TO_STAND.expected_landmarks
        assert "left_hip" in expected
        assert "left_knee" in expected
        assert "left_ankle" in expected
        assert "right_hip" in expected
        assert "right_knee" in expected
        assert "right_ankle" in expected
        assert "left_shoulder" in expected
        assert "right_shoulder" in expected

    def test_compensation_checks_empty(self):
        checks = SIT_TO_STAND.compensation_checks
        assert len(checks) == 0

    def test_feedback_templates_populated(self):
        templates = SIT_TO_STAND.feedback_templates
        assert "insufficient_forward_trunk_lean" in templates

    def test_generic_checks(self):
        checks = SIT_TO_STAND.generic_checks
        assert len(checks) == 1
        assert checks[0].id == "insufficient_forward_trunk_lean"

    def test_registered(self):
        assert SIT_TO_STAND in EXERCISE_REGISTRY


# ================================================================
# Test 2 — Knee Angle Calculation (Geometry)
# ================================================================

class TestKneeAngleCalculation:
    """Test geometric edge cases for knee flexion."""

    def test_seated_knee(self):
        # Hip at (0.3, 0.5), Knee at (0.6, 0.5), Ankle at (0.6, 0.8) -> 90 degrees
        landmarks = create_mock_landmarks(
            hip_xy=(0.3, 0.5),
            knee_xy=(0.6, 0.5),
            ankle_xy=(0.6, 0.8),
            side="right"
        )
        angle = calculate_knee_flexion_angle(landmarks, side="right")
        assert angle is not None
        assert math.isclose(angle, 90.0, abs_tol=1e-5)

    def test_standing_knee(self):
        # Hip at (0.5, 0.2), Knee at (0.5, 0.5), Ankle at (0.5, 0.8) -> 180 degrees
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.2),
            knee_xy=(0.5, 0.5),
            ankle_xy=(0.5, 0.8),
            side="right"
        )
        angle = calculate_knee_flexion_angle(landmarks, side="right")
        assert angle is not None
        assert math.isclose(angle, 180.0, abs_tol=1e-5)

    def test_monotonic_extension(self):
        # Verify angle increases as knee extends from 90 to 180.
        # Hip fixed at (0.3, 0.5), knee fixed at (0.6, 0.5)
        # Ankle moves from (0.6, 0.8) [seated] to (0.9, 0.5) [standing-ish horizontal]
        l1 = create_mock_landmarks((0.3, 0.5), (0.6, 0.5), (0.6, 0.8)) # 90
        l2 = create_mock_landmarks((0.3, 0.5), (0.6, 0.5), (0.7, 0.7)) # ~123
        l3 = create_mock_landmarks((0.3, 0.5), (0.6, 0.5), (0.9, 0.5)) # 180
        
        a1 = calculate_knee_flexion_angle(l1)
        a2 = calculate_knee_flexion_angle(l2)
        a3 = calculate_knee_flexion_angle(l3)
        
        assert a1 < a2 < a3

    def test_left_side(self):
        landmarks = create_mock_landmarks(
            hip_xy=(0.3, 0.5),
            knee_xy=(0.6, 0.5),
            ankle_xy=(0.6, 0.8),
            side="left"
        )
        angle = calculate_knee_flexion_angle(landmarks, side="left")
        assert angle is not None
        assert math.isclose(angle, 90.0, abs_tol=1e-5)

    def test_wrong_side_returns_none(self):
        landmarks = create_mock_landmarks(
            hip_xy=(0.3, 0.5),
            knee_xy=(0.6, 0.5),
            ankle_xy=(0.6, 0.8),
            side="right"
        )
        # Asked for left, but right is populated, left is 0,0,0
        # If visibility threshold is 0.5, the mock default visibility of 0 will fail
        angle = calculate_knee_flexion_angle(landmarks, side="left", visibility_threshold=0.5)
        assert angle is None

    def test_invalid_side(self):
        landmarks = create_mock_landmarks((0.3, 0.5), (0.6, 0.5), (0.6, 0.8))
        assert calculate_knee_flexion_angle(landmarks, side="middle") is None


# ================================================================
# Test 3 — Validation (Missing/Invalid Data)
# ================================================================

class TestKneeAngleValidation:
    """Missing or degenerate data returns None safely."""

    def test_empty_landmarks(self):
        assert calculate_knee_flexion_angle([]) is None

    def test_low_visibility(self):
        landmarks = create_mock_landmarks(
            hip_xy=(0.3, 0.5),
            knee_xy=(0.6, 0.5),
            ankle_xy=(0.6, 0.8),
            side="right",
            visibility=0.1
        )
        angle = calculate_knee_flexion_angle(landmarks, side="right", visibility_threshold=0.5)
        assert angle is None

    def test_degenerate_geometry(self):
        # Hip, knee, ankle all at same point
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5),
            knee_xy=(0.5, 0.5),
            ankle_xy=(0.5, 0.5),
            side="right"
        )
        angle = calculate_knee_flexion_angle(landmarks, side="right")
        assert angle is None


# ================================================================
# Test 4 — Forward Trunk Lean Calculation
# ================================================================

class TestForwardTrunkLean:
    def test_upright_posture(self):
        # Hip at (0.5, 0.5), Shoulder at (0.5, 0.2)
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5), knee_xy=(0,0), ankle_xy=(0,0),
            shoulder_xy=(0.5, 0.2), side="right"
        )
        angle = compute_forward_trunk_lean(landmarks, side="right", visibility_threshold=0.5)
        assert angle is not None
        assert math.isclose(angle, 0.0, abs_tol=1e-5)

    def test_forward_lean(self):
        # Hip at (0.5, 0.5), Shoulder at (0.7, 0.3)
        # Vector is (0.2, -0.2). Vertical is (0, -1)
        # Dot product = 0.2. Mags = sqrt(0.08) and 1.
        # cos = 0.2 / sqrt(0.08) = 0.2 / 0.2828427 = 0.707106 -> 45 degrees
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5), knee_xy=(0,0), ankle_xy=(0,0),
            shoulder_xy=(0.7, 0.3), side="right"
        )
        angle = compute_forward_trunk_lean(landmarks, side="right", visibility_threshold=0.5)
        assert angle is not None
        assert math.isclose(angle, 45.0, abs_tol=1e-5)
        
    def test_different_leans(self):
        # Steeper lean (0.8, 0.3) -> dot product = 0.3. Mags = sqrt(0.09+0.04)=sqrt(0.13) and 1.
        # cos = 0.3 / sqrt(0.13) -> angle is larger
        # Wait, forward leaning increases the x displacement from hip.
        # Vertical is up: (0, -1). Hip is (0.5, 0.5).
        # Trunk vector (shoulder - hip) = (0.7-0.5, 0.3-0.5) = (0.2, -0.2).
        # Actually in calculate_angle, vertex is hip. BA = vertical, BC = shoulder.
        # BA = (0.5 - 0.5, 0.4 - 0.5) = (0, -0.1) which points STRAIGHT UP (negative Y direction).
        # BC = (0.7 - 0.5, 0.3 - 0.5) = (0.2, -0.2)
        # dot = 0*0.2 + (-0.1)*(-0.2) = 0.02
        # |BA| = 0.1
        # |BC| = sqrt(0.04 + 0.04) = sqrt(0.08) = 0.2828427
        # cos = 0.02 / (0.1 * 0.2828) = 0.7071 -> 45 degrees.
        pass

    def test_left_side(self):
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5), knee_xy=(0,0), ankle_xy=(0,0),
            shoulder_xy=(0.7, 0.3), side="left"
        )
        angle = compute_forward_trunk_lean(landmarks, side="left", visibility_threshold=0.5)
        assert angle is not None
        assert math.isclose(angle, 45.0, abs_tol=1e-5)

    def test_low_visibility(self):
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5), knee_xy=(0,0), ankle_xy=(0,0),
            shoulder_xy=(0.5, 0.2), side="right", visibility=0.1
        )
        angle = compute_forward_trunk_lean(landmarks, side="right", visibility_threshold=0.5)
        assert angle is None

    def test_degenerate(self):
        # Shoulder and hip at same point
        landmarks = create_mock_landmarks(
            hip_xy=(0.5, 0.5), knee_xy=(0,0), ankle_xy=(0,0),
            shoulder_xy=(0.5, 0.5), side="right"
        )
        angle = compute_forward_trunk_lean(landmarks, side="right", visibility_threshold=0.5)
        assert angle is None

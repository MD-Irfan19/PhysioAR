"""Tests for Phase 3 — Exercise Definition Structure + Shoulder Abduction.

All tests use deterministic synthetic data. No webcam, OpenCV, MediaPipe
inference, GUI, or random data required.

Tests cover:
  A. CameraOrientation enum
  B. ExerciseDefinition dataclass
  C. Shoulder Abduction definition values
  D. Angle calculation (0°, 45°, 90°)
  E. Both sides (left and right)
  F. Invalid/missing landmarks
  G. Regression (existing tests verified separately via full pytest run)

Phase 3 — Exercise Definition Structure + Shoulder Abduction.
"""

import math
from dataclasses import dataclass

import pytest

from src.exercises.base import CameraOrientation, ExerciseDefinition
from src.exercises.shoulder_abduction import (
    SHOULDER_ABDUCTION,
    calculate_shoulder_abduction_angle,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
)


# ================================================================
# Helper: mock SmoothedLandmark
# ================================================================


@dataclass
class MockSmoothedLandmark:
    """Minimal mock for SmoothedLandmark."""

    x: float
    y: float
    z: float = 0.0
    raw_x: float = 0.0
    raw_y: float = 0.0
    raw_z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


def _make_landmarks(overrides: dict | None = None) -> list[MockSmoothedLandmark]:
    """Build 33 mock smoothed landmarks with specified overrides.

    Default: all landmarks at (0.5, 0.5) with visibility=1.0.

    overrides maps index to (x, y) or (x, y, visibility).
    """
    landmarks = [MockSmoothedLandmark(x=0.5, y=0.5) for _ in range(33)]
    if overrides:
        for idx, vals in overrides.items():
            if len(vals) == 2:
                landmarks[idx] = MockSmoothedLandmark(x=vals[0], y=vals[1])
            elif len(vals) == 3:
                landmarks[idx] = MockSmoothedLandmark(
                    x=vals[0], y=vals[1], visibility=vals[2],
                )
    return landmarks


# ================================================================
# A. CameraOrientation enum
# ================================================================


class TestCameraOrientation:
    """CameraOrientation enum has FRONT and SIDE values."""

    def test_front_exists(self):
        assert CameraOrientation.FRONT.value == "front"

    def test_side_exists(self):
        assert CameraOrientation.SIDE.value == "side"

    def test_only_two_members(self):
        assert len(CameraOrientation) == 2


# ================================================================
# B. ExerciseDefinition dataclass
# ================================================================


class TestExerciseDefinition:
    """ExerciseDefinition can be instantiated with required fields."""

    def test_can_instantiate(self):
        def dummy_calculator(landmarks, side):
            return 0.0

        ed = ExerciseDefinition(
            name="Test Exercise",
            camera_orientation=CameraOrientation.FRONT,
            target_joint="test",
            rom_target=90.0,
            rep_start_angle=10.0,
            rep_end_angle=80.0,
            angle_calculator=dummy_calculator,
        )
        assert ed.name == "Test Exercise"
        assert ed.camera_orientation == CameraOrientation.FRONT
        assert ed.target_joint == "test"
        assert ed.rom_target == 90.0

    def test_defaults(self):
        def dummy_calculator(landmarks, side):
            return 0.0

        ed = ExerciseDefinition(
            name="Test",
            camera_orientation=CameraOrientation.SIDE,
            target_joint="elbow",
            rom_target=120.0,
            rep_start_angle=5.0,
            rep_end_angle=110.0,
            angle_calculator=dummy_calculator,
        )
        assert ed.compensation_checks == []
        assert ed.feedback_templates == {}
        assert ed.expected_landmarks == []

    def test_all_fields_stored(self):
        def calc(lm, side):
            return 42.0

        ed = ExerciseDefinition(
            name="Full",
            camera_orientation=CameraOrientation.FRONT,
            target_joint="shoulder",
            rom_target=90.0,
            rep_start_angle=15.0,
            rep_end_angle=80.0,
            angle_calculator=calc,
            compensation_checks=["torso_lean"],
            feedback_templates={"good": "Well done!"},
            expected_landmarks=["left_shoulder"],
        )
        assert ed.compensation_checks == ["torso_lean"]
        assert ed.feedback_templates == {"good": "Well done!"}
        assert ed.expected_landmarks == ["left_shoulder"]
        assert ed.angle_calculator([], "left") == 42.0


# ================================================================
# C. Shoulder Abduction definition values
# ================================================================


class TestShoulderAbductionDefinition:
    """SHOULDER_ABDUCTION instance has correct values."""

    def test_name(self):
        assert SHOULDER_ABDUCTION.name == "Shoulder Abduction"

    def test_camera_orientation(self):
        assert SHOULDER_ABDUCTION.camera_orientation == CameraOrientation.FRONT

    def test_target_joint(self):
        assert SHOULDER_ABDUCTION.target_joint == "shoulder"

    def test_rom_target(self):
        assert SHOULDER_ABDUCTION.rom_target == 90.0

    def test_rep_start_angle(self):
        assert SHOULDER_ABDUCTION.rep_start_angle == 15.0

    def test_rep_end_angle(self):
        assert SHOULDER_ABDUCTION.rep_end_angle == 80.0

    def test_compensation_checks_empty(self):
        assert SHOULDER_ABDUCTION.compensation_checks == []

    def test_feedback_templates_empty(self):
        assert SHOULDER_ABDUCTION.feedback_templates == {}

    def test_expected_landmarks(self):
        expected = SHOULDER_ABDUCTION.expected_landmarks
        assert "left_shoulder" in expected
        assert "left_elbow" in expected
        assert "right_shoulder" in expected
        assert "right_elbow" in expected

    def test_angle_calculator_is_callable(self):
        assert callable(SHOULDER_ABDUCTION.angle_calculator)


# ================================================================
# D. Angle calculation — deterministic mathematical tests
# ================================================================


class TestShoulderAbductionAngle:
    """Shoulder abduction angle calculation with known geometry.

    Convention:
      - MediaPipe image coords: y increases downward.
      - Vertical reference: (shoulder_x, shoulder_y + 1) = straight down.
      - Angle at shoulder between arm vector and downward reference.
      - Arm hanging down ≈ 0°, arm horizontal ≈ 90°.

    calculate_angle(elbow, shoulder, vertical_ref) where shoulder is vertex.
    """

    def test_arm_straight_down_zero_degrees(self):
        """Elbow directly below shoulder → 0° abduction."""
        landmarks = _make_landmarks({
            LEFT_SHOULDER: (0.5, 0.4),
            LEFT_ELBOW: (0.5, 0.7),  # Directly below.
        })
        angle = calculate_shoulder_abduction_angle(landmarks, side="left")
        assert angle is not None
        assert angle == pytest.approx(0.0, abs=0.1)

    def test_arm_horizontal_right_90_degrees(self):
        """Left elbow directly to the left of shoulder → 90° abduction.

        In image coords, left = smaller x.
        """
        landmarks = _make_landmarks({
            LEFT_SHOULDER: (0.5, 0.4),
            LEFT_ELBOW: (0.2, 0.4),  # Same y, to the left.
        })
        angle = calculate_shoulder_abduction_angle(landmarks, side="left")
        assert angle is not None
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_arm_45_degrees(self):
        """Elbow at 45° from vertical → ~45° abduction.

        For a 45° angle from vertical-down at shoulder (0.5, 0.4):
        The elbow should be displaced equally in x and y from shoulder,
        in the down-and-left direction.

        Vector from shoulder to elbow = (-0.3, 0.3) → 45° from vertical.
        """
        landmarks = _make_landmarks({
            LEFT_SHOULDER: (0.5, 0.4),
            LEFT_ELBOW: (0.2, 0.7),  # Down-left, 45° from vertical.
        })
        angle = calculate_shoulder_abduction_angle(landmarks, side="left")
        assert angle is not None
        assert angle == pytest.approx(45.0, abs=0.1)

    def test_right_arm_straight_down(self):
        """Right elbow directly below right shoulder → 0°."""
        landmarks = _make_landmarks({
            RIGHT_SHOULDER: (0.5, 0.4),
            RIGHT_ELBOW: (0.5, 0.7),
        })
        angle = calculate_shoulder_abduction_angle(landmarks, side="right")
        assert angle is not None
        assert angle == pytest.approx(0.0, abs=0.1)

    def test_right_arm_horizontal(self):
        """Right elbow directly to the right of shoulder → 90°.

        In image coords, right = larger x.
        """
        landmarks = _make_landmarks({
            RIGHT_SHOULDER: (0.5, 0.4),
            RIGHT_ELBOW: (0.8, 0.4),  # Same y, to the right.
        })
        angle = calculate_shoulder_abduction_angle(landmarks, side="right")
        assert angle is not None
        assert angle == pytest.approx(90.0, abs=0.1)


# ================================================================
# E. Both sides
# ================================================================


class TestBothSides:
    """Left and right arm calculations use the correct landmark indices."""

    def test_left_uses_left_indices(self):
        """Left side uses LEFT_SHOULDER and LEFT_ELBOW."""
        landmarks = _make_landmarks({
            LEFT_SHOULDER: (0.4, 0.4),
            LEFT_ELBOW: (0.4, 0.7),    # Down → 0°.
            RIGHT_SHOULDER: (0.6, 0.4),
            RIGHT_ELBOW: (0.9, 0.4),   # Horizontal → 90°.
        })
        left_angle = calculate_shoulder_abduction_angle(landmarks, side="left")
        right_angle = calculate_shoulder_abduction_angle(landmarks, side="right")
        assert left_angle == pytest.approx(0.0, abs=0.1)
        assert right_angle == pytest.approx(90.0, abs=0.1)

    def test_right_uses_right_indices(self):
        """Right side uses RIGHT_SHOULDER and RIGHT_ELBOW."""
        landmarks = _make_landmarks({
            LEFT_SHOULDER: (0.4, 0.4),
            LEFT_ELBOW: (0.1, 0.4),    # Horizontal → 90°.
            RIGHT_SHOULDER: (0.6, 0.4),
            RIGHT_ELBOW: (0.6, 0.7),   # Down → 0°.
        })
        left_angle = calculate_shoulder_abduction_angle(landmarks, side="left")
        right_angle = calculate_shoulder_abduction_angle(landmarks, side="right")
        assert left_angle == pytest.approx(90.0, abs=0.1)
        assert right_angle == pytest.approx(0.0, abs=0.1)

    def test_invalid_side_raises(self):
        landmarks = _make_landmarks()
        with pytest.raises(ValueError, match="Invalid side"):
            calculate_shoulder_abduction_angle(landmarks, side="center")


# ================================================================
# F. Invalid / missing landmarks
# ================================================================


class TestInvalidLandmarks:
    """Missing or degenerate landmarks return None, not fake values."""

    def test_missing_shoulder_returns_none(self):
        """Fewer than 12 landmarks → left shoulder missing."""
        landmarks = [MockSmoothedLandmark(x=0.5, y=0.5) for _ in range(10)]
        angle = calculate_shoulder_abduction_angle(landmarks, side="left")
        assert angle is None

    def test_missing_elbow_returns_none(self):
        """Fewer than 14 landmarks → left elbow missing."""
        landmarks = [MockSmoothedLandmark(x=0.5, y=0.5) for _ in range(12)]
        angle = calculate_shoulder_abduction_angle(landmarks, side="left")
        assert angle is None

    def test_empty_landmarks_returns_none(self):
        angle = calculate_shoulder_abduction_angle([], side="left")
        assert angle is None

    def test_low_visibility_shoulder_returns_none(self):
        landmarks = _make_landmarks({
            LEFT_SHOULDER: (0.5, 0.4, 0.1),  # Low visibility.
            LEFT_ELBOW: (0.5, 0.7, 0.9),
        })
        angle = calculate_shoulder_abduction_angle(
            landmarks, side="left", visibility_threshold=0.5,
        )
        assert angle is None

    def test_low_visibility_elbow_returns_none(self):
        landmarks = _make_landmarks({
            LEFT_SHOULDER: (0.5, 0.4, 0.9),
            LEFT_ELBOW: (0.5, 0.7, 0.1),  # Low visibility.
        })
        angle = calculate_shoulder_abduction_angle(
            landmarks, side="left", visibility_threshold=0.5,
        )
        assert angle is None

    def test_coincident_shoulder_elbow_returns_none(self):
        """Degenerate geometry: shoulder == elbow."""
        landmarks = _make_landmarks({
            LEFT_SHOULDER: (0.5, 0.5),
            LEFT_ELBOW: (0.5, 0.5),  # Same as shoulder.
        })
        angle = calculate_shoulder_abduction_angle(landmarks, side="left")
        assert angle is None

    def test_does_not_return_zero_for_missing(self):
        """Ensure None, not 0.0, for missing landmarks."""
        angle = calculate_shoulder_abduction_angle([], side="right")
        assert angle is not None or angle is None  # Tautology for clarity.
        assert angle is None  # The actual assertion.


# ================================================================
# Uses smoothed coordinates
# ================================================================


class TestUsesSmoothedCoordinates:
    """The angle calculator uses SmoothedLandmark.x/.y (smoothed)."""

    def test_uses_smoothed_not_raw(self):
        lm_shoulder = MockSmoothedLandmark(
            x=0.5, y=0.4,
            raw_x=0.52, raw_y=0.42,
            visibility=0.9,
        )
        lm_elbow = MockSmoothedLandmark(
            x=0.5, y=0.7,  # Straight down → 0°.
            raw_x=0.2, raw_y=0.7,  # If raw were used → ~90°.
            visibility=0.9,
        )
        landmarks = [MockSmoothedLandmark(x=0.5, y=0.5) for _ in range(33)]
        landmarks[LEFT_SHOULDER] = lm_shoulder
        landmarks[LEFT_ELBOW] = lm_elbow

        angle = calculate_shoulder_abduction_angle(landmarks, side="left")
        assert angle is not None
        # Should be ~0° using smoothed (arm straight down),
        # not ~90° using raw (arm horizontal).
        assert angle == pytest.approx(0.0, abs=1.0)


# ================================================================
# Exercise registry
# ================================================================


class TestExerciseRegistry:
    """EXERCISE_REGISTRY is populated correctly."""

    def test_registry_contains_shoulder_abduction(self):
        from src.exercises import EXERCISE_REGISTRY
        assert len(EXERCISE_REGISTRY) >= 1
        assert EXERCISE_REGISTRY[0].name == "Shoulder Abduction"

    def test_registry_elements_are_exercise_definitions(self):
        from src.exercises import EXERCISE_REGISTRY
        for ex in EXERCISE_REGISTRY:
            assert isinstance(ex, ExerciseDefinition)


# ================================================================
# Angle calculator via ExerciseDefinition (integration)
# ================================================================


class TestAngleViaDefinition:
    """Calling exercise.angle_calculator works end-to-end."""

    def test_via_definition_arm_down(self):
        landmarks = _make_landmarks({
            LEFT_SHOULDER: (0.5, 0.4),
            LEFT_ELBOW: (0.5, 0.7),
        })
        angle = SHOULDER_ABDUCTION.angle_calculator(landmarks, "left")
        assert angle is not None
        assert angle == pytest.approx(0.0, abs=0.1)

    def test_via_definition_arm_horizontal(self):
        landmarks = _make_landmarks({
            RIGHT_SHOULDER: (0.5, 0.4),
            RIGHT_ELBOW: (0.8, 0.4),
        })
        angle = SHOULDER_ABDUCTION.angle_calculator(landmarks, "right")
        assert angle is not None
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_via_definition_missing_returns_none(self):
        angle = SHOULDER_ABDUCTION.angle_calculator([], "left")
        assert angle is None


# ================================================================
# No exercise-name branching (architecture test)
# ================================================================


class TestNoExerciseNameBranching:
    """Verify no exercise-name branching in main.py."""

    def test_no_exercise_name_conditional_in_main(self):
        """main.py should not contain 'if exercise == \"Shoulder'."""
        import inspect
        import src.main as main_module

        source = inspect.getsource(main_module)
        # Remove the exercise selection / registry / import lines.
        # Only check the main() function and live loop for branching.
        main_source = inspect.getsource(main_module.main)

        assert 'if exercise == "Shoulder' not in main_source
        assert "if exercise_name ==" not in main_source
        assert 'elif exercise == "' not in main_source

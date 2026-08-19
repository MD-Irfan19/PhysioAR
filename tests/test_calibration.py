"""Tests for src/calibration.py — Phase 2 + Phase 2.1.

All tests use synthetic, deterministic coordinates with hand-calculated
expected results. No webcam, OpenCV camera, MediaPipe inference,
real-time timing, GUI, or random data is used.

Coordinate system reminder:
    x: 0→1, left to right
    y: 0→1, top to bottom
    Upward = (0, -1)

Vertical reference rule:
    vertical_reference = vertex + (0, -1)

calculate_angle(point_a, point_b, point_c) computes the angle AT point_b.

Phase 2.1 additions:
    - Landmark confidence/visibility gating tests.
    - Required-landmark validation tests.
    - Low-confidence hip/shoulder/nose rejection tests.
    - Threshold boundary behavior tests.
    - Mathematical regression tests confirming metric formulas unchanged.
    - Recalibration orchestration tests with mocked camera/estimator.
"""

import math
import statistics
from dataclasses import dataclass
from unittest.mock import MagicMock, PropertyMock

import pytest

from src.calibration import (
    CalibrationResult,
    MetricBaseline,
    REQUIRED_LANDMARKS,
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    compute_frame_metrics,
    compute_hip_alignment,
    compute_neck_tilt,
    compute_shoulder_height_difference,
    compute_spine_angle,
    validate_required_landmarks,
)


# ================================================================
# Helper: lightweight mock SmoothedLandmark for testing
# ================================================================


@dataclass
class MockSmoothedLandmark:
    """Minimal mock for SmoothedLandmark with x, y, z and confidence fields."""

    x: float
    y: float
    z: float = 0.0
    raw_x: float = 0.0
    raw_y: float = 0.0
    raw_z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


def _make_landmarks(**kwargs) -> list[MockSmoothedLandmark]:
    """Build a list of 33 mock landmarks with specified overrides.

    kwargs maps name to (x, y) tuples OR (x, y, visibility) tuples:
        _make_landmarks(nose=(0.5, 0.3), left_shoulder=(0.4, 0.5), ...)
        _make_landmarks(left_hip=(0.4, 0.8, 0.2))  # low visibility

    Unspecified landmarks default to (0.5, 0.5) with visibility=1.0.
    """
    name_to_index = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
    }
    landmarks = [MockSmoothedLandmark(x=0.5, y=0.5) for _ in range(33)]
    for name, values in kwargs.items():
        idx = name_to_index[name]
        if len(values) == 2:
            landmarks[idx] = MockSmoothedLandmark(x=values[0], y=values[1])
        elif len(values) == 3:
            landmarks[idx] = MockSmoothedLandmark(
                x=values[0], y=values[1], visibility=values[2],
            )
    return landmarks


# ================================================================
# TEST A — Level shoulders (Phase 2)
# ================================================================


class TestLevelShoulders:
    """If left_shoulder.y == right_shoulder.y, difference must be 0."""

    def test_level_shoulders(self):
        result = compute_shoulder_height_difference((0.4, 0.5), (0.6, 0.5))
        assert result == pytest.approx(0.0)


# ================================================================
# TEST B — Level hips (Phase 2)
# ================================================================


class TestLevelHips:
    """If left_hip.y == right_hip.y, hip alignment must be 0."""

    def test_level_hips(self):
        result = compute_hip_alignment((0.4, 0.8), (0.6, 0.8))
        assert result == pytest.approx(0.0)


# ================================================================
# TEST C — Vertical torso (Phase 2)
# ================================================================


class TestVerticalTorso:
    """Shoulder midpoint directly above hip midpoint → spine ≈ 0°."""

    def test_vertical_torso(self):
        angle = compute_spine_angle(
            left_shoulder_xy=(0.4, 0.4),
            right_shoulder_xy=(0.6, 0.4),
            left_hip_xy=(0.4, 0.8),
            right_hip_xy=(0.6, 0.8),
        )
        assert angle == pytest.approx(0.0, abs=0.01)


# ================================================================
# TEST D — Tilted torso (Phase 2)
# ================================================================


class TestTiltedTorso:
    """Deliberately tilted torso produces a nonzero spine angle."""

    def test_tilted_torso_45_degrees(self):
        angle = compute_spine_angle(
            left_shoulder_xy=(0.8, 0.5),
            right_shoulder_xy=(0.8, 0.5),
            left_hip_xy=(0.5, 0.8),
            right_hip_xy=(0.5, 0.8),
        )
        assert angle == pytest.approx(45.0, abs=0.01)

    def test_tilted_torso_nonzero(self):
        """Any lateral shift from vertical should produce a positive angle."""
        angle = compute_spine_angle(
            left_shoulder_xy=(0.45, 0.4),
            right_shoulder_xy=(0.65, 0.4),
            left_hip_xy=(0.4, 0.8),
            right_hip_xy=(0.6, 0.8),
        )
        assert angle > 0.0


# ================================================================
# TEST E — Vertical neck (Phase 2)
# ================================================================


class TestVerticalNeck:
    """Nose directly above shoulder midpoint → neck tilt ≈ 0°."""

    def test_vertical_neck(self):
        angle = compute_neck_tilt(
            left_shoulder_xy=(0.4, 0.6),
            right_shoulder_xy=(0.6, 0.6),
            nose_xy=(0.5, 0.3),
        )
        assert angle == pytest.approx(0.0, abs=0.01)


# ================================================================
# TEST F — Tilted neck (Phase 2)
# ================================================================


class TestTiltedNeck:
    """Nose displaced laterally from shoulder midpoint → nonzero tilt."""

    def test_tilted_neck_45_degrees(self):
        angle = compute_neck_tilt(
            left_shoulder_xy=(0.5, 0.6),
            right_shoulder_xy=(0.5, 0.6),
            nose_xy=(0.8, 0.3),
        )
        assert angle == pytest.approx(45.0, abs=0.01)

    def test_tilted_neck_nonzero(self):
        """Any lateral offset → positive neck tilt."""
        angle = compute_neck_tilt(
            left_shoulder_xy=(0.4, 0.6),
            right_shoulder_xy=(0.6, 0.6),
            nose_xy=(0.55, 0.3),
        )
        assert angle > 0.0


# ================================================================
# TEST G — Statistics (Phase 2)
# ================================================================


class TestStatistics:
    """Verify mean and sample standard deviation calculations."""

    def test_known_mean(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert statistics.mean(data) == pytest.approx(3.0)

    def test_known_sample_stdev(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected = math.sqrt(2.5)
        assert statistics.stdev(data) == pytest.approx(expected)


# ================================================================
# TEST H — Invalid frame produces no metrics (Phase 2)
# ================================================================


class TestInvalidFrame:
    """Invalid frames must produce None (no metrics)."""

    def test_empty_landmarks_returns_none(self):
        result = compute_frame_metrics([])
        assert result is None

    def test_insufficient_landmarks_returns_none(self):
        """Fewer than 25 landmarks → required indices don't exist."""
        landmarks = [MockSmoothedLandmark(x=0.5, y=0.5) for _ in range(10)]
        result = compute_frame_metrics(landmarks)
        assert result is None


# ================================================================
# TEST I — Degenerate geometry handled as skip (Phase 2)
# ================================================================


class TestDegenerateGeometry:
    """Degenerate geometry → frame skipped (None), not a crash."""

    def test_coincident_midpoints_returns_none(self):
        """If hip midpoint == shoulder midpoint, angle raises ValueError."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.5),
            left_shoulder=(0.5, 0.5),
            right_shoulder=(0.5, 0.5),
            left_hip=(0.5, 0.5),
            right_hip=(0.5, 0.5),
        )
        result = compute_frame_metrics(landmarks)
        assert result is None


# ================================================================
# TEST J — Insufficient samples (Phase 2)
# ================================================================


class TestInsufficientSamples:
    """CalibrationResult requires min valid samples; fewer → failure."""

    def test_metric_baseline_requires_data(self):
        """statistics.stdev requires at least 2 data points."""
        with pytest.raises(statistics.StatisticsError):
            statistics.stdev([1.0])

    def test_min_calibration_samples_constant(self):
        """The config constant must exist and be reasonable."""
        from src.config import MIN_CALIBRATION_SAMPLES

        assert isinstance(MIN_CALIBRATION_SAMPLES, int)
        assert MIN_CALIBRATION_SAMPLES >= 2


# ================================================================
# TEST K — Result structure (Phase 2)
# ================================================================


class TestCalibrationResultStructure:
    """CalibrationResult and MetricBaseline expose all required fields."""

    def test_metric_baseline_fields(self):
        mb = MetricBaseline(mean=1.5, std=0.3)
        assert mb.mean == pytest.approx(1.5)
        assert mb.std == pytest.approx(0.3)

    def test_calibration_result_fields(self):
        cr = CalibrationResult(
            spine_angle=MetricBaseline(mean=2.0, std=0.5),
            shoulder_height_difference=MetricBaseline(mean=0.01, std=0.002),
            neck_tilt=MetricBaseline(mean=1.5, std=0.4),
            hip_alignment=MetricBaseline(mean=0.008, std=0.001),
            valid_samples=100,
            skipped_samples=5,
            duration_seconds=10.2,
        )
        assert cr.spine_angle.mean == pytest.approx(2.0)
        assert cr.spine_angle.std == pytest.approx(0.5)
        assert cr.shoulder_height_difference.mean == pytest.approx(0.01)
        assert cr.shoulder_height_difference.std == pytest.approx(0.002)
        assert cr.neck_tilt.mean == pytest.approx(1.5)
        assert cr.neck_tilt.std == pytest.approx(0.4)
        assert cr.hip_alignment.mean == pytest.approx(0.008)
        assert cr.hip_alignment.std == pytest.approx(0.001)
        assert cr.valid_samples == 100
        assert cr.skipped_samples == 5
        assert isinstance(cr.duration_seconds, float)


# ================================================================
# Critical regression: vertical reference from VERTEX (Phase 2)
# ================================================================


class TestVerticalReferenceFromVertex:
    """Verify vertical references are constructed from the angle vertex.

    These tests would FAIL if the vertical reference were incorrectly
    anchored to the opposite endpoint instead of the vertex.
    """

    def test_spine_vertical_reference_from_hip_midpoint(self):
        angle = compute_spine_angle(
            left_shoulder_xy=(0.5, 0.4),
            right_shoulder_xy=(0.5, 0.4),
            left_hip_xy=(0.5, 0.8),
            right_hip_xy=(0.5, 0.8),
        )
        assert angle == pytest.approx(0.0, abs=0.01)

    def test_neck_vertical_reference_from_shoulder_midpoint(self):
        angle = compute_neck_tilt(
            left_shoulder_xy=(0.5, 0.6),
            right_shoulder_xy=(0.5, 0.6),
            nose_xy=(0.5, 0.3),
        )
        assert angle == pytest.approx(0.0, abs=0.01)

    def test_spine_tilted_hand_calculated(self):
        angle = compute_spine_angle(
            left_shoulder_xy=(0.8, 0.5),
            right_shoulder_xy=(0.8, 0.5),
            left_hip_xy=(0.5, 0.8),
            right_hip_xy=(0.5, 0.8),
        )
        assert angle == pytest.approx(45.0, abs=0.01)

    def test_neck_tilted_hand_calculated(self):
        angle = compute_neck_tilt(
            left_shoulder_xy=(0.5, 0.6),
            right_shoulder_xy=(0.5, 0.6),
            nose_xy=(0.8, 0.3),
        )
        assert angle == pytest.approx(45.0, abs=0.01)


# ================================================================
# compute_frame_metrics with valid synthetic data (Phase 2)
# ================================================================


class TestComputeFrameMetrics:
    """Verify compute_frame_metrics produces correct tuple output."""

    def test_perfect_posture(self):
        """Symmetrical upright posture → small angles, zero asymmetry."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3),
            left_shoulder=(0.4, 0.5),
            right_shoulder=(0.6, 0.5),
            left_hip=(0.4, 0.8),
            right_hip=(0.6, 0.8),
        )
        result = compute_frame_metrics(landmarks)
        assert result is not None

        spine, shoulder, neck, hip = result
        assert spine == pytest.approx(0.0, abs=0.1)
        assert shoulder == pytest.approx(0.0)
        assert neck == pytest.approx(0.0, abs=0.1)
        assert hip == pytest.approx(0.0)

    def test_asymmetric_shoulders(self):
        """Uneven shoulders → nonzero shoulder height difference."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3),
            left_shoulder=(0.4, 0.48),
            right_shoulder=(0.6, 0.52),
            left_hip=(0.4, 0.8),
            right_hip=(0.6, 0.8),
        )
        result = compute_frame_metrics(landmarks)
        assert result is not None

        _, shoulder, _, _ = result
        assert shoulder == pytest.approx(0.04)

    def test_asymmetric_hips(self):
        """Uneven hips → nonzero hip alignment."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3),
            left_shoulder=(0.4, 0.5),
            right_shoulder=(0.6, 0.5),
            left_hip=(0.4, 0.78),
            right_hip=(0.6, 0.82),
        )
        result = compute_frame_metrics(landmarks)
        assert result is not None

        _, _, _, hip = result
        assert hip == pytest.approx(0.04)


# ================================================================
# Shoulder and hip height difference — asymmetry values (Phase 2)
# ================================================================


class TestAsymmetryMetrics:
    """Shoulder and hip difference metrics use absolute values."""

    def test_shoulder_left_higher(self):
        result = compute_shoulder_height_difference((0.4, 0.45), (0.6, 0.55))
        assert result == pytest.approx(0.1)

    def test_shoulder_right_higher(self):
        result = compute_shoulder_height_difference((0.4, 0.55), (0.6, 0.45))
        assert result == pytest.approx(0.1)

    def test_hip_left_higher(self):
        result = compute_hip_alignment((0.4, 0.75), (0.6, 0.85))
        assert result == pytest.approx(0.1)

    def test_hip_right_higher(self):
        result = compute_hip_alignment((0.4, 0.85), (0.6, 0.75))
        assert result == pytest.approx(0.1)


# ================================================================
# Configuration constants (Phase 2 + 2.1)
# ================================================================


class TestCalibrationConfig:
    """Configuration constants must exist and be reasonable."""

    def test_calibration_seconds(self):
        from src.config import CALIBRATION_SECONDS

        assert isinstance(CALIBRATION_SECONDS, (int, float))
        assert CALIBRATION_SECONDS > 0

    def test_min_calibration_samples(self):
        from src.config import MIN_CALIBRATION_SAMPLES

        assert isinstance(MIN_CALIBRATION_SAMPLES, int)
        assert MIN_CALIBRATION_SAMPLES >= 2

    def test_visibility_threshold_exists(self):
        """Phase 2.1: Threshold constant must exist in config."""
        from src.config import LANDMARK_VISIBILITY_THRESHOLD

        assert isinstance(LANDMARK_VISIBILITY_THRESHOLD, float)
        assert 0.0 <= LANDMARK_VISIBILITY_THRESHOLD <= 1.0


# ================================================================
# PHASE 2.1 — Landmark validity / confidence gating tests
# ================================================================


# ----------------------------------------------------------------
# 2.1-A: Valid landmark confidence → accepted
# ----------------------------------------------------------------


class TestValidLandmarkConfidence:
    """Landmarks above threshold are accepted."""

    def test_all_required_above_threshold(self):
        """All five required landmarks with visibility=1.0 → no rejections."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3),
            left_shoulder=(0.4, 0.5),
            right_shoulder=(0.6, 0.5),
            left_hip=(0.4, 0.8),
            right_hip=(0.6, 0.8),
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert reasons == []

    def test_visibility_exactly_at_threshold_accepted(self):
        """Visibility == threshold → accepted (threshold is minimum)."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.5),
            left_shoulder=(0.4, 0.5, 0.5),
            right_shoulder=(0.6, 0.5, 0.5),
            left_hip=(0.4, 0.8, 0.5),
            right_hip=(0.6, 0.8, 0.5),
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert reasons == []


# ----------------------------------------------------------------
# 2.1-B: Low-confidence landmark → frame rejected
# ----------------------------------------------------------------


class TestLowConfidenceLandmark:
    """A required landmark below threshold causes frame rejection."""

    def test_single_low_confidence_rejects_frame(self):
        """One landmark below threshold → frame rejected."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.1),  # low visibility
            left_shoulder=(0.4, 0.5),
            right_shoulder=(0.6, 0.5),
            left_hip=(0.4, 0.8),
            right_hip=(0.6, 0.8),
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert len(reasons) > 0

    def test_frame_metrics_returns_none_for_low_confidence(self):
        """compute_frame_metrics rejects when a landmark is unreliable."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.1),  # low visibility
            left_shoulder=(0.4, 0.5),
            right_shoulder=(0.6, 0.5),
            left_hip=(0.4, 0.8),
            right_hip=(0.6, 0.8),
        )
        result = compute_frame_metrics(landmarks, visibility_threshold=0.5)
        assert result is None


# ----------------------------------------------------------------
# 2.1-C: Missing landmark → frame rejected
# ----------------------------------------------------------------


class TestMissingLandmark:
    """Missing required landmark causes frame rejection."""

    def test_missing_landmark_index(self):
        """Fewer than max(required) + 1 landmarks → rejected."""
        landmarks = [MockSmoothedLandmark(x=0.5, y=0.5) for _ in range(10)]
        reasons = validate_required_landmarks(landmarks)
        assert len(reasons) > 0
        # Should mention missing landmarks.
        assert any("missing" in r for r in reasons)


# ----------------------------------------------------------------
# 2.1-D: Low-confidence LEFT_HIP → frame rejected
# ----------------------------------------------------------------


class TestLowConfidenceLeftHip:
    """Low-confidence left hip causes calibration frame rejection."""

    def test_low_visibility_left_hip(self):
        landmarks = _make_landmarks(
            nose=(0.5, 0.3),
            left_shoulder=(0.4, 0.5),
            right_shoulder=(0.6, 0.5),
            left_hip=(0.4, 0.8, 0.2),  # low visibility
            right_hip=(0.6, 0.8),
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert len(reasons) > 0
        assert any("LEFT_HIP" in r for r in reasons)

    def test_frame_metrics_none_for_low_left_hip(self):
        landmarks = _make_landmarks(
            nose=(0.5, 0.3),
            left_shoulder=(0.4, 0.5),
            right_shoulder=(0.6, 0.5),
            left_hip=(0.4, 0.8, 0.2),
            right_hip=(0.6, 0.8),
        )
        result = compute_frame_metrics(landmarks, visibility_threshold=0.5)
        assert result is None


# ----------------------------------------------------------------
# 2.1-E: Low-confidence RIGHT_HIP → frame rejected
# ----------------------------------------------------------------


class TestLowConfidenceRightHip:
    """Low-confidence right hip causes calibration frame rejection."""

    def test_low_visibility_right_hip(self):
        landmarks = _make_landmarks(
            nose=(0.5, 0.3),
            left_shoulder=(0.4, 0.5),
            right_shoulder=(0.6, 0.5),
            left_hip=(0.4, 0.8),
            right_hip=(0.6, 0.8, 0.1),  # low visibility
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert len(reasons) > 0
        assert any("RIGHT_HIP" in r for r in reasons)

    def test_frame_metrics_none_for_low_right_hip(self):
        landmarks = _make_landmarks(
            nose=(0.5, 0.3),
            left_shoulder=(0.4, 0.5),
            right_shoulder=(0.6, 0.5),
            left_hip=(0.4, 0.8),
            right_hip=(0.6, 0.8, 0.1),
        )
        result = compute_frame_metrics(landmarks, visibility_threshold=0.5)
        assert result is None


# ----------------------------------------------------------------
# 2.1-F: Low-confidence NOSE → frame rejected
# ----------------------------------------------------------------


class TestLowConfidenceNose:
    """Low-confidence nose causes calibration frame rejection."""

    def test_low_visibility_nose(self):
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.05),  # very low visibility
            left_shoulder=(0.4, 0.5),
            right_shoulder=(0.6, 0.5),
            left_hip=(0.4, 0.8),
            right_hip=(0.6, 0.8),
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert len(reasons) > 0
        assert any("NOSE" in r for r in reasons)


# ----------------------------------------------------------------
# 2.1-G: Low-confidence shoulder → frame rejected
# ----------------------------------------------------------------


class TestLowConfidenceShoulder:
    """Low-confidence shoulder causes calibration frame rejection."""

    def test_low_visibility_left_shoulder(self):
        landmarks = _make_landmarks(
            nose=(0.5, 0.3),
            left_shoulder=(0.4, 0.5, 0.3),  # low
            right_shoulder=(0.6, 0.5),
            left_hip=(0.4, 0.8),
            right_hip=(0.6, 0.8),
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert len(reasons) > 0
        assert any("LEFT_SHOULDER" in r for r in reasons)

    def test_low_visibility_right_shoulder(self):
        landmarks = _make_landmarks(
            nose=(0.5, 0.3),
            left_shoulder=(0.4, 0.5),
            right_shoulder=(0.6, 0.5, 0.3),  # low
            left_hip=(0.4, 0.8),
            right_hip=(0.6, 0.8),
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert len(reasons) > 0
        assert any("RIGHT_SHOULDER" in r for r in reasons)


# ----------------------------------------------------------------
# 2.1-H: All required landmarks valid → metrics calculated
# ----------------------------------------------------------------


class TestAllRequiredLandmarksValid:
    """All required landmarks above threshold → metrics computed."""

    def test_valid_landmarks_produce_metrics(self):
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.99),
            left_shoulder=(0.4, 0.5, 0.95),
            right_shoulder=(0.6, 0.5, 0.92),
            left_hip=(0.4, 0.8, 0.88),
            right_hip=(0.6, 0.8, 0.90),
        )
        result = compute_frame_metrics(landmarks, visibility_threshold=0.5)
        assert result is not None
        spine, shoulder, neck, hip = result
        assert isinstance(spine, float)
        assert isinstance(shoulder, float)
        assert isinstance(neck, float)
        assert isinstance(hip, float)


# ----------------------------------------------------------------
# 2.1-I: Hip-only invalidity rejects full frame
# ----------------------------------------------------------------


class TestHipOnlyInvalidity:
    """If shoulders and nose are valid but hips are below threshold,
    the full calibration frame is rejected.
    Hip alignment is NOT calculated from unreliable hip data.
    """

    def test_valid_upper_body_invalid_hips(self):
        """Camera shows face+shoulders but not hips reliably."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.99),
            left_shoulder=(0.4, 0.5, 0.95),
            right_shoulder=(0.6, 0.5, 0.92),
            left_hip=(0.4, 0.8, 0.1),   # unreliable
            right_hip=(0.6, 0.8, 0.15),  # unreliable
        )
        # Validation should reject.
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert len(reasons) >= 2
        assert any("LEFT_HIP" in r for r in reasons)
        assert any("RIGHT_HIP" in r for r in reasons)

        # Frame metrics should return None.
        result = compute_frame_metrics(landmarks, visibility_threshold=0.5)
        assert result is None

    def test_single_hip_invalid_rejects_frame(self):
        """Even one unreliable hip → entire frame rejected."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.99),
            left_shoulder=(0.4, 0.5, 0.95),
            right_shoulder=(0.6, 0.5, 0.92),
            left_hip=(0.4, 0.8, 0.9),    # valid
            right_hip=(0.6, 0.8, 0.2),   # invalid
        )
        result = compute_frame_metrics(landmarks, visibility_threshold=0.5)
        assert result is None


# ----------------------------------------------------------------
# 2.1-J: Confidence threshold from config
# ----------------------------------------------------------------


class TestConfidenceThresholdConfig:
    """Threshold is configurable from config.py."""

    def test_threshold_imported_from_config(self):
        from src.config import LANDMARK_VISIBILITY_THRESHOLD

        assert isinstance(LANDMARK_VISIBILITY_THRESHOLD, float)
        assert 0.0 <= LANDMARK_VISIBILITY_THRESHOLD <= 1.0

    def test_threshold_not_hardcoded_in_calibration(self):
        """validate_required_landmarks accepts threshold argument."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.3),
            left_shoulder=(0.4, 0.5, 0.3),
            right_shoulder=(0.6, 0.5, 0.3),
            left_hip=(0.4, 0.8, 0.3),
            right_hip=(0.6, 0.8, 0.3),
        )
        # threshold=0.2 → all pass
        reasons_low = validate_required_landmarks(landmarks, threshold=0.2)
        assert reasons_low == []

        # threshold=0.5 → all fail
        reasons_high = validate_required_landmarks(landmarks, threshold=0.5)
        assert len(reasons_high) == 5


# ----------------------------------------------------------------
# 2.1-K: Threshold boundary behavior
# ----------------------------------------------------------------


class TestThresholdBoundary:
    """Test behavior exactly at the threshold."""

    def test_visibility_equal_to_threshold_accepted(self):
        """visibility == threshold → accepted."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.5),
            left_shoulder=(0.4, 0.5, 0.5),
            right_shoulder=(0.6, 0.5, 0.5),
            left_hip=(0.4, 0.8, 0.5),
            right_hip=(0.6, 0.8, 0.5),
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert reasons == []

    def test_visibility_just_below_threshold_rejected(self):
        """visibility < threshold → rejected."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.499),
            left_shoulder=(0.4, 0.5, 0.8),
            right_shoulder=(0.6, 0.5, 0.8),
            left_hip=(0.4, 0.8, 0.8),
            right_hip=(0.6, 0.8, 0.8),
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert len(reasons) == 1
        assert "NOSE" in reasons[0]

    def test_visibility_just_above_threshold_accepted(self):
        """visibility > threshold → accepted."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.501),
            left_shoulder=(0.4, 0.5, 0.501),
            right_shoulder=(0.6, 0.5, 0.501),
            left_hip=(0.4, 0.8, 0.501),
            right_hip=(0.6, 0.8, 0.501),
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert reasons == []

    def test_threshold_zero_accepts_all(self):
        """threshold=0.0 → all landmarks accepted (even visibility 0)."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.0),
            left_shoulder=(0.4, 0.5, 0.0),
            right_shoulder=(0.6, 0.5, 0.0),
            left_hip=(0.4, 0.8, 0.0),
            right_hip=(0.6, 0.8, 0.0),
        )
        reasons = validate_required_landmarks(landmarks, threshold=0.0)
        assert reasons == []


# ----------------------------------------------------------------
# 2.1-L: Mathematical regression (unchanged formulas)
# ----------------------------------------------------------------


class TestMathematicalRegression:
    """Confirm Phase 2 metric formulas remain unchanged after Phase 2.1."""

    def test_spine_zero_unchanged(self):
        angle = compute_spine_angle(
            left_shoulder_xy=(0.4, 0.4),
            right_shoulder_xy=(0.6, 0.4),
            left_hip_xy=(0.4, 0.8),
            right_hip_xy=(0.6, 0.8),
        )
        assert angle == pytest.approx(0.0, abs=0.01)

    def test_spine_45_unchanged(self):
        angle = compute_spine_angle(
            left_shoulder_xy=(0.8, 0.5),
            right_shoulder_xy=(0.8, 0.5),
            left_hip_xy=(0.5, 0.8),
            right_hip_xy=(0.5, 0.8),
        )
        assert angle == pytest.approx(45.0, abs=0.01)

    def test_neck_zero_unchanged(self):
        angle = compute_neck_tilt(
            left_shoulder_xy=(0.4, 0.6),
            right_shoulder_xy=(0.6, 0.6),
            nose_xy=(0.5, 0.3),
        )
        assert angle == pytest.approx(0.0, abs=0.01)

    def test_neck_45_unchanged(self):
        angle = compute_neck_tilt(
            left_shoulder_xy=(0.5, 0.6),
            right_shoulder_xy=(0.5, 0.6),
            nose_xy=(0.8, 0.3),
        )
        assert angle == pytest.approx(45.0, abs=0.01)

    def test_shoulder_diff_unchanged(self):
        result = compute_shoulder_height_difference((0.4, 0.45), (0.6, 0.55))
        assert result == pytest.approx(0.1)

    def test_hip_diff_unchanged(self):
        result = compute_hip_alignment((0.4, 0.75), (0.6, 0.85))
        assert result == pytest.approx(0.1)


# ----------------------------------------------------------------
# 2.1-M: Existing invalid geometry still causes skip (not crash)
# ----------------------------------------------------------------


class TestDegenerateGeometryRegression:
    """ValueError from calculate_angle() still causes frame rejection."""

    def test_coincident_points_returns_none(self):
        """All points coincide → ValueError → frame skipped."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.5, 1.0),
            left_shoulder=(0.5, 0.5, 1.0),
            right_shoulder=(0.5, 0.5, 1.0),
            left_hip=(0.5, 0.5, 1.0),
            right_hip=(0.5, 0.5, 1.0),
        )
        # All pass visibility but geometry is degenerate.
        result = compute_frame_metrics(landmarks, visibility_threshold=0.5)
        assert result is None


# ----------------------------------------------------------------
# 2.1-N: No fake values from rejected frames
# ----------------------------------------------------------------


class TestNoFakeValues:
    """Rejected frames must not contribute any metric values."""

    def test_low_confidence_contributes_nothing(self):
        """Low-confidence frame returns None, not zeros."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.01),
            left_shoulder=(0.4, 0.5, 0.01),
            right_shoulder=(0.6, 0.5, 0.01),
            left_hip=(0.4, 0.8, 0.01),
            right_hip=(0.6, 0.8, 0.01),
        )
        result = compute_frame_metrics(landmarks, visibility_threshold=0.5)
        assert result is None
        # Explicitly NOT zero, not NaN, not infinity.
        # None means "skip this frame entirely".

    def test_degenerate_geometry_contributes_nothing(self):
        """Degenerate-geometry frame returns None, not zeros."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.5, 1.0),
            left_shoulder=(0.5, 0.5, 1.0),
            right_shoulder=(0.5, 0.5, 1.0),
            left_hip=(0.5, 0.5, 1.0),
            right_hip=(0.5, 0.5, 1.0),
        )
        result = compute_frame_metrics(landmarks, visibility_threshold=0.5)
        assert result is None


# ================================================================
# PHASE 2.1 — Recalibration orchestration tests
# ================================================================


class TestRecalibrationOrchestration:
    """Test runtime recalibration using mocked camera/estimator.

    These tests verify the recalibration state machine without
    requiring a real webcam, GUI, or MediaPipe model.
    """

    def _make_mock_pose_result(self, landmarks):
        """Create a mock PoseResult with given smoothed landmarks."""
        mock = MagicMock()
        mock.pose_detected = True
        mock.smoothed_landmarks = landmarks
        mock.mediapipe_result = MagicMock()
        mock.mediapipe_result.pose_landmarks = [[]]
        return mock

    def _run_calibration_with_mocks(self, landmark_sets, min_samples=30):
        """Run calibration with fully mocked camera, estimator, time, cv2.

        camera.read() returns (True, MagicMock()) for each landmark set,
        then time expires so the loop exits naturally.
        """
        import src.calibration as cal
        import unittest.mock as um

        camera = MagicMock()
        estimator = MagicMock()

        # Build pose results.
        pose_results = [self._make_mock_pose_result(lm) for lm in landmark_sets]

        # camera.read() returns valid frames, then the time check
        # causes the loop to exit.
        frame_index = [0]
        n_frames = len(landmark_sets)

        def fake_read():
            idx = frame_index[0]
            if idx < n_frames:
                frame_index[0] += 1
                return (True, MagicMock())
            return (True, MagicMock())  # Extra frames if time hasn't expired.

        camera.read.side_effect = fake_read

        # estimator.process() returns pose results, cycling if needed.
        process_index = [0]
        def fake_process(frame):
            idx = process_index[0]
            if idx < len(pose_results):
                process_index[0] += 1
                return pose_results[idx]
            return pose_results[-1]  # Repeat last result if needed.
        estimator.process.side_effect = fake_process

        # Time mock: start at 0, then after n_frames calls to the loop
        # body, make time exceed duration.
        time_call_count = [0]
        duration = 100.0

        def fake_time():
            time_call_count[0] += 1
            # First call is start_time (= 0.0).
            # After n_frames * 3 calls (roughly 3 time() calls per iteration),
            # exceed duration to end the loop.
            if time_call_count[0] <= 1:
                return 0.0
            if frame_index[0] >= n_frames:
                return duration + 1.0  # Exceed duration → exit.
            return 0.01 * time_call_count[0]

        with um.patch.object(cal, "time") as mock_time, \
             um.patch.object(cal, "cv2") as mock_cv2:
            mock_time.time.side_effect = fake_time
            mock_cv2.waitKey.return_value = 0

            result = cal.run_calibration(
                camera, estimator,
                duration_seconds=duration,
                min_samples=min_samples,
            )

        return result, camera, estimator

    def test_successful_calibration_produces_result(self):
        """Sufficient valid frames → CalibrationResult returned."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.99),
            left_shoulder=(0.4, 0.5, 0.95),
            right_shoulder=(0.6, 0.5, 0.92),
            left_hip=(0.4, 0.8, 0.88),
            right_hip=(0.6, 0.8, 0.90),
        )
        landmark_sets = [landmarks] * 35

        result, _, _ = self._run_calibration_with_mocks(landmark_sets)

        assert isinstance(result, CalibrationResult)
        assert result.valid_samples >= 30
        assert isinstance(result.spine_angle.mean, float)
        assert isinstance(result.spine_angle.std, float)

    def test_failed_calibration_raises_runtime_error(self):
        """Insufficient valid frames → RuntimeError."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.99),
            left_shoulder=(0.4, 0.5, 0.95),
            right_shoulder=(0.6, 0.5, 0.92),
            left_hip=(0.4, 0.8, 0.88),
            right_hip=(0.6, 0.8, 0.90),
        )
        landmark_sets = [landmarks] * 5  # Too few.

        with pytest.raises(RuntimeError, match="Calibration failed"):
            self._run_calibration_with_mocks(landmark_sets, min_samples=30)

    def test_recalibration_replaces_on_success(self):
        """Successful recalibration replaces the previous CalibrationResult."""
        lm1 = _make_landmarks(
            nose=(0.5, 0.3, 0.99),
            left_shoulder=(0.4, 0.5, 0.95),
            right_shoulder=(0.6, 0.5, 0.92),
            left_hip=(0.4, 0.8, 0.88),
            right_hip=(0.6, 0.8, 0.90),
        )
        lm2 = _make_landmarks(
            nose=(0.5, 0.3, 0.99),
            left_shoulder=(0.4, 0.48, 0.95),  # Different shoulder.
            right_shoulder=(0.6, 0.5, 0.92),
            left_hip=(0.4, 0.8, 0.88),
            right_hip=(0.6, 0.8, 0.90),
        )

        result1, _, _ = self._run_calibration_with_mocks([lm1] * 35)
        result2, _, _ = self._run_calibration_with_mocks([lm2] * 35)

        assert isinstance(result1, CalibrationResult)
        assert isinstance(result2, CalibrationResult)
        assert result1.shoulder_height_difference.mean != result2.shoulder_height_difference.mean

    def test_failed_recalibration_preserves_previous(self):
        """Failed recalibration → previous result preserved."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.99),
            left_shoulder=(0.4, 0.5, 0.95),
            right_shoulder=(0.6, 0.5, 0.92),
            left_hip=(0.4, 0.8, 0.88),
            right_hip=(0.6, 0.8, 0.90),
        )

        original_result, _, _ = self._run_calibration_with_mocks(
            [landmarks] * 35,
        )

        # Attempt recalibration with too few frames.
        calibration_result = original_result
        try:
            new_result, _, _ = self._run_calibration_with_mocks(
                [landmarks] * 3, min_samples=30,
            )
            calibration_result = new_result
        except RuntimeError:
            pass  # Preserve previous.

        assert calibration_result is original_result

    def test_ema_reset_called_during_calibration(self):
        """run_calibration should call reset_smoothing on the estimator."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.99),
            left_shoulder=(0.4, 0.5, 0.95),
            right_shoulder=(0.6, 0.5, 0.92),
            left_hip=(0.4, 0.8, 0.88),
            right_hip=(0.6, 0.8, 0.90),
        )

        _, _, estimator = self._run_calibration_with_mocks([landmarks] * 35)

        estimator.reset_smoothing.assert_called_once()

    def test_no_new_ema_filter_created(self):
        """Recalibration reuses the same estimator (no new EMA filter)."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.99),
            left_shoulder=(0.4, 0.5, 0.95),
            right_shoulder=(0.6, 0.5, 0.92),
            left_hip=(0.4, 0.8, 0.88),
            right_hip=(0.6, 0.8, 0.90),
        )

        _, _, estimator = self._run_calibration_with_mocks([landmarks] * 35)

        # The same estimator's process() was called (not a new one).
        assert estimator.process.call_count == 35

    def test_required_landmarks_constant(self):
        """REQUIRED_LANDMARKS must contain all five calibration landmarks."""
        assert set(REQUIRED_LANDMARKS) == {NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
                                           LEFT_HIP, RIGHT_HIP}


# ================================================================
# Validate_required_landmarks edge cases
# ================================================================


class TestValidateRequiredLandmarksEdgeCases:
    """Edge cases for the validation helper."""

    def test_empty_list(self):
        reasons = validate_required_landmarks([])
        assert len(reasons) == 5  # All five missing.

    def test_no_visibility_attribute(self):
        """If a landmark has no visibility → rejected."""
        @dataclass
        class BarePoint:
            x: float
            y: float
            z: float = 0.0

        landmarks = [BarePoint(x=0.5, y=0.5) for _ in range(33)]
        reasons = validate_required_landmarks(landmarks, threshold=0.5)
        assert len(reasons) == 5
        assert all("no visibility data" in r for r in reasons)

    def test_custom_required_indices(self):
        """Can validate a subset of landmarks."""
        landmarks = _make_landmarks(
            nose=(0.5, 0.3, 0.99),
            left_shoulder=(0.4, 0.5, 0.01),  # low
            right_shoulder=(0.6, 0.5, 0.01),  # low
            left_hip=(0.4, 0.8, 0.88),
            right_hip=(0.6, 0.8, 0.90),
        )
        # Only check hips — should pass.
        reasons = validate_required_landmarks(
            landmarks,
            required_indices=[LEFT_HIP, RIGHT_HIP],
            threshold=0.5,
        )
        assert reasons == []

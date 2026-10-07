"""Tests for Phase 4B — Compensation Flagging (Baseline-Relative Thresholding).

All tests use deterministic synthetic data. No webcam, OpenCV,
MediaPipe inference, GUI, or random data.

Phase 4B — Compensation Flagging.
"""

import pytest

from src.calibration import MetricBaseline, CalibrationResult
from src.compensation import (
    compute_deviation,
    compute_threshold,
    evaluate_metric,
    evaluate_compensation,
    CompensationMetricResult,
    CompensationResult,
)
from src.config import TORSO_LEAN_FLOOR, SHOULDER_HIKE_FLOOR, NECK_TILT_FLOOR
from src.metrics.posture import PostureMetrics


# ================================================================
# Helpers
# ================================================================


def _make_baseline(mean: float, std: float) -> MetricBaseline:
    return MetricBaseline(mean=mean, std=std)


def _make_calibration(
    spine_mean: float = 2.0,
    spine_std: float = 1.0,
    shoulder_mean: float = 0.01,
    shoulder_std: float = 0.005,
    neck_mean: float = 1.5,
    neck_std: float = 0.8,
) -> CalibrationResult:
    return CalibrationResult(
        spine_angle=_make_baseline(spine_mean, spine_std),
        shoulder_height_difference=_make_baseline(shoulder_mean, shoulder_std),
        neck_tilt=_make_baseline(neck_mean, neck_std),
        hip_alignment=_make_baseline(0.0, 0.0),
        hip_rotation=_make_baseline(0.0, 0.0),
        lateral_trunk_lean=_make_baseline(0.0, 0.0),
        valid_samples=100,
        skipped_samples=5,
        duration_seconds=10.0,
    )


# ================================================================
# DEVIATION
# ================================================================


class TestComputeDeviation:
    """Deviation = abs(current - baseline_mean)."""

    def test_basic_deviation(self):
        assert compute_deviation(12.0, 2.0) == pytest.approx(10.0)

    def test_zero_deviation(self):
        assert compute_deviation(5.0, 5.0) == pytest.approx(0.0)

    def test_negative_direction(self):
        """current < baseline_mean → still positive deviation."""
        assert compute_deviation(1.0, 5.0) == pytest.approx(4.0)

    def test_small_deviation(self):
        assert compute_deviation(2.5, 2.0) == pytest.approx(0.5)


# ================================================================
# THRESHOLD
# ================================================================


class TestComputeThreshold:
    """Threshold = max(2 × baseline_std, fixed_floor)."""

    def test_floor_dominates(self):
        """2 × 2 = 4 < floor 5 → threshold = 5."""
        assert compute_threshold(2.0, 5.0) == pytest.approx(5.0)

    def test_std_dominates(self):
        """2 × 4 = 8 > floor 5 → threshold = 8."""
        assert compute_threshold(4.0, 5.0) == pytest.approx(8.0)

    def test_equal(self):
        """2 × 2.5 = 5.0 == floor 5 → threshold = 5."""
        assert compute_threshold(2.5, 5.0) == pytest.approx(5.0)

    def test_zero_std(self):
        """Zero std → floor prevents zero-width threshold."""
        assert compute_threshold(0.0, 5.0) == pytest.approx(5.0)

    def test_very_large_std(self):
        """Large std → threshold = 2 × std."""
        assert compute_threshold(100.0, 5.0) == pytest.approx(200.0)


# ================================================================
# FLAGGING
# ================================================================


class TestFlagging:
    """Flag uses strict > (NOT >=)."""

    def test_below_threshold_not_flagged(self):
        result = evaluate_metric(5.0, _make_baseline(2.0, 1.0), 5.0)
        # deviation = 3.0, threshold = 5.0 → not flagged
        assert result.flagged is False

    def test_at_threshold_not_flagged(self):
        """deviation == threshold → NOT flagged (strict >)."""
        # baseline_mean=2, std=1 → threshold=max(2, 5)=5
        # current=7 → deviation=5 → 5 > 5 is False
        result = evaluate_metric(7.0, _make_baseline(2.0, 1.0), 5.0)
        assert result.deviation == pytest.approx(5.0)
        assert result.threshold == pytest.approx(5.0)
        assert result.flagged is False

    def test_above_threshold_flagged(self):
        # current=8 → deviation=6 → 6 > 5 is True
        result = evaluate_metric(8.0, _make_baseline(2.0, 1.0), 5.0)
        assert result.deviation == pytest.approx(6.0)
        assert result.flagged is True

    def test_just_above_threshold_flagged(self):
        # current=7.01 → deviation=5.01 → 5.01 > 5 is True
        result = evaluate_metric(7.01, _make_baseline(2.0, 1.0), 5.0)
        assert result.flagged is True


# ================================================================
# BASELINE-RELATIVE BEHAVIOR
# ================================================================


class TestBaselineRelative:
    """Verify baseline mean is NOT added to threshold."""

    def test_mean_plus_threshold_not_flagged(self):
        """current = baseline_mean + threshold → deviation == threshold → NOT flagged."""
        baseline = _make_baseline(mean=10.0, std=1.0)
        # threshold = max(2, 5) = 5
        # current = 10 + 5 = 15 → deviation = 5 → 5 > 5 is False
        result = evaluate_metric(15.0, baseline, 5.0)
        assert result.deviation == pytest.approx(5.0)
        assert result.flagged is False

    def test_above_mean_plus_threshold_flagged(self):
        """current > baseline_mean + threshold → flagged."""
        baseline = _make_baseline(mean=10.0, std=1.0)
        # current = 15.1 → deviation = 5.1 → 5.1 > 5 is True
        result = evaluate_metric(15.1, baseline, 5.0)
        assert result.flagged is True

    def test_deviation_computed_first(self):
        """Deviation = current - mean, THEN compared to threshold."""
        result = evaluate_metric(12.0, _make_baseline(2.0, 1.0), 5.0)
        assert result.deviation == pytest.approx(10.0)
        assert result.threshold == pytest.approx(5.0)
        assert result.flagged is True  # 10 > 5


# ================================================================
# NONE HANDLING
# ================================================================


class TestNoneHandling:
    """None metric → no flag, no fake deviation."""

    def test_none_current_not_flagged(self):
        result = evaluate_metric(None, _make_baseline(2.0, 1.0), 5.0)
        assert result.flagged is False

    def test_none_current_deviation_is_none(self):
        result = evaluate_metric(None, _make_baseline(2.0, 1.0), 5.0)
        assert result.deviation is None

    def test_none_current_preserves_baseline(self):
        result = evaluate_metric(None, _make_baseline(2.0, 1.0), 5.0)
        assert result.baseline_mean == pytest.approx(2.0)
        assert result.baseline_std == pytest.approx(1.0)
        assert result.threshold == pytest.approx(5.0)

    def test_none_current_value_stored(self):
        result = evaluate_metric(None, _make_baseline(2.0, 1.0), 5.0)
        assert result.current_value is None


# ================================================================
# EACH METRIC INDEPENDENTLY
# ================================================================


class TestTorsoLeanMapping:
    """Torso lean uses spine_angle baseline + TORSO_LEAN_FLOOR."""

    def test_torso_lean_uses_spine_angle(self):
        cal = _make_calibration(spine_mean=3.0, spine_std=1.0)
        posture = PostureMetrics(torso_lean=15.0, shoulder_height_difference=0.01, neck_tilt=1.5, hip_rotation=None, lateral_trunk_lean=None)
        comp = evaluate_compensation(posture, cal)
        assert comp is not None
        assert comp.torso_lean.baseline_mean == pytest.approx(3.0)
        assert comp.torso_lean.deviation == pytest.approx(12.0)

    def test_torso_lean_uses_correct_floor(self):
        cal = _make_calibration(spine_std=0.5)
        posture = PostureMetrics(torso_lean=10.0, shoulder_height_difference=0.01, neck_tilt=1.5, hip_rotation=None, lateral_trunk_lean=None)
        comp = evaluate_compensation(posture, cal)
        # threshold = max(2*0.5, TORSO_LEAN_FLOOR) = max(1, 5) = 5
        assert comp.torso_lean.threshold == pytest.approx(TORSO_LEAN_FLOOR)


class TestShoulderHikeMapping:
    """Shoulder hike uses shoulder_height_difference baseline + SHOULDER_HIKE_FLOOR."""

    def test_shoulder_hike_uses_correct_baseline(self):
        cal = _make_calibration(shoulder_mean=0.01, shoulder_std=0.003)
        posture = PostureMetrics(torso_lean=2.0, shoulder_height_difference=5.0, neck_tilt=1.5, hip_rotation=None, lateral_trunk_lean=None)
        comp = evaluate_compensation(posture, cal)
        assert comp.shoulder_hike.baseline_mean == pytest.approx(0.01)
        assert comp.shoulder_hike.deviation == pytest.approx(4.99)

    def test_shoulder_hike_uses_correct_floor(self):
        cal = _make_calibration(shoulder_std=0.001)
        posture = PostureMetrics(torso_lean=2.0, shoulder_height_difference=5.0, neck_tilt=1.5, hip_rotation=None, lateral_trunk_lean=None)
        comp = evaluate_compensation(posture, cal)
        # threshold = max(2*0.001, SHOULDER_HIKE_FLOOR) = max(0.002, 3) = 3
        assert comp.shoulder_hike.threshold == pytest.approx(SHOULDER_HIKE_FLOOR)


class TestNeckTiltMapping:
    """Neck tilt uses neck_tilt baseline + NECK_TILT_FLOOR."""

    def test_neck_tilt_uses_correct_baseline(self):
        cal = _make_calibration(neck_mean=1.0, neck_std=0.5)
        posture = PostureMetrics(torso_lean=2.0, shoulder_height_difference=0.01, neck_tilt=10.0, hip_rotation=None, lateral_trunk_lean=None)
        comp = evaluate_compensation(posture, cal)
        assert comp.neck_tilt.baseline_mean == pytest.approx(1.0)
        assert comp.neck_tilt.deviation == pytest.approx(9.0)

    def test_neck_tilt_uses_correct_floor(self):
        cal = _make_calibration(neck_std=0.5)
        posture = PostureMetrics(torso_lean=2.0, shoulder_height_difference=0.01, neck_tilt=10.0, hip_rotation=None, lateral_trunk_lean=None)
        comp = evaluate_compensation(posture, cal)
        # threshold = max(2*0.5, NECK_TILT_FLOOR) = max(1, 5) = 5
        assert comp.neck_tilt.threshold == pytest.approx(NECK_TILT_FLOOR)


# ================================================================
# METRIC CROSS-MAPPING
# ================================================================


class TestMetricCrossMapping:
    """Each metric uses its OWN baseline, not another metric's."""

    def test_torso_not_using_shoulder_baseline(self):
        cal = _make_calibration(spine_mean=2.0, shoulder_mean=100.0)
        posture = PostureMetrics(torso_lean=12.0, shoulder_height_difference=0.01, neck_tilt=1.5, hip_rotation=None, lateral_trunk_lean=None)
        comp = evaluate_compensation(posture, cal)
        # If torso accidentally used shoulder baseline (100), deviation would be 88
        assert comp.torso_lean.baseline_mean == pytest.approx(2.0)
        assert comp.torso_lean.deviation == pytest.approx(10.0)

    def test_shoulder_not_using_neck_baseline(self):
        cal = _make_calibration(shoulder_mean=0.01, neck_mean=100.0)
        posture = PostureMetrics(torso_lean=2.0, shoulder_height_difference=5.0, neck_tilt=1.5, hip_rotation=None, lateral_trunk_lean=None)
        comp = evaluate_compensation(posture, cal)
        assert comp.shoulder_hike.baseline_mean == pytest.approx(0.01)

    def test_neck_not_using_torso_baseline(self):
        cal = _make_calibration(neck_mean=1.0, spine_mean=100.0)
        posture = PostureMetrics(torso_lean=2.0, shoulder_height_difference=0.01, neck_tilt=10.0, hip_rotation=None, lateral_trunk_lean=None)
        comp = evaluate_compensation(posture, cal)
        assert comp.neck_tilt.baseline_mean == pytest.approx(1.0)
        assert comp.neck_tilt.deviation == pytest.approx(9.0)


# ================================================================
# ZERO STANDARD DEVIATION
# ================================================================


class TestZeroStandardDeviation:
    """Zero std → floor prevents zero-width threshold."""

    def test_zero_std_uses_floor(self):
        result = evaluate_metric(10.0, _make_baseline(2.0, 0.0), 5.0)
        assert result.threshold == pytest.approx(5.0)

    def test_zero_std_still_flags_when_exceeded(self):
        # deviation = 8, threshold = 5 → flagged
        result = evaluate_metric(10.0, _make_baseline(2.0, 0.0), 5.0)
        assert result.flagged is True


# ================================================================
# CALIBRATION UNAVAILABLE
# ================================================================


class TestCalibrationUnavailable:
    """No calibration result → no compensation evaluation."""

    def test_none_calibration_returns_none(self):
        posture = PostureMetrics(torso_lean=5.0, shoulder_height_difference=0.01, neck_tilt=1.0, hip_rotation=None, lateral_trunk_lean=None)
        comp = evaluate_compensation(posture, None)
        assert comp is None


# ================================================================
# REALISTIC EXAMPLES
# ================================================================


class TestRealisticCleanMovement:
    """Clean movement stays within threshold."""

    def test_clean_movement_no_flags(self):
        cal = _make_calibration(
            spine_mean=2.0, spine_std=1.0,
            shoulder_mean=0.01, shoulder_std=0.005,
            neck_mean=1.5, neck_std=0.8,
        )
        posture = PostureMetrics(
            torso_lean=3.0,     # deviation=1, threshold=max(2,5)=5 → CLEAR
            shoulder_height_difference=0.015,  # deviation=0.005, threshold=max(0.01,3)=3 → CLEAR
            neck_tilt=2.0,      # deviation=0.5, threshold=max(1.6,5)=5 → CLEAR
        )
        comp = evaluate_compensation(posture, cal)
        assert comp.torso_lean.flagged is False
        assert comp.shoulder_hike.flagged is False
        assert comp.neck_tilt.flagged is False


class TestRealisticTorsoCompensation:
    """Torso lean exceeds threshold, others clear."""

    def test_torso_compensation(self):
        cal = _make_calibration(spine_mean=2.0, spine_std=1.0)
        posture = PostureMetrics(
            torso_lean=15.0,    # deviation=13, threshold=5 → FLAGGED
            shoulder_height_difference=0.012,
            neck_tilt=2.0, hip_rotation=None, lateral_trunk_lean=None,
        )
        comp = evaluate_compensation(posture, cal)
        assert comp.torso_lean.flagged is True
        assert comp.shoulder_hike.flagged is False
        assert comp.neck_tilt.flagged is False


class TestRealisticShoulderHike:
    """Shoulder hike flagged, others clear."""

    def test_shoulder_hike(self):
        cal = _make_calibration(shoulder_mean=0.01, shoulder_std=0.003)
        posture = PostureMetrics(
            torso_lean=2.5,
            shoulder_height_difference=10.0,  # deviation≈10, threshold=3 → FLAGGED
            neck_tilt=1.8, hip_rotation=None, lateral_trunk_lean=None,
        )
        comp = evaluate_compensation(posture, cal)
        assert comp.torso_lean.flagged is False
        assert comp.shoulder_hike.flagged is True
        assert comp.neck_tilt.flagged is False


class TestRealisticNeckTilt:
    """Neck tilt flagged, others clear."""

    def test_neck_tilt_compensation(self):
        cal = _make_calibration(neck_mean=1.5, neck_std=0.8)
        posture = PostureMetrics(
            torso_lean=2.5,
            shoulder_height_difference=0.012,
            neck_tilt=12.0, hip_rotation=None, lateral_trunk_lean=None,     # deviation=10.5, threshold=5 → FLAGGED
        )
        comp = evaluate_compensation(posture, cal)
        assert comp.torso_lean.flagged is False
        assert comp.shoulder_hike.flagged is False
        assert comp.neck_tilt.flagged is True


# ================================================================
# RESULT STRUCTURE
# ================================================================


class TestResultStructure:
    """CompensationMetricResult retains all intermediate values."""

    def test_all_fields_present(self):
        result = evaluate_metric(12.0, _make_baseline(2.0, 3.0), 5.0)
        assert result.current_value == pytest.approx(12.0)
        assert result.baseline_mean == pytest.approx(2.0)
        assert result.baseline_std == pytest.approx(3.0)
        assert result.deviation == pytest.approx(10.0)
        assert result.threshold == pytest.approx(6.0)  # max(6, 5) = 6
        assert result.flagged is True  # 10 > 6

    def test_compensation_result_has_three_fields(self):
        cal = _make_calibration()
        posture = PostureMetrics(torso_lean=2.0, shoulder_height_difference=0.01, neck_tilt=1.5, hip_rotation=None, lateral_trunk_lean=None)
        comp = evaluate_compensation(posture, cal)
        assert isinstance(comp.torso_lean, CompensationMetricResult)
        assert isinstance(comp.shoulder_hike, CompensationMetricResult)
        assert isinstance(comp.neck_tilt, CompensationMetricResult)


# ================================================================
# PARTIAL NONE METRICS
# ================================================================


class TestPartialNone:
    """One metric None, others still evaluated."""

    def test_torso_none_others_evaluated(self):
        cal = _make_calibration()
        posture = PostureMetrics(
            torso_lean=None,
            shoulder_height_difference=0.01,
            neck_tilt=1.5, hip_rotation=None, lateral_trunk_lean=None,
        )
        comp = evaluate_compensation(posture, cal)
        assert comp.torso_lean.flagged is False
        assert comp.torso_lean.deviation is None
        assert comp.shoulder_hike.deviation is not None
        assert comp.neck_tilt.deviation is not None

    def test_all_none_no_flags(self):
        cal = _make_calibration()
        posture = PostureMetrics(
            torso_lean=None,
            shoulder_height_difference=None,
            neck_tilt=None, hip_rotation=None, lateral_trunk_lean=None,
        )
        comp = evaluate_compensation(posture, cal)
        assert comp.torso_lean.flagged is False
        assert comp.shoulder_hike.flagged is False
        assert comp.neck_tilt.flagged is False


# ================================================================
# CONFIG CONSTANTS
# ================================================================


class TestConfigConstants:
    """Verify config constants exist and have expected values."""

    def test_torso_lean_floor(self):
        assert TORSO_LEAN_FLOOR == 4

    def test_shoulder_hike_floor(self):
        assert SHOULDER_HIKE_FLOOR == 1

    def test_neck_tilt_floor(self):
        assert NECK_TILT_FLOOR == 4

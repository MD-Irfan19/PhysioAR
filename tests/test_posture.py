"""Tests for Phase 4A — Posture Metrics (Compute Only).

All tests use deterministic synthetic data. No webcam, OpenCV,
MediaPipe inference, GUI, or random data.

Tests cover:
  - Torso lean: upright, known lean, missing landmarks, degenerate
  - Shoulder height diff: level, displaced, missing landmarks
  - Neck tilt: upright, known tilt, missing landmarks, degenerate
  - Independence: cross-contamination checks
  - Smoothed input: uses supplied coordinates directly
  - Frame-level result: partial availability

Phase 4A — Posture Metrics (Compute Only).
"""

from dataclasses import dataclass

import pytest

from src.metrics.posture import (
    PostureMetrics,
    compute_torso_lean,
    compute_shoulder_height_diff,
    compute_neck_tilt_metric,
    compute_posture_metrics,
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


# Landmark indices (matching calibration.py / posture.py).
_NOSE = 0
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_HIP = 23
_R_HIP = 24


def _make_landmarks(overrides: dict | None = None) -> list[MockSmoothedLandmark]:
    """Build 33 mock landmarks. Default all at (0.5, 0.5) visibility=1.0.

    overrides: {index: (x, y)} or {index: (x, y, visibility)}.
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
# TORSO LEAN
# ================================================================


class TestTorsoLeanUpright:
    """Upright torso → approximately 0°."""

    def test_perfectly_vertical_torso(self):
        """Shoulders directly above hips → 0° lean."""
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.3),
            _R_SHOULDER: (0.6, 0.3),
            _L_HIP: (0.4, 0.7),
            _R_HIP: (0.6, 0.7),
        })
        lean = compute_torso_lean(landmarks)
        assert lean is not None
        assert lean == pytest.approx(0.0, abs=0.1)


class TestTorsoLeanKnownAngle:
    """Known lean angle → expected value."""

    def test_lean_right(self):
        """Shoulder midpoint shifted right of hip midpoint.

        Hip midpoint: (0.5, 0.7)
        Shoulder midpoint: (0.7, 0.3) — shifted right
        Vector: (0.2, -0.4) from hip to shoulder
        Vertical ref: (0.0, -1.0) from hip upward

        Angle between (0.2, -0.4) and (0.0, -1.0):
        cos(θ) = (0*0.2 + (-1)*(-0.4)) / (1 * √(0.04+0.16))
               = 0.4 / √0.2 = 0.4 / 0.4472 ≈ 0.8944
        θ ≈ 26.57°
        """
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.6, 0.3),
            _R_SHOULDER: (0.8, 0.3),
            _L_HIP: (0.4, 0.7),
            _R_HIP: (0.6, 0.7),
        })
        lean = compute_torso_lean(landmarks)
        assert lean is not None
        assert lean == pytest.approx(26.57, abs=0.5)

    def test_lean_left(self):
        """Shoulder midpoint shifted left of hip midpoint.

        Hip midpoint: (0.5, 0.7)
        Shoulder midpoint: (0.3, 0.3) — shifted left
        Magnitude should be the same as lean_right (symmetric).
        """
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.2, 0.3),
            _R_SHOULDER: (0.4, 0.3),
            _L_HIP: (0.4, 0.7),
            _R_HIP: (0.6, 0.7),
        })
        lean = compute_torso_lean(landmarks)
        assert lean is not None
        assert lean == pytest.approx(26.57, abs=0.5)


class TestTorsoLeanMissingLandmarks:
    """Missing landmarks → None."""

    def test_missing_left_shoulder(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.3, 0.1),  # Low visibility.
            _R_SHOULDER: (0.6, 0.3),
            _L_HIP: (0.4, 0.7),
            _R_HIP: (0.6, 0.7),
        })
        assert compute_torso_lean(landmarks, visibility_threshold=0.5) is None

    def test_missing_right_hip(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.3),
            _R_SHOULDER: (0.6, 0.3),
            _L_HIP: (0.4, 0.7),
            _R_HIP: (0.6, 0.7, 0.1),  # Low visibility.
        })
        assert compute_torso_lean(landmarks, visibility_threshold=0.5) is None

    def test_empty_landmarks(self):
        assert compute_torso_lean([]) is None


class TestTorsoLeanDegenerateGeometry:
    """Degenerate geometry → None."""

    def test_coincident_midpoints(self):
        """Shoulder midpoint == hip midpoint → None."""
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5),
            _R_SHOULDER: (0.6, 0.5),
            _L_HIP: (0.4, 0.5),
            _R_HIP: (0.6, 0.5),
        })
        assert compute_torso_lean(landmarks) is None


# ================================================================
# SHOULDER HEIGHT DIFFERENCE
# ================================================================


class TestShoulderHeightLevel:
    """Equal shoulder heights → 0."""

    def test_level_shoulders(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.35),
            _R_SHOULDER: (0.6, 0.35),
        })
        diff = compute_shoulder_height_diff(landmarks)
        assert diff is not None
        assert diff == pytest.approx(0.0)


class TestShoulderHeightDisplaced:
    """Vertical displacement → expected difference."""

    def test_left_higher(self):
        """Left shoulder higher (smaller y in image coords)."""
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.30),
            _R_SHOULDER: (0.6, 0.35),
        })
        diff = compute_shoulder_height_diff(landmarks)
        assert diff is not None
        assert diff == pytest.approx(0.05, abs=0.001)

    def test_right_higher(self):
        """Right shoulder higher → same magnitude."""
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.35),
            _R_SHOULDER: (0.6, 0.30),
        })
        diff = compute_shoulder_height_diff(landmarks)
        assert diff is not None
        assert diff == pytest.approx(0.05, abs=0.001)

    def test_horizontal_shift_no_effect(self):
        """Shoulders at same y but different x → 0 height diff."""
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.3, 0.35),
            _R_SHOULDER: (0.7, 0.35),
        })
        diff = compute_shoulder_height_diff(landmarks)
        assert diff is not None
        assert diff == pytest.approx(0.0)


class TestShoulderHeightMissing:
    """Missing shoulders → None."""

    def test_missing_left(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.3, 0.1),
            _R_SHOULDER: (0.6, 0.3),
        })
        assert compute_shoulder_height_diff(
            landmarks, visibility_threshold=0.5,
        ) is None

    def test_missing_right(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.3),
            _R_SHOULDER: (0.6, 0.3, 0.1),
        })
        assert compute_shoulder_height_diff(
            landmarks, visibility_threshold=0.5,
        ) is None


# ================================================================
# NECK TILT
# ================================================================


class TestNeckTiltUpright:
    """Upright head → approximately 0°."""

    def test_nose_directly_above_shoulder_midpoint(self):
        """Nose at same x as shoulder midpoint, above → 0°."""
        landmarks = _make_landmarks({
            _NOSE: (0.5, 0.15),
            _L_SHOULDER: (0.4, 0.35),
            _R_SHOULDER: (0.6, 0.35),
        })
        tilt = compute_neck_tilt_metric(landmarks)
        assert tilt is not None
        assert tilt == pytest.approx(0.0, abs=0.1)


class TestNeckTiltKnownAngle:
    """Known tilt → expected value."""

    def test_tilt_right(self):
        """Nose shifted right of shoulder midpoint.

        Shoulder midpoint: (0.5, 0.35)
        Nose: (0.7, 0.15) — shifted right
        Vector: (0.2, -0.2) from shoulder_mid to nose
        Vertical ref: (0.0, -1.0)

        cos(θ) = (0*0.2 + (-1)*(-0.2)) / (1 * √(0.04+0.04))
               = 0.2 / 0.2828 ≈ 0.7071
        θ ≈ 45°
        """
        landmarks = _make_landmarks({
            _NOSE: (0.7, 0.15),
            _L_SHOULDER: (0.4, 0.35),
            _R_SHOULDER: (0.6, 0.35),
        })
        tilt = compute_neck_tilt_metric(landmarks)
        assert tilt is not None
        assert tilt == pytest.approx(45.0, abs=0.5)

    def test_tilt_left(self):
        """Nose shifted left → same magnitude as right (symmetric)."""
        landmarks = _make_landmarks({
            _NOSE: (0.3, 0.15),
            _L_SHOULDER: (0.4, 0.35),
            _R_SHOULDER: (0.6, 0.35),
        })
        tilt = compute_neck_tilt_metric(landmarks)
        assert tilt is not None
        assert tilt == pytest.approx(45.0, abs=0.5)


class TestNeckTiltMissing:
    """Missing head/neck landmark → None."""

    def test_missing_nose(self):
        landmarks = _make_landmarks({
            _NOSE: (0.5, 0.15, 0.1),
            _L_SHOULDER: (0.4, 0.35),
            _R_SHOULDER: (0.6, 0.35),
        })
        assert compute_neck_tilt_metric(
            landmarks, visibility_threshold=0.5,
        ) is None

    def test_missing_left_shoulder(self):
        landmarks = _make_landmarks({
            _NOSE: (0.5, 0.15),
            _L_SHOULDER: (0.4, 0.35, 0.1),
            _R_SHOULDER: (0.6, 0.35),
        })
        assert compute_neck_tilt_metric(
            landmarks, visibility_threshold=0.5,
        ) is None


class TestNeckTiltDegenerate:
    """Degenerate geometry → None."""

    def test_nose_at_shoulder_midpoint(self):
        """Nose coincides with shoulder midpoint → None."""
        landmarks = _make_landmarks({
            _NOSE: (0.5, 0.35),
            _L_SHOULDER: (0.4, 0.35),
            _R_SHOULDER: (0.6, 0.35),
        })
        assert compute_neck_tilt_metric(landmarks) is None


# ================================================================
# INDEPENDENCE / CROSS-CONTAMINATION
# ================================================================


class TestMetricIndependence:
    """Each metric depends only on its required landmarks."""

    def test_torso_lean_independent_of_nose(self):
        """Moving nose should NOT change torso lean."""
        base = {
            _L_SHOULDER: (0.4, 0.3),
            _R_SHOULDER: (0.6, 0.3),
            _L_HIP: (0.4, 0.7),
            _R_HIP: (0.6, 0.7),
        }
        lm1 = _make_landmarks({**base, _NOSE: (0.5, 0.1)})
        lm2 = _make_landmarks({**base, _NOSE: (0.7, 0.1)})

        lean1 = compute_torso_lean(lm1)
        lean2 = compute_torso_lean(lm2)

        assert lean1 is not None and lean2 is not None
        assert lean1 == pytest.approx(lean2, abs=0.001)

    def test_neck_tilt_independent_of_hips(self):
        """Moving hips should NOT change neck tilt."""
        base = {
            _NOSE: (0.5, 0.15),
            _L_SHOULDER: (0.4, 0.35),
            _R_SHOULDER: (0.6, 0.35),
        }
        lm1 = _make_landmarks({**base, _L_HIP: (0.4, 0.7), _R_HIP: (0.6, 0.7)})
        lm2 = _make_landmarks({**base, _L_HIP: (0.3, 0.8), _R_HIP: (0.7, 0.8)})

        tilt1 = compute_neck_tilt_metric(lm1)
        tilt2 = compute_neck_tilt_metric(lm2)

        assert tilt1 is not None and tilt2 is not None
        assert tilt1 == pytest.approx(tilt2, abs=0.001)

    def test_shoulder_height_independent_of_elbows(self):
        """Shoulder height diff should NOT depend on elbow positions."""
        base = {
            _L_SHOULDER: (0.4, 0.30),
            _R_SHOULDER: (0.6, 0.35),
        }
        # Elbow indices: 13 (left), 14 (right).
        lm1 = _make_landmarks({**base, 13: (0.3, 0.5), 14: (0.7, 0.5)})
        lm2 = _make_landmarks({**base, 13: (0.3, 0.9), 14: (0.7, 0.1)})

        diff1 = compute_shoulder_height_diff(lm1)
        diff2 = compute_shoulder_height_diff(lm2)

        assert diff1 is not None and diff2 is not None
        assert diff1 == pytest.approx(diff2, abs=0.001)

    def test_shoulder_height_independent_of_wrists(self):
        """Shoulder height diff should NOT depend on wrist positions."""
        base = {
            _L_SHOULDER: (0.4, 0.30),
            _R_SHOULDER: (0.6, 0.35),
        }
        # Wrist indices: 15 (left), 16 (right).
        lm1 = _make_landmarks({**base, 15: (0.2, 0.6), 16: (0.8, 0.6)})
        lm2 = _make_landmarks({**base, 15: (0.5, 0.1), 16: (0.5, 0.9)})

        diff1 = compute_shoulder_height_diff(lm1)
        diff2 = compute_shoulder_height_diff(lm2)

        assert diff1 is not None and diff2 is not None
        assert diff1 == pytest.approx(diff2, abs=0.001)


# ================================================================
# SMOOTHED INPUT
# ================================================================


class TestUsesSmoothedCoordinates:
    """Computation uses supplied coordinates, not raw_x/raw_y."""

    def test_torso_lean_uses_smoothed(self):
        """SmoothedLandmark.x/.y should be used, not raw_x/raw_y."""
        lm_l_shoulder = MockSmoothedLandmark(
            x=0.4, y=0.3,       # Smoothed (upright).
            raw_x=0.6, raw_y=0.3,  # Raw (different).
        )
        lm_r_shoulder = MockSmoothedLandmark(x=0.6, y=0.3)
        lm_l_hip = MockSmoothedLandmark(x=0.4, y=0.7)
        lm_r_hip = MockSmoothedLandmark(x=0.6, y=0.7)

        landmarks = [MockSmoothedLandmark(x=0.5, y=0.5) for _ in range(33)]
        landmarks[_L_SHOULDER] = lm_l_shoulder
        landmarks[_R_SHOULDER] = lm_r_shoulder
        landmarks[_L_HIP] = lm_l_hip
        landmarks[_R_HIP] = lm_r_hip

        lean = compute_torso_lean(landmarks)
        assert lean is not None
        # Smoothed coords give upright → ~0°.
        assert lean == pytest.approx(0.0, abs=0.1)


# ================================================================
# FRAME-LEVEL RESULT
# ================================================================


class TestComputePostureMetrics:
    """compute_posture_metrics returns all three, allows partial None."""

    def test_all_available(self):
        landmarks = _make_landmarks({
            _NOSE: (0.5, 0.15),
            _L_SHOULDER: (0.4, 0.3),
            _R_SHOULDER: (0.6, 0.3),
            _L_HIP: (0.4, 0.7),
            _R_HIP: (0.6, 0.7),
        })
        metrics = compute_posture_metrics(landmarks)
        assert isinstance(metrics, PostureMetrics)
        assert metrics.torso_lean is not None
        assert metrics.shoulder_height_difference is not None
        assert metrics.neck_tilt is not None

    def test_hips_missing_other_metrics_available(self):
        """Hips unavailable → torso_lean=None, others still valid."""
        landmarks = _make_landmarks({
            _NOSE: (0.5, 0.15),
            _L_SHOULDER: (0.4, 0.3),
            _R_SHOULDER: (0.6, 0.3),
            _L_HIP: (0.4, 0.7, 0.1),  # Low vis.
            _R_HIP: (0.6, 0.7, 0.1),  # Low vis.
        })
        metrics = compute_posture_metrics(landmarks, visibility_threshold=0.5)
        assert metrics.torso_lean is None
        assert metrics.shoulder_height_difference is not None
        assert metrics.neck_tilt is not None

    def test_nose_missing_others_available(self):
        """Nose unavailable → neck_tilt=None, others still valid."""
        landmarks = _make_landmarks({
            _NOSE: (0.5, 0.15, 0.1),  # Low vis.
            _L_SHOULDER: (0.4, 0.3),
            _R_SHOULDER: (0.6, 0.3),
            _L_HIP: (0.4, 0.7),
            _R_HIP: (0.6, 0.7),
        })
        metrics = compute_posture_metrics(landmarks, visibility_threshold=0.5)
        assert metrics.torso_lean is not None
        assert metrics.shoulder_height_difference is not None
        assert metrics.neck_tilt is None

    def test_all_unavailable(self):
        metrics = compute_posture_metrics([])
        assert metrics.torso_lean is None
        assert metrics.shoulder_height_difference is None
        assert metrics.neck_tilt is None

    def test_returns_posture_metrics_type(self):
        landmarks = _make_landmarks()
        metrics = compute_posture_metrics(landmarks)
        assert isinstance(metrics, PostureMetrics)


# ================================================================
# REGRESSION: PostureMetrics dataclass
# ================================================================


class TestPostureMetricsDataclass:
    """PostureMetrics fields exist and are accessible."""

    def test_fields_exist(self):
        pm = PostureMetrics(torso_lean=1.0, shoulder_height_difference=0.01, neck_tilt=2.0)
        assert pm.torso_lean == 1.0
        assert pm.shoulder_height_difference == 0.01
        assert pm.neck_tilt == 2.0

    def test_none_values(self):
        pm = PostureMetrics(torso_lean=None, shoulder_height_difference=None, neck_tilt=None)
        assert pm.torso_lean is None
        assert pm.shoulder_height_difference is None
        assert pm.neck_tilt is None

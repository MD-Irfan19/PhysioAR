"""Tests for Phase 2.5 debug overlay visibility gating.

Tests the get_debug_landmark_info() helper from src.main.
All tests use synthetic, deterministic data. No webcam, OpenCV,
MediaPipe inference, GUI, or random data.

Phase 2.5 diagnostic — temporary landmark validation.
"""

from dataclasses import dataclass

import pytest

from src.main import get_debug_landmark_info


# ================================================================
# Helper: mock SmoothedLandmark
# ================================================================


@dataclass
class MockSmoothedLandmark:
    """Minimal mock for SmoothedLandmark with x, y, z, visibility."""

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

    overrides maps index to a dict of field values:
        {11: {"x": 0.4, "y": 0.5, "visibility": 0.9}, ...}

    For convenience, a tuple of (x, y) sets those plus default visibility,
    and a tuple of (x, y, vis) sets all three.
    """
    landmarks = [MockSmoothedLandmark(x=0.5, y=0.5) for _ in range(33)]
    if overrides:
        for idx, vals in overrides.items():
            if isinstance(vals, tuple):
                if len(vals) == 2:
                    landmarks[idx] = MockSmoothedLandmark(x=vals[0], y=vals[1])
                elif len(vals) == 3:
                    landmarks[idx] = MockSmoothedLandmark(
                        x=vals[0], y=vals[1], visibility=vals[2],
                    )
            elif isinstance(vals, dict):
                lm = MockSmoothedLandmark(x=0.5, y=0.5)
                for k, v in vals.items():
                    setattr(lm, k, v)
                landmarks[idx] = lm
    return landmarks


# Landmark indices (matching main.py).
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW = 13
_R_ELBOW = 14
_L_HIP = 23
_R_HIP = 24


# ================================================================
# Test 1: Landmark above threshold displays coordinates
# ================================================================


class TestLandmarkAboveThreshold:
    """A landmark with visibility above threshold displays coordinates."""

    def test_visible_landmark_has_coordinates(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.421, 0.312, 0.9),
            _R_SHOULDER: (0.579, 0.315, 0.8),
            _L_ELBOW: (0.365, 0.447, 0.85),
            _R_ELBOW: (0.635, 0.450, 0.88),
            _L_HIP: (0.438, 0.681, 0.75),
            _R_HIP: (0.562, 0.683, 0.72),
        })
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)

        # All 6 named landmarks + Neck should have coordinates.
        for label, x, y in info:
            assert x is not None, f"{label} should have x coordinate"
            assert y is not None, f"{label} should have y coordinate"

    def test_exactly_at_threshold_is_accepted(self):
        """visibility == threshold → accepted."""
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, 0.5),
            _R_SHOULDER: (0.6, 0.5, 0.5),
        })
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)

        # Find L-Shoulder and R-Shoulder.
        l_sh = [i for i in info if i[0] == "L-Shoulder"][0]
        r_sh = [i for i in info if i[0] == "R-Shoulder"][0]
        assert l_sh[1] is not None
        assert r_sh[1] is not None


# ================================================================
# Test 2: Landmark below threshold displays "unavailable"
# ================================================================


class TestLandmarkBelowThreshold:
    """A landmark with visibility below threshold has None coordinates."""

    def test_low_visibility_returns_none(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, 0.9),
            _R_SHOULDER: (0.6, 0.5, 0.9),
            _L_ELBOW: (0.365, 0.447, 0.1),  # LOW
            _R_ELBOW: (0.635, 0.450, 0.88),
            _L_HIP: (0.438, 0.681, 0.02),   # LOW
            _R_HIP: (0.562, 0.683, 0.72),
        })
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)

        l_elbow = [i for i in info if i[0] == "L-Elbow"][0]
        l_hip = [i for i in info if i[0] == "L-Hip"][0]
        assert l_elbow[1] is None and l_elbow[2] is None
        assert l_hip[1] is None and l_hip[2] is None

    def test_just_below_threshold_returns_none(self):
        """visibility 0.499 < threshold 0.5 → unavailable."""
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, 0.499),
        })
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)
        l_sh = [i for i in info if i[0] == "L-Shoulder"][0]
        assert l_sh[1] is None


# ================================================================
# Test 3: Missing landmark displays "unavailable"
# ================================================================


class TestMissingLandmark:
    """A missing landmark (index out of range) has None coordinates."""

    def test_short_landmark_list(self):
        """Fewer than 25 landmarks → hips are unavailable."""
        landmarks = [MockSmoothedLandmark(x=0.5, y=0.5) for _ in range(15)]
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)

        l_hip = [i for i in info if i[0] == "L-Hip"][0]
        r_hip = [i for i in info if i[0] == "R-Hip"][0]
        assert l_hip[1] is None
        assert r_hip[1] is None

    def test_empty_list(self):
        """Empty landmarks → all unavailable."""
        info = get_debug_landmark_info([], visibility_threshold=0.5)
        for label, x, y in info:
            assert x is None, f"{label} should be unavailable"
            assert y is None, f"{label} should be unavailable"


# ================================================================
# Test 4: Neck unavailable if either shoulder below threshold
# ================================================================


class TestNeckRequiresBothShoulders:
    """Neck requires BOTH shoulders to pass visibility check."""

    def test_left_shoulder_low_makes_neck_unavailable(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, 0.1),  # LOW
            _R_SHOULDER: (0.6, 0.5, 0.9),  # OK
        })
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)
        neck = [i for i in info if i[0] == "Neck"][0]
        assert neck[1] is None and neck[2] is None

    def test_right_shoulder_low_makes_neck_unavailable(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, 0.9),  # OK
            _R_SHOULDER: (0.6, 0.5, 0.1),  # LOW
        })
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)
        neck = [i for i in info if i[0] == "Neck"][0]
        assert neck[1] is None and neck[2] is None

    def test_both_shoulders_low_makes_neck_unavailable(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, 0.1),
            _R_SHOULDER: (0.6, 0.5, 0.1),
        })
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)
        neck = [i for i in info if i[0] == "Neck"][0]
        assert neck[1] is None and neck[2] is None


# ================================================================
# Test 5: Neck available when both shoulders pass
# ================================================================


class TestNeckAvailableWhenBothShouldersPass:
    """Neck coordinates are the midpoint of the smoothed shoulders."""

    def test_neck_is_shoulder_midpoint(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, 0.9),
            _R_SHOULDER: (0.6, 0.5, 0.9),
        })
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)
        neck = [i for i in info if i[0] == "Neck"][0]
        assert neck[1] == pytest.approx(0.5)  # midpoint x
        assert neck[2] == pytest.approx(0.5)  # midpoint y

    def test_neck_asymmetric_shoulders(self):
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.3, 0.4, 0.8),
            _R_SHOULDER: (0.7, 0.6, 0.8),
        })
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)
        neck = [i for i in info if i[0] == "Neck"][0]
        assert neck[1] == pytest.approx(0.5)   # midpoint x
        assert neck[2] == pytest.approx(0.5)   # midpoint y


# ================================================================
# Test 6: Coordinates come from SMOOTHED landmarks, not raw
# ================================================================


class TestCoordinatesAreSmoothed:
    """Displayed coordinates must be the smoothed (x, y), not raw."""

    def test_smoothed_not_raw_coordinates(self):
        """SmoothedLandmark with different x vs raw_x."""
        lm = MockSmoothedLandmark(
            x=0.450, y=0.310,  # Smoothed.
            raw_x=0.460, raw_y=0.320,  # Raw (different).
            visibility=0.9,
        )
        landmarks = [MockSmoothedLandmark(x=0.5, y=0.5) for _ in range(33)]
        landmarks[_L_SHOULDER] = lm

        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)
        l_sh = [i for i in info if i[0] == "L-Shoulder"][0]

        # Must match the smoothed coordinates, NOT raw.
        assert l_sh[1] == pytest.approx(0.450)
        assert l_sh[2] == pytest.approx(0.310)
        assert l_sh[1] != pytest.approx(0.460)  # NOT raw_x.


# ================================================================
# Test 7: No stale coordinates after landmark becomes unavailable
# ================================================================


class TestNoStaleCoordinates:
    """When a landmark goes from visible to invisible, it shows None."""

    def test_visible_then_invisible(self):
        """Two sequential calls: first visible, then not."""
        # Frame 1: visible.
        lm_vis = _make_landmarks({_L_HIP: (0.438, 0.681, 0.9)})
        info1 = get_debug_landmark_info(lm_vis, visibility_threshold=0.5)
        l_hip_1 = [i for i in info1 if i[0] == "L-Hip"][0]
        assert l_hip_1[1] is not None

        # Frame 2: no longer visible.
        lm_invis = _make_landmarks({_L_HIP: (0.438, 0.681, 0.05)})
        info2 = get_debug_landmark_info(lm_invis, visibility_threshold=0.5)
        l_hip_2 = [i for i in info2 if i[0] == "L-Hip"][0]
        assert l_hip_2[1] is None  # No stale coordinates.
        assert l_hip_2[2] is None

    def test_invisible_then_visible(self):
        """Landmark reappears → coordinates resume from smoothed."""
        lm_invis = _make_landmarks({_R_ELBOW: (0.635, 0.450, 0.05)})
        info1 = get_debug_landmark_info(lm_invis, visibility_threshold=0.5)
        r_elbow_1 = [i for i in info1 if i[0] == "R-Elbow"][0]
        assert r_elbow_1[1] is None

        lm_vis = _make_landmarks({_R_ELBOW: (0.640, 0.455, 0.85)})
        info2 = get_debug_landmark_info(lm_vis, visibility_threshold=0.5)
        r_elbow_2 = [i for i in info2 if i[0] == "R-Elbow"][0]
        assert r_elbow_2[1] == pytest.approx(0.640)
        assert r_elbow_2[2] == pytest.approx(0.455)


# ================================================================
# Test: Correct number of output items
# ================================================================


class TestOutputStructure:
    """get_debug_landmark_info always returns exactly 7 items."""

    def test_always_seven_items(self):
        landmarks = _make_landmarks()
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)
        assert len(info) == 7

    def test_seven_items_even_with_empty_input(self):
        info = get_debug_landmark_info([], visibility_threshold=0.5)
        assert len(info) == 7

    def test_labels_are_correct(self):
        landmarks = _make_landmarks()
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)
        labels = [i[0] for i in info]
        assert labels == [
            "L-Shoulder", "R-Shoulder",
            "L-Elbow", "R-Elbow",
            "L-Hip", "R-Hip",
            "Neck",
        ]


# ================================================================
# Test: Mixed visibility — partial body
# ================================================================


class TestPartialBodyVisibility:
    """Simulate upper-body-only and partial visibility scenarios."""

    def test_upper_body_only(self):
        """Shoulders visible, hips not visible."""
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, 0.9),
            _R_SHOULDER: (0.6, 0.5, 0.88),
            _L_ELBOW: (0.365, 0.447, 0.85),
            _R_ELBOW: (0.635, 0.450, 0.82),
            _L_HIP: (0.438, 0.681, 0.02),   # NOT visible
            _R_HIP: (0.562, 0.683, 0.03),   # NOT visible
        })
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)

        l_sh = [i for i in info if i[0] == "L-Shoulder"][0]
        r_sh = [i for i in info if i[0] == "R-Shoulder"][0]
        l_elbow = [i for i in info if i[0] == "L-Elbow"][0]
        r_elbow = [i for i in info if i[0] == "R-Elbow"][0]
        l_hip = [i for i in info if i[0] == "L-Hip"][0]
        r_hip = [i for i in info if i[0] == "R-Hip"][0]
        neck = [i for i in info if i[0] == "Neck"][0]

        # Upper body available.
        assert l_sh[1] is not None
        assert r_sh[1] is not None
        assert l_elbow[1] is not None
        assert r_elbow[1] is not None
        assert neck[1] is not None

        # Hips unavailable.
        assert l_hip[1] is None
        assert r_hip[1] is None

    def test_one_arm_hidden(self):
        """One elbow visible, the other not."""
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, 0.9),
            _R_SHOULDER: (0.6, 0.5, 0.88),
            _L_ELBOW: (0.365, 0.447, 0.85),
            _R_ELBOW: (0.635, 0.450, 0.04),   # NOT visible
            _L_HIP: (0.438, 0.681, 0.75),
            _R_HIP: (0.562, 0.683, 0.72),
        })
        info = get_debug_landmark_info(landmarks, visibility_threshold=0.5)

        l_elbow = [i for i in info if i[0] == "L-Elbow"][0]
        r_elbow = [i for i in info if i[0] == "R-Elbow"][0]

        assert l_elbow[1] is not None  # visible
        assert r_elbow[1] is None      # unavailable


# ================================================================
# Test: Threshold from config
# ================================================================


class TestThresholdFromConfig:
    """Verify the default threshold comes from config."""

    def test_uses_config_threshold_by_default(self):
        from src.config import LANDMARK_VISIBILITY_THRESHOLD

        # Landmark with visibility exactly at config threshold.
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, LANDMARK_VISIBILITY_THRESHOLD),
        })
        info = get_debug_landmark_info(landmarks)  # No explicit threshold.
        l_sh = [i for i in info if i[0] == "L-Shoulder"][0]
        assert l_sh[1] is not None  # At threshold → accepted.

    def test_custom_threshold_overrides_config(self):
        """Explicitly passed threshold overrides config default."""
        landmarks = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, 0.3),  # Below 0.5 but above 0.2.
        })
        # Default threshold (0.5) would reject.
        info_default = get_debug_landmark_info(landmarks, visibility_threshold=0.5)
        l_sh_default = [i for i in info_default if i[0] == "L-Shoulder"][0]
        assert l_sh_default[1] is None

        # Custom threshold (0.2) accepts.
        info_custom = get_debug_landmark_info(landmarks, visibility_threshold=0.2)
        l_sh_custom = [i for i in info_custom if i[0] == "L-Shoulder"][0]
        assert l_sh_custom[1] is not None

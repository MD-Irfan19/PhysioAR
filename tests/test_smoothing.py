"""Tests for src/utils/smoothing.py — Phase 1.5 EMA landmark smoothing.

All tests use synthetic, deterministic coordinates with hand-calculated
expected results. No webcam, camera, MediaPipe, or random data is used.
"""

from dataclasses import dataclass

import pytest

from src.utils.smoothing import EMAFilter, SmoothedLandmark


# ================================================================
# Helper: lightweight mock landmark for testing
# ================================================================


@dataclass
class MockLandmark:
    """Minimal landmark mock with x, y, z, visibility, presence."""

    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0


# ================================================================
# EMAFilter — First observation
# ================================================================


class TestEMAFilterFirstObservation:
    """The first observation must pass through unchanged."""

    def test_first_observation_passes_through(self):
        """First observation → filtered == current, no artificial movement."""
        ema = EMAFilter(alpha=0.5)
        landmarks = [MockLandmark(x=10.0, y=20.0, z=30.0)]

        result = ema.smooth(landmarks)

        assert len(result) == 1
        assert result[0].x == pytest.approx(10.0)
        assert result[0].y == pytest.approx(20.0)
        assert result[0].z == pytest.approx(30.0)

    def test_first_observation_raw_equals_smoothed(self):
        """On the first frame, raw and smoothed coordinates must match."""
        ema = EMAFilter(alpha=0.5)
        landmarks = [MockLandmark(x=5.0, y=10.0, z=15.0)]

        result = ema.smooth(landmarks)

        assert result[0].raw_x == pytest.approx(result[0].x)
        assert result[0].raw_y == pytest.approx(result[0].y)
        assert result[0].raw_z == pytest.approx(result[0].z)


# ================================================================
# EMAFilter — Basic EMA calculation
# ================================================================


class TestEMAFilterBasicCalculation:
    """Verify the EMA formula: Filtered = alpha * Current + (1 - alpha) * Previous."""

    def test_basic_ema(self):
        """alpha=0.5, prev=(10,20,30), current=(20,40,60) → (15,30,45)."""
        ema = EMAFilter(alpha=0.5)

        # First observation: initializes filter state.
        ema.smooth([MockLandmark(x=10.0, y=20.0, z=30.0)])

        # Second observation: EMA applied.
        result = ema.smooth([MockLandmark(x=20.0, y=40.0, z=60.0)])

        assert result[0].x == pytest.approx(15.0)
        assert result[0].y == pytest.approx(30.0)
        assert result[0].z == pytest.approx(45.0)

    def test_raw_coordinates_preserved(self):
        """Raw coordinates must reflect the current (unsmoothed) input."""
        ema = EMAFilter(alpha=0.5)
        ema.smooth([MockLandmark(x=10.0, y=20.0, z=30.0)])

        result = ema.smooth([MockLandmark(x=20.0, y=40.0, z=60.0)])

        assert result[0].raw_x == pytest.approx(20.0)
        assert result[0].raw_y == pytest.approx(40.0)
        assert result[0].raw_z == pytest.approx(60.0)


# ================================================================
# EMAFilter — Alpha = 1 (no smoothing)
# ================================================================


class TestEMAFilterAlphaOne:
    """Alpha = 1.0 means no smoothing; Filtered == Current."""

    def test_alpha_one_no_smoothing(self):
        """With alpha=1.0, every observation passes through unchanged."""
        ema = EMAFilter(alpha=1.0)

        ema.smooth([MockLandmark(x=10.0, y=20.0, z=30.0)])
        result = ema.smooth([MockLandmark(x=50.0, y=60.0, z=70.0)])

        assert result[0].x == pytest.approx(50.0)
        assert result[0].y == pytest.approx(60.0)
        assert result[0].z == pytest.approx(70.0)


# ================================================================
# EMAFilter — Alpha near 0 (heavy smoothing)
# ================================================================


class TestEMAFilterAlphaNearZero:
    """Small alpha means heavy smoothing; result stays close to previous."""

    def test_alpha_near_zero(self):
        """alpha=0.1 → result heavily weighted toward previous filtered value."""
        ema = EMAFilter(alpha=0.1)

        # Initialize at (0, 0, 0).
        ema.smooth([MockLandmark(x=0.0, y=0.0, z=0.0)])

        # Current jumps to (100, 100, 100).
        # Expected: 0.1 * 100 + 0.9 * 0 = 10.0
        result = ema.smooth([MockLandmark(x=100.0, y=100.0, z=100.0)])

        assert result[0].x == pytest.approx(10.0)
        assert result[0].y == pytest.approx(10.0)
        assert result[0].z == pytest.approx(10.0)


# ================================================================
# EMAFilter — Multiple sequential observations (convergence)
# ================================================================


class TestEMAFilterSequentialConvergence:
    """Verify a sequence converges toward the repeated current value."""

    def test_convergence_sequence(self):
        """alpha=0.5, repeated current=10 from initial=0 → converges."""
        ema = EMAFilter(alpha=0.5)

        # Frame 0: current=0, filtered=0
        result = ema.smooth([MockLandmark(x=0.0, y=0.0, z=0.0)])
        assert result[0].x == pytest.approx(0.0)

        # Frame 1: current=10, filtered = 0.5*10 + 0.5*0 = 5.0
        result = ema.smooth([MockLandmark(x=10.0, y=0.0, z=0.0)])
        assert result[0].x == pytest.approx(5.0)

        # Frame 2: current=10, filtered = 0.5*10 + 0.5*5 = 7.5
        result = ema.smooth([MockLandmark(x=10.0, y=0.0, z=0.0)])
        assert result[0].x == pytest.approx(7.5)

        # Frame 3: current=10, filtered = 0.5*10 + 0.5*7.5 = 8.75
        result = ema.smooth([MockLandmark(x=10.0, y=0.0, z=0.0)])
        assert result[0].x == pytest.approx(8.75)


# ================================================================
# EMAFilter — Independent x/y/z filtering
# ================================================================


class TestEMAFilterIndependentAxes:
    """Each coordinate must be filtered independently."""

    def test_independent_xyz(self):
        """Different x, y, z deltas produce independently smoothed results."""
        ema = EMAFilter(alpha=0.5)

        ema.smooth([MockLandmark(x=10.0, y=20.0, z=30.0)])
        result = ema.smooth([MockLandmark(x=20.0, y=40.0, z=60.0)])

        # x: 0.5*20 + 0.5*10 = 15
        assert result[0].x == pytest.approx(15.0)
        # y: 0.5*40 + 0.5*20 = 30
        assert result[0].y == pytest.approx(30.0)
        # z: 0.5*60 + 0.5*30 = 45
        assert result[0].z == pytest.approx(45.0)


# ================================================================
# EMAFilter — Reset
# ================================================================


class TestEMAFilterReset:
    """Reset must clear all state; next observation becomes new baseline."""

    def test_reset_clears_state(self):
        """After reset, the next observation passes through unchanged."""
        ema = EMAFilter(alpha=0.5)

        # Feed initial + second observation.
        ema.smooth([MockLandmark(x=10.0, y=10.0, z=10.0)])
        ema.smooth([MockLandmark(x=20.0, y=20.0, z=20.0)])

        # Reset.
        ema.reset()

        # Next observation becomes the new baseline (passes through).
        result = ema.smooth([MockLandmark(x=80.0, y=90.0, z=100.0)])

        assert result[0].x == pytest.approx(80.0)
        assert result[0].y == pytest.approx(90.0)
        assert result[0].z == pytest.approx(100.0)


# ================================================================
# EMAFilter — Multiple landmarks (independent state)
# ================================================================


class TestEMAFilterMultipleLandmarks:
    """Each landmark index must maintain its own independent EMA state."""

    def test_independent_landmark_state(self):
        """Changing landmark 0 must not affect landmarks 1 or 2."""
        ema = EMAFilter(alpha=0.5)

        # Initialize with 3 landmarks.
        ema.smooth([
            MockLandmark(x=10.0, y=10.0, z=10.0),
            MockLandmark(x=20.0, y=20.0, z=20.0),
            MockLandmark(x=30.0, y=30.0, z=30.0),
        ])

        # Only landmark 0 changes significantly.
        result = ema.smooth([
            MockLandmark(x=100.0, y=10.0, z=10.0),   # big jump
            MockLandmark(x=20.0, y=20.0, z=20.0),     # unchanged
            MockLandmark(x=30.0, y=30.0, z=30.0),     # unchanged
        ])

        # Landmark 0: x smoothed from 10→100 → 0.5*100 + 0.5*10 = 55
        assert result[0].x == pytest.approx(55.0)
        # Landmark 1: unchanged → 0.5*20 + 0.5*20 = 20
        assert result[1].x == pytest.approx(20.0)
        assert result[1].y == pytest.approx(20.0)
        # Landmark 2: unchanged → 0.5*30 + 0.5*30 = 30
        assert result[2].x == pytest.approx(30.0)
        assert result[2].z == pytest.approx(30.0)


# ================================================================
# EMAFilter — Visibility and presence are NOT smoothed
# ================================================================


class TestEMAFilterNoSmoothingOfConfidence:
    """Visibility and presence must be preserved as-is, not EMA-filtered."""

    def test_visibility_preserved(self):
        """Visibility should reflect the current frame, not be smoothed."""
        ema = EMAFilter(alpha=0.5)

        ema.smooth([MockLandmark(x=0.0, y=0.0, z=0.0, visibility=0.9, presence=0.8)])
        result = ema.smooth([MockLandmark(x=1.0, y=1.0, z=1.0, visibility=0.3, presence=0.4)])

        # Visibility and presence should be the current frame's values.
        assert result[0].visibility == pytest.approx(0.3)
        assert result[0].presence == pytest.approx(0.4)


# ================================================================
# EMAFilter — Oscillation damping
# ================================================================


class TestEMAFilterOscillationDamping:
    """Verify EMA reduces rapid frame-to-frame oscillation."""

    def test_alternating_oscillation(self):
        """Alternating 0/10 sequence with alpha=0.5 → diminishing oscillation."""
        ema = EMAFilter(alpha=0.5)

        # Raw:     0,    10,   0,     10,    0,     10
        # Smooth:  0,    5,    2.5,   6.25,  3.125, 6.5625
        expected = [0.0, 5.0, 2.5, 6.25, 3.125, 6.5625]
        raw_values = [0, 10, 0, 10, 0, 10]

        for i, raw_x in enumerate(raw_values):
            result = ema.smooth([MockLandmark(x=float(raw_x), y=0.0, z=0.0)])
            assert result[0].x == pytest.approx(expected[i]), (
                f"Frame {i}: expected {expected[i]}, got {result[0].x}"
            )


# ================================================================
# EMAFilter — Configuration from src.config
# ================================================================


class TestEMAFilterConfiguration:
    """Verify the production alpha comes from src.config.SMOOTHING_ALPHA."""

    def test_config_alpha_is_used(self):
        """src.config.SMOOTHING_ALPHA must be importable and valid."""
        from src.config import SMOOTHING_ALPHA

        assert isinstance(SMOOTHING_ALPHA, float)
        assert 0 < SMOOTHING_ALPHA <= 1.0

    def test_filter_accepts_config_alpha(self):
        """EMAFilter can be constructed with the config alpha."""
        from src.config import SMOOTHING_ALPHA

        ema = EMAFilter(alpha=SMOOTHING_ALPHA)
        assert ema.alpha == SMOOTHING_ALPHA


# ================================================================
# EMAFilter — Invalid alpha
# ================================================================


class TestEMAFilterInvalidAlpha:
    """Invalid alpha values must raise ValueError."""

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            EMAFilter(alpha=0.0)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError):
            EMAFilter(alpha=-0.5)

    def test_alpha_above_one_raises(self):
        with pytest.raises(ValueError):
            EMAFilter(alpha=1.5)


# ================================================================
# EMAFilter — Changing landmark count
# ================================================================


class TestEMAFilterChangingLandmarkCount:
    """Changing landmark count must reinitialize safely, not crash."""

    def test_landmark_count_change_reinitializes(self):
        """If landmark count changes, filter state is safely reinitialized."""
        ema = EMAFilter(alpha=0.5)

        # Start with 2 landmarks.
        ema.smooth([
            MockLandmark(x=10.0, y=10.0, z=10.0),
            MockLandmark(x=20.0, y=20.0, z=20.0),
        ])

        # Now provide 3 landmarks — count changed.
        result = ema.smooth([
            MockLandmark(x=50.0, y=50.0, z=50.0),
            MockLandmark(x=60.0, y=60.0, z=60.0),
            MockLandmark(x=70.0, y=70.0, z=70.0),
        ])

        # Should reinitialize: first observation passes through.
        assert result[0].x == pytest.approx(50.0)
        assert result[1].x == pytest.approx(60.0)
        assert result[2].x == pytest.approx(70.0)


# ================================================================
# SmoothedLandmark dataclass
# ================================================================


class TestSmoothedLandmark:
    """Verify SmoothedLandmark holds all required fields."""

    def test_fields_accessible(self):
        lm = SmoothedLandmark(
            x=1.0, y=2.0, z=3.0,
            raw_x=1.1, raw_y=2.1, raw_z=3.1,
            visibility=0.9, presence=0.8,
        )
        assert lm.x == 1.0
        assert lm.raw_x == 1.1
        assert lm.visibility == 0.9
        assert lm.presence == 0.8

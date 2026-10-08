import os
import pytest

test_path = r"c:\Users\irfan\OneDrive\Desktop\PhysioAR\tests\test_posture.py"

with open(test_path, "a", encoding="utf-8") as f:
    f.write("""

# ================================================================
# Phase 11 — Hip Hike
# ================================================================

from src.metrics.posture import compute_hip_hike_metric

class TestComputeHipHike:
    \"\"\"Tests for the Hip Hike metric.\"\"\"

    def test_level_hips(self):
        \"\"\"Level hips \\u2192 near-zero metric.\"\"\"
        lm = _make_landmarks({
            _L_HIP: (0.4, 0.8),
            _R_HIP: (0.6, 0.8),
        })
        val = compute_hip_hike_metric(lm)
        assert val is not None
        assert val == pytest.approx(0.0)

    def test_small_asymmetry(self):
        \"\"\"Small asymmetry \\u2192 small positive magnitude.\"\"\"
        lm = _make_landmarks({
            _L_HIP: (0.4, 0.8),
            _R_HIP: (0.6, 0.79),
        })
        val = compute_hip_hike_metric(lm)
        assert val is not None
        assert val == pytest.approx(0.01)

    def test_large_asymmetry(self):
        \"\"\"Large asymmetry \\u2192 larger magnitude.\"\"\"
        lm = _make_landmarks({
            _L_HIP: (0.4, 0.8),
            _R_HIP: (0.6, 0.7),
        })
        val = compute_hip_hike_metric(lm)
        assert val is not None
        assert val == pytest.approx(0.1)

    def test_missing_left_hip(self):
        lm = _make_landmarks({
            _L_HIP: (0.4, 0.8, 0.1),
            _R_HIP: (0.6, 0.8),
        })
        val = compute_hip_hike_metric(lm, visibility_threshold=0.5)
        assert val is None

    def test_missing_right_hip(self):
        lm = _make_landmarks({
            _L_HIP: (0.4, 0.8),
            _R_HIP: (0.6, 0.8, 0.1),
        })
        val = compute_hip_hike_metric(lm, visibility_threshold=0.5)
        assert val is None

    def test_independence(self):
        \"\"\"Changing hip alignment affects hip hike but shouldn't alter lateral trunk lean geometry incorrectly.\"\"\"
        # They share the hip midpoint, so lateral trunk lean WILL change if hips move.
        # But this test just makes sure we get the correct metric regardless.
        lm = _make_landmarks({
            _L_HIP: (0.4, 0.8),
            _R_HIP: (0.6, 0.7),
        })
        val = compute_hip_hike_metric(lm)
        assert val is not None
        assert val == pytest.approx(0.1)


# ================================================================
# Phase 11 — Trunk Lean
# ================================================================

# Note: Trunk lean uses lateral_trunk_lean_metric.

class TestComputeTrunkLean:
    \"\"\"Tests for the Trunk Lean metric (aliased to lateral_trunk_lean).\"\"\"

    def test_upright_torso(self):
        lm = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5),
            _R_SHOULDER: (0.6, 0.5),
            _L_HIP: (0.4, 0.8),
            _R_HIP: (0.6, 0.8),
        })
        val = compute_lateral_trunk_lean_metric(lm)
        assert val is not None
        assert val == pytest.approx(0.0)

    def test_left_lateral_lean(self):
        lm = _make_landmarks({
            _L_SHOULDER: (0.3, 0.5),
            _R_SHOULDER: (0.5, 0.5),
            _L_HIP: (0.4, 0.8),
            _R_HIP: (0.6, 0.8),
        })
        val = compute_lateral_trunk_lean_metric(lm)
        assert val is not None
        assert val > 10.0

    def test_right_lateral_lean(self):
        lm = _make_landmarks({
            _L_SHOULDER: (0.5, 0.5),
            _R_SHOULDER: (0.7, 0.5),
            _L_HIP: (0.4, 0.8),
            _R_HIP: (0.6, 0.8),
        })
        val = compute_lateral_trunk_lean_metric(lm)
        assert val is not None
        assert val > 10.0

    def test_missing_shoulders(self):
        lm = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5, 0.1),
            _R_SHOULDER: (0.6, 0.5),
            _L_HIP: (0.4, 0.8),
            _R_HIP: (0.6, 0.8),
        })
        val = compute_lateral_trunk_lean_metric(lm, visibility_threshold=0.5)
        assert val is None

    def test_missing_hips(self):
        lm = _make_landmarks({
            _L_SHOULDER: (0.4, 0.5),
            _R_SHOULDER: (0.6, 0.5),
            _L_HIP: (0.4, 0.8, 0.1),
            _R_HIP: (0.6, 0.8),
        })
        val = compute_lateral_trunk_lean_metric(lm, visibility_threshold=0.5)
        assert val is None

""")
print("Tests appended!")

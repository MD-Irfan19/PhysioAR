"""Tests for generic compensation framework."""

import math
import pytest
from unittest.mock import Mock

from src.exercises.base import CheckCondition, GenericCompensationCheck, RepState, ExerciseDefinition, CameraOrientation
from src.compensation import (
    evaluate_generic_checks,
    GenericCompensationResult,
    compute_threshold,
)
from src.calibration import CalibrationResult, MetricBaseline
from src.metrics.posture import PostureMetrics


# ================================================================
# Test 1 - Configuration Validation
# ================================================================

class TestGenericCheckConfiguration:
    def test_above_requires_upper(self):
        with pytest.raises(ValueError):
            GenericCompensationCheck(id="test", condition=CheckCondition.ABOVE)

    def test_below_requires_lower(self):
        with pytest.raises(ValueError):
            GenericCompensationCheck(id="test", condition=CheckCondition.BELOW)

    def test_range_requires_both(self):
        with pytest.raises(ValueError):
            GenericCompensationCheck(id="test", condition=CheckCondition.RANGE, lower_threshold=5.0)
        with pytest.raises(ValueError):
            GenericCompensationCheck(id="test", condition=CheckCondition.RANGE, upper_threshold=10.0)

    def test_range_requires_valid_bounds(self):
        with pytest.raises(ValueError):
            GenericCompensationCheck(id="test", condition=CheckCondition.RANGE, lower_threshold=10.0, upper_threshold=5.0)

    def test_valid_configuration(self):
        check = GenericCompensationCheck(id="test", condition=CheckCondition.RANGE, lower_threshold=5.0, upper_threshold=10.0)
        assert check.id == "test"


# ================================================================
# Test 2 - Directional Logic
# ================================================================

class TestGenericEvaluationLogic:
    def setup_method(self):
        self.calib = CalibrationResult(
            spine_angle=MetricBaseline(mean=10.0, std=1.0),
            shoulder_height_difference=MetricBaseline(mean=0.0, std=0.0),
            neck_tilt=MetricBaseline(mean=5.0, std=1.0),
            hip_rotation=MetricBaseline(mean=0.0, std=0.0),
            lateral_trunk_lean=MetricBaseline(mean=0.0, std=0.0),
        )
        self.posture = PostureMetrics(
            torso_lean=10.0,
            shoulder_height_difference=0.0,
            neck_tilt=5.0,
        )

    def _evaluate(self, check, val):
        def calc(lms, side, visibility_threshold): return val
        check.metric_calculator = calc
        res = evaluate_generic_checks([check], self.posture, self.calib, [], "right", 0.5, RepState.DOWN)
        return res[check.id]

    def test_above_threshold(self):
        check = GenericCompensationCheck(id="test", condition=CheckCondition.ABOVE, upper_threshold=10.0)
        assert not self._evaluate(check, 9.0).flagged
        assert not self._evaluate(check, 10.0).flagged  # strict boundary
        assert self._evaluate(check, 10.1).flagged

    def test_below_threshold(self):
        check = GenericCompensationCheck(id="test", condition=CheckCondition.BELOW, lower_threshold=5.0)
        assert not self._evaluate(check, 6.0).flagged
        assert not self._evaluate(check, 5.0).flagged   # strict boundary
        assert self._evaluate(check, 4.9).flagged

    def test_range(self):
        check = GenericCompensationCheck(id="test", condition=CheckCondition.RANGE, lower_threshold=5.0, upper_threshold=10.0)
        assert self._evaluate(check, 4.9).flagged       # outside low
        assert not self._evaluate(check, 5.0).flagged   # inside
        assert not self._evaluate(check, 7.5).flagged   # inside
        assert not self._evaluate(check, 10.0).flagged  # inside
        assert self._evaluate(check, 10.1).flagged      # outside high

    def test_baseline_deviation(self):
        # spine_angle baseline is mean=10, std=1
        # floor is 0 by default for unknown metrics, so threshold = 2 * std = 2.0
        check = GenericCompensationCheck(id="spine_angle", condition=CheckCondition.BASELINE_DEVIATION)
        # mapped to calibration_result.spine_angle
        # if val is 11, dev is 1, not flagged
        assert not self._evaluate(check, 11.0).flagged
        # if val is 12, dev is 2, threshold is 2 -> strict boundary not flagged
        assert not self._evaluate(check, 12.0).flagged
        # if val is 12.1, dev is 2.1, flagged
        assert self._evaluate(check, 12.1).flagged

    def test_baseline_deviation_missing_baseline(self):
        check = GenericCompensationCheck(id="missing_metric", condition=CheckCondition.BASELINE_DEVIATION)
        res = self._evaluate(check, 15.0)
        assert not res.flagged

    def test_missing_current_value(self):
        check = GenericCompensationCheck(id="test", condition=CheckCondition.ABOVE, upper_threshold=10.0)
        res = self._evaluate(check, None)
        assert not res.flagged

    def test_non_finite_current_value(self):
        check = GenericCompensationCheck(id="test", condition=CheckCondition.ABOVE, upper_threshold=10.0)
        assert not self._evaluate(check, math.nan).flagged
        assert not self._evaluate(check, math.inf).flagged


# ================================================================
# Test 3 - Phase Gating
# ================================================================

class TestGenericPhaseGating:
    def test_runs_in_configured_phase(self):
        check = GenericCompensationCheck(
            id="test", condition=CheckCondition.ABOVE, upper_threshold=10.0,
            allowed_phases={RepState.RISING},
        )
        def calc(lms, side, visibility_threshold): return 15.0
        check.metric_calculator = calc

        # Active in RISING
        res_active = evaluate_generic_checks([check], PostureMetrics(), None, [], "right", 0.5, RepState.RISING)
        assert res_active["test"].flagged
        
        # Ignored in FALLING
        res_ignored = evaluate_generic_checks([check], PostureMetrics(), None, [], "right", 0.5, RepState.FALLING)
        assert "test" not in res_ignored


# ================================================================
# Test 4 - Synthetic Integration
# ================================================================

def test_synthetic_exercise_integration():
    from src.main import FrameSample
    from src.quality import evaluate_rep_quality
    
    # Create an exercise with a generic check
    def mock_calc(lms, side, visibility_threshold): return 15.0
    check = GenericCompensationCheck(
        id="generic_test",
        condition=CheckCondition.ABOVE,
        upper_threshold=10.0,
        allowed_phases={RepState.RISING},
        metric_calculator=mock_calc
    )
    
    ex = ExerciseDefinition(
        name="Synthetic",
        camera_orientation=CameraOrientation.SIDE,
        target_joint="knee",
        rom_target=90.0,
        rep_start_angle=100.0,
        rep_end_angle=150.0,
        angle_calculator=lambda lms, side: 150.0,
        generic_checks=[check]
    )
    
    # We will simulate quality scoring with a FrameSample
    samples = []
    for i in range(10):
        s = FrameSample(
            frame_index=i,
            angle=100.0 + i * 5,
            generic_flags={"generic_test": True} if i == 5 else {}
        )
        samples.append(s)
        
    res = evaluate_rep_quality(
        rep_number=1,
        frame_history=samples,
        start_frame=0,
        top_frame=9,
        end_frame=9,
        target_rom=90.0
    )
    
    # distinct compensations should include 'generic_test'
    assert res.compensation_types == 1
    # score is penalized
    from src.config import COMPENSATION_PENALTY
    assert res.compensation_score == 100.0 - COMPENSATION_PENALTY

"""Compensation flagging for PhysioAR.

Compares Phase 4A raw posture metrics against the Phase 2/2.1
calibration baseline to produce compensation flags.

For each posture metric:

    1. Deviation = abs(Current Value - Baseline Mean)
    2. Threshold = max(2 × Baseline Std, Fixed Floor)
    3. Flagged   = Deviation > Threshold    (strict >)

This module is THRESHOLD-BASED FLAGGING ONLY.

A compensation flag means ONLY:
    "The current metric's baseline-relative deviation exceeded
     the configured threshold."

It does NOT mean the exercise is clinically incorrect, the user
is injured, or the repetition was performed incorrectly.

Phase 4B — Compensation Flagging (Baseline-Relative Thresholding).

============================================================
METRIC MAPPING
============================================================

    PostureMetrics.torso_lean
        ↔ CalibrationResult.spine_angle
        floor: TORSO_LEAN_FLOOR (degrees)

    PostureMetrics.shoulder_height_difference
        ↔ CalibrationResult.shoulder_height_difference
        floor: SHOULDER_HIKE_FLOOR (normalized units)

    PostureMetrics.neck_tilt
        ↔ CalibrationResult.neck_tilt
        floor: NECK_TILT_FLOOR (degrees)

============================================================
NONE / UNAVAILABLE HANDLING
============================================================

If the current metric is None:
    deviation = None, flagged = False

None does NOT mean 0. A missing metric never generates a flag.

If the calibration baseline is unavailable (calibration_result
is None), all flags are False and all deviations are None.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.calibration import CalibrationResult, MetricBaseline
from src.config import (
    TORSO_LEAN_FLOOR,
    SHOULDER_HIKE_FLOOR,
    NECK_TILT_FLOOR,
    HIP_ROTATION_FLOOR,
    LATERAL_TRUNK_LEAN_FLOOR,
    HIP_HIKE_FLOOR,
    SHOULDER_SUBSTITUTION_FLOOR,
    LANDMARK_VISIBILITY_THRESHOLD,
)
from src.metrics.posture import PostureMetrics
from src.exercises.base import CheckCondition, GenericCompensationCheck, RepState


# ============================================================
# Pure computation functions
# ============================================================


def compute_deviation(
    current_value: float,
    baseline_mean: float,
) -> float:
    """Compute the absolute deviation of a metric from its baseline.

    Args:
        current_value: The current frame's metric value.
        baseline_mean: The calibration baseline mean.

    Returns:
        abs(current_value - baseline_mean).
    """
    return abs(current_value - baseline_mean)


def compute_threshold(
    baseline_std: float,
    fixed_floor: float,
) -> float:
    """Compute the compensation threshold for a metric.

    Threshold = max(2 × baseline_std, fixed_floor).

    The fixed floor prevents zero-width thresholds when the
    calibration standard deviation is very small.

    Args:
        baseline_std: The calibration baseline sample standard
            deviation.
        fixed_floor: The minimum threshold (from config).

    Returns:
        The effective threshold in the same units as the metric.
    """
    return max(2.0 * baseline_std, fixed_floor)


def evaluate_metric(
    current_value: Optional[float],
    baseline: Optional[MetricBaseline],
    fixed_floor: float,
) -> CompensationMetricResult:
    """Evaluate a single metric against its calibration baseline.

    If current_value or baseline is None, returns an unavailable result
    (flagged=False).

    The flagging rule is strict >:
        flagged = deviation > threshold

    Args:
        current_value: The current metric value, or None if
            unavailable.
        baseline: The MetricBaseline (mean, std) from calibration, or None.
        fixed_floor: The minimum threshold floor from config.

    Returns:
        A CompensationMetricResult with all intermediate values.
    """
    if baseline is None:
        return CompensationMetricResult(
            current_value=current_value,
            baseline_mean=0.0,
            baseline_std=0.0,
            deviation=None,
            threshold=fixed_floor,
            flagged=False,
        )

    threshold = compute_threshold(baseline.std, fixed_floor)

    if current_value is None:
        return CompensationMetricResult(
            current_value=None,
            baseline_mean=baseline.mean,
            baseline_std=baseline.std,
            deviation=None,
            threshold=threshold,
            flagged=False,
        )

    deviation = compute_deviation(current_value, baseline.mean)
    flagged = deviation > threshold

    return CompensationMetricResult(
        current_value=current_value,
        baseline_mean=baseline.mean,
        baseline_std=baseline.std,
        deviation=deviation,
        threshold=threshold,
        flagged=flagged,
    )


# ============================================================
# Result data structures
# ============================================================


@dataclass
class CompensationMetricResult:
    """Result of evaluating a single metric against its baseline.

    Retains all intermediate values so the threshold decision
    is transparent and testable.

    Attributes:
        current_value: The current frame's raw metric value,
            or None if the metric was unavailable.
        baseline_mean: The calibration baseline mean.
        baseline_std: The calibration baseline standard deviation.
        deviation: abs(current_value - baseline_mean), or None
            if current_value is None.
        threshold: max(2 × baseline_std, fixed_floor).
        flagged: True if deviation > threshold, False otherwise.
            Always False when current_value is None.
    """

    current_value: Optional[float] = None
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    deviation: Optional[float] = None
    threshold: float = 0.0
    flagged: bool = False


@dataclass
class CompensationResult:
    """Frame-level compensation result for all three posture metrics.

    Attributes:
        torso_lean: Result for the torso lean metric.
        shoulder_hike: Result for the shoulder height difference metric.
        neck_tilt: Result for the neck tilt metric.
        hip_rotation: Result for the hip rotation metric.
        lateral_trunk_lean: Result for the lateral trunk lean metric.
    """

    torso_lean: CompensationMetricResult = None
    shoulder_hike: CompensationMetricResult = None
    neck_tilt: CompensationMetricResult = None
    hip_rotation: CompensationMetricResult = None
    lateral_trunk_lean: CompensationMetricResult = None
    hip_hike: CompensationMetricResult = None
    trunk_lean: CompensationMetricResult = None
    shoulder_substitution: CompensationMetricResult = None
    generic_results: dict[str, GenericCompensationResult] = field(default_factory=dict)

@dataclass
class GenericCompensationResult:
    """Result of evaluating a generic compensation check."""
    id: str
    condition: CheckCondition
    flagged: bool = False
    current_value: Optional[float] = None
    threshold_used: Optional[float] = None
    baseline_mean: Optional[float] = None
    deviation: Optional[float] = None
    phase_used: Optional[RepState] = None
    lower_bound_used: Optional[float] = None
    upper_bound_used: Optional[float] = None


# ============================================================
# Frame-level compensation evaluation
# ============================================================


def evaluate_compensation(
    posture: PostureMetrics,
    calibration_result: Optional[CalibrationResult],
) -> Optional[CompensationResult]:
    """Evaluate all three compensation flags for a single frame.

    Consumes Phase 4A raw posture metrics and Phase 2 calibration
    baselines. Does NOT calculate metrics itself.

    If calibration_result is None (calibration failed or not yet
    performed), returns None — compensation cannot be evaluated
    without a valid baseline.

    Each metric is evaluated independently using its own baseline
    and floor constant. A None metric value produces flagged=False.

    Args:
        posture: PostureMetrics from compute_posture_metrics().
        calibration_result: CalibrationResult from run_calibration(),
            or None if calibration is unavailable.

    Returns:
        A CompensationResult, or None if no baseline is available.
    """
    if calibration_result is None:
        return None

    return CompensationResult(
        torso_lean=evaluate_metric(
            posture.torso_lean,
            calibration_result.spine_angle,
            TORSO_LEAN_FLOOR,
        ),
        shoulder_hike=evaluate_metric(
            posture.shoulder_height_difference,
            calibration_result.shoulder_height_difference,
            SHOULDER_HIKE_FLOOR,
        ),
        neck_tilt=evaluate_metric(
            posture.neck_tilt,
            calibration_result.neck_tilt,
            NECK_TILT_FLOOR,
        ),
        hip_rotation=evaluate_metric(
            posture.hip_rotation,
            calibration_result.hip_rotation,
            HIP_ROTATION_FLOOR,
        ),
        lateral_trunk_lean=evaluate_metric(
            posture.lateral_trunk_lean,
            calibration_result.lateral_trunk_lean,
            LATERAL_TRUNK_LEAN_FLOOR,
        ),
        hip_hike=evaluate_metric(
            posture.hip_hike,
            calibration_result.hip_alignment,
            HIP_HIKE_FLOOR,
        ),
        trunk_lean=evaluate_metric(
            posture.trunk_lean,
            calibration_result.lateral_trunk_lean,
            LATERAL_TRUNK_LEAN_FLOOR,
        ),
        shoulder_substitution=evaluate_metric(
            posture.shoulder_substitution,
            calibration_result.shoulder_substitution,
            SHOULDER_SUBSTITUTION_FLOOR,
        ),
    )


def evaluate_generic_checks(
    checks: list[GenericCompensationCheck],
    posture: PostureMetrics,
    calibration_result: Optional[CalibrationResult],
    smoothed_landmarks: list,
    side: str,
    visibility_threshold: float,
    current_phase: RepState,
) -> dict[str, GenericCompensationResult]:
    """Evaluate generic compensation checks."""
    results = {}
    
    for check in checks:
        # Phase gating
        if check.allowed_phases is not None:
            if current_phase not in check.allowed_phases:
                # Phase not matched, skip evaluation
                continue
                
        # Calculate or map the metric
        current_val = None
        if check.metric_calculator is not None:
            try:
                current_val = check.metric_calculator(smoothed_landmarks, side=side, visibility_threshold=visibility_threshold)
            except Exception:
                current_val = None
        else:
            # If no calculator, we could map from PostureMetrics, but for now 
            # we rely on the calculator or it's unmapped.
            # To support legacy mapping if requested, we could check hasattr(posture, check.id).
            # But the prompt says "rely on existing mapped metric if None".
            if hasattr(posture, check.id):
                current_val = getattr(posture, check.id)
            else:
                # E.g. torso_lean -> trunk_lean fallback, etc.
                pass
                
        if current_val is None or not math.isfinite(current_val):
            results[check.id] = GenericCompensationResult(
                id=check.id, condition=check.condition, flagged=False, phase_used=current_phase
            )
            continue
            
        flagged = False
        threshold_used = None
        deviation = None
        baseline_mean = None
        lower_bound = None
        upper_bound = None
        
        if check.condition == CheckCondition.ABOVE:
            threshold_used = check.upper_threshold
            flagged = current_val > threshold_used
            
        elif check.condition == CheckCondition.BELOW:
            threshold_used = check.lower_threshold
            flagged = current_val < threshold_used
            
        elif check.condition == CheckCondition.RANGE:
            lower_bound = check.lower_threshold
            upper_bound = check.upper_threshold
            flagged = current_val < lower_bound or current_val > upper_bound
            
        elif check.condition == CheckCondition.BASELINE_DEVIATION:
            # Lookup baseline
            if calibration_result is None or not hasattr(calibration_result, check.id):
                # Cannot evaluate without baseline
                results[check.id] = GenericCompensationResult(
                    id=check.id, condition=check.condition, flagged=False, phase_used=current_phase
                )
                continue
                
            baseline = getattr(calibration_result, check.id)
            if baseline is None:
                results[check.id] = GenericCompensationResult(
                    id=check.id, condition=check.condition, flagged=False, phase_used=current_phase
                )
                continue
                
            # Default floor for unknown baseline deviation checks
            floor = 0.0
            deviation = compute_deviation(current_val, baseline.mean)
            threshold_used = compute_threshold(baseline.std, floor)
            baseline_mean = baseline.mean
            flagged = deviation > threshold_used
            
        results[check.id] = GenericCompensationResult(
            id=check.id,
            condition=check.condition,
            flagged=flagged,
            current_value=current_val,
            threshold_used=threshold_used,
            baseline_mean=baseline_mean,
            deviation=deviation,
            phase_used=current_phase,
            lower_bound_used=lower_bound,
            upper_bound_used=upper_bound,
        )
        
    return results


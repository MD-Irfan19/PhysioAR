"""Movement quality assessment for PhysioAR.

Calculates per-repetition movement quality using the repetition
boundaries from Phase 5 and the frame-level data from Phases 3,
4A, and 4B.

For each completed repetition, five scores are computed:

1. ROM Score         — how much of the target ROM was achieved
2. Alignment Score   — posture deviation from calibration baseline
3. Stability Score   — consistency of the tracked joint angle
4. Compensation Score — penalties for distinct compensations
5. Overall Quality   — weighted combination of all four

All scores are bounded to [0, 100].

Phase 6 — Movement Quality Assessment.

============================================================
FORMULAS (from Project Overview Module 7)
============================================================

ROM:
    min(max_angle / target_rom, 1.0) × 100

Alignment:
    max(0, 100 − (avg_deviation / normalization) × 100)

Stability:
    max(0, 100 − (angle_stddev / normalization) × 100)

Compensation:
    max(0, 100 − count_distinct_flagged × penalty)

Overall:
    0.3 × ROM + 0.2 × Alignment + 0.2 × Stability + 0.3 × Compensation

============================================================
ALIGNMENT — CROSS-UNIT HANDLING
============================================================

The three posture metrics have mixed units:
  - torso_lean: degrees
  - shoulder_height_difference: normalized image-coordinate units
  - neck_tilt: degrees

To produce a single alignment score, each metric's deviation
is individually divided by the ALIGNMENT_NORMALIZATION_CONSTANT
to obtain a [0, 1+] ratio, then the ratios are averaged.

The shoulder_height_difference deviation is multiplied by a
SHOULDER_UNIT_SCALE factor (100.0) before dividing by the
normalization constant to approximate degree-like units.
This is a pragmatic engineering choice documented here and
configurable for future tuning. It is NOT clinically validated.

The resulting average ratio represents:
  "How much of the maximum tolerable deviation was observed."

Alignment = max(0, 100 − average_ratio × 100)

============================================================
NONE / MISSING DATA
============================================================

- None angle values are excluded from ROM, stability, and
  alignment calculations (NOT treated as 0).
- If no valid angle samples exist in the rep window,
  stability defaults to 100 (no variation observed).
- If no valid posture samples exist, alignment defaults to 100.
- If compensation result is None, no penalty is applied.
- NaN and Infinity are never produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.compensation import CompensationResult
from src.config import (
    ALIGNMENT_NORMALIZATION_CONSTANT,
    STABILITY_NORMALIZATION_CONSTANT,
    COMPENSATION_PENALTY,
)


# Pragmatic scale factor to convert shoulder_height_difference
# (normalized image-coordinate units) to approximate degree-like
# units before alignment normalization. NOT clinically validated.
# Configurable for future tuning.
SHOULDER_UNIT_SCALE = 100.0


# ============================================================
# Frame sample record
# ============================================================


@dataclass
class FrameSample:
    """Per-frame data retained for quality scoring.

    Stored in a bounded buffer during the live loop.
    Only the information needed by Phase 6 is retained.

    Attributes:
        frame_index: The frame counter value.
        angle: The tracked exercise joint angle (degrees),
            or None if unavailable.
        torso_lean: Torso lean metric (degrees), or None.
        shoulder_height_diff: Shoulder height difference
            (normalized image units), or None.
        neck_tilt: Neck tilt metric (degrees), or None.
        torso_lean_flagged: Whether torso lean compensation
            was flagged this frame.
        shoulder_hike_flagged: Whether shoulder hike compensation
            was flagged this frame.
        neck_tilt_flagged: Whether neck tilt compensation
            was flagged this frame.
        hip_rotation_flagged: Whether hip rotation compensation
            was flagged this frame.
        lateral_trunk_lean_flagged: Whether lateral trunk lean compensation
            was flagged this frame.
        torso_deviation: abs(torso_lean - baseline_mean), or None.
        shoulder_deviation: abs(shoulder_diff - baseline_mean), or None.
        neck_deviation: abs(neck_tilt - baseline_mean), or None.
        hip_rotation_deviation: abs(hip_rotation - baseline_mean), or None.
        lateral_trunk_lean_deviation: abs(lateral_trunk_lean - baseline_mean), or None.
    """

    frame_index: int = 0
    angle: Optional[float] = None
    torso_lean: Optional[float] = None
    shoulder_height_diff: Optional[float] = None
    neck_tilt: Optional[float] = None
    torso_lean_flagged: bool = False
    shoulder_hike_flagged: bool = False
    neck_tilt_flagged: bool = False
    hip_rotation_flagged: bool = False
    lateral_trunk_lean_flagged: bool = False
    torso_deviation: Optional[float] = None
    shoulder_deviation: Optional[float] = None
    neck_deviation: Optional[float] = None
    hip_rotation_deviation: Optional[float] = None
    lateral_trunk_lean_deviation: Optional[float] = None


# ============================================================
# Quality result
# ============================================================


@dataclass(frozen=True)
class QualityResult:
    """Per-repetition movement quality assessment.

    All score fields are bounded to [0, 100].

    Attributes:
        rep_number: The 1-based repetition number.
        rom_score: Range of motion score.
        alignment_score: Posture alignment score.
        stability_score: Joint angle stability score.
        compensation_score: Compensation penalty score.
        overall_score: Weighted combination of all four.
        max_angle: Maximum exercise angle during the rep.
        target_rom: Target ROM from the exercise definition.
        avg_alignment_deviation: Average normalized deviation ratio.
        angle_std: Standard deviation of exercise angle during rep.
        compensation_types: Number of distinct compensation types flagged.
        start_frame: Rep start frame index.
        top_frame: Rep top frame index.
        end_frame: Rep end frame index.
    """

    rep_number: int
    rom_score: float
    alignment_score: float
    stability_score: float
    compensation_score: float
    overall_score: float
    max_angle: Optional[float]
    target_rom: float
    avg_alignment_deviation: Optional[float]
    angle_std: Optional[float]
    compensation_types: int
    start_frame: Optional[int]
    top_frame: Optional[int]
    end_frame: Optional[int]


# ============================================================
# Pure scoring functions
# ============================================================


def compute_rom_score(
    max_angle: Optional[float],
    target_rom: float,
) -> float:
    """Compute the ROM score for a completed repetition.

    ROM = min(max_angle / target_rom, 1.0) × 100

    Args:
        max_angle: Maximum tracked exercise angle during the rep,
            or None if no valid angle samples existed.
        target_rom: Target range of motion from ExerciseDefinition.

    Returns:
        ROM score bounded to [0, 100]. Returns 0 if max_angle is
        None or target_rom is invalid (≤ 0).
    """
    if max_angle is None or target_rom <= 0:
        return 0.0
    if max_angle < 0:
        return 0.0
    return min(max_angle / target_rom, 1.0) * 100.0


def compute_alignment_score(
    avg_deviation_ratio: Optional[float],
    normalization_constant: float = ALIGNMENT_NORMALIZATION_CONSTANT,
) -> float:
    """Compute the alignment score for a completed repetition.

    Alignment = max(0, 100 − avg_deviation_ratio × 100)

    where avg_deviation_ratio is the mean of per-metric
    (deviation / normalization_constant) values.

    Note: The avg_deviation_ratio is already normalized BEFORE
    being passed to this function. This function applies the
    final score formula.

    Args:
        avg_deviation_ratio: Average of (deviation / constant)
            across valid posture metrics during the rep, or None
            if no valid samples existed.
        normalization_constant: Not used here (normalization is
            done upstream), kept for interface consistency.

    Returns:
        Alignment score bounded to [0, 100]. Returns 100 if
        avg_deviation_ratio is None (no deviation observed).
    """
    if avg_deviation_ratio is None:
        return 100.0
    return max(0.0, 100.0 - avg_deviation_ratio * 100.0)


def compute_stability_score(
    angle_std: Optional[float],
    normalization_constant: float = STABILITY_NORMALIZATION_CONSTANT,
) -> float:
    """Compute the stability score for a completed repetition.

    Stability = max(0, 100 − (angle_std / normalization) × 100)

    Args:
        angle_std: Standard deviation of the tracked exercise
            angle during the rep, or None if insufficient samples.
        normalization_constant: The stability normalization constant
            from config. Must be > 0.

    Returns:
        Stability score bounded to [0, 100]. Returns 100 if
        angle_std is None (no variation observed).
    """
    if angle_std is None:
        return 100.0
    if normalization_constant <= 0:
        return 0.0
    return max(0.0, 100.0 - (angle_std / normalization_constant) * 100.0)


def compute_compensation_score(
    distinct_flagged_types: int,
    penalty: float = COMPENSATION_PENALTY,
) -> float:
    """Compute the compensation score for a completed repetition.

    Compensation = max(0, 100 − count × penalty)

    One penalty per DISTINCT compensation type flagged at least
    once during the rep.

    Args:
        distinct_flagged_types: Number of distinct compensation
            types that were flagged at least once during the rep.
        penalty: Points deducted per flagged type (from config).

    Returns:
        Compensation score bounded to [0, 100].
    """
    return max(0.0, 100.0 - distinct_flagged_types * penalty)


def compute_overall_quality_score(
    rom: float,
    alignment: float,
    stability: float,
    compensation: float,
) -> float:
    """Compute the weighted overall quality score.

    Quality = 0.3×ROM + 0.2×Alignment + 0.2×Stability + 0.3×Compensation

    Args:
        rom: ROM score [0, 100].
        alignment: Alignment score [0, 100].
        stability: Stability score [0, 100].
        compensation: Compensation score [0, 100].

    Returns:
        Overall quality score bounded to [0, 100].
    """
    raw = 0.3 * rom + 0.2 * alignment + 0.2 * stability + 0.3 * compensation
    return max(0.0, min(100.0, raw))


# ============================================================
# Aggregation helpers
# ============================================================


def _extract_rep_samples(
    frame_history: list[FrameSample],
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> list[FrameSample]:
    """Extract samples within the rep window [start_frame, end_frame].

    Args:
        frame_history: The bounded frame history buffer.
        start_frame: Rep start frame index (inclusive).
        end_frame: Rep end frame index (inclusive).

    Returns:
        List of FrameSample objects within the window.
        Empty list if boundaries are None.
    """
    if start_frame is None or end_frame is None:
        return []
    return [
        s for s in frame_history
        if s.frame_index >= start_frame and s.frame_index <= end_frame
    ]


def _compute_max_angle(samples: list[FrameSample]) -> Optional[float]:
    """Find the maximum exercise angle in the rep window.

    Ignores None angle values.

    Returns None if no valid angles exist.
    """
    valid = [s.angle for s in samples if s.angle is not None]
    if not valid:
        return None
    return max(valid)


def _compute_angle_std(samples: list[FrameSample]) -> Optional[float]:
    """Compute standard deviation of exercise angle in the rep window.

    Uses population standard deviation (matching the documented formula
    for "StdDev of Tracked Joint Angle During Rep").

    Ignores None angle values.

    Returns None if fewer than 2 valid samples exist.
    """
    valid = [s.angle for s in samples if s.angle is not None]
    if len(valid) < 2:
        return None
    mean = sum(valid) / len(valid)
    variance = sum((x - mean) ** 2 for x in valid) / len(valid)
    return math.sqrt(variance)


def _compute_avg_alignment_deviation_ratio(
    samples: list[FrameSample],
    normalization_constant: float = ALIGNMENT_NORMALIZATION_CONSTANT,
) -> Optional[float]:
    """Compute average alignment deviation ratio across the rep window.

    For each valid frame, computes per-metric deviation ratios:
        torso_ratio     = torso_deviation / normalization_constant
        shoulder_ratio  = (shoulder_deviation × SHOULDER_UNIT_SCALE) / normalization_constant
        neck_ratio      = neck_deviation / normalization_constant

    Then averages all valid ratios across all frames and all metrics.

    Returns None if no valid deviation samples exist.
    """
    if normalization_constant <= 0:
        return None

    ratios = []
    for s in samples:
        if s.torso_deviation is not None:
            ratios.append(s.torso_deviation / normalization_constant)
        if s.shoulder_deviation is not None:
            ratios.append(
                (s.shoulder_deviation * SHOULDER_UNIT_SCALE)
                / normalization_constant
            )
        if s.neck_deviation is not None:
            ratios.append(s.neck_deviation / normalization_constant)
        if s.hip_rotation_deviation is not None:
            ratios.append(s.hip_rotation_deviation / normalization_constant)
        if s.lateral_trunk_lean_deviation is not None:
            ratios.append(s.lateral_trunk_lean_deviation / normalization_constant)

    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def _count_distinct_compensations(samples: list[FrameSample]) -> int:
    """Count distinct compensation types flagged at least once in the rep.

    Returns:
        Integer in [0, 3].
    """
    torso_flagged = any(s.torso_lean_flagged for s in samples)
    shoulder_flagged = any(s.shoulder_hike_flagged for s in samples)
    neck_flagged = any(s.neck_tilt_flagged for s in samples)
    hip_flagged = any(s.hip_rotation_flagged for s in samples)
    lat_lean_flagged = any(s.lateral_trunk_lean_flagged for s in samples)
    return sum([torso_flagged, shoulder_flagged, neck_flagged, hip_flagged, lat_lean_flagged])


# ============================================================
# Per-rep quality evaluation
# ============================================================


def evaluate_rep_quality(
    rep_number: int,
    frame_history: list[FrameSample],
    start_frame: Optional[int],
    top_frame: Optional[int],
    end_frame: Optional[int],
    target_rom: float,
) -> QualityResult:
    """Evaluate movement quality for a single completed repetition.

    Extracts samples from the frame history within the rep window,
    then calculates all five quality metrics.

    Args:
        rep_number: The 1-based repetition number.
        frame_history: The bounded frame history buffer.
        start_frame: Rep start frame index.
        top_frame: Rep top frame index.
        end_frame: Rep end frame index.
        target_rom: Target ROM from ExerciseDefinition.

    Returns:
        A QualityResult with all scores bounded to [0, 100].
    """
    samples = _extract_rep_samples(frame_history, start_frame, end_frame)

    # ROM.
    max_angle = _compute_max_angle(samples)
    rom = compute_rom_score(max_angle, target_rom)

    # Alignment.
    avg_dev_ratio = _compute_avg_alignment_deviation_ratio(samples)
    alignment = compute_alignment_score(avg_dev_ratio)

    # Stability.
    angle_std = _compute_angle_std(samples)
    stability = compute_stability_score(angle_std)

    # Compensation.
    comp_types = _count_distinct_compensations(samples)
    compensation = compute_compensation_score(comp_types)

    # Overall.
    overall = compute_overall_quality_score(rom, alignment, stability, compensation)

    return QualityResult(
        rep_number=rep_number,
        rom_score=rom,
        alignment_score=alignment,
        stability_score=stability,
        compensation_score=compensation,
        overall_score=overall,
        max_angle=max_angle,
        target_rom=target_rom,
        avg_alignment_deviation=avg_dev_ratio,
        angle_std=angle_std,
        compensation_types=comp_types,
        start_frame=start_frame,
        top_frame=top_frame,
        end_frame=end_frame,
    )

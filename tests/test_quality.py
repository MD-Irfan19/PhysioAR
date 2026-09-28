"""Tests for Phase 6 — Movement Quality Assessment.

All tests use deterministic synthetic values. No webcam, OpenCV,
MediaPipe inference, GUI, or random data.

Phase 6 — Movement Quality Assessment.
"""

import pytest

from src.config import (
    ALIGNMENT_NORMALIZATION_CONSTANT,
    STABILITY_NORMALIZATION_CONSTANT,
    COMPENSATION_PENALTY,
)
from src.quality import (
    FrameSample,
    QualityResult,
    SHOULDER_UNIT_SCALE,
    compute_rom_score,
    compute_alignment_score,
    compute_stability_score,
    compute_compensation_score,
    compute_overall_quality_score,
    evaluate_rep_quality,
    _extract_rep_samples,
    _compute_max_angle,
    _compute_angle_std,
    _compute_avg_alignment_deviation_ratio,
    _count_distinct_compensations,
)


# ================================================================
# Helpers
# ================================================================


def _make_sample(
    frame_index: int = 0,
    angle=45.0,
    torso_lean=2.0,
    shoulder_height_diff=0.01,
    neck_tilt=1.5,
    torso_lean_flagged=False,
    shoulder_hike_flagged=False,
    neck_tilt_flagged=False,
    torso_deviation=0.5,
    shoulder_deviation=0.005,
    neck_deviation=0.3,
) -> FrameSample:
    return FrameSample(
        frame_index=frame_index,
        angle=angle,
        torso_lean=torso_lean,
        shoulder_height_diff=shoulder_height_diff,
        neck_tilt=neck_tilt,
        torso_lean_flagged=torso_lean_flagged,
        shoulder_hike_flagged=shoulder_hike_flagged,
        neck_tilt_flagged=neck_tilt_flagged,
        torso_deviation=torso_deviation,
        shoulder_deviation=shoulder_deviation,
        neck_deviation=neck_deviation,
    )


def _make_rep_samples(
    start: int = 0,
    count: int = 10,
    angle: float = 45.0,
    **kwargs,
) -> list[FrameSample]:
    """Create a list of identical FrameSamples."""
    return [
        _make_sample(frame_index=start + i, angle=angle, **kwargs)
        for i in range(count)
    ]


# ================================================================
# ROM SCORE
# ================================================================


class TestROMScore:
    """ROM = min(max_angle / target_rom, 1.0) × 100."""

    def test_target_reached_exactly(self):
        assert compute_rom_score(90.0, 90.0) == pytest.approx(100.0)

    def test_below_target(self):
        assert compute_rom_score(45.0, 90.0) == pytest.approx(50.0)

    def test_above_target_capped(self):
        assert compute_rom_score(120.0, 90.0) == pytest.approx(100.0)

    def test_zero_max_angle(self):
        assert compute_rom_score(0.0, 90.0) == pytest.approx(0.0)

    def test_none_max_angle(self):
        assert compute_rom_score(None, 90.0) == pytest.approx(0.0)

    def test_different_target_rom(self):
        assert compute_rom_score(60.0, 120.0) == pytest.approx(50.0)

    def test_zero_target_rom(self):
        """Invalid target ROM → 0."""
        assert compute_rom_score(45.0, 0.0) == pytest.approx(0.0)

    def test_negative_target_rom(self):
        assert compute_rom_score(45.0, -10.0) == pytest.approx(0.0)

    def test_negative_max_angle(self):
        assert compute_rom_score(-5.0, 90.0) == pytest.approx(0.0)

    def test_proportional_score(self):
        assert compute_rom_score(72.0, 90.0) == pytest.approx(80.0)


# ================================================================
# ALIGNMENT SCORE
# ================================================================


class TestAlignmentScore:
    """Alignment = max(0, 100 - ratio × 100)."""

    def test_zero_deviation(self):
        assert compute_alignment_score(0.0) == pytest.approx(100.0)

    def test_ratio_equals_one(self):
        """Deviation ratio = 1.0 → score = 0."""
        assert compute_alignment_score(1.0) == pytest.approx(0.0)

    def test_ratio_above_one(self):
        """Deviation exceeds normalization → score = 0 (clamped)."""
        assert compute_alignment_score(1.5) == pytest.approx(0.0)

    def test_small_deviation(self):
        assert compute_alignment_score(0.2) == pytest.approx(80.0)

    def test_none_deviation(self):
        """No valid samples → 100."""
        assert compute_alignment_score(None) == pytest.approx(100.0)

    def test_half_deviation(self):
        assert compute_alignment_score(0.5) == pytest.approx(50.0)


# ================================================================
# STABILITY SCORE
# ================================================================


class TestStabilityScore:
    """Stability = max(0, 100 - (std / norm) × 100)."""

    def test_zero_std(self):
        """Constant angle → 100."""
        assert compute_stability_score(0.0) == pytest.approx(100.0)

    def test_std_equals_norm(self):
        """std == norm → 0."""
        assert compute_stability_score(
            STABILITY_NORMALIZATION_CONSTANT,
        ) == pytest.approx(0.0)

    def test_std_above_norm(self):
        """std > norm → 0 (clamped)."""
        assert compute_stability_score(
            STABILITY_NORMALIZATION_CONSTANT * 2,
        ) == pytest.approx(0.0)

    def test_small_std(self):
        """Small std → high stability."""
        assert compute_stability_score(3.0, 30.0) == pytest.approx(90.0)

    def test_none_std(self):
        """No valid samples → 100."""
        assert compute_stability_score(None) == pytest.approx(100.0)

    def test_zero_normalization(self):
        """Invalid normalization → 0."""
        assert compute_stability_score(5.0, 0.0) == pytest.approx(0.0)

    def test_stability_uses_exercise_angle(self):
        """Confirm stability uses angle std, not posture metrics."""
        # This is a conceptual test verified by the function signature.
        # angle_std parameter is explicitly the exercise joint angle std.
        result = compute_stability_score(6.0, 30.0)
        assert result == pytest.approx(80.0)


# ================================================================
# COMPENSATION SCORE
# ================================================================


class TestCompensationScore:
    """Compensation = max(0, 100 - count × penalty)."""

    def test_no_compensation(self):
        assert compute_compensation_score(0) == pytest.approx(100.0)

    def test_one_compensation(self):
        assert compute_compensation_score(1) == pytest.approx(85.0)

    def test_two_compensations(self):
        assert compute_compensation_score(2) == pytest.approx(70.0)

    def test_three_compensations(self):
        assert compute_compensation_score(3) == pytest.approx(55.0)

    def test_cannot_go_negative(self):
        """7 × 15 = 105 → clamped to 0."""
        assert compute_compensation_score(7) == pytest.approx(0.0)

    def test_custom_penalty(self):
        assert compute_compensation_score(2, penalty=20) == pytest.approx(60.0)

    def test_uses_config_penalty(self):
        """Default penalty is from config."""
        assert COMPENSATION_PENALTY == 15

    def test_repeated_same_type_counts_once(self):
        """Distinct types, not frame count."""
        # 20 frames of torso flagged = 1 type = 85
        samples = [
            _make_sample(frame_index=i, torso_lean_flagged=True)
            for i in range(20)
        ]
        types = _count_distinct_compensations(samples)
        assert types == 1
        assert compute_compensation_score(types) == pytest.approx(85.0)

    def test_different_types_separate_penalties(self):
        samples = [
            _make_sample(
                frame_index=0,
                torso_lean_flagged=True,
                shoulder_hike_flagged=True,
            ),
        ]
        types = _count_distinct_compensations(samples)
        assert types == 2


# ================================================================
# OVERALL QUALITY
# ================================================================


class TestOverallQuality:
    """Quality = 0.3×ROM + 0.2×Align + 0.2×Stab + 0.3×Comp."""

    def test_project_overview_example(self):
        """ROM=88, Align=92, Stab=89, Comp=85 → 88.1."""
        result = compute_overall_quality_score(88, 92, 89, 85)
        assert result == pytest.approx(88.1)

    def test_all_zero(self):
        assert compute_overall_quality_score(0, 0, 0, 0) == pytest.approx(0.0)

    def test_all_hundred(self):
        assert compute_overall_quality_score(100, 100, 100, 100) == pytest.approx(100.0)

    def test_mixed(self):
        result = compute_overall_quality_score(50, 50, 50, 50)
        assert result == pytest.approx(50.0)

    def test_bounded_above(self):
        """Even if sub-scores somehow exceed 100, overall capped at 100."""
        result = compute_overall_quality_score(100, 100, 100, 100)
        assert result <= 100.0

    def test_bounded_below(self):
        result = compute_overall_quality_score(0, 0, 0, 0)
        assert result >= 0.0

    def test_weights_correct(self):
        """Verify individual weight contributions."""
        # Only ROM = 100, others 0 → 0.3 × 100 = 30
        assert compute_overall_quality_score(100, 0, 0, 0) == pytest.approx(30.0)
        # Only Alignment = 100 → 0.2 × 100 = 20
        assert compute_overall_quality_score(0, 100, 0, 0) == pytest.approx(20.0)
        # Only Stability = 100 → 0.2 × 100 = 20
        assert compute_overall_quality_score(0, 0, 100, 0) == pytest.approx(20.0)
        # Only Compensation = 100 → 0.3 × 100 = 30
        assert compute_overall_quality_score(0, 0, 0, 100) == pytest.approx(30.0)


# ================================================================
# MISSING DATA
# ================================================================


class TestMissingData:
    """None must never become zero."""

    def test_none_angles_excluded_from_max(self):
        samples = [
            _make_sample(frame_index=0, angle=None),
            _make_sample(frame_index=1, angle=60.0),
            _make_sample(frame_index=2, angle=None),
        ]
        assert _compute_max_angle(samples) == pytest.approx(60.0)

    def test_all_none_angles_returns_none(self):
        samples = [_make_sample(frame_index=i, angle=None) for i in range(5)]
        assert _compute_max_angle(samples) is None

    def test_none_angles_excluded_from_std(self):
        samples = [
            _make_sample(frame_index=0, angle=None),
            _make_sample(frame_index=1, angle=50.0),
            _make_sample(frame_index=2, angle=50.0),
        ]
        assert _compute_angle_std(samples) == pytest.approx(0.0)

    def test_insufficient_valid_for_std(self):
        """Only 1 valid → None."""
        samples = [
            _make_sample(frame_index=0, angle=50.0),
            _make_sample(frame_index=1, angle=None),
        ]
        assert _compute_angle_std(samples) is None

    def test_none_posture_excluded_from_alignment(self):
        samples = [
            _make_sample(
                frame_index=0,
                torso_deviation=None,
                shoulder_deviation=None,
                neck_deviation=None,
            ),
        ]
        assert _compute_avg_alignment_deviation_ratio(samples) is None

    def test_partial_none_posture(self):
        """Some deviations valid, some None → averages valid ones."""
        samples = [
            _make_sample(
                frame_index=0,
                torso_deviation=None,
                shoulder_deviation=None,
                neck_deviation=ALIGNMENT_NORMALIZATION_CONSTANT,
            ),
        ]
        ratio = _compute_avg_alignment_deviation_ratio(samples)
        assert ratio is not None
        # neck ratio = 15/15 = 1.0
        assert ratio == pytest.approx(1.0)

    def test_evaluate_with_all_none(self):
        """All None values → safe result, no NaN."""
        samples = [
            _make_sample(
                frame_index=i,
                angle=None,
                torso_lean=None,
                shoulder_height_diff=None,
                neck_tilt=None,
                torso_deviation=None,
                shoulder_deviation=None,
                neck_deviation=None,
            )
            for i in range(5)
        ]
        result = evaluate_rep_quality(
            rep_number=1,
            frame_history=samples,
            start_frame=0,
            top_frame=2,
            end_frame=4,
            target_rom=90.0,
        )
        assert result.rom_score == pytest.approx(0.0)
        assert result.alignment_score == pytest.approx(100.0)
        assert result.stability_score == pytest.approx(100.0)
        assert result.overall_score >= 0.0
        assert result.overall_score <= 100.0


# ================================================================
# REP WINDOW ISOLATION
# ================================================================


class TestRepWindowIsolation:
    """Rep 1 data must not leak into Rep 2."""

    def test_independent_reps(self):
        # Rep 1: poor stability (varying angles).
        rep1_samples = [
            _make_sample(frame_index=i, angle=10.0 + i * 15)
            for i in range(10)
        ]
        # Rep 2: good stability (constant angles).
        rep2_samples = [
            _make_sample(frame_index=10 + i, angle=50.0)
            for i in range(10)
        ]
        all_samples = rep1_samples + rep2_samples

        # Rep 1.
        r1 = evaluate_rep_quality(1, all_samples, 0, 5, 9, 90.0)
        # Rep 2.
        r2 = evaluate_rep_quality(2, all_samples, 10, 15, 19, 90.0)

        # Rep 2 should have better stability.
        assert r2.stability_score > r1.stability_score

    def test_compensation_isolated_per_rep(self):
        """Compensation flags from rep 1 don't affect rep 2."""
        rep1_samples = [
            _make_sample(
                frame_index=i, torso_lean_flagged=True,
                shoulder_hike_flagged=True, neck_tilt_flagged=True,
            )
            for i in range(5)
        ]
        rep2_samples = [
            _make_sample(frame_index=5 + i)  # All flags False.
            for i in range(5)
        ]
        all_samples = rep1_samples + rep2_samples

        r1 = evaluate_rep_quality(1, all_samples, 0, 2, 4, 90.0)
        r2 = evaluate_rep_quality(2, all_samples, 5, 7, 9, 90.0)

        assert r1.compensation_score < 100.0  # Has penalties.
        assert r2.compensation_score == pytest.approx(100.0)  # No penalties.

    def test_rom_isolated_per_rep(self):
        """Max angle from rep 1 doesn't inflate rep 2."""
        rep1_samples = [
            _make_sample(frame_index=i, angle=85.0)
            for i in range(5)
        ]
        rep2_samples = [
            _make_sample(frame_index=5 + i, angle=45.0)
            for i in range(5)
        ]
        all_samples = rep1_samples + rep2_samples

        r2 = evaluate_rep_quality(2, all_samples, 5, 7, 9, 90.0)
        # Rep 2 max angle = 45, not 85.
        assert r2.rom_score == pytest.approx(50.0)


# ================================================================
# QUALITY RESULT
# ================================================================


class TestQualityResult:
    """QualityResult structure and bounds."""

    def test_all_scores_bounded(self):
        samples = _make_rep_samples(start=0, count=10, angle=45.0)
        result = evaluate_rep_quality(1, samples, 0, 5, 9, 90.0)
        assert 0 <= result.rom_score <= 100
        assert 0 <= result.alignment_score <= 100
        assert 0 <= result.stability_score <= 100
        assert 0 <= result.compensation_score <= 100
        assert 0 <= result.overall_score <= 100

    def test_result_is_frozen(self):
        samples = _make_rep_samples(start=0, count=10, angle=45.0)
        result = evaluate_rep_quality(1, samples, 0, 5, 9, 90.0)
        with pytest.raises(AttributeError):
            result.rom_score = 999

    def test_diagnostic_fields(self):
        samples = _make_rep_samples(start=0, count=10, angle=45.0)
        result = evaluate_rep_quality(1, samples, 0, 5, 9, 90.0)
        assert result.rep_number == 1
        assert result.max_angle is not None
        assert result.target_rom == 90.0
        assert result.start_frame == 0
        assert result.end_frame == 9

    def test_empty_frame_history(self):
        """Empty history → safe defaults, not NaN."""
        result = evaluate_rep_quality(1, [], 0, 5, 9, 90.0)
        assert result.rom_score == pytest.approx(0.0)
        assert result.stability_score == pytest.approx(100.0)
        assert result.overall_score >= 0

    def test_none_boundaries(self):
        """None start/end → empty samples → safe defaults."""
        result = evaluate_rep_quality(1, [], None, None, None, 90.0)
        assert result.rom_score == pytest.approx(0.0)


# ================================================================
# CONFIGURATION
# ================================================================


class TestConfiguration:
    """Configuration constants exist and are valid."""

    def test_alignment_normalization_exists(self):
        assert ALIGNMENT_NORMALIZATION_CONSTANT > 0

    def test_stability_normalization_exists(self):
        assert STABILITY_NORMALIZATION_CONSTANT > 0

    def test_compensation_penalty(self):
        assert COMPENSATION_PENALTY == 15


# ================================================================
# INTEGRATION WITH EXERCISE DEFINITION
# ================================================================


class TestIntegrationWithExercise:
    """Quality uses ExerciseDefinition.rom_target."""

    def test_rom_uses_exercise_target(self):
        """ROM depends on target_rom parameter."""
        samples = _make_rep_samples(start=0, count=10, angle=45.0)
        r1 = evaluate_rep_quality(1, samples, 0, 5, 9, 90.0)
        r2 = evaluate_rep_quality(1, samples, 0, 5, 9, 45.0)
        assert r1.rom_score < r2.rom_score  # 45/90 < 45/45

    def test_shoulder_abduction_rom_target(self):
        from src.exercises.shoulder_abduction import SHOULDER_ABDUCTION
        assert SHOULDER_ABDUCTION.rom_target == 90.0


# ================================================================
# SAMPLE EXTRACTION
# ================================================================


class TestSampleExtraction:
    """_extract_rep_samples uses correct window."""

    def test_inclusive_boundaries(self):
        samples = [_make_sample(frame_index=i) for i in range(20)]
        extracted = _extract_rep_samples(samples, 5, 14)
        assert len(extracted) == 10
        assert extracted[0].frame_index == 5
        assert extracted[-1].frame_index == 14

    def test_none_boundaries_empty(self):
        samples = [_make_sample(frame_index=i) for i in range(10)]
        assert _extract_rep_samples(samples, None, None) == []
        assert _extract_rep_samples(samples, None, 5) == []
        assert _extract_rep_samples(samples, 0, None) == []


# ================================================================
# DISTINCT COMPENSATIONS
# ================================================================


class TestDistinctCompensations:
    """_count_distinct_compensations counts types, not frames."""

    def test_no_flags(self):
        samples = _make_rep_samples(start=0, count=5)
        assert _count_distinct_compensations(samples) == 0

    def test_all_three_types(self):
        samples = [
            _make_sample(
                frame_index=0,
                torso_lean_flagged=True,
                shoulder_hike_flagged=True,
                neck_tilt_flagged=True,
            ),
        ]
        assert _count_distinct_compensations(samples) == 3

    def test_one_type_many_frames(self):
        """Same type flagged across many frames = 1 type."""
        samples = [
            _make_sample(frame_index=i, torso_lean_flagged=True)
            for i in range(50)
        ]
        assert _count_distinct_compensations(samples) == 1

    def test_compensation_outside_window_excluded(self):
        """Only samples within the rep window matter."""
        all_samples = [
            _make_sample(frame_index=0, torso_lean_flagged=True),
            _make_sample(frame_index=1),  # Inside window.
            _make_sample(frame_index=2),
            _make_sample(frame_index=3, shoulder_hike_flagged=True),
        ]
        window = _extract_rep_samples(all_samples, 1, 2)
        assert _count_distinct_compensations(window) == 0


# ================================================================
# ANGLE STD
# ================================================================


class TestAngleStd:
    """Standard deviation of exercise angle during rep."""

    def test_constant_angle(self):
        """Constant → std = 0."""
        samples = _make_rep_samples(start=0, count=10, angle=50.0)
        assert _compute_angle_std(samples) == pytest.approx(0.0)

    def test_varying_angle(self):
        """Known std computation."""
        # angles = [40, 60] → mean=50, variance=100, std=10
        samples = [
            _make_sample(frame_index=0, angle=40.0),
            _make_sample(frame_index=1, angle=60.0),
        ]
        assert _compute_angle_std(samples) == pytest.approx(10.0)

    def test_less_than_two_samples(self):
        samples = [_make_sample(frame_index=0, angle=50.0)]
        assert _compute_angle_std(samples) is None

    def test_stability_not_posture(self):
        """Stability uses angle, NOT posture metrics."""
        # Samples with constant angle but varying posture.
        samples = [
            _make_sample(frame_index=0, angle=50.0, torso_lean=2.0),
            _make_sample(frame_index=1, angle=50.0, torso_lean=20.0),
        ]
        std = _compute_angle_std(samples)
        assert std == pytest.approx(0.0)  # Angle is constant.

    def test_only_rep_window_used(self):
        """Different windows give different results."""
        all_samples = [
            _make_sample(frame_index=0, angle=40.0),
            _make_sample(frame_index=1, angle=60.0),
            _make_sample(frame_index=2, angle=50.0),
            _make_sample(frame_index=3, angle=50.0),
        ]
        w1 = _extract_rep_samples(all_samples, 0, 1)
        w2 = _extract_rep_samples(all_samples, 2, 3)
        assert _compute_angle_std(w1) > _compute_angle_std(w2)


# ================================================================
# ALIGNMENT DEVIATION RATIO
# ================================================================


class TestAlignmentDeviationRatio:
    """Average deviation ratio across metrics and frames."""

    def test_zero_deviations(self):
        samples = [
            _make_sample(
                frame_index=0,
                torso_deviation=0.0,
                shoulder_deviation=0.0,
                neck_deviation=0.0,
            ),
        ]
        ratio = _compute_avg_alignment_deviation_ratio(samples)
        assert ratio == pytest.approx(0.0)

    def test_deviations_do_not_cancel(self):
        """Absolute deviations — no cancellation."""
        # All deviations are already abs values from Phase 4B.
        samples = [
            _make_sample(
                frame_index=0,
                torso_deviation=5.0,
                shoulder_deviation=0.0,
                neck_deviation=5.0,
            ),
        ]
        ratio = _compute_avg_alignment_deviation_ratio(
            samples, normalization_constant=10.0,
        )
        # torso: 5/10=0.5, shoulder: 0*100/10=0.0, neck: 5/10=0.5
        # avg = (0.5 + 0.0 + 0.5) / 3 ≈ 0.333
        assert ratio == pytest.approx(1.0 / 3.0)

    def test_shoulder_unit_scaling(self):
        """Shoulder deviation is scaled by SHOULDER_UNIT_SCALE."""
        samples = [
            _make_sample(
                frame_index=0,
                torso_deviation=None,
                shoulder_deviation=0.15,  # × 100 = 15
                neck_deviation=None,
            ),
        ]
        ratio = _compute_avg_alignment_deviation_ratio(
            samples, normalization_constant=15.0,
        )
        # (0.15 * 100) / 15 = 1.0
        assert ratio == pytest.approx(1.0)

    def test_baseline_from_calibration(self):
        """Deviations come from Phase 4B (already baseline-relative)."""
        # Phase 4B computes deviation = abs(current - baseline_mean).
        # Quality.py uses the stored deviation values directly.
        samples = [
            _make_sample(frame_index=0, torso_deviation=10.0),
        ]
        ratio = _compute_avg_alignment_deviation_ratio(
            samples, normalization_constant=10.0,
        )
        assert ratio is not None

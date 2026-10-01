"""Tests for Phase 7 — Explainable Feedback Engine.

All tests use deterministic synthetic values. No webcam, OpenCV,
MediaPipe inference, GUI, or random data.

Phase 7 — Explainable Feedback Engine.
"""

import pytest

from src.exercises.base import CameraOrientation, ExerciseDefinition
from src.exercises.shoulder_abduction import SHOULDER_ABDUCTION
from src.compensation import CompensationMetricResult, CompensationResult
from src.feedback import (
    generate_feedback,
    generate_all_feedback,
    format_deviation,
    COMPENSATION_TYPES,
)


# ================================================================
# Helpers
# ================================================================


def _make_exercise(templates=None) -> ExerciseDefinition:
    """Create a minimal ExerciseDefinition with specified templates."""
    return ExerciseDefinition(
        name="Test Exercise",
        camera_orientation=CameraOrientation.FRONT,
        target_joint="test",
        rom_target=90.0,
        rep_start_angle=15.0,
        rep_end_angle=80.0,
        angle_calculator=lambda lm, side: 0.0,
        feedback_templates=templates or {},
    )


def _make_metric(
    flagged=False,
    deviation=None,
    current_value=None,
    baseline_mean=0.0,
    baseline_std=1.0,
    threshold=5.0,
) -> CompensationMetricResult:
    return CompensationMetricResult(
        current_value=current_value,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        deviation=deviation,
        threshold=threshold,
        flagged=flagged,
    )


def _make_comp(
    torso_flagged=False, torso_deviation=None,
    shoulder_flagged=False, shoulder_deviation=None,
    neck_flagged=False, neck_deviation=None,
) -> CompensationResult:
    return CompensationResult(
        torso_lean=_make_metric(
            flagged=torso_flagged, deviation=torso_deviation,
        ),
        shoulder_hike=_make_metric(
            flagged=shoulder_flagged, deviation=shoulder_deviation,
        ),
        neck_tilt=_make_metric(
            flagged=neck_flagged, deviation=neck_deviation,
        ),
    )


# ================================================================
# Test 1 — Torso Lean
# ================================================================


class TestTorsoLean:
    """Torso lean feedback generation."""

    def test_correct_template_selected(self):
        msg = generate_feedback(SHOULDER_ABDUCTION, "torso_lean", 11.0)
        assert msg is not None
        assert "Torso Lean" in msg

    def test_actual_value_appears(self):
        msg = generate_feedback(SHOULDER_ABDUCTION, "torso_lean", 11.0)
        assert "11°" in msg

    def test_message_is_not_generic(self):
        msg = generate_feedback(SHOULDER_ABDUCTION, "torso_lean", 11.0)
        assert "shoulder isolation" in msg
        assert msg != "Please correct your posture."

    def test_corrective_instruction_present(self):
        msg = generate_feedback(SHOULDER_ABDUCTION, "torso_lean", 11.0)
        assert "reduce torso lean" in msg


# ================================================================
# Test 2 — Shoulder Hike
# ================================================================


class TestShoulderHike:
    """Shoulder hike feedback generation."""

    def test_correct_template_selected(self):
        msg = generate_feedback(SHOULDER_ABDUCTION, "shoulder_hike", 4.0)
        assert msg is not None
        assert "Shoulder Hike" in msg

    def test_measured_value_appears(self):
        msg = generate_feedback(SHOULDER_ABDUCTION, "shoulder_hike", 4.0)
        # Shoulder hike is NOT in degrees — no ° symbol.
        assert "4" in msg

    def test_message_differs_from_torso(self):
        torso = generate_feedback(SHOULDER_ABDUCTION, "torso_lean", 11.0)
        shoulder = generate_feedback(SHOULDER_ABDUCTION, "shoulder_hike", 4.0)
        assert torso != shoulder

    def test_no_degree_symbol(self):
        """Shoulder hike is in normalized image units, not degrees."""
        msg = generate_feedback(SHOULDER_ABDUCTION, "shoulder_hike", 4.0)
        # The value itself should not have a ° suffix.
        assert "4°" not in msg


# ================================================================
# Test 3 — Neck Tilt
# ================================================================


class TestNeckTilt:
    """Neck tilt feedback generation."""

    def test_correct_template_selected(self):
        msg = generate_feedback(SHOULDER_ABDUCTION, "neck_tilt", 7.0)
        assert msg is not None
        assert "Neck Tilt" in msg

    def test_measured_value_appears(self):
        msg = generate_feedback(SHOULDER_ABDUCTION, "neck_tilt", 7.0)
        assert "7°" in msg

    def test_correct_feedback_template(self):
        msg = generate_feedback(SHOULDER_ABDUCTION, "neck_tilt", 7.0)
        assert "head and neck aligned" in msg


# ================================================================
# Test 4 — Distinct Messages
# ================================================================


class TestDistinctMessages:
    """Each compensation type produces a distinct message."""

    def test_all_types_produce_distinct_messages(self):
        messages = set()
        for comp_type in COMPENSATION_TYPES:
            msg = generate_feedback(
                SHOULDER_ABDUCTION, comp_type, 10.0,
            )
            assert msg is not None
            messages.add(msg)
        # All three messages should be different.
        assert len(messages) == 3

    def test_messages_contain_type_name(self):
        names = {
            "torso_lean": "Torso Lean",
            "shoulder_hike": "Shoulder Hike",
            "neck_tilt": "Neck Tilt",
        }
        for comp_type, expected_name in names.items():
            msg = generate_feedback(
                SHOULDER_ABDUCTION, comp_type, 10.0,
            )
            assert expected_name in msg


# ================================================================
# Test 5 — Actual Value Formatting
# ================================================================


class TestValueFormatting:
    """Measured deviation values formatted correctly."""

    def test_integer_value(self):
        msg = generate_feedback(SHOULDER_ABDUCTION, "torso_lean", 11.0)
        assert "11°" in msg

    def test_decimal_value(self):
        msg = generate_feedback(SHOULDER_ABDUCTION, "torso_lean", 11.37)
        assert "11.37°" in msg

    def test_trailing_zeros_stripped(self):
        assert format_deviation("torso_lean", 11.0) == "11°"
        assert format_deviation("torso_lean", 11.10) == "11.1°"
        assert format_deviation("torso_lean", 11.50) == "11.5°"

    def test_degree_symbol_for_torso(self):
        assert "°" in format_deviation("torso_lean", 5.0)

    def test_degree_symbol_for_neck(self):
        assert "°" in format_deviation("neck_tilt", 5.0)

    def test_no_degree_for_shoulder(self):
        assert "°" not in format_deviation("shoulder_hike", 5.0)

    def test_small_value(self):
        result = format_deviation("shoulder_hike", 0.03456)
        assert result == "0.03"

    def test_zero_value(self):
        result = format_deviation("torso_lean", 0.0)
        assert result == "0°"


# ================================================================
# Test 6 — None Deviation
# ================================================================


class TestNoneDeviation:
    """deviation = None must be handled safely."""

    def test_none_returns_none(self):
        result = generate_feedback(SHOULDER_ABDUCTION, "torso_lean", None)
        assert result is None

    def test_no_none_in_output(self):
        result = generate_feedback(SHOULDER_ABDUCTION, "torso_lean", None)
        assert result is None  # Never produces "None°".

    def test_no_crash(self):
        # Should not raise any exception.
        for comp_type in COMPENSATION_TYPES:
            result = generate_feedback(SHOULDER_ABDUCTION, comp_type, None)
            assert result is None


# ================================================================
# Test 7 — Unknown Compensation
# ================================================================


class TestUnknownCompensation:
    """Unknown compensation type handled safely."""

    def test_unknown_returns_none(self):
        result = generate_feedback(
            SHOULDER_ABDUCTION, "unknown_compensation", 10.0,
        )
        assert result is None

    def test_no_crash(self):
        # Should not raise.
        result = generate_feedback(SHOULDER_ABDUCTION, "nonexistent", 5.0)
        assert result is None

    def test_empty_string_type(self):
        result = generate_feedback(SHOULDER_ABDUCTION, "", 5.0)
        assert result is None


# ================================================================
# Test 8 — Missing Template
# ================================================================


class TestMissingTemplate:
    """Exercise without a template for a compensation type."""

    def test_no_template_returns_none(self):
        exercise = _make_exercise(templates={})
        result = generate_feedback(exercise, "torso_lean", 11.0)
        assert result is None

    def test_partial_templates(self):
        """Exercise has some templates but not all."""
        exercise = _make_exercise(templates={
            "torso_lean": "Torso = {value} — fix it.",
        })
        assert generate_feedback(exercise, "torso_lean", 5.0) is not None
        assert generate_feedback(exercise, "neck_tilt", 5.0) is None


# ================================================================
# Test 9 — ExerciseDefinition Integration
# ================================================================


class TestExerciseDefinition:
    """Feedback comes from ExerciseDefinition.feedback_templates."""

    def test_templates_from_exercise(self):
        """Templates are read from the exercise, not hardcoded."""
        custom = _make_exercise(templates={
            "torso_lean": "CUSTOM: torso = {value}.",
        })
        msg = generate_feedback(custom, "torso_lean", 5.0)
        assert "CUSTOM" in msg

    def test_real_shoulder_abduction_templates(self):
        """SHOULDER_ABDUCTION has all three templates."""
        assert "torso_lean" in SHOULDER_ABDUCTION.feedback_templates
        assert "shoulder_hike" in SHOULDER_ABDUCTION.feedback_templates
        assert "neck_tilt" in SHOULDER_ABDUCTION.feedback_templates

    def test_templates_have_value_placeholder(self):
        """All templates contain {value}."""
        for template in SHOULDER_ABDUCTION.feedback_templates.values():
            assert "{value}" in template


# ================================================================
# Test 10 — Multiple Compensation Types
# ================================================================


class TestMultipleCompensations:
    """Independent feedback for multiple compensations."""

    def test_all_three_flagged(self):
        comp = _make_comp(
            torso_flagged=True, torso_deviation=11.0,
            shoulder_flagged=True, shoulder_deviation=4.0,
            neck_flagged=True, neck_deviation=7.0,
        )
        msgs = generate_all_feedback(SHOULDER_ABDUCTION, comp)
        assert len(msgs) == 3

    def test_one_flagged(self):
        comp = _make_comp(
            torso_flagged=True, torso_deviation=11.0,
        )
        msgs = generate_all_feedback(SHOULDER_ABDUCTION, comp)
        assert len(msgs) == 1
        assert "Torso Lean" in msgs[0]

    def test_none_flagged(self):
        comp = _make_comp()
        msgs = generate_all_feedback(SHOULDER_ABDUCTION, comp)
        assert len(msgs) == 0

    def test_comp_is_none(self):
        msgs = generate_all_feedback(SHOULDER_ABDUCTION, None)
        assert msgs == []

    def test_each_message_contains_its_value(self):
        comp = _make_comp(
            torso_flagged=True, torso_deviation=11.0,
            shoulder_flagged=True, shoulder_deviation=0.04,
            neck_flagged=True, neck_deviation=7.5,
        )
        msgs = generate_all_feedback(SHOULDER_ABDUCTION, comp)
        values_found = {
            "torso": any("11°" in m for m in msgs),
            "shoulder": any("0.04" in m for m in msgs),
            "neck": any("7.5°" in m for m in msgs),
        }
        assert all(values_found.values())

    def test_two_flagged(self):
        comp = _make_comp(
            torso_flagged=True, torso_deviation=5.0,
            neck_flagged=True, neck_deviation=3.0,
        )
        msgs = generate_all_feedback(SHOULDER_ABDUCTION, comp)
        assert len(msgs) == 2


# ================================================================
# Additional edge cases
# ================================================================


class TestEdgeCases:
    """Additional edge case coverage."""

    def test_flagged_but_deviation_none(self):
        """Flagged=True but deviation=None → no feedback (can't format)."""
        comp = _make_comp(
            torso_flagged=True, torso_deviation=None,
        )
        msgs = generate_all_feedback(SHOULDER_ABDUCTION, comp)
        # generate_feedback returns None for None deviation.
        assert len(msgs) == 0

    def test_deviation_present_but_not_flagged(self):
        """Deviation exists but not flagged → no feedback."""
        comp = _make_comp(
            torso_flagged=False, torso_deviation=3.0,
        )
        msgs = generate_all_feedback(SHOULDER_ABDUCTION, comp)
        assert len(msgs) == 0

    def test_compensation_types_constant(self):
        """COMPENSATION_TYPES matches expected identifiers."""
        assert "torso_lean" in COMPENSATION_TYPES
        assert "shoulder_hike" in COMPENSATION_TYPES
        assert "neck_tilt" in COMPENSATION_TYPES
        assert len(COMPENSATION_TYPES) == 3

"""Explainable feedback engine for PhysioAR.

Converts an existing compensation flag and its measured deviation
into a specific, explainable corrective feedback message using
the active ExerciseDefinition's feedback_templates.

The feedback engine is a pure translation layer:

    Compensation type  +  Measured deviation  +  ExerciseDefinition
                            ↓
                    Corrective message

It does NOT:
- Detect compensations (Phase 4B does that)
- Calculate deviations (Phase 4B does that)
- Score quality (Phase 6 does that)
- Provide clinical diagnosis

Phase 7 — Explainable Feedback Engine.

============================================================
VALUE FORMATTING
============================================================

Torso lean and neck tilt are measured in degrees → formatted
with "°" suffix.

Shoulder hike (shoulder_height_difference) is in normalized
image-coordinate units → formatted without "°".

============================================================
COMPENSATION TYPE IDENTIFIERS
============================================================

The compensation types match the field names on CompensationResult:

    "torso_lean"      → CompensationResult.torso_lean
    "shoulder_hike"   → CompensationResult.shoulder_hike
    "neck_tilt"       → CompensationResult.neck_tilt

============================================================
"""

from __future__ import annotations

from typing import Optional

from src.exercises.base import ExerciseDefinition
from src.compensation import CompensationResult


# Compensation types whose deviation is measured in degrees.
_DEGREE_TYPES = frozenset({"torso_lean", "neck_tilt"})

# All recognized compensation types and their corresponding
# CompensationResult attribute names (they match exactly).
COMPENSATION_TYPES = ("torso_lean", "shoulder_hike", "neck_tilt")


def format_deviation(
    compensation_type: str,
    deviation: float,
) -> str:
    """Format a measured deviation value for human display.

    Degree-based metrics (torso_lean, neck_tilt) receive a "°" suffix.
    Other metrics (shoulder_hike) are displayed without a degree symbol.

    Values are formatted to at most 2 decimal places, with trailing
    zeros stripped for cleanliness:
        11.0     → "11°"
        11.25    → "11.25°"
        0.03456  → "0.03"

    Args:
        compensation_type: The compensation type identifier.
        deviation: The measured deviation value.

    Returns:
        A human-readable string representation.
    """
    # Format to 2 decimal places, strip trailing zeros.
    formatted = f"{deviation:.2f}".rstrip("0").rstrip(".")

    if compensation_type in _DEGREE_TYPES:
        return f"{formatted}°"
    return formatted


def generate_feedback(
    exercise: ExerciseDefinition,
    compensation_type: str,
    deviation: Optional[float],
) -> Optional[str]:
    """Generate an explainable corrective feedback message.

    Looks up the feedback template from the active ExerciseDefinition,
    formats the measured deviation, and substitutes it into the
    template's ``{value}`` placeholder.

    Args:
        exercise: The active ExerciseDefinition containing
            feedback_templates.
        compensation_type: The compensation type identifier
            (e.g., "torso_lean", "shoulder_hike", "neck_tilt").
        deviation: The measured deviation value, or None if
            unavailable.

    Returns:
        A corrective feedback string containing the actual measured
        value, or None if:
        - The deviation is None (no measurement available).
        - The compensation_type is unknown/unsupported.
        - The exercise has no feedback template for this type.
    """
    # No deviation → no feedback (never fabricate a measurement).
    if deviation is None:
        return None

    # Look up template from the exercise definition.
    template = exercise.feedback_templates.get(compensation_type)
    if template is None:
        return None

    # Format the measured value and substitute into the template.
    formatted_value = format_deviation(compensation_type, deviation)
    return template.format(value=formatted_value)


def generate_all_feedback(
    exercise: ExerciseDefinition,
    comp: Optional[CompensationResult],
) -> list[str]:
    """Generate feedback messages for all flagged compensations.

    Iterates over the recognized compensation types, checks whether
    each is flagged in the CompensationResult, and generates feedback
    for those that are flagged.

    Args:
        exercise: The active ExerciseDefinition.
        comp: The current frame's CompensationResult, or None if
            compensation evaluation is unavailable.

    Returns:
        A list of feedback message strings for flagged compensations.
        Empty list if no compensations are flagged or comp is None.
    """
    if comp is None:
        return []

    messages = []

    # Map compensation type identifiers to their CompensationResult fields.
    type_to_result = {
        "torso_lean": comp.torso_lean,
        "shoulder_hike": comp.shoulder_hike,
        "neck_tilt": comp.neck_tilt,
    }

    for comp_type, metric_result in type_to_result.items():
        if metric_result.flagged:
            msg = generate_feedback(
                exercise, comp_type, metric_result.deviation,
            )
            if msg is not None:
                messages.append(msg)

    return messages

"""PhysioAR project configuration.

Central location for tunable parameters and constants used across
the PhysioAR pipeline.
"""

# ============================================================
# EMA Landmark Smoothing
# ============================================================

# EMA landmark smoothing factor (alpha).
# Higher values respond faster but smooth less.
# Lower values smooth more but introduce more lag.
# Initial value only; final tuning is deferred to Phase 9 validation.
SMOOTHING_ALPHA = 0.5

# ============================================================
# Calibration
# ============================================================

# Duration (in seconds) of the neutral-posture calibration capture.
# The system collects posture metric samples for approximately this
# many seconds to establish a session baseline.
CALIBRATION_SECONDS = 10

# Minimum number of valid calibration samples required to produce a
# reliable baseline. If fewer valid frames are collected (e.g., due
# to the user being out of frame), calibration fails rather than
# producing a misleading baseline from insufficient data.
MIN_CALIBRATION_SAMPLES = 30

# ============================================================
# Landmark Validity / Confidence Gating
# ============================================================

# Minimum visibility value a landmark must have to be considered
# reliable for calibration metric computation. Landmarks with
# visibility below this threshold are treated as unreliable and
# cause the frame to be rejected from calibration.
#
# This is an engineering validity gate intended to prevent obviously
# unreliable landmark observations from entering the calibration
# baseline. It does NOT guarantee anatomical correctness.
#
# Initial value only; not experimentally validated.
# Final tuning is deferred to a later validation phase.
LANDMARK_VISIBILITY_THRESHOLD = 0.5

# ============================================================
# Compensation Threshold Floors (Phase 4B)
# ============================================================

# Fixed minimum thresholds for compensation flagging.
# The effective threshold for each metric is:
#     max(2 × baseline_std, fixed_floor)
#
# These floors prevent zero-width thresholds when the
# calibration standard deviation is very small (e.g., the user
# stood very still during calibration).
#
# A compensation flag means ONLY: "The current metric's
# baseline-relative deviation exceeded the configured threshold."
# It does NOT imply clinical incorrectness or injury.
#
# Units:
#   TORSO_LEAN_FLOOR:    degrees
#   SHOULDER_HIKE_FLOOR: normalized image-coordinate units
#                        (same units as Phase 4A shoulder height diff)
#   NECK_TILT_FLOOR:     degrees
#
# Initial values only; not experimentally validated.
TORSO_LEAN_FLOOR = 5
SHOULDER_HIKE_FLOOR = 3
NECK_TILT_FLOOR = 5

# ============================================================
# Movement Quality Scoring (Phase 6)
# ============================================================

# Alignment normalization constant.
# Used in: Alignment = max(0, 100 - (avg_deviation / constant) * 100)
#
# The "average deviation" is the mean of absolute posture deviations
# across all three metrics (torso lean, shoulder height diff, neck tilt)
# during a completed rep. Each metric is individually normalized to
# [0, 1] before averaging to handle different units:
#   - torso_lean deviation / ALIGNMENT_NORMALIZATION_CONSTANT (degrees)
#   - shoulder_height_diff deviation / ALIGNMENT_NORMALIZATION_CONSTANT (degrees)
#     (shoulder height diff is scaled by 100 to approximate degree-like units)
#   - neck_tilt deviation / ALIGNMENT_NORMALIZATION_CONSTANT (degrees)
#
# Units: degrees (or degree-equivalent after unit scaling).
#
# Initial placeholder value. NOT experimentally validated.
# Intended to be tuned during MVP validation.
ALIGNMENT_NORMALIZATION_CONSTANT = 15.0

# Stability normalization constant.
# Used in: Stability = max(0, 100 - (angle_stddev / constant) * 100)
#
# Units: degrees (standard deviation of the tracked exercise angle).
#
# Initial placeholder value. NOT experimentally validated.
# Intended to be tuned during MVP validation.
STABILITY_NORMALIZATION_CONSTANT = 30.0

# Compensation penalty per distinct flagged compensation type.
# Used in: Compensation Score = max(0, 100 - count * penalty)
#
# One penalty is applied per DISTINCT compensation type flagged
# at least once during the completed repetition:
#   0 flagged types → 100
#   1 flagged type  → 85
#   2 flagged types → 70
#   3 flagged types → 55
#
# Units: score points (out of 100).
COMPENSATION_PENALTY = 15

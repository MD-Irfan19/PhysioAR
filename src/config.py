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

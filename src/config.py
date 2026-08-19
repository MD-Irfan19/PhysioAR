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

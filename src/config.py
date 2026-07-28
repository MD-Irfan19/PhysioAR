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

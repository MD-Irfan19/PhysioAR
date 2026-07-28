"""Exponential Moving Average (EMA) landmark smoothing for PhysioAR.

This module provides a lightweight EMA filter for reducing frame-to-frame
numerical jitter in MediaPipe landmark coordinates. The filter operates
independently on each landmark's x, y, z spatial coordinates.

The fundamental EMA equation is:

    Filtered_t = alpha × Current_t + (1 - alpha) × Filtered_(t-1)

Visibility and presence values are NOT smoothed — they represent
detection confidence and should be preserved as-is.

The production alpha value is obtained from ``src.config.SMOOTHING_ALPHA``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SmoothedLandmark:
    """Application-level landmark representation with raw and smoothed coordinates.

    Attributes:
        x: Smoothed x coordinate.
        y: Smoothed y coordinate.
        z: Smoothed z coordinate.
        raw_x: Original unsmoothed x coordinate from MediaPipe.
        raw_y: Original unsmoothed y coordinate from MediaPipe.
        raw_z: Original unsmoothed z coordinate from MediaPipe.
        visibility: Original visibility value (NOT smoothed).
        presence: Original presence value (NOT smoothed).
    """

    x: float
    y: float
    z: float
    raw_x: float
    raw_y: float
    raw_z: float
    visibility: float
    presence: float


class EMAFilter:
    """Exponential Moving Average filter for landmark coordinate smoothing.

    Maintains per-landmark filter state so that each landmark index is
    smoothed independently. The filter applies EMA only to spatial
    coordinates (x, y, z), not to visibility or presence values.

    First-observation behavior:
        The first observation for each landmark passes through unchanged,
        becoming the initial filtered value. The filter does NOT initialize
        to (0, 0, 0).

    Reset behavior:
        Calling ``reset()`` clears all internal state. The next valid
        observation will again pass through unchanged as a new baseline.

    Missing pose handling:
        If no pose is detected in a given frame, the filter is NOT updated
        with fake coordinates. The filter retains its previous state until
        the next valid observation. The caller should not call ``smooth()``
        when no landmarks are available.

    Changing landmark counts:
        If the incoming landmark count differs from the existing filter
        state, the filter state is safely reinitialized to prevent index
        errors. This covers edge cases where the model configuration or
        number of detected poses changes.

    Args:
        alpha: The smoothing factor. Must be in the range (0, 1].
            Higher values respond faster but smooth less.
            Lower values smooth more but add more lag.
            A value of 1.0 means no smoothing (immediate response).
    """

    def __init__(self, alpha: float) -> None:
        if not (0 < alpha <= 1.0):
            raise ValueError(
                f"alpha must be in the range (0, 1], got {alpha}"
            )
        self._alpha = alpha
        # Per-landmark previous filtered coordinates: list of (x, y, z).
        self._previous: list[tuple[float, float, float]] | None = None

    @property
    def alpha(self) -> float:
        """The current smoothing factor."""
        return self._alpha

    def smooth(
        self,
        landmarks: list,
    ) -> list[SmoothedLandmark]:
        """Apply EMA smoothing to a list of landmarks.

        Each landmark object must have at least ``x``, ``y``, ``z``
        attributes. ``visibility`` and ``presence`` are preserved
        from the original landmark if available, defaulting to 0.0.

        Args:
            landmarks: A list of landmark objects (e.g., MediaPipe
                NormalizedLandmark instances) for a single detected
                pose.

        Returns:
            A list of SmoothedLandmark objects with both raw and
            smoothed coordinates. On the first call (or after reset),
            smoothed coordinates equal raw coordinates.
        """
        count = len(landmarks)

        # If the filter has no previous state or the landmark count
        # changed, initialize from the current observation.
        if self._previous is None or len(self._previous) != count:
            self._previous = [
                (lm.x, lm.y, lm.z) for lm in landmarks
            ]
            return [
                SmoothedLandmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    raw_x=lm.x,
                    raw_y=lm.y,
                    raw_z=lm.z,
                    visibility=getattr(lm, "visibility", 0.0),
                    presence=getattr(lm, "presence", 0.0),
                )
                for lm in landmarks
            ]

        # Apply EMA to each landmark's spatial coordinates.
        alpha = self._alpha
        one_minus_alpha = 1.0 - alpha
        smoothed_list: list[SmoothedLandmark] = []

        for i, lm in enumerate(landmarks):
            prev_x, prev_y, prev_z = self._previous[i]

            filtered_x = alpha * lm.x + one_minus_alpha * prev_x
            filtered_y = alpha * lm.y + one_minus_alpha * prev_y
            filtered_z = alpha * lm.z + one_minus_alpha * prev_z

            self._previous[i] = (filtered_x, filtered_y, filtered_z)

            smoothed_list.append(
                SmoothedLandmark(
                    x=filtered_x,
                    y=filtered_y,
                    z=filtered_z,
                    raw_x=lm.x,
                    raw_y=lm.y,
                    raw_z=lm.z,
                    visibility=getattr(lm, "visibility", 0.0),
                    presence=getattr(lm, "presence", 0.0),
                )
            )

        return smoothed_list

    def reset(self) -> None:
        """Clear all internal filter state.

        After calling reset(), the next valid observation will pass
        through unchanged as a new filtered baseline.
        """
        self._previous = None

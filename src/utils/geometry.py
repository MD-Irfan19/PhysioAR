"""
Geometry utilities for the PhysioAR project.

This module provides pure mathematical functions for 2D geometry
calculations. These functions serve as foundational building blocks
for posture metrics, joint-angle calculations, compensation detection,
exercise repetition detection, movement quality analysis, and feedback
generation in later PhysioAR modules.

All functions are deterministic, stateless, and use only the Python
standard library.

Dependencies: None (Python standard library only).
"""

import math
from typing import Tuple

# Type alias for a 2D point represented as (x, y).
Point2D = Tuple[float, float]


def calculate_angle(
    point_a: Point2D,
    point_b: Point2D,
    point_c: Point2D,
) -> float:
    """Calculate the angle at point_b formed by rays point_b→point_a and point_b→point_c.

    The second argument, point_b, is ALWAYS the vertex. The angle is
    measured between the two vectors BA and BC, where:

        BA = point_a - point_b
        BC = point_c - point_b

    The calculation uses the dot-product formula:

        cos(theta) = (BA · BC) / (|BA| * |BC|)

    The cosine value is clamped to [-1, 1] before applying acos() to
    guard against floating-point rounding errors.

    Diagram::

                    point_a
                       ●
                      /
                     /
                    /
                   ● point_b  (vertex)
                    \\
                     \\
                      \\
                       ●
                    point_c

    Args:
        point_a: First outer point (x, y).
        point_b: Vertex point (x, y). This is where the angle is measured.
        point_c: Second outer point (x, y).

    Returns:
        The angle at point_b in degrees, in the range [0, 180].

    Raises:
        ValueError: If point_a == point_b or point_c == point_b, because
            the corresponding vector would have zero length and the angle
            is mathematically undefined. A coincident landmark must never
            be silently interpreted as a valid 0-degree measurement.

    Notes:
        - point_b is the vertex.
        - The angle is measured between vectors point_b→point_a and
          point_b→point_c.
        - The returned value is in degrees.
        - The valid result range is 0 to 180 degrees.
        - The function is symmetric: calculate_angle(a, b, c) ==
          calculate_angle(c, b, a) for any valid inputs.
        - ValueError is raised when point_a == point_b or
          point_c == point_b.
    """
    # Unpack point coordinates.
    ax, ay = point_a
    bx, by = point_b
    cx, cy = point_c

    # Construct vector BA = point_a - point_b.
    ba_x = ax - bx
    ba_y = ay - by

    # Construct vector BC = point_c - point_b.
    bc_x = cx - bx
    bc_y = cy - by

    # Check for degenerate vectors (zero-length).
    magnitude_ba = math.hypot(ba_x, ba_y)
    magnitude_bc = math.hypot(bc_x, bc_y)

    if magnitude_ba == 0.0:
        raise ValueError(
            "Cannot calculate angle: point_a and point_b are identical, "
            "so vector BA has zero length and the angle is undefined."
        )
    if magnitude_bc == 0.0:
        raise ValueError(
            "Cannot calculate angle: point_c and point_b are identical, "
            "so vector BC has zero length and the angle is undefined."
        )

    # Dot product of BA and BC.
    dot_product = ba_x * bc_x + ba_y * bc_y

    # Cosine of the angle.
    cosine_value = dot_product / (magnitude_ba * magnitude_bc)

    # Clamp to [-1, 1] to guard against floating-point rounding errors.
    clamped_cosine = max(-1.0, min(1.0, cosine_value))

    # Calculate the angle in radians, then convert to degrees.
    angle_radians = math.acos(clamped_cosine)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def calculate_distance(point_a: Point2D, point_b: Point2D) -> float:
    """Calculate the Euclidean distance between two 2D points.

    Uses the formula:

        distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)

    Works with both integer and floating-point coordinates.

    Args:
        point_a: First point (x, y).
        point_b: Second point (x, y).

    Returns:
        The Euclidean distance between point_a and point_b.
        Returns 0.0 when the points are identical.
    """
    ax, ay = point_a
    bx, by = point_b

    return math.hypot(bx - ax, by - ay)


def calculate_slope(point_a: Point2D, point_b: Point2D) -> float:
    """Calculate the slope of the line passing through two 2D points.

    Uses the formula:

        slope = (y2 - y1) / (x2 - x1)

    For a horizontal line (y1 == y2), returns 0.0.

    Args:
        point_a: First point (x, y).
        point_b: Second point (x, y).

    Returns:
        The slope of the line through point_a and point_b.

    Raises:
        ValueError: If the line is vertical (x1 == x2), because the
            slope is mathematically undefined for a vertical line.
    """
    x1, y1 = point_a
    x2, y2 = point_b

    delta_x = x2 - x1

    if delta_x == 0:
        raise ValueError(
            "Cannot calculate slope: the line is vertical "
            f"(x1 == x2 == {x1}), so the slope is undefined."
        )

    return (y2 - y1) / delta_x


def calculate_midpoint(point_a: Point2D, point_b: Point2D) -> Tuple[float, float]:
    """Calculate the midpoint between two 2D points.

    Uses the formula:

        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)

    Works with both integer and floating-point coordinates.
    The result is always returned as a tuple of floats.

    Args:
        point_a: First point (x, y).
        point_b: Second point (x, y).

    Returns:
        A tuple (midpoint_x, midpoint_y) of floats representing the
        midpoint between point_a and point_b.
    """
    x1, y1 = point_a
    x2, y2 = point_b

    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2

    return (midpoint_x, midpoint_y)

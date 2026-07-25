"""Tests for src/utils/geometry.py — Phase 0.5 geometry utilities.

All tests use synthetic, deterministic coordinates with hand-calculated
expected results. No webcam, camera, MediaPipe, or random data is used.
"""

import pytest

from src.utils.geometry import (
    calculate_angle,
    calculate_distance,
    calculate_midpoint,
    calculate_slope,
)


# ================================================================
# ANGLE TESTS
# ================================================================


class TestCalculateAngle:
    """Tests for calculate_angle()."""

    def test_right_angle(self):
        """A = (0,1), B = (0,0), C = (1,0) → 90 degrees at B."""
        angle = calculate_angle((0, 1), (0, 0), (1, 0))
        assert angle == pytest.approx(90.0)

    def test_straight_angle(self):
        """A = (-1,0), B = (0,0), C = (1,0) → 180 degrees (collinear, opposite rays)."""
        angle = calculate_angle((-1, 0), (0, 0), (1, 0))
        assert angle == pytest.approx(180.0)

    def test_zero_degree_collinear(self):
        """A = (1,0), B = (0,0), C = (1,0) → 0 degrees (same direction)."""
        angle = calculate_angle((1, 0), (0, 0), (1, 0))
        assert angle == pytest.approx(0.0)

    def test_45_degree_angle(self):
        """A = (1,1), B = (0,0), C = (1,0) → 45 degrees."""
        angle = calculate_angle((1, 1), (0, 0), (1, 0))
        assert angle == pytest.approx(45.0)

    def test_symmetry(self):
        """Swapping point_a and point_c must produce the same angle.

        This test guards against a direction-sensitive implementation.
        Later callers may not always provide the two outer points in a
        fixed order, so the function must be symmetric with respect to
        point_a and point_c.
        """
        a = (1, 1)
        b = (0, 0)
        c = (1, 0)

        angle_abc = calculate_angle(a, b, c)
        angle_cba = calculate_angle(c, b, a)

        assert angle_abc == pytest.approx(angle_cba)

    def test_vertex_convention(self):
        """The second argument (point_b) is the vertex.

        A = (0,1), B = (0,0), C = (1,0) → the angle at B is 90 degrees.
        This test makes the vertex convention explicit and prevents
        regressions where the implementation accidentally calculates
        the angle at point_a or point_c.
        """
        # The second argument (point_b) is the vertex.
        angle = calculate_angle((0, 1), (0, 0), (1, 0))
        assert angle == pytest.approx(90.0)


# ================================================================
# ANGLE EDGE-CASE TESTS
# ================================================================


class TestCalculateAngleEdgeCases:
    """Edge-case tests for calculate_angle()."""

    def test_point_a_equals_point_b_raises_value_error(self):
        """When point_a == point_b, vector BA has zero length → ValueError."""
        with pytest.raises(ValueError):
            calculate_angle((0, 0), (0, 0), (1, 0))

    def test_point_c_equals_point_b_raises_value_error(self):
        """When point_c == point_b, vector BC has zero length → ValueError."""
        with pytest.raises(ValueError):
            calculate_angle((1, 0), (0, 0), (0, 0))


# ================================================================
# DISTANCE TESTS
# ================================================================


class TestCalculateDistance:
    """Tests for calculate_distance()."""

    def test_3_4_5_triangle(self):
        """Distance from (0,0) to (3,4) is 5.0 (classic 3-4-5 triangle)."""
        distance = calculate_distance((0, 0), (3, 4))
        assert distance == pytest.approx(5.0)

    def test_same_point(self):
        """Distance from a point to itself is 0.0."""
        distance = calculate_distance((2, 3), (2, 3))
        assert distance == pytest.approx(0.0)

    def test_floating_point_coordinates(self):
        """Distance between (0.0, 0.0) and (1.0, 1.0) is sqrt(2) ≈ 1.41421356."""
        import math

        distance = calculate_distance((0.0, 0.0), (1.0, 1.0))
        assert distance == pytest.approx(math.sqrt(2))


# ================================================================
# SLOPE TESTS
# ================================================================


class TestCalculateSlope:
    """Tests for calculate_slope()."""

    def test_horizontal_line(self):
        """A = (0,5), B = (10,5) → slope = 0.0."""
        slope = calculate_slope((0, 5), (10, 5))
        assert slope == pytest.approx(0.0)

    def test_slope_one(self):
        """A = (0,0), B = (5,5) → slope = 1.0."""
        slope = calculate_slope((0, 0), (5, 5))
        assert slope == pytest.approx(1.0)

    def test_slope_half(self):
        """A = (0,0), B = (4,2) → slope = 0.5."""
        slope = calculate_slope((0, 0), (4, 2))
        assert slope == pytest.approx(0.5)

    def test_negative_slope(self):
        """A = (0,0), B = (4,-2) → slope = -0.5."""
        slope = calculate_slope((0, 0), (4, -2))
        assert slope == pytest.approx(-0.5)

    def test_vertical_line_raises_value_error(self):
        """Vertical line (x1 == x2) → ValueError, not infinity or NaN."""
        with pytest.raises(ValueError):
            calculate_slope((2, 0), (2, 5))


# ================================================================
# MIDPOINT TESTS
# ================================================================


class TestCalculateMidpoint:
    """Tests for calculate_midpoint()."""

    def test_simple_midpoint(self):
        """Midpoint of (0,0) and (4,4) is (2.0, 2.0)."""
        mid = calculate_midpoint((0, 0), (4, 4))
        assert mid[0] == pytest.approx(2.0)
        assert mid[1] == pytest.approx(2.0)

    def test_negative_coordinates(self):
        """Midpoint of (-2,-4) and (2,4) is (0.0, 0.0)."""
        mid = calculate_midpoint((-2, -4), (2, 4))
        assert mid[0] == pytest.approx(0.0)
        assert mid[1] == pytest.approx(0.0)

    def test_non_integer_result(self):
        """Midpoint of (1,2) and (4,7) is (2.5, 4.5)."""
        mid = calculate_midpoint((1, 2), (4, 7))
        assert mid[0] == pytest.approx(2.5)
        assert mid[1] == pytest.approx(4.5)

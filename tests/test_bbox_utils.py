"""
Tests for utils/bbox_utils.py

All expected values are hand-computed so the tests act as a specification,
not just a round-trip check.
"""

import math
import sys
import os

import numpy as np
import pytest

# Allow running from project root or from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.bbox_utils import (
    get_center_of_bbox,
    get_bbox_width,
    get_foot_position,
    measure_distance,
    measure_xy_distance,
)


# ---------------------------------------------------------------------------
# get_center_of_bbox
# ---------------------------------------------------------------------------

class TestGetCenterOfBbox:

    def test_simple_even(self):
        # x: (100+200)/2 = 150, y: (50+150)/2 = 100
        assert get_center_of_bbox([100, 50, 200, 150]) == (150, 100)

    def test_simple_odd_truncates(self):
        # x: (101+200)/2 = 150.5 → 150 (int truncation)
        assert get_center_of_bbox([101, 50, 200, 150]) == (150, 100)

    def test_float_input(self):
        # x: (10.0+20.0)/2 = 15, y: (5.0+25.0)/2 = 15
        cx, cy = get_center_of_bbox([10.0, 5.0, 20.0, 25.0])
        assert cx == 15
        assert cy == 15

    def test_numpy_input(self):
        bbox = np.array([0, 0, 100, 80], dtype=float)
        cx, cy = get_center_of_bbox(bbox)
        assert cx == 50
        assert cy == 40

    def test_tuple_input(self):
        assert get_center_of_bbox((0, 0, 60, 40)) == (30, 20)

    def test_returns_int_types(self):
        cx, cy = get_center_of_bbox([0, 0, 10, 10])
        assert isinstance(cx, int)
        assert isinstance(cy, int)

    def test_zero_width_height(self):
        # Degenerate bbox — single point
        assert get_center_of_bbox([5, 5, 5, 5]) == (5, 5)

    def test_large_coords(self):
        # 1920x1080 frame corner to corner
        cx, cy = get_center_of_bbox([0, 0, 1920, 1080])
        assert cx == 960
        assert cy == 540


# ---------------------------------------------------------------------------
# get_bbox_width
# ---------------------------------------------------------------------------

class TestGetBboxWidth:

    def test_basic(self):
        assert get_bbox_width([100, 50, 160, 200]) == 60

    def test_float_input(self):
        # 150.7 - 100.3 = 50.4 → int(50.4) = 50
        assert get_bbox_width([100.3, 0, 150.7, 0]) == 50

    def test_numpy_input(self):
        assert get_bbox_width(np.array([200.0, 0.0, 250.0, 0.0])) == 50

    def test_returns_int(self):
        assert isinstance(get_bbox_width([0, 0, 40, 100]), int)

    def test_zero_width(self):
        assert get_bbox_width([50, 10, 50, 90]) == 0


# ---------------------------------------------------------------------------
# get_foot_position
# ---------------------------------------------------------------------------

class TestGetFootPosition:

    def test_basic(self):
        # x_centre = int((100+200)/2) = 150, y_bottom = int(300) = 300
        assert get_foot_position([100, 100, 200, 300]) == (150, 300)

    def test_float_input(self):
        # x_centre = int((100.5+199.5)/2) = int(150.0) = 150
        # y_bottom = int(299.9) = 299
        assert get_foot_position([100.5, 50.0, 199.5, 299.9]) == (150, 299)

    def test_numpy_input(self):
        bbox = np.array([0, 0, 80, 120], dtype=float)
        assert get_foot_position(bbox) == (40, 120)

    def test_returns_int_types(self):
        fx, fy = get_foot_position([0, 0, 50, 100])
        assert isinstance(fx, int)
        assert isinstance(fy, int)

    def test_x_centre_and_y_bottom_are_independent(self):
        fx, fy = get_foot_position([10, 20, 30, 80])
        # x_centre = int((10+30)/2) = 20
        assert fx == 20
        # y_bottom = int(80) = 80
        assert fy == 80


# ---------------------------------------------------------------------------
# measure_distance
# ---------------------------------------------------------------------------

class TestMeasureDistance:

    def test_horizontal(self):
        # Points (0,0) and (3,0) → distance = 3.0
        assert measure_distance((0, 0), (3, 0)) == pytest.approx(3.0)

    def test_vertical(self):
        assert measure_distance((0, 0), (0, 4)) == pytest.approx(4.0)

    def test_diagonal_3_4_5(self):
        # 3-4-5 right triangle
        assert measure_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_symmetry(self):
        p1, p2 = (10, 20), (30, 50)
        assert measure_distance(p1, p2) == pytest.approx(measure_distance(p2, p1))

    def test_same_point(self):
        assert measure_distance((7, 7), (7, 7)) == pytest.approx(0.0)

    def test_float_points(self):
        # sqrt((1.5-0)^2 + (0-2)^2) = sqrt(2.25+4) = sqrt(6.25) = 2.5
        assert measure_distance((0, 2), (1.5, 0)) == pytest.approx(2.5)

    def test_numpy_points(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([5.0, 12.0])
        assert measure_distance(p1, p2) == pytest.approx(13.0)

    def test_large_coords(self):
        # Across a 1920x1080 frame
        d = measure_distance((0, 0), (1920, 1080))
        expected = math.sqrt(1920**2 + 1080**2)
        assert d == pytest.approx(expected)


# ---------------------------------------------------------------------------
# measure_xy_distance
# ---------------------------------------------------------------------------

class TestMeasureXyDistance:

    def test_basic_positive(self):
        # p1=(5,8), p2=(2,3) → dx=3, dy=5
        dx, dy = measure_xy_distance((5, 8), (2, 3))
        assert dx == pytest.approx(3.0)
        assert dy == pytest.approx(5.0)

    def test_basic_negative(self):
        # p1=(2,3), p2=(5,8) → dx=-3, dy=-5
        dx, dy = measure_xy_distance((2, 3), (5, 8))
        assert dx == pytest.approx(-3.0)
        assert dy == pytest.approx(-5.0)

    def test_zero_distance(self):
        dx, dy = measure_xy_distance((10, 10), (10, 10))
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_float_coords(self):
        dx, dy = measure_xy_distance((1.5, 2.5), (0.5, 1.0))
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(1.5)

    def test_numpy_points(self):
        p1 = np.array([10.0, 20.0])
        p2 = np.array([3.0, 7.0])
        dx, dy = measure_xy_distance(p1, p2)
        assert dx == pytest.approx(7.0)
        assert dy == pytest.approx(13.0)

    def test_tuple_of_two_values_returned(self):
        result = measure_xy_distance((1, 2), (3, 4))
        assert len(result) == 2

    def test_camera_estimator_usage(self):
        """
        Simulate how Person A's camera estimator uses this function:
        new_point - old_point should give positive dx/dy when the new feature
        has moved right and down.
        """
        new_point = (110.5, 55.3)
        old_point = (100.0, 50.0)
        dx, dy = measure_xy_distance(new_point, old_point)
        assert dx == pytest.approx(10.5)
        assert dy == pytest.approx(5.3)


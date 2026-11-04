"""
Bounding box utility functions.

Pure functions — accept list, tuple, or numpy array input.
All coordinate values are interpreted as [x1, y1, x2, y2].
"""

import math


def get_center_of_bbox(bbox):
    """
    Get the centre coordinates of a bounding box.

    Args:
        bbox (list | tuple | np.ndarray): [x1, y1, x2, y2].

    Returns:
        tuple[int, int]: (x_center, y_center) as integers.
    """
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    """
    Get the width of a bounding box.

    Args:
        bbox (list | tuple | np.ndarray): [x1, y1, x2, y2].

    Returns:
        int: Width (x2 - x1), truncated to integer.
    """
    return int(bbox[2] - bbox[0])


def get_foot_position(bbox):
    """
    Get the foot (ground-anchor) position of a bounding box.

    Returns the bottom-centre point, i.e. where a standing player's
    feet touch the ground.

    Args:
        bbox (list | tuple | np.ndarray): [x1, y1, x2, y2].

    Returns:
        tuple[int, int]: (x_center, y_bottom) as integers.
    """
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1, p2):
    """
    Compute the Euclidean distance between two 2-D points.

    Args:
        p1 (tuple | list | np.ndarray): First point (x, y).
        p2 (tuple | list | np.ndarray): Second point (x, y).

    Returns:
        float: sqrt((x1-x2)^2 + (y1-y2)^2).
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def measure_xy_distance(p1, p2):
    """
    Compute the signed component-wise distance from p2 to p1 (p1 - p2).

    Used by Person A's camera estimator to decompose optical-flow shift
    into horizontal and vertical components.

    Args:
        p1 (tuple | list | np.ndarray): First point (x, y).
        p2 (tuple | list | np.ndarray): Second point (x, y).

    Returns:
        tuple[float, float]: (dx, dy) where dx = p1.x - p2.x, dy = p1.y - p2.y.
    """
    return (p1[0] - p2[0]), (p1[1] - p2[1])

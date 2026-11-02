import math


def get_center_of_bbox(bbox):
    """
    Get the center coordinates of a bounding box.

    Args:
        bbox (list): Bounding box as [x1, y1, x2, y2].

    Returns:
        tuple: (x_center, y_center) as integers.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    """
    Get the width of a bounding box.

    Args:
        bbox (list): Bounding box as [x1, y1, x2, y2].

    Returns:
        float: Width (x2 - x1).
    """
    return bbox[2] - bbox[0]


def get_foot_position(bbox):
    """
    Get the foot (ground anchor) position of a bounding box.

    Returns the bottom-center point, which represents where a
    player's feet touch the ground.

    Args:
        bbox (list): Bounding box as [x1, y1, x2, y2].

    Returns:
        tuple: (x_center, y2) as integers.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1, p2):
    """
    Compute Euclidean distance between two 2D points.

    Args:
        p1 (tuple): First point as (x, y).
        p2 (tuple): Second point as (x, y).

    Returns:
        float: Euclidean distance sqrt((x2-x1)^2 + (y2-y1)^2).
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


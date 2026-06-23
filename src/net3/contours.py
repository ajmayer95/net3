"""
Contour extraction and processing for vectorization.

Functions for finding, flattening, and filtering contours from binary images.
"""

import numpy as np
import cv2
import math
from typing import List, Tuple
from shapely.geometry import Point, Polygon


def get_contours(image: np.ndarray) -> np.ndarray:
    """
    Extract contours from binary image using OpenCV.

    Uses Teh-Chin dominant point detection for contour approximation.

    Parameters
    ----------
    image : np.ndarray
        Binary image (uint8)

    Returns
    -------
    np.ndarray
        Array of contours (each contour is array of points)
    """
    image = image.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )
    return np.array(contours, dtype=object)


def flatten_contours(raw_contours: np.ndarray) -> List[List[List[float]]]:
    """
    Flatten nested contour arrays to simple point lists.

    Parameters
    ----------
    raw_contours : np.ndarray
        Raw contours from cv2.findContours

    Returns
    -------
    List[List[List[float]]]
        List of contours, each contour is list of [x, y] points
    """
    converted = []
    for contour in raw_contours:
        new_contour = []
        for point in contour:
            x, y = float(point[0][0]), float(point[0][1])
            new_contour.append([x, y])
        converted.append(new_contour)
    return converted


def threshold_contours(contours: List, min_length: int = 3) -> List:
    """
    Filter contours by minimum length.

    Parameters
    ----------
    contours : List
        List of contours
    min_length : int
        Minimum number of points in contour

    Returns
    -------
    List
        Filtered contours
    """
    return [c for c in contours if len(c) > min_length]


def find_longest_contour(contours: List) -> int:
    """
    Find index of longest contour.

    Parameters
    ----------
    contours : List
        List of contours

    Returns
    -------
    int
        Index of longest contour
    """
    longest_idx = 0
    longest_len = 0
    for i, c in enumerate(contours):
        if len(c) > longest_len:
            longest_len = len(c)
            longest_idx = i
    return longest_idx


def round_trip_connect(start: int, end: int) -> List[Tuple[int, int]]:
    """
    Create facets connecting points in a closed loop.

    Parameters
    ----------
    start : int
        Index of first point
    end : int
        Index of last point

    Returns
    -------
    List[Tuple[int, int]]
        List of (i, j) index pairs forming closed loop
    """
    return [(i, i + 1) for i in range(start, end)] + [(end, start)]


def get_interior_point(contour: List[List[float]]) -> Tuple[float, float]:
    """
    Find a point inside a polygon contour.

    First tries centroid, then uses geometric methods if centroid
    is outside the polygon.

    Parameters
    ----------
    contour : List[List[float]]
        Contour as list of [x, y] points

    Returns
    -------
    Tuple[float, float]
        (x, y) coordinates of interior point
    """
    poly = Polygon(contour)

    # Try centroid first
    x = sum(p[0] for p in contour) / len(contour)
    y = sum(p[1] for p in contour) / len(contour)
    center = Point(x, y)

    if center.within(poly):
        return (x, y)

    # If centroid fails, use rotation method
    def rotate(angle_deg, vec):
        angle = math.radians(angle_deg)
        rot = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)]
        ])
        return np.dot(rot, vec)

    # Get three consecutive points
    p1 = np.array(contour[0])
    cp = np.array(contour[1])
    p2 = np.array(contour[2])

    seg1 = cp - p1
    seg2 = p2 - cp

    phi_plus = math.atan2(seg2[1], seg2[0])
    phi_minus = math.atan2(seg1[1], seg1[0])
    phi = math.degrees((math.pi - phi_plus + phi_minus) % (2 * math.pi))

    N = 0.5  # Distance from corner point

    # Try rotating to find interior point
    for angle in [0.5 * phi, -0.5 * phi] + list(range(360)):
        rot_seg = rotate(angle if isinstance(angle, (int, float)) else angle, seg2)
        int_point = [cp[0] + N * rot_seg[0], cp[1] + N * rot_seg[1]]
        test_point = Point(int_point)
        if test_point.within(poly):
            return (int_point[0], int_point[1])

    # Fallback to centroid
    return (x, y)


def add_noise_to_contours(contours: List, noise_scale: float = 0.1) -> List:
    """
    Add small noise to contour points for triangulation stability.

    Parameters
    ----------
    contours : List
        List of contours
    noise_scale : float
        Scale of random noise to add

    Returns
    -------
    List
        Contours with noise added
    """
    for c in contours:
        for p in c:
            p[0] = p[0] + noise_scale * np.random.rand()
            p[1] = p[1] + noise_scale * np.random.rand()
    return contours


def extract_and_process_contours(
    image: np.ndarray,
    min_contour_length: int = 3
) -> Tuple[List, int]:
    """
    Full contour extraction pipeline.

    Parameters
    ----------
    image : np.ndarray
        Binary image
    min_contour_length : int
        Minimum contour length to keep

    Returns
    -------
    Tuple[List, int]
        (processed_contours, longest_contour_index)
    """
    raw = get_contours(image)
    flat = flatten_contours(raw)
    filtered = threshold_contours(flat, min_contour_length)
    longest_idx = find_longest_contour(filtered)

    return filtered, longest_idx

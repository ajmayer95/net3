"""
Image preprocessing for vectorization.

Functions for loading, binarizing, cleaning, and computing distance maps.
"""

import numpy as np
import cv2
from PIL import Image
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects
from typing import Optional, Tuple


def load_binary_image(path: str, invert: bool = False) -> np.ndarray:
    """
    Load image and convert to binary.

    Parameters
    ----------
    path : str
        Path to image file
    invert : bool
        If True, invert the mask (swap foreground/background).
        Use this when vessels are black on white background.

    Returns
    -------
    np.ndarray
        Binary image (0 or 1), dtype uint8
    """
    image = Image.open(path)
    image = image.convert('L')
    image = np.asarray(image, dtype=np.uint8)

    # Convert to binary if not already
    if image.max() > 1:
        image = np.where(image < 127, 0, 1).astype(np.uint8)

    # Invert if requested (for masks with black vessels on white background)
    if invert:
        image = 1 - image

    return image


def clean_binary_image(
    image: np.ndarray,
    min_size: int = 3000,
    smoothing: Optional[int] = None,
    bridge_gaps: Optional[int] = None
) -> np.ndarray:
    """
    Clean binary image by removing small foreground objects and optional smoothing.

    Parameters
    ----------
    image : np.ndarray
        Binary image
    min_size : int
        Minimum feature size in pixels (smaller features removed)
    smoothing : int, optional
        Kernel size for morphological smoothing (opening + closing)
    bridge_gaps : int, optional
        Morphological closing radius to bridge small gaps between regions.
        Useful for connecting disconnected vessel segments. Typical values: 5-15.

    Returns
    -------
    np.ndarray
        Cleaned binary image
    """
    image = image.astype(bool)

    # Bridge gaps first (before any filtering that might break connections)
    if bridge_gaps:
        image = binary_closing(image, disk(bridge_gaps))

    # Remove small foreground objects (noise)
    image = remove_small_objects(image, min_size=min_size, connectivity=1)

    # Optional smoothing
    if smoothing:
        image = binary_opening(image, disk(smoothing))
        image = binary_closing(image, disk(smoothing))
        image = remove_small_objects(image, min_size=min_size, connectivity=1)

    return image.astype(np.uint8)


def compute_distance_map(image: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance transform of binary image.

    The distance map gives the distance from each foreground pixel
    to the nearest background pixel - this approximates vessel radius.

    Parameters
    ----------
    image : np.ndarray
        Binary image (foreground = 1, background = 0)

    Returns
    -------
    np.ndarray
        Distance map (float32)
    """
    image = image.astype(np.uint8)
    # distanceType=2 is CV_DIST_L2 (Euclidean)
    # maskSize=0 is CV_DIST_MASK_PRECISE
    distance_map = cv2.distanceTransform(image, distanceType=2, maskSize=0)
    return distance_map.astype(np.int32)


def preprocess_for_vectorization(
    image_path: str,
    min_feature_size: int = 3000,
    smoothing: Optional[int] = None,
    bridge_gaps: Optional[int] = None,
    invert: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline: load, clean, compute distance map.

    Parameters
    ----------
    image_path : str
        Path to binary mask image
    min_feature_size : int
        Remove features smaller than this (pixels)
    smoothing : int, optional
        Morphological smoothing kernel size
    bridge_gaps : int, optional
        Closing radius to bridge gaps between disconnected regions (e.g., 10)
    invert : bool
        If True, invert the mask (swap foreground/background).
        Use this when vessels are black on white background.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (cleaned_binary_image, distance_map)
    """
    # Load
    image = load_binary_image(image_path, invert=invert)

    # Clean
    image = clean_binary_image(
        image, min_size=min_feature_size, smoothing=smoothing, bridge_gaps=bridge_gaps
    )

    # Distance map
    distance_map = compute_distance_map(image)

    return image, distance_map

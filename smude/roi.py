__author__ = "Simon Waloschek"

from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.exposure import equalize_adapthist
from skimage.feature import canny
from skimage.io import imread, imsave
from skimage.morphology import disk, square
from skimage.segmentation import felzenszwalb, slic, watershed


def get_border(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get border widths of binary image for all 4 edges.

    Parameters
    ----------
    image : np.ndarray
        Binary image.

    Returns
    -------
    x_start : int
        X-Coordinate of first non-white pixel.
    x_end : int
        X-Coordinate of last non-white pixel.
    y_start : int
        Y-Coordinate of first non-white pixel.
    y_end : int
        Y-Coordinate of last non-white pixel.
    """
    cols, rows = np.where(image == 0)
    x_start = np.min(cols)
    x_end = np.max(cols) + 1

    y_start = np.min(rows)
    y_end = np.max(rows) + 1

    return x_start, x_end, y_start, y_end


def extract_roi_mask(image: np.ndarray, min_hull_ratio: float = 0.3) -> Tuple[np.ndarray, float]:
    """
    Extract region of interest (ROI) for the given image.

    Parameters
    ----------
    image : np.ndarray
        Input document image covering the entire ROI.
    min_hull_ratio : float, optional
        Minimum ratio All/ROI for counting as "success", by default 0.3.

    Returns
    -------
    mask_fullsize : np.ndarray
        Binary image respresenting the ROI. White pixels (1) = ROI.
    mask_ratio : float
        Pixel ratio (widht * height) / ROI.

    Raises
    ------
    Exception
        If the minimum desired ratio could not be achieved, an error is raised.
    """
    # Scale image to fixed size
    size = 1000
    width, height, _ = image.shape
    image_resized = cv2.resize(image, (size, size))

    # Search for edges with canny filter on each channel of HSV image
    image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)
    image_canny = np.empty_like(image_hsv)
    for channel in range(3):
        # Apply adaptive histogram equalization
        channel_eq = equalize_adapthist(image_hsv[:,:,channel], kernel_size=200)
        # Apply canny filter
        channel_canny = canny(channel_eq, sigma=3)
        image_canny[:,:,channel] = channel_canny
    
    # Perform segmentation using the Felzenszwalb-Huttenlocher algorithm
    # and sort segments by size in descending order
    image_segmented = felzenszwalb(image_canny, scale=1000, sigma=0.3, min_size=50)

    segment_sizes = np.bincount(image_segmented.flatten())
    segments = np.argsort(-segment_sizes)

    # Iterate over segments, starting from largest
    for s in segments:
        # Get segment and fill all holes
        segment = image_segmented == s
        hull = ndimage.binary_fill_holes(segment)

        # Exit if hull_ratio is sufficient
        hull_ratio = np.sum(hull) / (size**2)
        if hull_ratio >= min_hull_ratio:
          break
    
    # Raise error if hull_ratio criterion could not be met
    if hull_ratio < min_hull_ratio:
        raise Exception('ROI could not be computed')

    ### Postprocessing

    # Removes areas that are only connected by few pixels to the hull
    hull_opened = cv2.morphologyEx(hull.astype(np.uint8), cv2.MORPH_OPEN, kernel=disk(20))

    # Take center blob
    blobs_segmented = measure.label(hull_opened)
    center_blob_label = blobs_segmented[size // 2, size // 2]
    mask = blobs_segmented == center_blob_label

    # Resize mask back to original image size
    mask_fullsize = cv2.resize(mask.astype(np.uint8), (height, width))
    mask_ratio = np.sum(mask) / (size**2)

    return mask_fullsize, mask_ratio

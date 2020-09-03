__author__ = "Simon Waloschek"

import numpy as np
from skimage.color import rgb2hsv
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_sauvola, unsharp_mask
from skimage.morphology import remove_small_holes
from skimage.segmentation import flood_fill


def binarize(image: np.ndarray, holes_threshold: float = 20) -> np.ndarray:
    """
    Binarize image using Sauvola algorithm.

    Parameters
    ----------
    image : np.ndarray
        RGB image to binarize.
    holes_threshold : float, optional
        Pixel areas covering less than the given number of pixels are removed in the process, by default 20.

    Returns
    -------
    binarized : np.ndarray
        Binarized and filtered image.
    """

    # Extract brightness channel from HSV-converted image
    image_gray = rgb2hsv(image)[:,:,2]

    # Enhance contrast
    image_gray = equalize_adapthist(image_gray, kernel_size=100)

    # Threshold using Sauvola algorithm
    thresh_sauvola = threshold_sauvola(image_gray, window_size=51, k=0.25)
    binary_sauvola = image_gray > thresh_sauvola

    # Remove small objects
    binary_cleaned = 1.0 * remove_small_holes(binary_sauvola, area_threshold=holes_threshold)

    # Remove thick black border (introduced during thresholding)
    binary_cleaned = flood_fill(binary_cleaned, (0, 0), 0)
    binary_cleaned = flood_fill(binary_cleaned, (0, 0), 1)
    
    return binary_cleaned.astype(np.bool)

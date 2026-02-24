__author__ = "Simon Waloschek"

import numpy as np
import cv2


def binarize(image: np.ndarray, roi_mask: np.ndarray = None) -> np.ndarray:
    """
    Binarize image using Sauvola algorithm with adaptive fallback.

    Primary path: masks the image first (zeroing background), then binarizes
    with the original CLAHE + Sauvola parameters the U-Net was trained on.

    Fallback: if the primary result has too little black content (<3%),
    binarizes the full unmasked image with stronger contrast enhancement
    and a lower Sauvola k, then applies the ROI mask afterwards.

    Parameters
    ----------
    image : np.ndarray
        RGB image to binarize.
    roi_mask : np.ndarray, optional
        Binary mask (1 = ROI, 0 = background). If provided, non-ROI pixels
        are zeroed before binarization (primary) or set to white after (fallback).

    Returns
    -------
    binarized : np.ndarray
        Binarized boolean image.
    """

    # --- Primary path: mask first, then binarize (original training pipeline) ---
    if roi_mask is not None:
        mask_3c = np.broadcast_to(roi_mask[..., None], roi_mask.shape + (3,))
        masked = image * mask_3c
    else:
        masked = image

    gray_masked = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(256, 256))
    eq = clahe.apply(gray_masked)

    binary = cv2.ximgproc.niBlackThreshold(
        eq, 255, k=0.25, blockSize=51,
        type=cv2.THRESH_BINARY,
        binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA,
    )
    binary = binary.astype(np.uint8)

    cv2.floodFill(binary, None, (0, 0), 0)
    cv2.floodFill(binary, None, (0, 0), 1)

    # Check if the result has enough content
    black_ratio = 1.0 - np.mean(binary > 0)

    if black_ratio < 0.03:
        # Too little content — binarize the full unmasked image with stronger params
        gray_full = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        clahe_strong = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_full = clahe_strong.apply(gray_full)

        binary = cv2.ximgproc.niBlackThreshold(
            eq_full, 255, k=0.08, blockSize=51,
            type=cv2.THRESH_BINARY,
            binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA,
        )
        binary = binary.astype(np.uint8)

        # Apply ROI mask after binarization
        if roi_mask is not None:
            binary[roi_mask == 0] = 255

        cv2.floodFill(binary, None, (0, 0), 0)
        cv2.floodFill(binary, None, (0, 0), 1)

    return binary.astype(bool)

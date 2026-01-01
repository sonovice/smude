__author__ = "Simon Waloschek"
"""
Perform metric rectification on given sheet music images.
The algorithm is based on the paper
    "Metric Rectification of Curved Document Images"
    by Gaofeng Meng et al. (2012)
    https://doi.org/10.1109/TPAMI.2011.151
"""

import logging
import math
from functools import lru_cache
from typing import Callable, List, Optional, Tuple


import cv2 as cv
import numpy as np
from scipy import interpolate
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage import label
from scipy.optimize import fsolve, minimize
from scipy.spatial.distance import euclidean
from skimage.transform import hough_line, hough_line_peaks

from .utils import *


def _fast_trapz_integrate(func: Callable, a: float, b: float, n: int = 100) -> float:
    """
    Fast trapezoidal integration as a replacement for scipy.integrate.quad.
    
    Parameters
    ----------
    func : Callable
        Function to integrate.
    a : float
        Lower bound.
    b : float
        Upper bound.
    n : int
        Number of sample points for integration.
    
    Returns
    -------
    float
        Approximate integral value.
    """
    t = np.linspace(a, b, n)
    # Pre-allocate and vectorize when possible
    y = np.empty(n)
    for i, ti in enumerate(t):
        y[i] = func(ti)
    # Use trapezoid for numpy 2.0+, fall back to trapz for older versions
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, t)
    else:
        return np.trapz(y, t)


def get_outer_barlines(barline_img: np.ndarray) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """
    Return the outer barlines as lines each defined by slope and one point.

    Parameters
    ----------
    barline_img : np.ndarray
        Grayscale barline image.

    Returns
    -------
    left : function
        Function for the leftmost barline.
    right: function
        Function for the rightmost barline.
    """

    # Compute bar lines using straight line Hough transform
    # Restrict computation to -30° to 30° line angle with 0.05 degree precision
    tested_angles = np.linspace(-np.pi / 6, np.pi / 6, int(60 / 0.025))
    h, theta, d = hough_line(barline_img, theta=tested_angles)
    # Get peaks from Hough transform
    peaks = np.stack(hough_line_peaks(h, theta, d, threshold=np.max(h) * 0.45))

    origin = np.array((0, barline_img.shape[1]))

    # Leftmost vertical line
    l_idx = np.argmin(peaks[2])
    _, angle, dist = peaks[:, l_idx]
    y1, y2 = (dist - origin * np.cos(angle)) / np.sin(angle)
    left = line(x1=origin[0], x2=origin[1], y1=y1, y2=y2)


    # Rightmost vertical line
    r_idx = np.argmax(peaks[2])
    _, angle, dist = peaks[:, r_idx]
    y1, y2 = (dist - origin * np.cos(angle)) / np.sin(angle)
    right = line(x1=origin[0], x2=origin[1], y1=y1, y2=y2)


    return left, right


def get_stafflines(upper_img: np.ndarray, lower_img: np.ndarray, step_size: int) -> List[UnivariateSpline]:
    """
    Return the top and bottom stafflines from images and return the as
    cubic splines.

    Parameters
    ----------
    upper_img : ndarray
        Grayscale image containing all upper staff lines.
    lower_img : ndarray
        Grayscale image containing all lower staff lines.
    step_size : int, optional
        Picks every step_size-ths (x, y) pair from pixels for spline
        interpolation.

    Returns
    -------
    splines : List[UnivariateSpline]
        A list of UnivariateSpline instances.
    """

    splines = []
    morph_kernel = np.ones((25, 25), dtype=np.uint8)
    
    for image in [upper_img, lower_img]:
        # Morphological operation to close potential gaps
        image_closed = cv.morphologyEx(image, cv.MORPH_CLOSE, morph_kernel)

        # Segmentize image to get individual staff line instances
        labels, count = label(image_closed)

        # Compute univariate spline for each staff line
        for l in range(1, count + 1):
            line_pixels = (labels == l).astype(np.uint8) * 255

            # Ensure that the staff line has 1px thickness using OpenCV thinning (faster than skimage skeletonize)
            line_pixels = cv.ximgproc.thinning(line_pixels, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)

            y, x = np.nonzero(line_pixels)
            
            if len(x) == 0:
                continue
            
            # Remove duplicates and sort
            x, idx = np.unique(x, return_index=True)
            y = y[idx]

            # Sample
            x_sampled = x[::step_size]
            y_sampled = y[::step_size]

            # Add rightmost point from pixels in case sampling missed it
            last_x = x[-1]
            if x_sampled[-1] != last_x:
                last_y = y[-1]
                x_sampled = np.append(x_sampled, last_x)
                y_sampled = np.append(y_sampled, last_y)

            # Check if enough data is present for splines, otherwise skip
            if len(x_sampled) <= 5:
                continue

            spline = UnivariateSpline(x_sampled, y_sampled, k=3, s=1)
            splines.append(spline)

    # Sort splines from top to bottom
    splines.sort(key=lambda i : i(i.get_knots()[0]))

    return splines


def get_top_bottom_stafflines(stafflines: List[UnivariateSpline], left: Callable[[float], float], right: Callable[[float], float], max_dist: float = 5) -> Tuple[Tuple[UnivariateSpline, float, float], Tuple[UnivariateSpline, float, float]]:
    """
    Return the topmost and bottommost 'complete' staff lines. 'Complete' means
    that the endings of the staff lines should be very close to the given left
    and right boundaries.

    Parameters
    ----------
    stafflines : List[UnivariateSpline]
        List of staff line splines.
    left : function
        Line function for the leftmost bar line of the sheet music.
    right : function
        Line function for the rightmost bar line of the sheet music.
    max_dist : float, optional
        Maximum allowed distance from spline end points to left/right
        boundaries to count as 'complete' staff line.

    Returns
    -------
    top : Tuple[UnivariateSpline, float, float]
        Tuple consisting of (top staff line spline, spline parameter for left
        (start) point, spline parameter for right (end) point).
    bottom : Tuple[UnivariateSpline, float, float]
        Tuple consisting of (bottom staff line spline, spline parameter for
        left (start) point, spline parameter for right (end) point).
    """

    success = False
    top = None

    for spline in stafflines:
        knots = spline.get_knots()

        # Use spline start point as initial guess for left intersection
        left_x, left_y = func_intersection(spline, left, x0=knots[0])
        distance = euclidean((left_x, left_y), (knots[0], spline(knots[0])))

        if distance > max_dist:
            continue

        # Use spline end point as initial guess for right intersection
        right_x, right_y = func_intersection(spline, right, x0=knots[-1])
        distance = euclidean((right_x, right_y), (knots[-1], spline(knots[-1])))

        if distance > max_dist:
            continue

        if top is None:
            top = (spline, left_x, right_x)
            continue

        bottom = (spline, left_x, right_x)
        success = True

    if not success:
        raise ValueError('Staff lines could not be detected!')

    return top, bottom



def cost_function(v_x: float, v_y: float, f: float, top: Tuple[UnivariateSpline, float, float], bottom: Tuple[UnivariateSpline, float, float], num_samples: int = 10) -> float:
    """
    Compute the costs for a given vanishing point and the focal distance. Used for optimization.
    
    Vectorized implementation for better performance.

    Parameters
    ----------
    v_x : float
        X coordinate of vanishing point.
    v_y : float
        Y coordinate of vanishing point.
    f : float
        Focal length
    top : Tuple[UnivariateSpline, float, float]
        Tuple consisting of (top staff line spline, spline parameter for left
        (start) point, spline parameter for right (end) point).
    bottom : Tuple[UnivariateSpline, float, float]
        Tuple consisting of (bottom staff line spline, spline parameter for
        left (start) point, spline parameter for right (end) point).
    num_samples : int, optional
        Number of sampling points along top/bottom splines, by default 10.

    Returns
    -------
    float
        Costs.
    """
    spline_top, top_left, top_right = top
    spline_bottom, bottom_left, bottom_right = bottom

    V = np.array([v_x, v_y, f])
    length_V = np.linalg.norm(V)

    # Pre-compute x_top values and corresponding y_top values
    x_top_arr = np.linspace(top_left, top_right, num_samples)
    y_top_arr = spline_top(x_top_arr)
    
    # Compute slopes for longitudes from vanishing point to top spline points
    # Longitude: y = m * (x - v_x) + v_y where m = (y_top - v_y) / (x_top - v_x)
    dx = x_top_arr - v_x
    dy = y_top_arr - v_y
    lon_slopes = dy / dx
    lon_intercepts = v_y - lon_slopes * v_x
    
    # Find intersections with bottom spline using vectorized Newton-Raphson
    x0 = np.linspace(bottom_left, bottom_right, num_samples)
    x_bottom_arr, y_bottom_arr = _find_spline_line_intersection_batch(
        spline_bottom, lon_slopes, lon_intercepts, x0
    )
    
    # Get derivatives at sample points
    m_top_arr = spline_top(x_top_arr, 1)
    m_bottom_arr = spline_bottom(x_bottom_arr, 1)
    
    # Compute tangent intersections analytically (line-line intersection)
    # Tangent at top: y = m_top * (x - x_top) + y_top
    # Tangent at bottom: y = m_bottom * (x - x_bottom) + y_bottom
    # Intersection: x = (m_top * x_top - m_bottom * x_bottom - y_top + y_bottom) / (m_top - m_bottom)
    
    dm = m_top_arr - m_bottom_arr
    # Avoid division by zero for parallel tangents
    dm = np.where(np.abs(dm) < 1e-10, 1e-10, dm)
    
    P_x = (m_top_arr * x_top_arr - m_bottom_arr * x_bottom_arr - y_top_arr + y_bottom_arr) / dm
    P_y = m_top_arr * (P_x - x_top_arr) + y_top_arr
    
    # Stack P coordinates
    P = np.column_stack([P_x, P_y, np.full(num_samples, f)])
    p = P[:, :2]
    
    # Vectorized E_c computation
    length_p = np.linalg.norm(p, axis=1)
    
    # For m_top
    l_top = np.column_stack([np.ones(num_samples), m_top_arr])
    length_l_top = np.linalg.norm(l_top, axis=1)
    cos_P_l_top = np.abs(np.sum(p * l_top, axis=1) / (length_p * length_l_top))
    
    # For m_bottom
    l_bottom = np.column_stack([np.ones(num_samples), m_bottom_arr])
    length_l_bottom = np.linalg.norm(l_bottom, axis=1)
    cos_P_l_bottom = np.abs(np.sum(p * l_bottom, axis=1) / (length_p * length_l_bottom))
    
    E_c = (np.sum(cos_P_l_top) + np.sum(cos_P_l_bottom)) / 2
    
    # Vectorized E_o computation
    length_P = np.linalg.norm(P, axis=1)
    cos_P_V = np.abs(np.dot(P, V) / (length_P * length_V))
    E_o = np.sum(cos_P_V)

    return E_c + 0.1 * E_o


def estimate_focal_length(v_x: float, v_y: float, top: Tuple[UnivariateSpline, float, float], bottom: Tuple[UnivariateSpline, float, float], f: float = 3760) -> float:
    """
    Estimate the focal length for a given vanishing point and a pair of staff
    lines.

    Parameters
    ----------
    v_x : float
        X coordinate of vanishing point.
    v_y : float
        Y coordinate of vanishing point.
    top : Tuple[UnivariateSpline, float, float]
        Tuple consisting of (top staff line spline, spline parameter for left
        (start) point, spline parameter for right (end) point).
    bottom : Tuple[UnivariateSpline, float, float]
        Tuple consisting of (bottom staff line spline, spline parameter for
        left (start) point, spline parameter for right (end) point).
    f : float, optional
        Initial guess for the focal length estimation, by default 3760.

    Returns
    -------
    f : float
        Estimated focal length.
    """
    # Cyclic coordinate descent until convergence
    counter = 1
    while(True):
        old_f = f
        f = minimize(lambda f, v_x=v_x, v_y=v_y: cost_function(v_x, v_y, f, top, bottom), f).x[0]

        old_v_x = v_x
        v_x = minimize(lambda v_x, v_y=v_y, f=f: cost_function(v_x, v_y, f, top, bottom), v_x).x[0]

        old_v_y = v_y
        v_y = minimize(lambda v_y, v_x=v_x, f=f: cost_function(v_x, v_y, f, top, bottom), v_y).x[0]

        logging.debug(f'Iteration {counter}')
        logging.debug(f'x: {v_x}')
        logging.debug(f'y: {v_y}')
        logging.debug(f'f: {f}')
        logging.debug(f'-- {cost_function(v_x, v_y, f, top, bottom)}')

        counter += 1

        if old_f == f and old_v_x == v_x and old_v_y == v_y:
            break

    return f


class LatitudeCache:
    """Cache for latitude splines to avoid redundant computation."""
    
    def __init__(self, v_x: float, v_y: float, top: Derivable, bottom: Derivable):
        self.v_x = v_x
        self.v_y = v_y
        self.top = top
        self.bottom = bottom
        self._cache = {}
        self._parametric_cache = {}
        
        # Pre-compute sampling points for faster spline creation
        n_samples = 30
        self._t_samples = np.linspace(0, 1, n_samples)
        
        # Pre-compute bottom and top values at sample points - avoid list comprehension
        self._bottom_samples = np.empty((n_samples, 2))
        self._top_samples = np.empty((n_samples, 2))
        for i, t in enumerate(self._t_samples):
            self._bottom_samples[i] = bottom(t)
            self._top_samples[i] = top(t)
        
        # Pre-compute distances from bottom to vanishing point
        self._dist_bottom_v = np.sqrt(
            (self._bottom_samples[:, 0] - v_x) ** 2 + 
            (self._bottom_samples[:, 1] - v_y) ** 2
        )
        
        # Pre-compute distances from bottom to top
        self._dist_bottom_top = np.sqrt(
            (self._bottom_samples[:, 0] - self._top_samples[:, 0]) ** 2 + 
            (self._bottom_samples[:, 1] - self._top_samples[:, 1]) ** 2
        )
        
        # Pre-compute lambda for faster latitude computation
        self._lambda = self._dist_bottom_v / self._dist_bottom_top
    
    def _compute_latitude_points(self, mu: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute latitude points for a given mu value."""
        # Use pre-computed lambda
        alpha = (mu * self._lambda) / (mu + self._lambda - 1)
        
        # Vectorized computation of all points
        points = (1 - alpha[:, None]) * self._bottom_samples + alpha[:, None] * self._top_samples
        return points[:, 0], points[:, 1]
    
    def get_latitude(self, mu: float) -> UnivariateSpline:
        """Get cached or compute latitude spline."""
        # Handle numpy arrays from fsolve/minimize
        if hasattr(mu, '__len__'):
            mu = float(mu[0]) if len(mu) == 1 else float(mu)
        else:
            mu = float(mu)
        
        # Round mu to avoid floating point key issues
        mu_key = round(mu, 6)
        
        if mu_key not in self._cache:
            x_sampled, y_sampled = self._compute_latitude_points(mu)
            # Ensure x is monotonically increasing by sorting
            sort_idx = np.argsort(x_sampled)
            x_sampled = x_sampled[sort_idx]
            y_sampled = y_sampled[sort_idx]
            # Remove duplicates
            x_sampled, unique_idx = np.unique(x_sampled, return_index=True)
            y_sampled = y_sampled[unique_idx]
            self._cache[mu_key] = UnivariateSpline(x_sampled, y_sampled, k=3, s=1)
        
        return self._cache[mu_key]
    
    def get_latitude_parametric(self, mu: float) -> Derivable:
        """Get cached or compute parametric latitude spline."""
        # Handle numpy arrays from fsolve/minimize
        if hasattr(mu, '__len__'):
            mu = float(mu[0]) if len(mu) == 1 else float(mu)
        else:
            mu = float(mu)
            
        mu_key = round(mu, 6)
        
        if mu_key not in self._parametric_cache:
            latitude = self.get_latitude(mu)
            self._parametric_cache[mu_key] = to_parametric_spline(latitude)
        
        return self._parametric_cache[mu_key]


def _get_latitude_parmetric(v_x: float, v_y: float, top: Derivable, bottom: Derivable, mu: float) -> Derivable:
    """
    Return parametric spline representing a latitude defined by parameter 'mu'.

    Parameters
    ----------
    v_x : float
        X coordinate of vanishing point.
    v_y : float
        Y coordinate of vanishing point.
    top : Derivable
        Top parametric staff line spline.
    bottom : Derivable
        Bottom parametric staff line spline.
    mu : float
        Inter-/Extrapolation parameter to get latitudes between bottom (mu=0)
        and top (mu=1) or higher.

    Returns
    -------
    latitude : Derivable
        Parametric spline representing latitude.
    """
    n_samples = 30
    t_samples = np.linspace(0, 1, n_samples)
    
    # Pre-allocate arrays instead of list comprehension
    bottom_pts = np.empty((n_samples, 2))
    top_pts = np.empty((n_samples, 2))
    for i, t in enumerate(t_samples):
        bottom_pts[i] = bottom(t)
        top_pts[i] = top(t)
    
    dist_bottom_v = np.sqrt((bottom_pts[:, 0] - v_x) ** 2 + (bottom_pts[:, 1] - v_y) ** 2)
    dist_bottom_top = np.sqrt((bottom_pts[:, 0] - top_pts[:, 0]) ** 2 + (bottom_pts[:, 1] - top_pts[:, 1]) ** 2)
    
    lambd = dist_bottom_v / dist_bottom_top
    alpha = (mu * lambd) / (mu + lambd - 1)
    
    points = (1 - alpha[:, None]) * bottom_pts + alpha[:, None] * top_pts
    x_sampled, y_sampled = points[:, 0], points[:, 1]
    
    latitude = UnivariateSpline(x_sampled, y_sampled, k=3, s=1)
    latitude_parametric = to_parametric_spline(latitude)

    return latitude_parametric


def _get_latitude(v_x: float, v_y: float, top: Derivable, bottom: Derivable, mu: float) -> UnivariateSpline:
    """
    Return spline representing a latitude defined by parameter 'mu'.

    Parameters
    ----------
    v_x : float
        X coordinate of vanishing point.
    v_y : float
        Y coordinate of vanishing point.
    top : Derivable
        Top parametric staff line spline.
    bottom : Derivable
        Bottom parametric staff line spline.
    mu : float
        Inter-/Extrapolation parameter to get latitudes between bottom (mu=0)
        and top (mu=1) or higher.

    Returns
    -------
    latitude : UnivariateSpline
        Spline representing latitude.
    """
    n_samples = 30
    t_samples = np.linspace(0, 1, n_samples)
    
    # Pre-allocate arrays instead of list comprehension
    bottom_pts = np.empty((n_samples, 2))
    top_pts = np.empty((n_samples, 2))
    for i, t in enumerate(t_samples):
        bottom_pts[i] = bottom(t)
        top_pts[i] = top(t)
    
    dist_bottom_v = np.sqrt((bottom_pts[:, 0] - v_x) ** 2 + (bottom_pts[:, 1] - v_y) ** 2)
    dist_bottom_top = np.sqrt((bottom_pts[:, 0] - top_pts[:, 0]) ** 2 + (bottom_pts[:, 1] - top_pts[:, 1]) ** 2)
    
    lambd = dist_bottom_v / dist_bottom_top
    alpha = (mu * lambd) / (mu + lambd - 1)
    
    points = (1 - alpha[:, None]) * bottom_pts + alpha[:, None] * top_pts
    x_sampled, y_sampled = points[:, 0], points[:, 1]

    latitude = UnivariateSpline(x_sampled, y_sampled, k=3, s=1)

    return latitude


def get_longitudes(v_x: float, v_y: float, f: float, C: Derivable, num: int) -> Tuple[List[Callable[[float], float]], Derivable]:
    """
    Return longitudes as a list of line functions.

    Parameters
    ----------
    v_x : float
        X coordinate of vanishing point.
    v_y : float
        Y coordinate of vanishing point.
    f : float
        Focal length.
    C : Derivable
        Distant parametric spline (typically mu >= 20), extrapolated from staff
        lines.
    num : int
        Number of desired longitudes.

    Returns
    -------
    longitudes : List[Callable[[float], float]]
        List of line functions representing the longitudes.
    D : Derivable
        Parametric spline representing the curvature of the page.
    """

    theta = math.acos(f / (math.sqrt(v_x * v_x + v_y * v_y + f * f)))

    # Compute centroid of curve C - pre-evaluate C values once
    n_int = 50
    t_int = np.linspace(0, 1, n_int)
    C_vals = np.empty((n_int, 2))
    for i, t in enumerate(t_int):
        C_vals[i] = C(t)
    
    # Use trapezoidal integration directly on pre-computed values
    if hasattr(np, 'trapezoid'):
        x_0 = np.trapezoid(C_vals[:, 0], t_int)
        y_0 = np.trapezoid(C_vals[:, 1], t_int)
    else:
        x_0 = np.trapz(C_vals[:, 0], t_int)
        y_0 = np.trapz(C_vals[:, 1], t_int)

    norm = math.sqrt(x_0 * x_0 + y_0 * y_0 + f * f)
    t_1 = x_0 / norm
    t_2 = y_0 / norm
    t_3 = f / norm

    A = np.array([
        [1, -t_1 / t_3 * math.sin(theta)],
        [0, math.cos(theta) - t_2 / t_3 * math.sin(theta)]
    ])

    A_inv = np.array([
        [1, (t_1 * math.sin(theta)) / (t_3 * math.cos(theta) - t_2 * math.sin(theta))],
        [0, t_3 / (t_3 * math.cos(theta) - t_2 * math.sin(theta))]
    ])

    # Sample values and approximate new spline - use pre-computed C values
    # Apply A_inv transformation vectorized
    D_vals = np.dot(A_inv, C_vals.T).T
    x_sampled, y_sampled = D_vals[:, 0], D_vals[:, 1]
    
    D = UnivariateSpline(x_sampled, y_sampled, k=3, s=1)
    D_parametric = to_parametric_spline(D)
    t_sample_points = sample_spline_arc(D_parametric, num)
    
    # Pre-allocate and avoid list comprehension
    D_sampled = np.empty((num, 2))
    for i, t in enumerate(t_sample_points):
        D_sampled[i] = D_parametric(t)
    C_sampled = np.dot(A, D_sampled.T).T

    # Create longitude line functions
    longitudes = [line(x1=v_x, y1=v_y, x2=x, y2=y) for x, y in C_sampled]

    return longitudes, D_parametric


def expand_lr_boundaries(v_x: float, v_y: float, w: float, h: float) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """
    Expand left/right page boundaries to cover the entire page.

    Parameters
    ----------
    v_x : float
        X coordinate of vanishing point.
    v_y : float
        Y coordinate of vanishing point.
    w : float
        Width of input image.
    h : float
        Height of input image.

    Returns
    -------
    left : Callable[[float], float]
        New line function representing the left boundary.
    right : Callable[[float], float]
        New line function representing the right boundary.
    """

    if v_x < 0:
        if v_y < 0:
            left  = line(x1=v_x, y1=v_y, x2=0, y2=0)
            right = line(x1=v_x, y1=v_y, x2=w, y2=h)
        else:
            left  = line(x1=v_x, y1=v_y, x2=0, y2=h)
            right = line(x1=v_x, y1=v_y, x2=w, y2=0)
    elif v_x < w:
        if v_y < 0:
            left  = line(x1=v_x, y1=v_y, x2=0, y2=h)
            right = line(x1=v_x, y1=v_y, x2=w, y2=h)
        else:
            left  = line(x1=v_x, y1=v_y, x2=0, y2=0)
            right = line(x1=v_x, y1=v_y, x2=w, y2=0)
    else:
        if v_y < 0:
            left  = line(x1=v_x, y1=v_y, x2=0, y2=h)
            right = line(x1=v_x, y1=v_y, x2=w, y2=0)
        else:
            left  = line(x1=v_x, y1=v_y, x2=0, y2=0)
            right = line(x1=v_x, y1=v_y, x2=w, y2=h)    

    return left, right


def compute_aspect_ratio(v_x: float, v_y: float, f: float, h: int, w: int, top: Derivable, bottom: Derivable, D: Derivable) -> float:
    """
    Compute the aspect ratio of the page.

    Parameters
    ----------
    v_x : float
        X coordinate of vanishing point.
    v_y : float
        Y coordinate of vanishing point.
    f : float
        Focal length.
    w : int
        Width of input image.
    h : int
        Height of input image.
    top : Derivable
        Parametric spline representing the top of the page.
    bottom : Derivable
        Parametric spline representing the bottom of the page.
    D : Derivable
        Parametric spline representing the curvature of the page.

    Returns
    -------
    r : float
        Aspect ratio
    """

    # Compute convergence line L - cache repeated function calls
    top_025, top_025_d = top(0.25), top(0.25, 1)
    top_075, top_075_d = top(0.75), top(0.75, 1)
    bottom_025, bottom_025_d = bottom(0.25), bottom(0.25, 1)
    bottom_075, bottom_075_d = bottom(0.75), bottom(0.75, 1)
    
    m_top_025 = top_025_d[1]
    m_bottom_025 = bottom_025_d[1]
    tangent_top_025 = line(m=m_top_025, x1=top_025[0], y1=top_025[1])
    tangent_bottom_025 = line(m=m_bottom_025, x1=bottom_025[0], y1=bottom_025[1])
    point_025 = line_intersection(tangent_top_025, tangent_bottom_025)
    
    m_top_075 = top_075_d[1]
    m_bottom_075 = bottom_075_d[1]
    tangent_top_075 = line(m=m_top_075, x1=top_075[0], y1=top_075[1])
    tangent_bottom_075 = line(m=m_bottom_075, x1=bottom_075[0], y1=bottom_075[1])
    point_075 = line_intersection(tangent_top_075, tangent_bottom_075)
    
    m_L = (point_075[1] - point_025[1]) / (point_075[0] - point_025[0])
    L = line(m=m_L, x1=point_025[0], y1=point_025[1])

    # Compute line vF which is perpendicular to L through the vanishing point
    vF = line(m=-1/m_L, x1=v_x, y1=v_y)
    F = line_intersection(L, vF)

    # Cache bottom and top evaluations at 0 and 1
    bottom_0 = bottom(0)
    bottom_1 = bottom(1)
    top_0 = top(0)
    
    L_0 = line(m=m_L, x1=bottom_0[0], y1=bottom_0[1])
    L_1 = line(m=m_L, x1=top_0[0], y1=top_0[1])
    p_0 = line_intersection(L_0, vF)
    p_1 = line_intersection(L_1, vF)
    q_0 = line_intersection(L_0, line(x1=F[0], y1=F[1], x2=bottom_1[0], y2=bottom_1[1]))

    d = math.sqrt(p_1[0]**2 + p_1[1]**2 + f**2)  # euclidean from origin
    theta = math.acos(f / (math.sqrt(v_x * v_x + v_y * v_y + f * f)))
    alpha = math.atan(d / f)
    beta = math.pi / 2 - theta

    h_img = math.sqrt((p_0[0] - p_1[0])**2 + (p_0[1] - p_1[1])**2)  # euclidean
    l_img = math.sqrt((q_0[0] - bottom_0[0])**2 + (q_0[1] - bottom_0[1])**2)  # euclidean

    # Length of directrix D using vectorized integration
    n_int = 100
    t_int = np.linspace(0, 1, n_int)
    D_derivs = np.empty((n_int, 2))
    for i, t in enumerate(t_int):
        D_derivs[i] = D(t, 1)
    ds_magnitudes = np.sqrt(D_derivs[:, 0]**2 + D_derivs[:, 1]**2)
    if hasattr(np, 'trapezoid'):
        wl_img = np.trapezoid(ds_magnitudes, t_int)
    else:
        wl_img = np.trapz(ds_magnitudes, t_int)

    D_0 = D(0)
    D_1 = D(1)
    ll_img = D_1[0] - D_0[0]

    r = (h_img / l_img) * (ll_img / wl_img) * (math.cos(alpha) / math.cos(alpha + beta))

    return r


def _find_spline_line_intersection_batch(spline: UnivariateSpline, line_slopes: np.ndarray, line_intercepts: np.ndarray, x0: np.ndarray, max_iter: int = 30, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find intersections between a spline and multiple lines using vectorized Newton-Raphson.
    
    Each line is defined as y = slope * x + intercept.
    
    Parameters
    ----------
    spline : UnivariateSpline
        The spline function.
    line_slopes : np.ndarray
        Array of line slopes.
    line_intercepts : np.ndarray
        Array of line y-intercepts.
    x0 : np.ndarray
        Initial guesses for x coordinates.
    max_iter : int
        Maximum iterations for Newton-Raphson.
    tol : float
        Convergence tolerance.
    
    Returns
    -------
    x : np.ndarray
        X coordinates of intersections.
    y : np.ndarray
        Y coordinates of intersections.
    """
    x = x0.copy()
    h = 1e-7
    
    for _ in range(max_iter):
        # f(x) = spline(x) - (slope * x + intercept)
        spline_vals = spline(x)
        f_val = spline_vals - (line_slopes * x + line_intercepts)
        
        # Check convergence
        if np.all(np.abs(f_val) < tol):
            break
        
        # Numerical derivative: f'(x) = spline'(x) - slope
        spline_deriv = (spline(x + h) - spline(x - h)) / (2 * h)
        df_val = spline_deriv - line_slopes
        
        # Avoid division by zero
        df_val = np.where(np.abs(df_val) < 1e-12, 1e-12, df_val)
        
        # Newton-Raphson update
        x = x - f_val / df_val
    
    y = spline(x)
    return x, y


def generate_mesh(num_latitudes: int, num_longitudes: int, longitudes: List[Callable[[float], float]], mu_top: float, mu_bottom: float, w: int, h: int, orig_w: int, orig_h: int, get_latitude: Callable[[float], Derivable]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return warped mesh.

    Parameters
    ----------
    num_latitudes : int
        Number of desired latitudes.
    num_longitudes : int
        Number of desired longitudes.
    longitudes : List[Callable[[float], float]]
        List of line functions representing longitudes.
    mu_top : float
        Parameter for top page boundary.
    mu_bottom : float
        Parameter for bottom page boundary.
    w : int
        Width of input image.
    h : int
        Height of input image.
    orig_w : int
        Width of original image.
    orig_h : int
        Height of original image.
    get_latitude : Callable[[float], Derivable]
        Function to get parametric splines representing latitudes for a given
        parameter 'mu'.

    Returns
    -------
    cols : np.ndarray
        Adressing columns for mesh.
    rows : np.ndarray
        Adressing rows for mesh.
    """
    coords = np.empty((num_latitudes, num_longitudes, 2), dtype=np.float32)
    
    # Pre-compute line parameters (y = slope * x + intercept) - vectorized
    line_slopes = np.array([lon(1) - lon(0) for lon in longitudes], dtype=np.float64)
    line_intercepts = np.array([lon(0) for lon in longitudes], dtype=np.float64)
    
    # Initial guesses based on longitude x-values (estimate from image center)
    x0_base = np.linspace(0, w, num_longitudes, dtype=np.float64)
    
    # Pre-compute all mu values
    mu_values = np.linspace(mu_top, mu_bottom, num_latitudes)
    
    # Pre-compute all latitude splines to avoid repeated computation
    latitude_splines = [get_latitude(mu) for mu in mu_values]
    
    # Process all latitudes with pre-computed splines
    for idx_lat, lat in enumerate(latitude_splines):
        # Use vectorized intersection finding
        x_intersect, y_intersect = _find_spline_line_intersection_batch(
            lat, line_slopes, line_intercepts, x0_base
        )
        
        coords[idx_lat, :, 0] = x_intersect
        coords[idx_lat, :, 1] = y_intersect

    # Scale mesh - use in-place operations for efficiency
    coords[:, :, 0] *= (orig_w / w)
    coords[:, :, 1] *= (orig_h / h)
    coords = cv.resize(coords, (orig_w, orig_h), interpolation=cv.INTER_LINEAR).astype(np.float32)
    
    return coords[:, :, 0], coords[:, :, 1]


def mrcdi(input_img: np.ndarray, barlines_img: np.ndarray, upper_img: np.ndarray, lower_img: np.ndarray, background_img: np.ndarray, original_img: np.ndarray, optimize_f: bool = False) -> np.ndarray:
    """
    Perform metric rectification on given sheet music images.
    The algorithm is based on the paper
        "Metric Rectification of Curved Document Images"
        by Gaofeng Meng et al. (2012)
        https://doi.org/10.1109/TPAMI.2011.151

    Parameters
    ----------
    input_img : np.ndarray
        Resized version of the original input image.
    barlines_img : np.ndarray
        Barlines class output of segmentation network.
    upper_img : np.ndarray
        Upper staff lines class output of segmentation network.
    lower_img : np.ndarray
        Lower staff lines class output of segmentation network.
    background_img : np.ndarray
        Background class output of segmentation network.
    original_img : np.ndarray
        Full size original (binarized) score image.

    Returns
    -------
    score : np.ndarray
        Rectified score image.
    """
    
    h, w, _ = input_img.shape
    min_dim = min(h, w)
    num_longitudes = int(w / min_dim * 20)
    num_latitudes = int(h / min_dim * 20)


    logging.info('Estimating vanishing point')
    left, right = get_outer_barlines(barlines_img)
    v_x, v_y = line_intersection(left, right)


    logging.info('Getting top and bottom stafflines')
    stafflines = get_stafflines(upper_img, lower_img, w//num_longitudes)
    # import matplotlib.pyplot as plt
    # plt.imshow(input_img, cmap='Greys')
    # for s in stafflines:
    #     x = s.get_knots()
    #     y = [s(x) for x in x]
    #     plt.scatter(x, y)

    #     x = np.linspace(x[0], x[-1], 100)
    #     y = [s(x) for x in x]
    #     plt.plot(x, y)
    # plt.show()

    top, bottom = get_top_bottom_stafflines(stafflines, left, right)


    # logging.info('Estimating focal length')
    if optimize_f:
        f = estimate_focal_length(v_x, v_y, top, bottom, f=3760)
    else:
        f = 3760  # Value has so little influence, just fix it...


    logging.info('Expand left/right boundaries')
    # ... to cover the entire page
    left, right = expand_lr_boundaries(v_x, v_y, w, h)


    logging.info('Computing distant latitude')
    # Convert top/bottom to parametric splines
    top_x_start, _ = func_intersection(top[0], left)
    top_x_end  , _ = func_intersection(top[0], right)
    top_parametric = to_parametric_spline(top[0], top_x_start, top_x_end)

    bottom_x_start, _ = func_intersection(bottom[0], left)
    bottom_x_end  , _ = func_intersection(bottom[0], right)
    bottom_parametric = to_parametric_spline(bottom[0], bottom_x_start, bottom_x_end)

    # Use cached latitude computation for better performance
    lat_cache = LatitudeCache(v_x, v_y, top_parametric, bottom_parametric)
    get_latitude_parametric = lat_cache.get_latitude_parametric
    get_latitude = lat_cache.get_latitude
    C_20 = get_latitude_parametric(20)


    logging.info('Expanding upper/lower boundaries')
    # ... to cover the entire page
    
    # TODO Make faster and possibly more elegant
    # mu_top = fsolve(lambda mu: max([get_latitude_parametric(mu)(t)[1] for t in np.linspace(0, 1, 20)]), 1)[0]
    # mu_bottom = fsolve(lambda mu: min([get_latitude_parametric(mu)(t)[1] for t in np.linspace(0, 1, 20)]) - h, 0)[0]

    # Another approach, needs testing
    # Note: minimize passes t as an array, so we need to extract the scalar with [0]
    t_max_dist = minimize(lambda t: euclidean(top_parametric(t[0]), bottom_parametric(t[0])), [0.5], bounds=[(0, 1)]).x[0]
    mu_top = fsolve(lambda mu: get_latitude_parametric(mu)(t_max_dist)[1], 1)[0]
    mu_bottom = fsolve(lambda mu: get_latitude_parametric(mu)(t_max_dist)[1] - h, 0)[0]


    logging.info('Computing longitudes')
    longitudes, D_parametric = get_longitudes(v_x, v_y, f, C_20, num_longitudes)


    logging.info('Computing aspect ratio')
    ratio = compute_aspect_ratio(v_x, v_y, f, h, w, get_latitude_parametric(mu_top), get_latitude_parametric(mu_bottom), D_parametric)


    logging.info('Generating mesh')
    orig_h, orig_w = original_img.shape
    cols, rows = generate_mesh(num_latitudes, num_longitudes, longitudes, mu_top, mu_bottom, w, h, orig_w, orig_h, get_latitude)

    return cols, rows

    #logging.info('Dewarping image')
    #result = cv.remap(original_img, cols, rows, cv.INTER_CUBIC, None, cv.BORDER_CONSTANT, 255)


    # logging.info('Drawing output')
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    # plt.imshow(input_img, interpolation='bilinear', cmap='Greys')

    # t = np.linspace(0, 1, 500)
    # x = [bottom_parametric(t)[0] for t in t]
    # y = [bottom_parametric(t)[1] for t in t]
    # plt.plot(x, y)
    # x = [top_parametric(t)[0] for t in t]
    # y = [top_parametric(t)[1] for t in t]
    # plt.plot(x, y)

    # # Latitudes
    # for mu in np.linspace(mu_bottom, mu_top, num_latitudes):
    #     latitude_mu = get_latitude_parametric(mu)
    #     x, y = np.array([latitude_mu(t) for t in np.linspace(0, 1, 100)]).transpose()
    #     plt.plot(x, y, color='blue')

    # # Longitudes
    # top_latitude = get_latitude(mu_top)
    # bottom_latitude = get_latitude(mu_bottom)
    # for l in longitudes:
    #     y = [func_intersection(top_latitude, l)[1], func_intersection(bottom_latitude, l)[1]]
    #     m = l(1) - l(0)
    #     plt.plot((m * v_x + y - v_y) / m, y, color='blue')

    # plt.subplot(1, 2, 2)
    # ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    # plt.imshow(result, interpolation='bilinear', cmap='Greys_r')

    # plt.tight_layout()
    # plt.show()


    return result




if __name__ == '__main__':
    from skimage.io import imread
    from cv2 import imwrite

    result = mrcdi(
        input_img      = imread('test1/input.png'),
        barlines_img   = imread('test1/barlines.png'),
        upper_img      = imread('test1/upper.png'),
        lower_img      = imread('test1/lower.png'),
        background_img = imread('test1/background.png'),
        original_img   = imread('test1/binarized.png'),
    )

    imwrite('result.png', result)

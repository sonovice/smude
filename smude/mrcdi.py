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
from typing import Callable, List, Optional, Tuple


import cv2 as cv
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage.measurements import label
from scipy.optimize import fsolve, minimize
from scipy.spatial.distance import euclidean
from skimage.morphology import skeletonize, square
from skimage.transform import hough_line, hough_line_peaks

from .utils import *


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
    for image in [upper_img, lower_img]:
        # Morphological operation to close potential gaps
        #image_closed = cv.dilate(image, square(35))
        image_closed = cv.morphologyEx(image, cv.MORPH_CLOSE, square(25))

        # Segmentize image to get individual staff line instances
        labels, count = label(image_closed)

        # Compute univariate spline for each staff line
        for l in range(1, count + 1):
            line_pixels = labels == l

            # Ensure that the staff line has 1px thickness
            line_pixels = skeletonize(line_pixels)

            y, x = np.nonzero(line_pixels)
            
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

        left_x, left_y = func_intersection(spline, left)
        distance = euclidean((left_x, left_y), (knots[0], spline(knots[0])))

        if distance > max_dist:
            continue

        right_x, right_y = func_intersection(spline, right)
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



def cost_function(v_x: float, v_y: float, f: float, top: Tuple[UnivariateSpline, float, float], bottom: Tuple[UnivariateSpline, float, float], m: int = 10) -> float:
    """
    Compute the costs for a given vanishing point and the focal distance. Used for optimization.

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
    m : int, optional
        Number of sampling points along top/bottom splines, by default 10.

    Returns
    -------
    float
        Costs.
    """
    spline_top   , top_left,    top_right    = top
    spline_bottom, bottom_left, bottom_right = bottom

    V = np.array([v_x, v_y, f])

    E_o = 0.0
    E_c = 0.0

    length_V = np.linalg.norm(V)

    for x_top in np.linspace(top_left, top_right, m):
        longitude = line(x1=v_x, y1=v_y, x2=x_top, y2=spline_top(x_top))
        intersection = func_intersection(spline_bottom, longitude)
        x_bottom = intersection[0]

        m_top    = spline_top(x_top, 1)
        m_bottom = spline_bottom(x_bottom, 1)

        tangent_top    = line(m=m_top, x1=x_top, y1=spline_top(x_top))
        tangent_bottom = line(m=m_bottom, x1=x_bottom, y1=spline_bottom(x_bottom))

        intersection = func_intersection(tangent_top, tangent_bottom)
        P = np.array([intersection[0], intersection[1], f])
        p = P[:2]

        # E_c
        length_p = np.linalg.norm(p)
        for m in [m_top, m_bottom]:
            l = np.array([1, m])
            length_l = np.linalg.norm(l)

            cos_P_l = np.dot(p, l) / (length_p * length_l)
            E_c += abs(cos_P_l)

        # E_o
        length_P = np.linalg.norm(P)
        cos_P_V = np.dot(P, V) / (length_P * length_V)
        E_o += abs(cos_P_V)

    E_c /= 2

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

    def temp(t):
        lambd = euclidean(bottom(t), (v_x, v_y)) / euclidean(bottom(t), top(t))
        alpha = (mu * lambd) / (mu + lambd - 1)
        p_a = (1 - alpha) * bottom(t) + alpha * top(t)
        return p_a

    # Sample values and approximate new spline
    # (faster in the long run but slightly less accurate)
    x_sampled, y_sampled = np.array([temp(t) for t in np.linspace(0, 1, 30)]).transpose()
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

    def temp(t):
        lambd = euclidean(bottom(t), (v_x, v_y)) / euclidean(bottom(t), top(t))
        alpha = (mu * lambd) / (mu + lambd - 1)
        p_a = (1 - alpha) * bottom(t) + alpha * top(t)
        return p_a

    # Sample values and approximate new spline
    # (faster in the long run but slightly less accurate)
    x_sampled, y_sampled = np.array([temp(t) for t in np.linspace(0, 1, 30)]).transpose()
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

    # Compute centroid of curve C
    x_0 = quad(lambda t: C(t)[0], 0, 1)[0]
    y_0 = quad(lambda t: C(t)[1], 0, 1)[0]

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

    D = lambda t: np.matmul(A_inv, C(t))

    # Sample values and approximate new spline
    # (faster in the long run but slightly less accurate)
    x_sampled, y_sampled = np.array([D(t) for t in np.linspace(0, 1, 50)]).transpose()
    D = UnivariateSpline(x_sampled, y_sampled, k=3, s=1)
    D_parametric = to_parametric_spline(D)
    t_sample_points = sample_spline_arc(D_parametric, num)
    D_sampled = np.array([D_parametric(t) for t in t_sample_points]).transpose()
    C_sampled = np.matmul(A, D_sampled).transpose()

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

    # Compute convergence line L
    points = []
    for t in [0.25, 0.75]:
        m_top    = top(t, 1)[1]
        m_bottom = bottom(t, 1)[1]
        tangent_top    = line(m=m_top,    x1=top(t)[0],    y1=top(t)[1])
        tangent_bottom = line(m=m_bottom, x1=bottom(t)[0], y1=bottom(t)[1])
        points.append(line_intersection(tangent_top, tangent_bottom))
    m_L = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
    L   = line(m=m_L, x1=points[0][0], y1=points[0][1])

    # Compute line vF which is perpendicular to L through the vanishing point
    vF  = line(m=-1/m_L, x1=v_x, y1=v_y)
    F   = line_intersection(L, vF)

    L_0 = line(m=m_L, x1=bottom(0)[0], y1=bottom(0)[1])
    L_1 = line(m=m_L, x1=top(0)[0],    y1=top(0)[1])
    p_0 = line_intersection(L_0, vF)
    p_1 = line_intersection(L_1, vF)
    q_0 = line_intersection(L_0, line(x1=F[0], y1=F[1], x2=bottom(1)[0], y2=bottom(1)[1]))

    d = euclidean(np.array([0,0, 0]), np.array([p_1[0], p_1[1], f]))
    theta = math.acos(f / (math.sqrt(v_x * v_x + v_y * v_y + f * f)))
    alpha = math.atan(d / f)
    beta  = math.pi / 2 - theta

    h_img = euclidean(p_0, p_1)
    l_img = euclidean(q_0, bottom(0))

    # Length of directrix D
    wl_img = quad(lambda t: np.sqrt(np.power(D(t, 1)[0], 2) + np.power(D(t, 1)[1], 2)), 0, 1)[0]

    ll_img = D(1)[0] - D(0)[0]

    r = (h_img / l_img) * (ll_img / wl_img) * (math.cos(alpha) / math.cos(alpha + beta))

    return r


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
        [description]
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
    coords = np.empty((num_latitudes, num_longitudes, 2))
    for idx_lat, mu in enumerate(np.linspace(mu_top, mu_bottom, num_latitudes)):
        lat = get_latitude(mu)
        for idx_lon, lon in enumerate(longitudes):
            coords[idx_lat, idx_lon] = func_intersection(lat, lon)

    # Scale mesh
    coords[:,:,0] = coords[:,:,0] / w * orig_w
    coords[:,:,1] = coords[:,:,1] / h * orig_h
    coords = cv.resize(coords, (orig_w, orig_h)).astype(np.float32)
    cols = coords[:,:,0]
    rows = coords[:,:,1]

    return cols, rows


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

    get_latitude_parametric = lambda mu, v_x=v_x, v_y=v_y, top=top, bottom=bottom: _get_latitude_parmetric(v_x, v_y, top_parametric, bottom_parametric, mu)
    get_latitude = lambda mu, v_x=v_x, v_y=v_y, top=top, bottom=bottom: _get_latitude(v_x, v_y, top_parametric, bottom_parametric, mu)
    C_20 = get_latitude_parametric(20)


    logging.info('Expanding upper/lower boundaries')
    # ... to cover the entire page
    
    # TODO Make faster and possibly more elegant
    # mu_top = fsolve(lambda mu: max([get_latitude_parametric(mu)(t)[1] for t in np.linspace(0, 1, 20)]), 1)[0]
    # mu_bottom = fsolve(lambda mu: min([get_latitude_parametric(mu)(t)[1] for t in np.linspace(0, 1, 20)]) - h, 0)[0]

    # Another approach, needs testing
    t_max_dist = minimize(lambda t: euclidean(top_parametric(t), bottom_parametric(t)), 0.5, bounds=[(0, 1)]).x[0]
    mu_top = fsolve(lambda mu: get_latitude_parametric(mu)(t_max_dist)[1], 1)[0]
    mu_bottom = fsolve(lambda mu: get_latitude_parametric(mu)(t_max_dist)[1] - h, 0)[0]


    logging.info('Computing longitudes')
    longitudes, D_parametric = get_longitudes(v_x, v_y, f, C_20, num_longitudes)


    logging.info('Computing aspect ratio')
    ratio = compute_aspect_ratio(v_x, v_y, f, h, w, get_latitude_parametric(mu_top), get_latitude_parametric(mu_bottom), D_parametric)


    logging.info('Generating mesh')
    orig_h, orig_w = original_img.shape
    cols, rows = generate_mesh(num_latitudes, num_longitudes, longitudes, mu_top, mu_bottom, w, h, orig_w, orig_h, get_latitude)


    logging.info('Dewarping image')
    result = cv.remap(original_img, cols, rows, cv.INTER_CUBIC, None, cv.BORDER_CONSTANT, 255)


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

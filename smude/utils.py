__author__ = "Simon Waloschek"

import datetime
import logging
import time
from functools import partial
from multiprocessing import Pool
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.misc import derivative
from scipy.optimize import fsolve
from typing_extensions import Protocol


class Derivable(Protocol):
    def __call__(self, val: float, derivative: Optional[int] = None) -> np.ndarray: ...


class RuntimeFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        duration = datetime.datetime.utcfromtimestamp(record.created - self.start_time)
        elapsed = duration.strftime('%H:%M:%S.%f')
        return "{}".format(elapsed)


def get_logger():
    LOGFORMAT = '%(asctime)s - %(levelname)-9s: %(message)s'
    handler = logging.StreamHandler()
    fmt = RuntimeFormatter(LOGFORMAT)
    handler.setFormatter(fmt)
    logging.getLogger().addHandler(handler)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger


def line(m: Optional[float] = None, x1: Optional[float] = 0, y1: Optional[float] = 0, x2: Optional[float] = None, y2: Optional[float] = None) -> Callable[[float], float]:
    """
    Construct a 2D-line. Parameters can be given as
    - Slope and one (x, y) point on line
    or
    - Two (x, y) points on line

    Parameters
    ----------
    m : float, optional
        Slope of the new line, by default None
    x1 : float, optional
        X coordinate of first given point on line, by default 0
    y1 : float, optional
        Y coordinate of first given point on line, by default 0
    x2 : float, optional
        X coordinate of second given point on line, by default None
    y2 : flaot, optional
        Y coordinate of second given point on line, by default None

    Returns
    -------
    line : function
        2D-line as function f(x) = y.
    """
    def func(x):
        return m * (x - x1) + y1

    if x2 is not None and y2 is not None:
        m = (y2 - y1) / (x2 - x1)

    return func


def sample_spline_arc(spline: UnivariateSpline, num_samples: int) -> np.ndarray:
    """
    Sample equidistant dividing points along the arc of a parametric spline.

    Parameters
    ----------
    spline : UnivariateSpline
        A function representing a parametric spline. Takes an input `t` and
        returns a touple `(x, y)`.
    num_samples : int
        Number of desired sampling points in parametric space.

    Returns
    -------
    t : np.ndarray
        An array of values representing the t values of the spline function
        evaluated at the equidistant diving points.
    """

    # The partial arc length L of a parametric function (parameter t = [0, 1])
    # is defined by
    #
    #     b      _____________
    #     ⌠     ╱    2       2      b
    #     ⎮    ╱ ⎛dx⎞    ⎛dy⎞       ⌠
    # L = ⎮   ╱  ⎜──⎟  + ⎜──⎟  dt = ⎮ ds dt
    #     ⌡ ╲╱   ⎝dt⎠    ⎝dt⎠       ⌡
    #     0                         0
    #
    # (see https://tutorial.math.lamar.edu/classes/calcii/ParaArcLength.aspx)

    # TODO Find workaround, very ugly! (But needed for Pool)
    global _solve_b, _ds, _L_diff

    def _L_diff(t, ds, b):
        return quad(ds, 0, t)[0] - b

    def _solve_b(f, ds, length):
        return fsolve(_L_diff, 0, (ds, f * length))[0]

    def _ds(t):
        return np.sqrt(np.power(spline(t, 1)[0], 2) + np.power(spline(t, 1)[1], 2))

    length = quad(_ds, 0, 1)[0]

    pool = Pool()
    b_vals = pool.map(partial(_solve_b, ds=_ds, length=length), np.linspace(0, 1, num_samples).tolist())
    #b_vals = [solve_b(f, ds, length) for f in np.linspace(0, 1, num_samples)]

    return np.asarray(b_vals)


def func_intersection(func1: Callable[[float], float], func2: Callable[[float], float]) -> Tuple[float, float]:
    """Return the intersection of two functions in R². Does find at most one intersection.

    Parameters
    ----------
    func1 : Callable[[float], float]
        First function.
    func2 : Callable[[float], float]
        Second function.

    Returns
    -------
    x : float
        X coordinate of the intersection.
    y : float
        Y coordinate of the intersection.
    """

    diff = lambda x: func1(x) - func2(x)
    x = fsolve(diff, 0)[0]
    y = func1(x)

    return x, y


def line_intersection(line1: Callable[[float], float], line2: Callable[[float], float]) -> Tuple[float, float]:
    """
    Return the intersection point of two lines in R².

    Parameters
    ----------
    line1 : Callable[[float], float]
        First line function.
    line2 : Callable[[float], float]
        Second line function.

    Returns
    -------
    x : float
        X coordinate of the line intersection.
    y : float
        Y coordinate of the line intersection.
    """

    x1 = 0
    y1 = line1(0)
    m1 = line1(1) - y1

    x2 = 0
    y2 = line2(0)
    m2 = line2(1) - y2

    #print(y1, y2)

    # Check if lines are parallel
    if m1 == m2:
        return np.nan, np.nan

    x = (m1 * x1 - m2 * x2 - y1 + y2) / (m1 - m2)
    y = m1 * (x - x1) + y1

    return x, y


def to_parametric_spline(spline: UnivariateSpline, x_start: Optional[float] = None, x_end: Optional[float] = None) -> Derivable:
    """
    Convert a regular 's(x) = y' spline into a 's(t) = (x, y)' parametric spline.

    Parameters
    ----------
    spline : UnivariateSpline
        Regular spline.
    x_start : float, optional
        X coordinate of left start point of the spline. Will be accessible by
        t = 0.
    x_end : float, optional
        X coordinate of right end point of the spline. Will be accessible by
        t = 1.

    Returns
    -------
    spline : Derivable
        Parametric spline.
    """
    if x_start is None:
        knots = spline.get_knots()
        x_start = knots[0]

    if x_end is None:
        knots = spline.get_knots()
        x_end = knots[-1]

    interpolator = interp1d((0, 1), (x_start, x_end), fill_value='extrapolate')

    def parametric_spline(t, d=0):
        x = interpolator(t)
        if d > 0:
            x_dt = derivative(interpolator, t, d)
        else:
            x_dt = x
        y_dt = spline(x, d)
        return np.array([x_dt, y_dt])

    return parametric_spline

import numpy as np
from oplab import Console
from numba import njit
from scipy import optimize


@njit
def exp_curve(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Compute exponent value with respect to a distance value

    Parameters
    -----------
    x : np.ndarray
        distance value
    a : float
        coefficient
    b : float
        coefficient
    c : float
        coefficient

    Returns
    --------
    np.ndarray
    """
    return a * np.exp(b * x) + c


@njit
def residual_exp_curve(params: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Compute residuals with respect to init params

    Parameters
    -----------
    params : numpy.ndarray
        array of init params
    x : numpy.ndarray
        array of distance values
    y : numpy.ndarray
        array of intensity values

    Returns
    --------
    numpy.ndarray
        residual
    """

    residual = exp_curve(x, params[0], params[1], params[2]) - y
    return residual


# compute attenuation correction parameters through regression
def curve_fitting(altitudes: np.ndarray, intensities: np.ndarray) -> np.ndarray:
    """Compute attenuation coefficients with respect to distance values

    Parameters
    -----------
    altitudes : list
        list of distance values
    intensities : list
        list of intensity values

    Returns
    --------
    numpy.ndarray
        parameters
    """
    loss = "soft_l1"
    method = "trf"
    bound_lower = [1, -np.inf, 0]
    bound_upper = [np.inf, 0, np.inf]

    n = len(intensities)
    idx_0 = int(n * 0.3)
    idx_1 = int(n * 0.7)

    # Avoid zero divisions
    b = 0
    c = 0
    if intensities[idx_1] != 0:
        b = (np.log((intensities[idx_0] - c) / (intensities[idx_1] - c))) / (
            altitudes[idx_0] - altitudes[idx_1]
        )
    a = (intensities[idx_1] - c) / np.exp(b * altitudes[idx_1])
    if a < 1 or b > 0 or np.isnan(a) or np.isnan(b):
        a = 1.01
        b = -0.01

    init_params = np.array([a, b, c], dtype=np.float32)
    try:
        tmp_params = optimize.least_squares(
            residual_exp_curve,
            init_params,
            loss=loss,
            method=method,
            args=(altitudes, intensities),
            bounds=(bound_lower, bound_upper),
        )
        return tmp_params.x
    except (ValueError, UnboundLocalError) as e:
        print("ERROR: Value Error due to Overflow", a, b, c)
        print('Parameters calculated are unoptimised because of Value Error', e)
        return init_params

# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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
def curve_fitting(
    altitudes: np.ndarray, intensities: np.ndarray, figure_path: Optional[Path]
) -> np.ndarray:
    """Compute attenuation coefficients with respect to distance values

    Parameters
    -----------
    altitudes : list
        list of distance values
    intensities : list
        list of intensity values
    figure_path: Path | None
        Path where curve fitting figure is written; or None to not write figure to file

    Returns
    --------
    numpy.ndarray
        parameters
    """
    altitudes = altitudes[np.isfinite(altitudes)]
    intensities = intensities[np.isfinite(intensities)]

    if altitudes.size == 0:
        print("---------\naltitudes: ", altitudes, "\nintensities: ", intensities)
        print("ERROR: Empty non-nan altitudes in curve fitting")
        return np.array([1, 0, 0])
    if intensities.size == 0:
        print("---------\naltitudes: ", altitudes, "\nintensities: ", intensities)
        print("ERROR: Empty non-nan intensities in curve fitting")
        return np.array([1, 0, 0])

    altitudes_filt = []
    intensities_filt = []
    for x, y in zip(altitudes, intensities):
        if x > 0 and y > 0:
            altitudes_filt.append(x)
            intensities_filt.append(y)

    if not altitudes_filt or not intensities_filt:
        if not altitudes_filt:
            print(
                "---------\naltitudes: ",
                altitudes,
                "\nintensities: ",
                intensities,
                "\naltitudes_filt: ",
                altitudes_filt,
                "\nintensities_filt: ",
                intensities_filt,
            )
            print("ERROR: Altitudes are negative in curve fitting")
        if not intensities_filt:
            print(
                "---------\naltitudes: ",
                altitudes,
                "\nintensities: ",
                intensities,
                "\naltitudes_filt: ",
                altitudes_filt,
                "\nintensities_filt: ",
                intensities_filt,
            )
            print("ERROR: Intensities are negative in curve fitting")
        return np.array([1, 0, 0])

    altitudes_filt = np.array(altitudes_filt)
    intensities_filt = np.array(intensities_filt)

    try:
        c_upper_bound = intensities_filt.min()
    except ValueError:  # raised if it is empty.
        c_upper_bound = np.finfo(float).eps

    if c_upper_bound <= 0:
        # c should be slightly greater than zero to avoid error
        # 'Each lower bound must be strictly less than each upper bound.'
        c_upper_bound = np.finfo(float).eps

    loss = "soft_l1"
    method = "trf"
    bound_lower = [1e-6, -np.inf, 0]
    bound_upper = [np.inf, 0, c_upper_bound]

    n = len(intensities_filt)
    idx_0 = int(n * 0.3)
    idx_1 = int(n * 0.7)

    int_0 = intensities_filt[idx_0]
    int_1 = intensities_filt[idx_1]
    alt_0 = altitudes_filt[idx_0]
    alt_1 = altitudes_filt[idx_1]

    # Avoid zero divisions
    b = 0.0
    try:
        c = intensities_filt.min() * 0.5
    except ValueError:  # raised if it is empty.
        c = np.finfo(float).eps

    if intensities_filt[idx_1] != 0:
        b = (np.log((int_0 - c) / (int_1 - c))) / (alt_0 - alt_1)
    a = (int_1 - c) / np.exp(b * alt_1)
    if a <= 0 or b > 0 or np.isnan(a) or np.isnan(b):
        a = 1.01
        b = -0.01

    init_params = np.array([a, b, c], dtype=np.float32)

    try:
        tmp_params = optimize.least_squares(
            residual_exp_curve,
            init_params,
            loss=loss,
            method=method,
            args=(altitudes_filt, intensities_filt),
            bounds=(bound_lower, bound_upper),
        )
        if figure_path:
            fig = plt.figure()
            plt.plot(altitudes_filt, intensities_filt, "c.", label="Intensities")
            xs = np.arange(2, 10, 0.1)
            ys = exp_curve(xs, tmp_params.x[0], tmp_params.x[1], tmp_params.x[2])
            plt.plot(xs, ys, "-m", label="Exp curve")
            plt.plot(xs, np.ones(xs.shape[0]) * tmp_params.x[2], "-y", label="C term")
            plt.legend()
            plt.savefig(str(figure_path), dpi=600)
            plt.close(fig)

        return tmp_params.x
    except (ValueError, UnboundLocalError) as e:
        print("ERROR: Value Error due to Overflow", a, b, c)
        print("Parameters calculated are unoptimised because of error", e)
        return init_params

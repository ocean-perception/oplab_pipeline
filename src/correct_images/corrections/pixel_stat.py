# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2025, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import numpy as np


def pixel_stat(
    bayer_img: np.ndarray,
    bayer_img_mean: np.ndarray,
    bayer_img_std: np.ndarray,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """Generate target stats for images

    Parameters
    -----------
    img : numpy.ndarray
        Image data to be corrected for target stats
    img_mean : np.ndarray
        Mean image of reference image dataset
    img_std : np.ndarray
        Standard deviation image of reference image dataset
    target_mean : float
        Desired mean on scale from 0 to 100
    target_std : float
        Desired standard deviation on scale from 0 to 100

    Returns
    -------
    numpy.ndarray
        Corrected image
    """

    # target_mean and target std should be given in 0 - 100 scale
    target_mean_unitary = target_mean / 100.0
    target_std_unitary = target_std / 100.0
    ret = (
        bayer_img - bayer_img_mean
    ) / bayer_img_std * target_std_unitary + target_mean_unitary
    ret = np.clip(ret, 0, 1)
    return ret

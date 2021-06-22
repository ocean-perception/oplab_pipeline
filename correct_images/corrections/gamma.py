# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import numpy as np


def gamma_correct(image):
    """performs gamma correction for images
    Parameters
    -----------
    image : numpy.ndarray
        image data to be corrected for gamma
    bitdepth : int
        target bitdepth for output image


    Returns
    -------
    numpy.ndarray
        Image
    """
    ret_image = image.copy()  # np.divide(image, (2 ** bitdepth - 1))
    if all(i < 0.0031308 for i in image.flatten()):
        ret_image = 12.92 * ret_image
    else:
        ret_image = 1.055 * np.power(ret_image, (1 / 1.5)) - 0.055
    # ret_image = np.multiply(np.array(ret_image), np.array(2 ** bitdepth - 1))
    # ret_image = np.clip(ret_image, 0, 2 ** bitdepth - 1)
    return ret_image

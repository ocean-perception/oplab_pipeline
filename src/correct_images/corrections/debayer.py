# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import cv2
import numpy as np

from oplab import Console


# convert bayer image to RGB based
# on the bayer pattern for the camera
def debayer(image: np.ndarray, pattern: str) -> np.ndarray:
    """Perform debayering of input image

    Parameters
    -----------
    image : numpy.ndarray
        image data to be debayered
    pattern : string
        bayer pattern

    Returns
    -------
    numpy.ndarray
        Debayered image
    """

    if image is None:
        Console.warn("Image is None")
        return None

    # Make use of 16 bit debayering
    image16_float = image.astype(np.float32) * (2**16 - 1)
    image16 = image16_float.clip(0, 2**16 - 1).astype(np.uint16)

    corrected_rgb_img = None
    if pattern == "rggb" or pattern == "RGGB":
        corrected_rgb_img = cv2.cvtColor(image16, cv2.COLOR_BAYER_BG2RGB_EA)
    elif pattern == "grbg" or pattern == "GRBG":
        corrected_rgb_img = cv2.cvtColor(image16, cv2.COLOR_BAYER_GB2RGB_EA)
    elif pattern == "bggr" or pattern == "BGGR":
        corrected_rgb_img = cv2.cvtColor(image16, cv2.COLOR_BAYER_RG2RGB_EA)
    elif pattern == "gbrg" or pattern == "GBRG":
        corrected_rgb_img = cv2.cvtColor(image16, cv2.COLOR_BAYER_GR2RGB_EA)
    else:
        Console.quit("Bayer pattern not supported (", pattern, ")")

    # Scale down to unitary
    corrected_rgb_img = corrected_rgb_img.astype(np.float32) * (2 ** (-16))
    return corrected_rgb_img

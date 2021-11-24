# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import numpy as np
from numba import njit


# read binary raw image files for xviii camera
@njit
def load_xviii_bayer_from_binary(binary_data, image_height, image_width):
    """Read XVIII binary images into bayer array

    Parameters
    -----------
    binary_data : numpy.ndarray
        binary image data from XVIII
    image_height : int
        image height
    image_width : int
        image width

    Returns
    --------
    numpy.ndarray
        Bayer image
    """

    img_h = image_height
    img_w = image_width
    bayer_img = np.zeros((img_h, img_w), dtype=np.uint32)

    # read raw data and put them into bayer pattern.
    count = 0
    for i in range(0, img_h, 1):
        for j in range(0, img_w, 4):
            chunk = binary_data[count : count + 12]  # noqa
            bayer_img[i, j] = (
                ((chunk[3] & 0xFF) << 16) | ((chunk[2] & 0xFF) << 8) | (chunk[1] & 0xFF)
            )
            bayer_img[i, j + 1] = (
                ((chunk[0] & 0xFF) << 16) | ((chunk[7] & 0xFF) << 8) | (chunk[6] & 0xFF)
            )
            bayer_img[i, j + 2] = (
                ((chunk[5] & 0xFF) << 16)
                | ((chunk[4] & 0xFF) << 8)
                | (chunk[11] & 0xFF)
            )
            bayer_img[i, j + 3] = (
                ((chunk[10] & 0xFF) << 16)
                | ((chunk[9] & 0xFF) << 8)
                | (chunk[8] & 0xFF)
            )
            count += 12

    bayer_img = bayer_img.astype(np.float32)
    return bayer_img


def loader(raw_filename, image_width=1280, image_height=1024, src_bit=18):
    """XVIII image loader

    Parameters
    ----------
    raw_filename : Path
        Image file path
    image_width : int
        Image height
    image_height : int
        Image height

    Returns
    -------
    np.ndarray
        Loaded image in matrix form (numpy)
    """
    binary_data = np.fromfile(raw_filename, dtype=np.uint8)
    bayer_img = load_xviii_bayer_from_binary(binary_data[:], image_height, image_width)
    # Scale down from 18 bits to unitary to process with OpenCV debayer
    bayer_img *= 2 ** (-src_bit)
    return bayer_img

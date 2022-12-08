# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import numpy as np

try:
    # Try using the v2 API directly to avoid a warning from imageio >= 2.16.2
    from imageio.v2 import imread
except ImportError:
    from imageio import imread

from oplab import Console


def loader(image_filepath, image_width=None, image_height=None, src_bit=8):
    """Default image loader using imageio

    Parameters
    ----------
    image_filepath : Path
        Image file path
    image_width : int
        Image width
    image_height : int
        Image height

    Returns
    -------
    np.ndarray
        Loaded image in matrix form (numpy)
    """
    # Clip the image to remove any unwanted data
    image = imread(str(image_filepath))

    # Check bit depth
    read_bit_depth = None
    if image.dtype == 'uint8':
        read_bit_depth = 8
    elif image.dtype == 'uint16':
        read_bit_depth = 16
    elif image.dtype == 'uint32':
        read_bit_depth = 32
    elif image.dtype == 'uint64':
        read_bit_depth = 64
    else:
        Console.quit("Image dtype not implemented:", image.dtype)

    bit_shift = read_bit_depth - src_bit
    if bit_shift > 0:
        image = np.right_shift(image, bit_shift).astype(np.float32)
    else:
        image = image.astype(np.float32)
    image = image * 2 ** (-src_bit)
    #print("Image min/max: ", np.min(image), np.max(image), np.min(image2), np.max(image2), src_bit)
    return image

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
    image = imread(str(image_filepath)).astype(np.float32)
    image *= 2 ** (-src_bit)
    return image

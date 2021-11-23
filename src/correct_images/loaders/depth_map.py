# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import numpy as np
from skimage.transform import resize


def loader(depth_map_filename, image_width, image_height):
    """Depth map image loader

    Parameters
    ----------
    depth_map_filename : Path
        Image file path
    image_width : int
        Image height
    image_height : int
        Image height

    Returns
    -------
    np.ndarray
        Loaded depth map in matrix form (numpy)
    """
    depth_array = np.load(depth_map_filename)
    distance_matrix_size = (image_height, image_width)
    distance_matrix = resize(depth_array, distance_matrix_size, preserve_range=True)
    return distance_matrix

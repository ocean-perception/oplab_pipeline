# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import cv2

from oplab import MonoCamera


# correct image for distortions using camera calibration parameters
def distortion_correct(camera_params_file_path, image):
    """Perform distortion correction for images

    Parameters
    -----------
    camera_params_file_path: str
        Path to the camera parameters file
    image : numpy.ndarray
        image data to be corrected for distortion
    dst_bit : int
        target bitdepth for output image

    Returns
    -------
    numpy.ndarray
        Image
    """

    monocam = MonoCamera(camera_params_file_path)
    map_x, map_y = monocam.rectification_maps
    # ret_image = np.clip(image, 0, 2 ** dst_bit - 1)
    ret_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return ret_image

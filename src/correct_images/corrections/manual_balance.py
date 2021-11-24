# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import numpy


def manual_balance(
    image: numpy.ndarray,
    gain_matrix_rgb: numpy.ndarray,
    negative_offset_rgb: numpy.ndarray,
) -> numpy.ndarray:
    """Perform manual balance of input image

    Parameters
    -----------
    image : numpy.ndarray
        image data to be debayered
    gain_matrix_rgb : numpy.ndarray
        3x3 matrix with the colour gains
        for B/W images, default values are the ones for red channel
    negative_offset_rgb : numpy.ndarray
        3x1 vector with the colour subtractors
        for B/W images, default values are the ones for red channel

    Returns
    -------
    numpy.ndarray
        Corrected image
    """

    image_shape = image.shape
    image_height = image_shape[0]
    image_width = image_shape[1]
    image_channels = 1
    if len(image_shape) > 2:
        image_channels = image_shape[2]

    input_image = image.copy()

    if image_channels == 3:
        # corrections for RGB images
        input_image = input_image.reshape((image_height * image_width, 3))
        for i in range(image_height * image_width):
            intensity_vector = input_image[i, :]
            intensity_vector = intensity_vector - negative_offset_rgb
            intensity_vector = gain_matrix_rgb.dot(intensity_vector)
            input_image[i, :] = intensity_vector
        input_image = input_image.reshape((image_height, image_width, 3))
    else:
        input_image = input_image - negative_offset_rgb[0]
        input_image = input_image * gain_matrix_rgb[0, 0]

    return input_image

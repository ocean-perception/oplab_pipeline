# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import uuid

import cv2
import numpy as np

from oplab import Console

from ..loaders import default


def create_memmap(image_list, dimensions, loader=default.loader):
    filename_map = "memmap_" + str(uuid.uuid4()) + ".map"
    # If only 1 channel, do not create a 3D array
    if dimensions[-1] == 1:
        dimensions = dimensions[:-1]
    list_shape = [len(image_list)] + list(dimensions)
    size = 1
    for i in list_shape:
        size *= i
    image_memmap = np.memmap(
        filename=filename_map,
        mode="w+",
        shape=tuple(list_shape),
        dtype=np.float32,
    )
    """
    joblib.Parallel(n_jobs=1, verbose=0)(
        joblib.delayed(memmap_loader)(
            image_list, image_memmap, idx, loader, dimensions[1], dimensions[0]
        )
    )
    """

    # The parent process/function is paralelised, so this one should not be!
    for idx in range(len(image_list)):
        memmap_loader(
            image_list, image_memmap, idx, loader, dimensions[1], dimensions[0]
        )
    return filename_map, image_memmap


def open_memmap(shape, dtype):
    filename_map = "memmap_" + str(uuid.uuid4()) + ".map"
    image_memmap = np.memmap(filename=filename_map, mode="w+", shape=shape, dtype=dtype)
    return filename_map, image_memmap


def convert_to_memmap(array, loader=default.loader):
    filename_map = "memmap_" + str(uuid.uuid4()) + ".map"
    image_memmap = np.memmap(
        filename=filename_map, mode="w+", shape=array.shape, dtype=array.dtype
    )
    image_memmap[:] = array[:]
    return filename_map, image_memmap


def memmap_loader(
    image_list,
    memmap_handle,
    idx,
    loader=default.loader,
    new_width=None,
    new_height=None,
):
    if image_list[idx] is None:
        Console.error("Image at idx", idx, "is None")
        Console.error("Please check your navigation CSV for any missing values")
        Console.quit("Image list is malformed")
    np_im = loader(image_list[idx]).astype(np.float32)

    dimensions = np_im.shape

    if new_height is not None and new_width is not None:
        same_dimensions = (new_width == dimensions[1]) and (new_height == dimensions[0])
        if not same_dimensions:
            im2 = cv2.resize(np_im, (new_width, new_height), cv2.INTER_CUBIC)
            memmap_handle[idx, ...] = im2
        else:
            memmap_handle[idx, ...] = np_im
    else:
        memmap_handle[idx, ...] = np_im

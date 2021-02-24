# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import argparse
import contextlib
import math
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Tuple

import cv2
import imageio
import joblib
import numpy as np
from numba import njit, prange
from tqdm import tqdm

# Parameters
brightness = 40.0  # over 100 of the entire image values
contrast = 10.0  # over 100 of the entire image values
max_space_gb = 60.0  # Max space to use in your hard drive
scale_factor = 0.25
use_random_sample = True


# Joblib and tqdm solution to progressbars
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given
    as argument
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def bytescaling(data: np.ndarray, cmin=None, cmax=None, high=255, low=0):
    if data.dtype == np.uint8:
        return data
    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")
    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()
    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1
    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


@njit
def np_clip(a, a_min, a_max, out=None):
    if out is None:
        out = np.empty_like(a)
    out[:] = np.minimum(np.maximum(a, a_min), a_max)
    return out


@njit
def pixel_stat(img, img_mean, img_std, target_mean, target_std, dst_bit=8):
    target_mean = target_mean / 100.0 * 2 ** dst_bit
    target_mean = target_std / 100.0 * 2 ** dst_bit
    image = (((img - img_mean) / img_std) * target_std) + target_mean
    image = np_clip(image, 0, 2 ** dst_bit - 1)
    return image


@njit(parallel=True)
def mean_std_array(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    [n, a, b] = data.shape

    mean_array = np.zeros((a, b), dtype=np.float32)
    std_array = np.zeros((a, b), dtype=np.float32)

    # Welford's online algorithm
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    for i in prange(a):
        for j in prange(b):
            mean = 0.0
            mean_sq = 0.0
            for k in range(n):
                count = k + 1
                new_value = data[k, i, j]
                delta = new_value - mean
                mean += delta / count
                delta2 = new_value - mean
                mean_sq += delta * delta2
            mean_array[i, j] = mean
            std_array[i, j] = math.sqrt(mean_sq / n)
    return mean_array, std_array


def memmap_loader(image_list, memmap_handle, idx, new_width, new_height):
    np_im = imageio.imread(image_list[idx])
    im2 = cv2.resize(np_im, (new_width, new_height), cv2.INTER_CUBIC)
    memmap_handle[idx, ...] = im2


def correct_image(
    image_raw_mean,
    image_raw_std,
    brightness,
    contrast,
    image_name,
    output_folder,
):
    image = imageio.imread(image_name)
    (image_height, image_width, image_channels) = image.shape

    output_image = np.empty((image_height, image_width, image_channels))

    intensities = None
    for i in range(image_channels):
        if image_channels == 3:
            intensities = image[:, :, i]
        else:
            intensities = image[:, :]
        intensities = pixel_stat(
            intensities,
            image_raw_mean[i],
            image_raw_std[i],
            brightness,
            contrast,
        )
        if image_channels == 3:
            output_image[:, :, i] = intensities
        else:
            output_image[:, :] = intensities

    # apply scaling to 8 bit and format image to unit8
    output_image_bt = bytescaling(output_image)
    filename = Path(output_folder) / image_name.name
    imageio.imwrite(filename, output_image_bt)


parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to images.")
parser.add_argument("extension", help="extension of images (e.g. jpg, png)")
parser.add_argument(
    "output_folder", help="Output folder to write processed images"
)
args = parser.parse_args()

output_folder = Path(args.output_folder)
if not output_folder.exists():
    output_folder.mkdir(parents=True, exist_ok=True)

image_folder = Path(args.path)
image_list = [p for p in image_folder.glob("*." + args.extension)]
print("Found", len(image_list), "images")

tmp_image = imageio.imread(image_list[0])
(orig_image_height, orig_image_width, image_channels) = tmp_image.shape
print(
    "Images are",
    orig_image_width,
    "x",
    orig_image_height,
    "with",
    image_channels,
    "channels",
)

image_height = int(scale_factor * float(orig_image_height))
image_width = int(scale_factor * float(orig_image_width))

print("Scaling to", image_width, "x", image_height)

image_raw_mean = np.empty(
    (image_channels, orig_image_height, orig_image_width)
)
image_raw_std = np.empty((image_channels, orig_image_height, orig_image_width))

image_raw_mean_scaled = np.empty((image_channels, image_height, image_width))
image_raw_std_scaled = np.empty((image_channels, image_height, image_width))

# Subsammple the list:
total, used, free = shutil.disk_usage("/")
free = free // (2 ** 30)

print("Free disk space: ", free, "Gb")
if free < max_space_gb:
    print(
        "Free disk space is below",
        max_space_gb,
        "Gb. you might have problems.",
    )

num_images_to_compute_mean = int(
    max_space_gb
    / (image_height * image_width * image_channels * 8 / (1024 ** 3))
)

print(
    "In a max space of",
    max_space_gb,
    "Gb we can fit",
    num_images_to_compute_mean,
    "images.",
)

image_list_sampled = []

if num_images_to_compute_mean >= len(image_list):
    image_list_sampled = image_list
else:
    if not use_random_sample:
        increment = int(len(image_list) / num_images_to_compute_mean) + 1
        i = 0
        while i < len(image_list):
            image_list_sampled.append(image_list[i])
            i += increment
    else:
        image_list_sampled = random.sample(
            image_list, num_images_to_compute_mean
        )

filename_map = "memmap_" + str(uuid.uuid4()) + ".map"
list_shape = [
    len(image_list_sampled),
    image_height,
    image_width,
    image_channels,
]

size = 1
for i in list_shape:
    size *= i

print(
    "Creating memmap of",
    size * 8 / (1024 ** 3),
    "Gb on the filesystem. Do not worry, it will be deleted later.",
)
image_memmap = np.memmap(
    filename=filename_map, mode="w+", shape=tuple(list_shape), dtype=np.float
)
with tqdm_joblib(
    tqdm(desc="Loading images to memmap", total=len(image_list_sampled))
) as progress_bar:
    joblib.Parallel(n_jobs=-2, verbose=0)(
        joblib.delayed(memmap_loader)(
            image_list_sampled, image_memmap, idx, image_width, image_height
        )
        for idx in range(len(image_list_sampled))
    )

print("Computing global mean and std...")
for i in range(image_channels):
    image_memmap_per_channel = None
    if image_channels == 1:
        image_memmap_per_channel = image_memmap
    else:
        image_memmap_per_channel = image_memmap[:, :, :, i]
    raw_image_mean, raw_image_std = mean_std_array(image_memmap_per_channel)
    image_raw_mean_scaled[i] = raw_image_mean
    image_raw_std_scaled[i] = raw_image_std

    image_raw_mean_scaled_t = image_raw_mean_scaled.transpose(1, 2, 0)
    image_raw_std_scaled_t = image_raw_std_scaled.transpose(1, 2, 0)

    image_raw_mean = cv2.resize(
        image_raw_mean_scaled_t,
        (orig_image_height, orig_image_width),
        cv2.INTER_CUBIC,
    ).transpose(2, 1, 0)
    image_raw_std = cv2.resize(
        image_raw_std_scaled_t,
        (orig_image_height, orig_image_width),
        cv2.INTER_CUBIC,
    ).transpose(2, 1, 0)

    image1 = bytescaling(image_raw_mean)
    image2 = bytescaling(image_raw_std)

    imageio.imwrite("mean.jpg", image1.transpose(1, 2, 0))
    imageio.imwrite("std.jpg", image2.transpose(1, 2, 0))

os.remove(filename_map)

with tqdm_joblib(
    tqdm(desc="Correcting images", total=len(image_list))
) as progress_bar:
    joblib.Parallel(n_jobs=-2, verbose=0)(
        joblib.delayed(correct_image)(
            image_raw_mean,
            image_raw_std,
            brightness,
            contrast,
            image_list[idx],
            output_folder,
        )
        for idx in range(0, len(image_list))
    )

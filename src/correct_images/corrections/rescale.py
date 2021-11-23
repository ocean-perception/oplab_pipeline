# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import os
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange

from oplab import Console, MonoCamera, get_processed_folder


def rescale(
    image_array: np.ndarray,
    interpolate_method: str,
    target_pixel_size_m: float,
    altitude: float,
    f_x: float,
    f_y: float,
    maintain_pixels: bool,
) -> np.ndarray:
    image_shape = image_array.shape
    image_height = image_shape[0]
    image_width = image_shape[1]
    pixel_height = altitude / f_y
    pixel_width = altitude / f_x
    vertical_rescale = pixel_height / target_pixel_size_m
    horizontal_rescale = pixel_width / target_pixel_size_m

    method = None
    if interpolate_method == "bicubic":
        method = Image.BICUBIC
    elif interpolate_method == "bilinear":
        method = Image.BILINEAR
    elif interpolate_method == "nearest_neighbour":
        method = Image.NEAREST
    elif interpolate_method == "lanczos":
        method = Image.LANCZOS
    else:
        Console.error("The requested rescaling method is not implemented.")
        Console.error("Valid methods are: ")
        Console.error("  * bicubic")
        Console.error("  * bilinear")
        Console.error("  * nearest_neighbour")
        Console.error("  * lanczos")
        Console.quit("Rescaling method not implemented.")

    image_rgb = Image.fromarray(image_array, "RGB")

    if maintain_pixels:
        if vertical_rescale < 1 or horizontal_rescale < 1:
            size = (
                int(image_width * horizontal_rescale),
                int(image_height * vertical_rescale),
            )
            image_rgb = image_rgb.resize(size, resample=method)
            size = (image_width, image_height)
            image_rgb = image_rgb.resize(size, resample=method)
        else:
            crop_width = int((1 / horizontal_rescale) * image_width)
            crop_height = int((1 / vertical_rescale) * image_height)

            # find crop box dimensions
            box_left = int((image_width - crop_width) / 2)
            box_upper = int((image_height - crop_height) / 2)
            box_right = image_width - box_left
            box_lower = image_height - box_upper

            # crop the image to the center
            box = (box_left, box_upper, box_right, box_lower)
            cropped_image = image_rgb.crop(box)

            # resize the cropped image to the size of original image
            size = (image_width, image_height)
            image_rgb = cropped_image.resize(size, resample=method)
    else:
        size = (
            int(image_width * horizontal_rescale),
            int(image_height * vertical_rescale),
        )
        image_rgb = image_rgb.resize(size, resample=method)

    image = np.array(image_rgb, dtype=np.uint8)
    return image


def rescale_images(
    imagenames_list,
    image_directory,
    interpolate_method,
    target_pixel_size_m,
    dataframe,
    output_directory,
    f_x,
    f_y,
    maintain_pixels,
):
    Console.info("Rescaling images...")

    for idx in trange(len(imagenames_list)):
        image_name = imagenames_list[idx]
        source_image_path = Path(image_directory) / image_name
        output_image_path = Path(output_directory) / image_name
        image_path_list = dataframe["relative_path"]
        trimmed_path_list = [
            path for path in image_path_list if Path(path).stem in image_name
        ]
        trimmed_dataframe = dataframe.loc[
            dataframe["relative_path"].isin(trimmed_path_list)
        ]
        altitude = trimmed_dataframe["altitude [m]"]
        if len(altitude) > 0:
            image = imageio.imread(source_image_path).astype("uint8")
            rescaled_image = rescale(
                image,
                interpolate_method,
                target_pixel_size_m,
                altitude,
                f_x,
                f_y,
                maintain_pixels,
            )
            imageio.imwrite(output_image_path, rescaled_image, format="PNG-FI")
        else:
            Console.warn("Did not get distance values for image: " + image_name)


def rescale_camera(path, camera_system, camera):
    name = camera.camera_name
    distance_path = camera.distance_path
    interpolate_method = camera.interpolate_method
    image_path = camera.path
    target_pixel_size = camera.target_pixel_size
    maintain_pixels = bool(camera.maintain_pixels)
    output_folder = camera.output_folder

    idx = [i for i, camera in enumerate(camera_system.cameras) if camera.name == name]

    if len(idx) > 0:
        Console.info("Camera found in camera.yaml file...")
    else:
        Console.warn(
            "Camera not found in camera.yaml file. Please provide a relevant \
            camera.yaml file..."
        )
        return False

    # obtain images to be rescaled
    path_processed = get_processed_folder(path)
    image_path = path_processed / image_path

    # obtain path to distance / altitude values
    full_distance_path = path_processed / distance_path
    full_distance_path = full_distance_path / "csv" / "ekf"
    distance_file = "auv_ekf_" + name + ".csv"
    distance_path = full_distance_path / distance_file

    # obtain focal lengths from calibration file
    camera_params_folder = path_processed / "calibration"
    camera_params_filename = "mono_" + name + ".yaml"
    camera_params_file_path = camera_params_folder / camera_params_filename

    if not camera_params_file_path.exists():
        Console.quit("Calibration file not found...")
    else:
        Console.info("Calibration file found...")

    monocam = MonoCamera(camera_params_file_path)
    focal_length_x = monocam.K[0, 0]
    focal_length_y = monocam.K[1, 1]

    # create output path
    output_directory = path_processed / output_folder
    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    # call rescale function
    dataframe = pd.read_csv(Path(distance_path))
    imagenames_list = [
        filename
        for filename in os.listdir(image_path)
        if filename[-4:] == ".jpg" or filename[-4:] == ".png" or filename[-4:] == ".tif"
    ]
    Console.info("Distance values loaded...")
    rescale_images(
        imagenames_list,
        image_path,
        interpolate_method,
        target_pixel_size,
        dataframe,
        output_directory,
        focal_length_x,
        focal_length_y,
        maintain_pixels,
    )
    return True

# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from pathlib import Path

import joblib
import numpy as np
import psutil
from matplotlib import pyplot as plt
from tqdm import tqdm

from correct_images.tools.curve_fitting import curve_fitting
from correct_images.tools.joblib_tqdm import tqdm_joblib
from oplab import Console


def attenuation_correct(
    img: np.ndarray,
    altitude: np.ndarray,
    atn_crr_params: np.ndarray,
    gain: np.ndarray,
) -> np.ndarray:
    """apply attenuation coefficients to an input image

    Parameters
    -----------
    img : numpy.ndarray
        input image
    altitude :
        distance matrix corresponding to the image
    atn_crr_params : numpy.ndarray
        attenuation coefficients
    gain : numpy.ndarray
        gain value for the image

    Returns
    -------
    numpy.ndarray
        Corrected image
    """
    # attenuation correction and gains matrices start with the channel
    # so we need to remove that first layer
    # (e.g. [1, 1024, 1280, 3] -> [1024, 1280, 3]])

    img_float32 = img.astype(np.float32)

    dims = img_float32.shape

    is_rgb = False
    if len(dims) == 3:
        is_rgb = True

    if is_rgb:
        for i_channel in range(atn_crr_params.shape[0]):
            atn_crr_params_ch = atn_crr_params[i_channel]
            gain_ch = gain[i_channel]
            img_float32[:, :, i_channel] = (
                gain_ch
                / (
                    atn_crr_params_ch[:, :, 0]
                    * np.exp(atn_crr_params_ch[:, :, 1] * altitude)
                    + atn_crr_params_ch[:, :, 2]
                )
            ) * img_float32[:, :, i_channel]
    else:
        img_float32[:, :] = (
            gain[0]
            / (
                atn_crr_params[0, :, :, 0]
                * np.exp(atn_crr_params[0, :, :, 1] * altitude)
                + atn_crr_params[0, :, :, 2]
            )
        ) * img_float32[:, :]
    return img_float32


# compute gain values for each pixel for a targeted altitude using the
# attenuation parameters
def calculate_correction_gains(
    target_altitude: np.ndarray,
    attenuation_parameters: np.ndarray,
    image_height: int,
    image_width: int,
    image_channels: int,
) -> np.ndarray:
    """Compute correction gains for an image

    Parameters
    -----------
    target_altitude : numpy.ndarray
        target distance for which the images will be corrected
    attenuation_parameters : numpy.ndarray
        attenuation coefficients

    Returns
    -------
    numpy.ndarray
        The correction gains
    """

    image_correction_gains = np.empty(
        (image_channels, image_height, image_width), dtype=np.float64
    )

    for i_channel in range(image_channels):

        # attenuation_parameters = attenuation_parameters.squeeze()
        correction_gains = (
            attenuation_parameters[i_channel, :, :, 0]
            * np.exp(attenuation_parameters[i_channel, :, :, 1] * target_altitude)
            + attenuation_parameters[i_channel, :, :, 2]
        )
        image_correction_gains[i_channel] = correction_gains
    return image_correction_gains


# calculate image attenuation parameters
def calculate_attenuation_parameters(
    images: np.ndarray,
    distances: np.ndarray,
    image_height: int,
    image_width: int,
    image_channels: int,
    output_folder: Path,
):
    """Compute attenuation parameters for all images

    Parameters
    -----------
    images : numpy.ndarray
        image memmap reshaped as a vector
    distances : numpy.ndarray
        distance memmap reshaped as a vector
    image_height : int
        height of an image
    image_width : int
        width of an image

    Returns
    -------
    numpy.ndarray
        attenuation_parameters
    """

    image_attenuation_parameters = np.empty(
        (image_channels, image_height, image_width, 3), dtype=np.float32
    )

    # Check available RAM and allocate threads accordingly
    mem = psutil.virtual_memory()
    available_bytes = mem.available  # in bytes
    required_bytes = image_channels * image_height * image_width * 4 * len(images)
    num_jobs = min(int(available_bytes / required_bytes), 12)

    # Keep one alive!
    cpus = psutil.cpu_count() - 1

    if num_jobs > cpus:
        num_jobs = cpus
    elif num_jobs <= 0:
        num_jobs = 1
        Console.warn("You might have not enough available RAM to continue.")

    if num_jobs < cpus - 1:
        Console.info("Assigning", num_jobs, "jobs to your CPU to save RAM")
    else:
        Console.info("Assigning", num_jobs, "jobs to your CPU")

    for i_channel in range(image_channels):
        # Populate paths for plots of intensities curves
        if image_channels > 1:
            channel_str = "c_" + str(i_channel) + "_"
        else:
            channel_str = ""
        figure_paths = []
        slope = image_height / image_width
        for y in range(image_height):
            x_diag = round(y / slope)
            for x in range(image_width):
                if x == x_diag:
                    # On image diagonal -> generate path so figure is written to file
                    filename = f"intensities_curve_{channel_str}x_{x:04}_y_{y:04}.png"
                    figure_paths.append(output_folder / Path(filename))
                else:
                    # Not on diagonal -> don't output figure / generate path for file
                    figure_paths.append(None)
        with tqdm_joblib(tqdm(desc="Curve fitting", total=image_height * image_width)):
            results = joblib.Parallel(n_jobs=num_jobs, verbose=0)(
                [
                    joblib.delayed(curve_fitting)(
                        distances[:, i_pixel],
                        images[:, i_pixel, i_channel],
                        figure_paths[i_pixel],
                    )
                    for i_pixel in range(image_height * image_width)
                ]
            )

            attenuation_parameters = np.array(results)
            attenuation_parameters = attenuation_parameters.reshape(
                [image_height, image_width, 3]
            )
        image_attenuation_parameters[i_channel] = attenuation_parameters

    return image_attenuation_parameters


def save_attenuation_plots(
    output_dir, attn=None, gains=None, img_mean=None, img_std=None
):
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if gains is not None:
        fig = plt.figure()
        plt.imshow(gains[0, :, :])
        plt.colorbar()
        plt.title("Gain")
        plt.savefig(output_dir / "gain.png", dpi=600)
        plt.close(fig)

    if attn is not None:
        fig = plt.figure()
        plt.imshow(attn[0, :, :, 0])
        plt.colorbar()
        plt.title("Attenuation coeff 0")
        plt.savefig(output_dir / "attenuation_coeff_0.png", dpi=600)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(attn[0, :, :, 1])
        plt.colorbar()
        plt.title("Attenuation coeff 1")
        plt.savefig(output_dir / "attenuation_coeff_1.png", dpi=600)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(attn[0, :, :, 2])
        plt.colorbar()
        plt.title("Attenuation coeff 2")
        plt.savefig(output_dir / "attenuation_coeff_2.png", dpi=600)
        plt.close(fig)

    if img_mean is not None:
        fig = plt.figure()
        if len(img_mean.shape) == 3:
            plt.imshow(img_mean[:, :, 0])
        else:
            plt.imshow(img_mean[:, :])
        plt.colorbar()
        plt.title("Mean 0")
        plt.savefig(output_dir / "image_corrected_mean_0.png", dpi=600)
        plt.close(fig)

        if len(img_mean.shape) == 3:
            if img_mean.shape[2] > 1:
                fig = plt.figure()
                plt.imshow(img_mean[:, :, 1])
                plt.colorbar()
                plt.title("Mean 1")
                plt.savefig(output_dir / "image_corrected_mean_1.png", dpi=600)
                plt.close(fig)

            if img_mean.shape[2] > 2:
                fig = plt.figure()
                plt.imshow(img_mean[:, :, 2])
                plt.colorbar()
                plt.title("Mean 2")
                plt.savefig(output_dir / "image_corrected_mean_2.png", dpi=600)
                plt.close(fig)

    if img_std is not None:
        fig = plt.figure()
        if len(img_std.shape) == 3:
            plt.imshow(img_std[:, :, 0])
        else:
            plt.imshow(img_std[:, :])
        plt.colorbar()
        plt.title("Std 0")
        plt.savefig(output_dir / "image_corrected_std_0.png", dpi=600)
        plt.close(fig)

        if len(img_std.shape) == 3:
            if img_std.shape[2] > 1:
                fig = plt.figure()
                plt.imshow(img_std[:, :, 0])
                plt.colorbar()
                plt.title("Std 1")
                plt.savefig(output_dir / "image_corrected_std_1.png", dpi=600)
                plt.close(fig)
            if img_std.shape[2] > 2:
                fig = plt.figure()
                plt.imshow(img_std[:, :, 0])
                plt.colorbar()
                plt.title("Std 2")
                plt.savefig(output_dir / "image_corrected_std_2.png", dpi=600)
                plt.close(fig)

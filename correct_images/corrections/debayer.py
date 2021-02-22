# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from pathlib import Path

import cv2
import joblib
import numpy as np
import tqdm
from correct_images.loaders import default, xviii
from correct_images.tools.file_handlers import write_output_image
from correct_images.tools.joblib_tqdm import tqdm_joblib
from oplab import Console


# convert bayer image to RGB based
# on the bayer pattern for the camera
def debayer(
    image: np.ndarray, pattern: str, src_bit=8, dst_bit=8
) -> np.ndarray:
    """Perform debayering of input image

    Parameters
    -----------
    image : numpy.ndarray
        image data to be debayered
    pattern : string
        bayer pattern

    Returns
    -------
    numpy.ndarray
        Debayered image
    """

    # Make use of 16 bit debayering
    image16_float = image.astype(np.float32) * (2 ** (16 - src_bit))
    image16 = image16_float.astype(np.uint16)

    corrected_rgb_img = None
    if pattern == "rggb" or pattern == "RGGB":
        corrected_rgb_img = cv2.cvtColor(image16, cv2.COLOR_BAYER_BG2RGB_EA)
    elif pattern == "grbg" or pattern == "GRBG":
        corrected_rgb_img = cv2.cvtColor(image16, cv2.COLOR_BAYER_GB2RGB_EA)
    elif pattern == "bggr" or pattern == "BGGR":
        corrected_rgb_img = cv2.cvtColor(image16, cv2.COLOR_BAYER_RG2RGB_EA)
    elif pattern == "gbrg" or pattern == "GBRG":
        corrected_rgb_img = cv2.cvtColor(image16, cv2.COLOR_BAYER_GR2RGB_EA)
    elif pattern == "mono" or pattern == "MONO":
        return image
    else:
        Console.quit("Bayer pattern not supported (", pattern, ")")

    # Scale down to 8 bits.
    corrected_rgb_img = corrected_rgb_img.astype(np.float32) * (
        2 ** (dst_bit - 16)
    )
    return corrected_rgb_img


def debayer_folder(
    output_dir: Path,
    filetype,
    pattern,
    output_format,
    image=None,
    image_dir=None,
) -> None:
    """Perform debayer of input bayer images without going through correction
    pipeline

    Parameters
    -----------
    output_dir : str
        Output directory where to save the debayered images
    filetype : str
        Input filetype (e.g. extension)
    pattern : str
        Bayer pattern ('bggr', 'rggb',...)
    output_format : str
        Output format (e.g. extension)
    image : str
        Optional parameter. Whether to process just one image. Useful to debug
        and check the patterns.
    image_dir : str
        Optional parameter. Folder with images to process.
    """

    def debayer_image(
        image_path, _filetype, _pattern, _output_dir, _output_format
    ):
        Console.info("Debayering image {}".format(image_path.name))

        # Define image loader
        loader = default.loader
        if _filetype == "raw":
            loader = xviii.loader

        img = loader(image_path)
        img_rgb = debayer(img, _pattern)
        img_rgb = img_rgb.astype(np.uint8)
        image_name = str(image_path.stem)
        write_output_image(img_rgb, image_name, _output_dir, _output_format)

    output_dir = Path(output_dir)
    image_list = []
    if not output_dir.exists():
        Console.info("Creating output dir {}".format(output_dir))
        output_dir.mkdir(parents=True)
    else:
        Console.info("Using output dir {}".format(output_dir))
    if not image:
        image_dir = Path(image_dir)
        Console.info(
            "Debayering folder {} to {}".format(image_dir, output_dir)
        )
        image_list = list(image_dir.glob("*." + filetype))
    else:
        single_image = Path(image)
        image_list.append(single_image)
    Console.info("Found " + str(len(image_list)) + " image(s)...")

    with tqdm_joblib(tqdm(desc="Debayering inages", total=len(image_list))):
        joblib.Parallel(n_jobs=-2, verbose=0)(
            [
                joblib.delayed(debayer_image)(
                    image_path, filetype, pattern, output_dir, output_format
                )
                for image_path in image_list
            ]
        )

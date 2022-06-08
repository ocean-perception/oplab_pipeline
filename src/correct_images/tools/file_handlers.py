# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import uuid
from pathlib import Path

import imageio
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from correct_images.tools.joblib_tqdm import tqdm_joblib
from oplab import Console


# functions used to create a trimmed auv_ekf_<camera_name>.csv file based on
# user's selection of images
def trim_csv_files(image_files_paths, original_csv_path, trimmed_csv_path):
    """Trim csv files based on the list of images provided by user

    Parameters
    -----------
    image_files_paths : str
        user provided list of imagenames which need to be processed
    original_csv_path : str
        path to the auv_ekf_<camera_name.csv>
    trimmed_csv_path : str
        path to trimmed csv which needs to be created
    """

    image_name_list = get_imagename_list(image_files_paths)
    dataframe = pd.read_csv(original_csv_path)
    image_path_list = dataframe["relative_path"]
    trimmed_path_list = [
        path for path in image_path_list if Path(path).stem in image_name_list
    ]
    trimmed_dataframe = dataframe.loc[
        dataframe["relative_path"].isin(trimmed_path_list)
    ]
    # trimmed_dataframe.to_csv(trimmed_csv_path, index=False, header=True)
    return trimmed_dataframe


def get_imagename_list(image_files_paths):
    """get list of imagenames from the filelist provided by user

    Parameters
    -----------
    image_files_paths : Path to the filelist provided in correct_images.yaml
    """

    with open(image_files_paths, "r") as image_file:
        image_name_list = image_file.read().splitlines()
    image_name_list = [Path(x).stem for x in image_name_list]
    return image_name_list


# TODO is this used?
# store into memmaps the distance and image numpy files
def load_memmap_from_numpyfilelist(filepath, numpyfilelist: list):
    """Generate memmaps from numpy arrays

    Parameters
    -----------
    filepath : Path
        path to output memmap folder
    numpyfilelist : list
        list of paths to numpy files

    Returns
    --------
    Path, numpy.ndarray
        memmap_path and memmap_handle
    """

    image = np.load(str(numpyfilelist[0]))
    list_shape = [len(numpyfilelist)]
    list_shape = list_shape + list(image.shape)

    filename_map = "memmap_" + str(uuid.uuid4()) + ".map"
    memmap_path = Path(filepath) / filename_map

    memmap_handle = np.memmap(
        filename=memmap_path,
        mode="w+",
        shape=tuple(list_shape),
        dtype=np.float32,
    )
    Console.info("Loading memmaps from numpy files...")

    def memmap_loader(numpyfilelist, memmap_handle, idx):
        memmap_handle[idx, ...] = np.load(numpyfilelist[idx])

    with tqdm_joblib(tqdm(desc="numpy images to memmap", total=len(numpyfilelist))):
        joblib.Parallel(n_jobs=-2, verbose=0)(
            joblib.delayed(memmap_loader)(numpyfilelist, memmap_handle, idx)
            for idx in range(len(numpyfilelist))
        )

    return memmap_path, memmap_handle


# save processed image in an output file with
# given output format
def write_output_image(image, filename, dest_path, dest_format):
    """Write into output images

    Parameters
    -----------
    image : numpy.ndarray
        image data to be written
    filename : string
        name of output image file
    dest_path : Path
        path to the output folder
    dest_format : string
        output image format
    """

    file = filename + "." + dest_format
    file_path = dest_path / file
    ch = image.shape[0]
    if ch == 3:
        image = image.transpose((1, 2, 0))
    imageio.imwrite(file_path, image)
    return file_path

# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import os
from pathlib import Path

import pandas as pd
import glob

from oplab import Console, get_raw_folder
from oplab import FilenameToDate


def parse_voyis_images(mission, vehicle, category, ftype, outpath):
    # parser meta data
    class_string = "measurement"
    frame_string = "body"
    sensor_string = "voyis"

    stills_filepath = Path(mission.image.cameras[0].path)
    laser_filepath = Path(mission.image.cameras[1].path)

    stills_format = 'xxxxxxxxxxxxxxxxxxxYYYYxMMxDDxhhmmssxfffuuuxx.xxx'
    laser_format = 'xxxxxxxxxxxxxxxxxxxxYYYYxMMxDDxhhmmssxfffuuuxx.xxx'

    stills_filename_to_date = FilenameToDate(stills_format)
    laser_filename_to_date = FilenameToDate(laser_format)

    Console.info("... parsing " + sensor_string + " images")

    # Find all *.tif images in stills_filepath and subfolders
    stills_image_list = glob.glob(str(stills_filepath)+'/**/*.tif', recursive=True)
    laser_image_list = glob.glob(str(laser_filepath)+'/**/*.tif', recursive=True)
    data_list = []
    for img in stills_image_list:
        epoch = stills_filename_to_date(str(img.name))
        data = {
            "epoch_timestamp": float(epoch),
            "class": class_string,
            "sensor": sensor_string,
            "frame": frame_string,
            "category": "image",
            "camera1": [
                {
                    "epoch_timestamp": float(epoch),
                    "filename": str(img),
                }
            ],
        }
        data_list.append(data)
    for img in laser_image_list:
        epoch = laser_filename_to_date(str(img.name))
        data = {
            "epoch_timestamp": float(epoch),
            "class": class_string,
            "sensor": sensor_string,
            "frame": frame_string,
            "category": "laser",
            "camera1": [
                {
                    "epoch_timestamp": float(epoch),
                    "filename": str(img),
                }
            ],
        }
        data_list.append(data)
    return data_list
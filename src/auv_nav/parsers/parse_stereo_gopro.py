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

from oplab import Console, get_raw_folder


def parse_stereo_gopro_images(mission, vehicle, category, ftype, outpath):
    # parser meta data
    class_string = "measurement"
    frame_string = "body"
    category = "image"
    sensor_string = "stereo_gopro"

    camera1_filepath = Path(mission.image.cameras[0].path)
    camera2_filepath = Path(mission.image.cameras[1].path)

    Console.info("... parsing " + sensor_string + " images")

    # Read filelist.csv
    df = pd.read_csv(get_raw_folder(camera1_filepath.parent / "filelist.csv"))

    data_list = []

    for _, row in df.iterrows():
        epoch = row["epoch"]
        file_left = row["file_left"]
        file_right = row["file_right"]
        # Check files exist
        if not os.path.isfile(camera1_filepath / file_left):
            Console.warn("File " + file_left + " does not exist")
            continue
        if not os.path.isfile(camera2_filepath / file_right):
            Console.warn("File " + file_right + " does not exist")
            continue
        data = {
            "epoch_timestamp": float(epoch),
            "class": class_string,
            "sensor": sensor_string,
            "frame": frame_string,
            "category": category,
            "camera1": [
                {
                    "epoch_timestamp": float(epoch),
                    "filename": str(camera1_filepath / file_left),
                }
            ],
            "camera2": [
                {
                    "epoch_timestamp": float(epoch),
                    "filename": str(camera2_filepath / file_right),
                }
            ],
        }
        data_list.append(data)
    return data_list

# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import glob
from pathlib import Path

from oplab import Console, FilenameToDate, get_raw_folder
from auv_nav.tools.time_conversions import read_timezone

def pathlist_relativeto(input_pathlist, base_path):
    out_list = []
    for x in input_pathlist:
        p = Path(x)
        pr = p.relative_to(base_path)
        out_list.append(str(pr))
    return out_list

def parse_images(mission, vehicle, category, ftype, outpath):
    # parser meta data
    class_string = "measurement"
    frame_string = "body"
    sensor_string = "unknown"

    dive_folder = get_raw_folder(outpath.parent)

    stills_filepath = dive_folder / Path(mission.image.cameras[0].path)
    Console.info("... parsing " + sensor_string + " images")

    # Find all *.tif images in stills_filepath and subfolders
    stills_image_list = glob.glob(str(stills_filepath) + "/**/*.jpg", recursive=True)
    stills_image_list_rel = pathlist_relativeto(stills_image_list, dive_folder)
    Console.info(f" .. found {len(stills_image_list)} stills images in {stills_filepath}")

    data_list = []
    timezone = mission.image.timezone
    timeoffset = mission.image.timeoffset_s

    timezone_offset_h = read_timezone(timezone)
    timeoffset_s = -timezone_offset_h * 60 * 60 - timeoffset
    for i, img in enumerate(stills_image_list):
        string_epoch_time = str(Path(img).stem)[12:22]+"."+str(Path(img).stem)[23:29]
        data = {
            "epoch_timestamp": float(string_epoch_time)+timeoffset_s,
            "class": class_string,
            "sensor": sensor_string,
            "frame": frame_string,
            "category": "image",
            "camera1": [
                {
                    "epoch_timestamp": float(string_epoch_time)+timeoffset_s,
                    "filename": str(stills_image_list_rel[i]),
                }
            ],
        }
        data_list.append(data)

    return data_list

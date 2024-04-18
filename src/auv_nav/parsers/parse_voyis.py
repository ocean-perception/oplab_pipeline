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


def pathlist_relativeto(input_pathlist, base_path):                   
    out_list = []                                                       
    for x in input_pathlist:                                                   
        p = Path(x)                                      
        pr = p.relative_to(base_path)                                       
        out_list.append(str(pr))                                            
    return out_list


def parse_voyis_images(mission, vehicle, category, ftype, outpath):
    # parser meta data
    class_string = "measurement"
    frame_string = "body"
    sensor_string = "voyis"

    dive_folder = get_raw_folder(outpath.parent)

    stills_filepath = dive_folder / Path(mission.image.cameras[0].path)
    laser_filepath = dive_folder / Path(mission.image.cameras[1].path)

    stills_format = "xxxxxxxxxxxxxxxxxxxYYYYxMMxDDxhhmmssxfffuuuxx.xxx"
    laser_format = "xxxxxxxxxxxxxxxxxxxxYYYYxMMxDDxhhmmssxfffuuuxx.xxx"

    stills_filename_to_date = FilenameToDate(stills_format)
    laser_filename_to_date = FilenameToDate(laser_format)

    Console.info("... parsing " + sensor_string + " images")

    # Find all *.tif images in stills_filepath and subfolders
    stills_image_list = glob.glob(str(stills_filepath) + "/**/*.tif", recursive=True)
    laser_image_list = glob.glob(str(laser_filepath) + "/**/*.tif", recursive=True)

    stills_image_list_rel = pathlist_relativeto(stills_image_list, dive_folder)
    laser_image_list_rel = pathlist_relativeto(laser_image_list, dive_folder)

    Console.info(f" .. found {len(stills_image_list)} stills images in {stills_filepath}")
    Console.info(f" .. found {len(laser_image_list)} laser images in {laser_filepath}")

    data_list = []
    for i, img in enumerate(stills_image_list):
        epoch = stills_filename_to_date(str(Path(img).name))
        data = {
            "epoch_timestamp": float(epoch),
            "class": class_string,
            "sensor": sensor_string,
            "frame": frame_string,
            "category": "image",
            "camera1": [
                {
                    "epoch_timestamp": float(epoch),
                    "filename": str(stills_image_list_rel[i]),
                }
            ],
        }
        data_list.append(data)
    for img in laser_image_list:
        epoch = laser_filename_to_date(str(Path(img).name))
        data = {
            "epoch_timestamp": float(epoch),
            "class": class_string,
            "sensor": sensor_string,
            "frame": frame_string,
            "category": "laser",
            "camera1": [
                {
                    "epoch_timestamp": float(epoch),
                    "filename": str(laser_image_list_rel[i]),
                }
            ],
        }
        data_list.append(data)
    return data_list

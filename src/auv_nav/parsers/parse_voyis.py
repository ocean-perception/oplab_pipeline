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


def parse_voyis_images(mission, vehicle, category, ftype, outpath):
    # parser meta data
    class_string = "measurement"
    frame_string = "body"
    sensor_string = "voyis"

    dive_folder = get_raw_folder(outpath.parent)

    stills_filepath = dive_folder / Path(mission.image.cameras[0].path)
    laser_filepath = dive_folder / Path(mission.image.cameras[1].path)

    stills_format_PPS = "xxxxxxxxxxxxxxxNNNxYYYYxMMxDDxhhmmssxfffuuuxx.xxx"                            
    stills_format_SYS = "xxxxxxxxxxxxxxxNNNNNNxYYYYxMMxDDxhhmmssxfffuuuxx.xxx"                     

    laser_format_PPS  = "xxxxxxxxxxxxxxxxNNNxYYYYxMMxDDxhhmmssxfffuuuxx.xxx"
    laser_format_SYS  = "xxxxxxxxxxxxxxxxNNNNNNxYYYYxMMxDDxhhmmssxfffuuuxx.xxx"



    stills_filename_to_date_PPS = FilenameToDate(stills_format_PPS)
    laser_filename_to_date_PPS = FilenameToDate(laser_format_PPS)

    stills_filename_to_date_SYS = FilenameToDate(stills_format_SYS)
    laser_filename_to_date_SYS = FilenameToDate(laser_format_SYS)

    Console.info("... parsing " + sensor_string + " images")

    # Find all *.tif images in stills_filepath and subfolders
    stills_image_list = glob.glob(str(stills_filepath) + "/**/*.tif", recursive=True)
    laser_image_list = glob.glob(str(laser_filepath) + "/**/*.tif", recursive=True)

    stills_image_list_rel = pathlist_relativeto(stills_image_list, dive_folder)
    laser_image_list_rel = pathlist_relativeto(laser_image_list, dive_folder)

    Console.info(f" .. found {len(stills_image_list)} stills images in {stills_filepath}")
    Console.info(f" .. found {len(laser_image_list)} laser images in {laser_filepath}")

    data_list = []
    timezone = mission.image.timezone
    timeoffset = mission.image.timeoffset_s

    timezone_offset_h = read_timezone(timezone)
    timeoffset_s = -timezone_offset_h * 60 * 60 - timeoffset
    for i, img in enumerate(stills_image_list):
        if str(Path(img).name)[15:18]=='PPS':
            epoch = stills_filename_to_date_PPS(str(Path(img).name))
        elif str(Path(img).name)[15:18]=='SYSTEM':
            epoch = stills_filename_to_date_SYS(str(Path(img).name))
        data = {
            "epoch_timestamp": float(epoch)+timeoffset_s,
            "class": class_string,
            "sensor": sensor_string,
            "frame": frame_string,
            "category": "image",
            "camera1": [
                {
                    "epoch_timestamp": float(epoch)+timeoffset_s,
                    "filename": str(stills_image_list_rel[i]),
                }
            ],
        }
        data_list.append(data)
    for img in laser_image_list:
        if str(Path(img).name[16:19])=='PPS':
            epoch = laser_filename_to_date_PPS(str(Path(img).name))
        elif str(Path(img).name[16:19])=='SYSTEM':
            epoch = laser_filename_to_date_SYS(str(Path(img).name))        
        data = {
            "epoch_timestamp": float(epoch)+timeoffset_s,
            "class": class_string,
            "sensor": sensor_string,
            "frame": frame_string,
            "category": "laser",
            "camera1": [
                {
                    "epoch_timestamp": float(epoch)+timeoffset_s,
                    "filename": str(laser_image_list_rel[i]),
                }
            ],
        }        
        data_list.append(data)
    return data_list

# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import glob
from pathlib import Path
import yaml
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
    
    camera_yaml = dive_folder / "camera.yaml"
    
    stills_flag = False
    laser_flag = False    
    
    with open(camera_yaml,"r") as file:
        camera = yaml.safe_load(file)
        for i in range(len(camera["cameras"])):
            if camera["cameras"][i]["name"] == "stills":
                stills_filepath = dive_folder / camera["cameras"][i]["path"]
                stills_format_PPS = camera["cameras"][i]["filename_to_date_pps"]
                stills_format_SYS = camera["cameras"][i]["filename_to_date_system"]
                stills_flag = True
            if camera["cameras"][i]["name"] == "laser":
                laser_filepath = dive_folder / camera["cameras"][i]["path"]
                laser_format_PPS = camera["cameras"][i]["filename_to_date_pps"]        	
                laser_format_SYS = camera["cameras"][i]["filename_to_date_system"]        	
                laser_flag = True

    Console.info("... parsing " + sensor_string + " images")

    data_list = []
    timezone = mission.image.timezone
    timeoffset = mission.image.timeoffset_s

    timezone_offset_h = read_timezone(timezone)
    timeoffset_s = -timezone_offset_h * 60 * 60 - timeoffset

    if stills_flag == True:
        stills_filename_to_date_PPS = FilenameToDate(stills_format_PPS)
        stills_filename_to_date_SYS = FilenameToDate(stills_format_SYS)
	# Find all *.tif images in stills_filepath and subfolders
        stills_image_list = glob.glob(str(stills_filepath) + "/**/*.tif", recursive=True)
        stills_image_list_rel = pathlist_relativeto(stills_image_list, dive_folder)
        Console.info(f" .. found {len(stills_image_list)} stills images in {stills_filepath}")

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



    if laser_flag == True:        
        laser_filename_to_date_PPS = FilenameToDate(laser_format_PPS)
        laser_filename_to_date_SYS = FilenameToDate(laser_format_SYS)
        laser_image_list = glob.glob(str(laser_filepath) + "/**/*.tif", recursive=True)
        laser_image_list_rel = pathlist_relativeto(laser_image_list, dive_folder)
        Console.info(f" .. found {len(laser_image_list)} laser images in {laser_filepath}")
        for i, img in enumerate(laser_image_list):
            if str(Path(img).name[16:19])=='PPS':
                epoch = laser_filename_to_date_PPS(str(Path(img).name))
            elif str(Path(img).name[16:19])=='SYSTEM':
                epoch = laser_filename_to_date_SYS(str(Path(img).name))        
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

     
    return data_list

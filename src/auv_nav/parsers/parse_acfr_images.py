# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

# Scripts to parse acfr image acquisition data

# Author: Blair Thornton
# Date: 31/08/2017

import os

from auv_nav.tools.time_conversions import date_time_to_epoch
from oplab import Console, get_raw_folder

epoch_timestamp_camera1 = []
epoch_timestamp_camera2 = []
values = []
data_list = []
tolerance = 0.05  # 0.01 # stereo pair must be within 10ms of each other


def acfr_timestamp_from_filename(filename, timezone_offset, timeoffset):
    filename_split = filename.strip().split("_")
    date_string = filename_split[1]
    time_string = filename_split[2]
    ms_time_string = filename_split[3]

    # read in date
    yyyy = int(date_string[0:4])
    mm = int(date_string[4:6])
    dd = int(date_string[6:8])

    # read in time
    hour = int(time_string[0:2])
    mins = int(time_string[2:4])
    secs = int(time_string[4:6])
    msec = int(ms_time_string[0:3])

    epoch_time = date_time_to_epoch(yyyy, mm, dd, hour, mins, secs, timezone_offset)
    # dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
    # time_tuple = dt_obj.timetuple()
    # epoch_time = time.mktime(time_tuple)
    epoch_timestamp = float(epoch_time + msec / 1000 + timeoffset)
    return epoch_timestamp


def parse_acfr_images(mission, vehicle, category, ftype, outpath):
    # parser meta data
    class_string = "measurement"
    frame_string = "body"
    category = "image"
    sensor_string = "acfr_standard"

    timezone = mission.image.timezone
    timeoffset = mission.image.timeoffset
    filepath = mission.image.cameras[0].path
    camera1_label = mission.image.cameras[0].name
    camera2_label = mission.image.cameras[1].name

    # read in timezone
    if isinstance(timezone, str):
        if timezone == "utc" or timezone == "UTC":
            timezone_offset = 0
        elif timezone == "jst" or timezone == "JST":
            timezone_offset = 9
    else:
        try:
            timezone_offset = float(timezone)
        except ValueError:
            print(
                "Error: timezone",
                timezone,
                "in mission.cfg not recognised, please enter value from UTC",
                "in hours",
            )
            return

    # convert to seconds from utc
    # timeoffset = -timezone_offset*60*60 + timeoffset

    Console.info("... parsing " + sensor_string + " images")

    # determine file paths

    filepath = get_raw_folder(outpath / ".." / filepath)
    all_list = os.listdir(str(filepath))

    camera1_filename = [
        line
        for line in all_list
        if camera1_label in line and ".txt" not in line and "._" not in line
    ]
    camera2_filename = [
        line
        for line in all_list
        if camera2_label in line and ".txt" not in line and "._" not in line
    ]

    data_list = []
    if ftype == "acfr":
        data_list = ""

    for i in range(len(camera1_filename)):
        epoch_timestamp = acfr_timestamp_from_filename(
            camera1_filename[i], timezone_offset, timeoffset
        )
        epoch_timestamp_camera1.append(str(epoch_timestamp))

    for i in range(len(camera2_filename)):
        epoch_timestamp = acfr_timestamp_from_filename(
            camera2_filename[i], timezone_offset, timeoffset
        )
        epoch_timestamp_camera2.append(str(epoch_timestamp))

    for i in range(len(camera1_filename)):
        # print(epoch_timestamp_camera1[i])
        values = []
        for j in range(len(camera2_filename)):
            # print(epoch_timestamp_camera2[j])
            values.append(
                abs(
                    float(epoch_timestamp_camera1[i])
                    - float(epoch_timestamp_camera2[j])
                )
            )
        (sync_difference, sync_pair) = min((v, k) for k, v in enumerate(values))
        if sync_difference > tolerance:
            # Skip the pair
            continue
        if ftype == "oplab":
            data = {
                "epoch_timestamp": float(epoch_timestamp_camera1[i]),
                "class": class_string,
                "sensor": sensor_string,
                "frame": frame_string,
                "category": category,
                "camera1": [
                    {
                        "epoch_timestamp": float(epoch_timestamp_camera1[i]),
                        "filename": str(filepath) + "/" + str(camera1_filename[i]),
                    }
                ],
                "camera2": [
                    {
                        "epoch_timestamp": float(epoch_timestamp_camera2[sync_pair]),
                        "filename": str(filepath)
                        + "/"
                        + str(camera2_filename[sync_pair]),
                    }
                ],
            }
            data_list.append(data)
        if ftype == "acfr":
            data = (
                "VIS: "
                + str(float(epoch_timestamp_camera1[i]))
                + " ["
                + str(float(epoch_timestamp_camera1[i]))
                + "] "
                + str(camera1_filename[i])
                + " exp: 0\n"
            )
            # fileout.write(data)
            data_list += data
            data = (
                "VIS: "
                + str(float(epoch_timestamp_camera2[sync_pair]))
                + " ["
                + str(float(epoch_timestamp_camera2[sync_pair]))
                + "] "
                + str(camera2_filename[sync_pair])
                + " exp: 0\n"
            )
            # fileout.write(data)
            data_list += data

    return data_list

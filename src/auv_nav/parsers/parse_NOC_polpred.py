# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

# Script to parse NOC tide data generated with POLPRED Continental Shelf
# Extended Model (CSX)
# Author: Jose Cappelletto
# Date: 11/09/2019
from auv_nav.sensors import Category, Tide
from auv_nav.tools.time_conversions import date_time_to_epoch, read_timezone
from oplab import Console, get_file_list, get_raw_folder


def parse_NOC_polpred(mission, vehicle, category, ftype, outpath):
    # parser meta data
    sensor_string = "autosub"
    category = category
    output_format = ftype

    if category == Category.TIDE:
        filepath = mission.tide.filepath
        timezone = mission.tide.timezone
        timeoffset = mission.tide.timeoffset
        timezone_offset = read_timezone(timezone)

        tide = Tide(mission.tide.std_offset)
        tide.sensor_string = sensor_string

        path = get_raw_folder(outpath / ".." / filepath)

        file_list = get_file_list(path)

        data_list = []

        Console.info("... parsing NOC tide data")
        # Data format sample
        #  Date      Time      Level     Speed    Direc'n
        #                         m        m/s       deg
        # 6/ 9/2019  00:00       0.74      0.14       51
        # 6/ 9/2019  01:00       0.58      0.15       56

        for file in file_list:
            with file.open("r", errors="ignore") as tide_file:
                for line in tide_file.readlines()[6:]:
                    # we have to skip the first 5 rows of the file
                    dd = int(line[0:2])
                    mm = int(line[3:5])
                    yyyy = int(line[6:10])
                    hour = int(line[12:14])
                    mins = int(line[15:17])
                    secs = 0  # current models only provide resolution in minutes
                    msec = 0
                    epoch_time = date_time_to_epoch(
                        yyyy, mm, dd, hour, mins, secs, timezone_offset
                    )
                    epoch_timestamp = epoch_time + msec / 1000 + timeoffset
                    tide.epoch_timestamp = epoch_timestamp
                    tide.height = float(line[22:28])
                    tide.height_std = tide.height * tide.height_std_factor

                    data = tide.export(output_format)
                    data_list.append(data)
        return data_list

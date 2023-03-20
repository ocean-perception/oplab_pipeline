# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

# Script to parse DY152 tide data exported from CTI / CTI Editor v6.2
# Author: Jose Cappelletto
# Date: 17/07/2022
from auv_nav.sensors import Category, Tide
from auv_nav.tools.time_conversions import date_time_to_epoch, read_timezone
from oplab import Console, get_file_list, get_raw_folder


def parse_tide_CTI(mission, vehicle, category, ftype, outpath):
    # parser meta data
    sensor_string = "alr"
    category = category
    output_format = ftype

    if category == Category.TIDE:
        filepath = mission.tide.filepath
        timezone = mission.tide.timezone
        timeoffset = mission.tide.timeoffset_s
        timezone_offset = read_timezone(timezone)

        tide = Tide(mission.tide.std_offset)
        tide.sensor_string = sensor_string

        path = get_raw_folder(outpath / ".." / filepath)
        file_list = get_file_list(path)

        data_list = []

        # Data comes in two flavors: CSV and XML.
        # XML contains exporter metadata information and tide-only information
        # CSV contains height, current speed and direction information. Let's parse the CSV
        Console.info("... parsing NOC CTI tide data")
        # "Computation Type","Time Series"
        # "Model","GLBF_5HC"
        # "Datum","Mean Sea Level"
        # ""
        # "Date/Time (UTC)","Height (m)","Current Speed (m/s)","Direction (Deg)"
        # 2022-07-09T00:00:00Z,1.11398,0.17887,74
        # 2022-07-09T00:15:00Z,1.13467,0.16603,78
        # 2022-07-09T00:30:00Z,1.13817,0.15147,82
        # 2022-07-09T00:45:00Z,1.12438,0.13565,87
        # 2022-07-09T01:00:00Z,1.09346,0.11924,94
        for file in file_list:
            with file.open("r", errors="ignore") as tide_file:
                for line in tide_file.readlines()[
                    6:
                ]:  # Skip first 5 lines of header information
                    # YYYY-MM-DD
                    yyyy = int(line[0:4])
                    mm = int(line[5:7])
                    dd = int(line[8:10])
                    # Skip one (T)
                    hour = int(line[11:13])
                    mins = int(line[14:16])
                    secs = int(line[17:19])
                    # current models only provide resolution in minutes, but we parse seconds anyways
                    msec = 0
                    epoch_time = date_time_to_epoch(
                        yyyy, mm, dd, hour, mins, secs, timezone_offset
                    )
                    epoch_timestamp = epoch_time + msec / 1000 + timeoffset
                    tide.epoch_timestamp = epoch_timestamp

                    # Warning: NOC/CTI format does not enforce fixed field width.
                    # Use the comma(,) as field delimiters
                    _ini = line.find(",") + 1
                    _end = line.find(",", _ini)
                    tide.height = float(line[_ini:_end])
                    tide.height_std = tide.height * tide.height_std_factor

                    data = tide.export(output_format)
                    data_list.append(data)
        return data_list

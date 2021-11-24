# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import calendar
import math
from datetime import datetime

from auv_nav.sensors import Altitude, BodyVelocity, Category, Orientation
from oplab import Console, get_raw_folder


def parse_rdi(mission, vehicle, category, ftype, outpath):

    # parser meta data
    sensor_string = "rdi"
    output_format = ftype
    filename = ""
    filepath = ""

    timeoffset_s = 0

    # RDI format only has two digits for year,
    # extract the other two from dive folder
    # name
    prepend_year = outpath.parents[0].name[0:2]

    # autosub std models
    velocity_std_factor = mission.velocity.std_factor
    velocity_std_offset = mission.velocity.std_offset
    orientation_std_offset = mission.orientation.std_offset
    altitude_std_factor = mission.altitude.std_factor
    headingoffset = vehicle.dvl.yaw

    bv = BodyVelocity(velocity_std_factor, velocity_std_offset, headingoffset)
    ot = Orientation(headingoffset, orientation_std_offset)
    al = Altitude(altitude_std_factor)

    bv.sensor_string = sensor_string
    ot.sensor_string = sensor_string
    al.sensor_string = sensor_string

    if category == Category.VELOCITY:
        Console.info("... parsing RDI velocity")
        filename = mission.velocity.filename
        filepath = mission.velocity.filepath
        timeoffset_s = mission.velocity.timeoffset_s
    elif category == Category.ORIENTATION:
        Console.info("... parsing RDI orientation")
        filename = mission.orientation.filename
        filepath = mission.orientation.filepath
        timeoffset_s = mission.orientation.timeoffset_s
    elif category == Category.ALTITUDE:
        Console.info("... parsing RDI altitude")
        filename = mission.altitude.filename
        timeoffset_s = mission.altitude.timeoffset_s
        filepath = mission.altitude.filepath

    logfile = get_raw_folder(outpath / ".." / filepath / filename)
    data_list = []
    altitude_valid = False
    with logfile.open("r", errors="ignore") as rdi_file:
        for line in rdi_file.readlines():
            # Lines start with TS, BI, BS, BE, BD, SA
            parts = line.split(",")
            if parts[0] == ":SA" and len(parts) == 4:
                ot.roll = math.radians(float(parts[2]))
                ot.pitch = math.radians(float(parts[1]))
                ot.yaw = math.radians(float(parts[3]))
            elif parts[0] == ":TS" and len(parts) == 7:

                # Every time a new TimeStamp is received, send the previous
                # data packet
                if category == Category.VELOCITY:
                    data = bv.export(output_format)
                    if data is not None:
                        data_list.append(data)
                if category == Category.ORIENTATION:
                    data = ot.export(output_format)
                    if data is not None:
                        data_list.append(data)
                if category == Category.ALTITUDE:
                    data = al.export(output_format)
                    if data is not None:
                        data_list.append(data)

                date = parts[1]
                year = int(prepend_year + date[0:2])
                month = int(date[2:4])
                day = int(date[4:6])
                hour = int(date[6:8])
                minute = int(date[8:10])
                second = int(date[10:12])
                millisecond = float(date[12:14]) * 1e-2
                date = datetime(
                    int(year),
                    int(month),
                    int(day),
                    int(hour),
                    int(minute),
                    int(second),
                )
                stamp = (
                    float(calendar.timegm(date.timetuple()))
                    + millisecond
                    + timeoffset_s
                )
                bv.epoch_timestamp = stamp
                bv.epoch_timestamp_dvl = stamp
                ot.epoch_timestamp = stamp
                al.epoch_timestamp = stamp
                al.altitude_timestamp = stamp
                al.sound_velocity = float(parts[5])
                altitude_valid = False
            elif parts[0] == ":BI" and len(parts) == 6:
                status = parts[5].strip()
                if status == "A":
                    altitude_valid = True
                    x = float(parts[1]) * 0.001
                    y = float(parts[2]) * 0.001
                    bv.x_velocity = x * math.cos(headingoffset) - y * math.sin(
                        headingoffset
                    )
                    bv.y_velocity = x * math.sin(headingoffset) + y * math.cos(
                        headingoffset
                    )
                    bv.z_velocity = float(parts[3]) * 0.001
                    bv.x_velocity_std = float(parts[4]) * 0.001
                    bv.y_velocity_std = float(parts[4]) * 0.001
                    bv.z_velocity_std = float(parts[4]) * 0.001
            elif parts[0] == ":BD" and len(parts) == 6 and altitude_valid:
                al.altitude = float(parts[4])
                al.altitude_std = altitude_std_factor * al.altitude
    # print(data_list)
    return data_list

# -*- coding: utf-8 -*-
"""
Copyright (c) 2021, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from math import isnan

from auv_nav.parsers.load_matlab_file import loadmat
from auv_nav.sensors import Altitude, BodyVelocity, Category, Depth, Orientation
from oplab import Console, get_raw_folder


def parse_alr(mission, vehicle, category, ftype, outpath):
    # parser meta data
    sensor_string = "alr"
    category = category
    output_format = ftype
    filename = mission.velocity.filename
    filepath = mission.velocity.filepath

    # ALR std models
    depth_std_factor = mission.depth.std_factor
    velocity_std_factor = mission.velocity.std_factor
    velocity_std_offset = mission.velocity.std_offset
    orientation_std_offset = mission.orientation.std_offset
    altitude_std_factor = mission.altitude.std_factor
    headingoffset = vehicle.dvl.yaw

    body_velocity = BodyVelocity(
        velocity_std_factor, velocity_std_offset, headingoffset
    )
    orientation = Orientation(headingoffset, orientation_std_offset)
    depth = Depth(depth_std_factor)
    altitude = Altitude(altitude_std_factor)

    body_velocity.sensor_string = sensor_string
    orientation.sensor_string = sensor_string
    depth.sensor_string = sensor_string
    altitude.sensor_string = sensor_string

    path = get_raw_folder(outpath / ".." / filepath / filename)

    alr_log = loadmat(str(path))
    mission_data = alr_log["MissionData"]

    data_list = []
    if category == Category.VELOCITY:
        Console.info("Parsing alr velocity...")
        previous_timestamp = 0
        for i in range(len(mission_data["epoch_timestamp"])):
            if mission_data["DVL_down_BT_is_good"][i] == 1:
                vx = mission_data["DVL_down_BT_x"][i]
                vy = mission_data["DVL_down_BT_y"][i]
                vz = mission_data["DVL_down_BT_z"][i]
                # vx, vy, vz should not be NaN on lines with bottom lock,
                # but check to be on the safe side:
                if not isnan(vx) and not isnan(vy) and not isnan(vz):
                    t = mission_data["epoch_timestamp"][i]
                    body_velocity.from_alr(t, vx, vy, vz)
                    data = body_velocity.export(output_format)
                    if body_velocity.epoch_timestamp > previous_timestamp:
                        data_list.append(data)
                    else:
                        data_list[-1] = data
                    previous_timestamp = body_velocity.epoch_timestamp
        Console.info("...done parsing alr velocity")
    if category == Category.ORIENTATION:
        Console.info("Parsing alr orientation...")
        previous_timestamp = 0
        for i in range(len(mission_data["epoch_timestamp"])):
            roll = mission_data["AUV_roll"][i]
            pitch = mission_data["AUV_pitch"][i]
            yaw = mission_data["AUV_heading"][i]
            if not isnan(roll) and not isnan(pitch) and not isnan(yaw):
                t = mission_data["epoch_timestamp"][i]
                orientation.from_alr(t, roll, pitch, yaw)
                data = orientation.export(output_format)
                if orientation.epoch_timestamp > previous_timestamp:
                    data_list.append(data)
                else:
                    data_list[-1] = data
                previous_timestamp = orientation.epoch_timestamp
        Console.info("...done parsing alr orientation")
    if category == Category.DEPTH:
        Console.info("Parsing alr depth...")
        previous_timestamp = 0
        for i in range(len(mission_data["epoch_timestamp"])):
            d = mission_data["AUV_depth"][i]
            if not isnan(d):
                t = mission_data["epoch_timestamp"][i]
                depth.from_alr(t, d)
                data = depth.export(output_format)
                if depth.epoch_timestamp > previous_timestamp:
                    data_list.append(data)
                else:
                    data_list[-1] = data
                previous_timestamp = depth.epoch_timestamp
        Console.info("...done parsing alr depth")
    if category == Category.ALTITUDE:
        Console.info("Parsing alr altitude...")
        previous_timestamp = 0
        for i in range(len(mission_data["epoch_timestamp"])):
            if mission_data["DVL_down_BT_is_good"][i] == 1:
                a = mission_data["AUV_altitude"][i]
                # The altitude should not be NaN on lines with bottom lock,
                # but check to be on the safe side:
                if not isnan(a):
                    t = mission_data["epoch_timestamp"][i]
                    altitude.from_alr(t, a)
                    data = altitude.export(output_format)
                    if altitude.epoch_timestamp > previous_timestamp:
                        data_list.append(data)
                    else:
                        data_list[-1] = data
                    previous_timestamp = altitude.epoch_timestamp
        Console.info("...done parsing alr altitude")
    return data_list

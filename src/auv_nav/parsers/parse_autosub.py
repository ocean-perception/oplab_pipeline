# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from auv_nav.parsers.load_matlab_file import loadmat

# fmt: off
from auv_nav.sensors import Altitude, BodyVelocity, Category, Depth, Orientation

# fmt: on
from oplab import Console, get_raw_folder


def parse_autosub(mission, vehicle, category, ftype, outpath):
    # parser meta data
    sensor_string = "autosub"
    category = category
    output_format = ftype
    filename = mission.velocity.filename
    filepath = mission.velocity.filepath

    # autosub std models
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

    autosub_log = loadmat(str(path))
    autosub_adcp = autosub_log["missionData"]["ADCPbin00"]
    autosub_ins = autosub_log["missionData"]["INSAttitude"]
    autosub_dep_ctl = autosub_log["missionData"]["DepCtlNode"]
    autosub_adcp_log1 = autosub_log["missionData"]["ADCPLog_1"]

    data_list = []
    if category == Category.VELOCITY:
        Console.info("... parsing autosub velocity")
        previous_timestamp = 0
        for i in range(len(autosub_adcp["eTime"])):
            body_velocity.from_autosub(autosub_adcp, i)
            data = body_velocity.export(output_format)
            if body_velocity.epoch_timestamp > previous_timestamp:
                data_list.append(data)
            else:
                data_list[-1] = data
            previous_timestamp = body_velocity.epoch_timestamp
    if category == Category.ORIENTATION:
        Console.info("... parsing autosub orientation")
        previous_timestamp = 0
        for i in range(len(autosub_ins["eTime"])):
            orientation.from_autosub(autosub_ins, i)
            data = orientation.export(output_format)
            if orientation.epoch_timestamp > previous_timestamp:
                data_list.append(data)
            else:
                data_list[-1] = data
            previous_timestamp = orientation.epoch_timestamp
    if category == Category.DEPTH:
        Console.info("... parsing autosub depth")
        previous_timestamp = 0
        for i in range(len(autosub_dep_ctl["eTime"])):
            depth.from_autosub(autosub_dep_ctl, i)
            data = depth.export(output_format)
            if depth.epoch_timestamp > previous_timestamp:
                data_list.append(data)
            else:
                data_list[-1] = data
            previous_timestamp = depth.epoch_timestamp
    if category == Category.ALTITUDE:
        Console.info("... parsing autosub altitude")
        previous_timestamp = 0
        for i in range(len(autosub_adcp_log1["eTime"])):
            altitude.from_autosub(autosub_adcp_log1, i)
            data = altitude.export(output_format)
            if altitude.epoch_timestamp > previous_timestamp:
                data_list.append(data)
            else:
                data_list[-1] = data
            previous_timestamp = altitude.epoch_timestamp
    return data_list

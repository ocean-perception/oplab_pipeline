# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import scipy.io as spio
from auv_nav.sensors import BodyVelocity
from auv_nav.sensors import Orientation, Depth, Altitude
from auv_nav.sensors import Category
from oplab import get_raw_folder
from oplab import Console


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def parse_autosub(mission, vehicle, category, ftype, outpath):
    # parser meta data
    class_string = "measurement"
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

    alr_log = loadmat(str(path))
    alr_adcp = alr_log["missionData"]["ADCPbin00"]
    alr_ins = alr_log["missionData"]["INSAttitude"]
    alr_dep_ctl = alr_log["missionData"]["DepCtlNode"]
    alr_adcp_log1 = alr_log["missionData"]["ADCPLog_1"]

    data_list = []
    if category == Category.VELOCITY:
        Console.info("... parsing autosub velocity")
        previous_timestamp = 0
        for i in range(len(alr_adcp["eTime"])):
            body_velocity.from_autosub(alr_adcp, i)
            data = body_velocity.export(output_format)
            if body_velocity.epoch_timestamp > previous_timestamp:
                data_list.append(data)
            else:
                data_list[-1] = data
            previous_timestamp = body_velocity.epoch_timestamp;
    if category == Category.ORIENTATION:
        Console.info("... parsing autosub orientation")
        previous_timestamp = 0
        for i in range(len(alr_ins["eTime"])):
            orientation.from_autosub(alr_ins, i)
            data = orientation.export(output_format)
            if orientation.epoch_timestamp > previous_timestamp:
                data_list.append(data)
            else:
                data_list[-1] = data
            previous_timestamp = orientation.epoch_timestamp;
    if category == Category.DEPTH:
        Console.info("... parsing autosub depth")
        previous_timestamp = 0
        for i in range(len(alr_dep_ctl["eTime"])):
            depth.from_autosub(alr_dep_ctl, i)
            data = depth.export(output_format)
            if depth.epoch_timestamp > previous_timestamp:
                data_list.append(data)
            else:
                data_list[-1] = data
            previous_timestamp = depth.epoch_timestamp;
    if category == Category.ALTITUDE:
        Console.info("... parsing autosub altitude")
        previous_timestamp = 0
        for i in range(len(alr_adcp_log1["eTime"])):
            altitude.from_autosub(alr_adcp_log1, i)
            data = altitude.export(output_format)
            if altitude.epoch_timestamp > previous_timestamp:
                data_list.append(data)
            else:
                data_list[-1] = data
            previous_timestamp = altitude.epoch_timestamp;
    return data_list

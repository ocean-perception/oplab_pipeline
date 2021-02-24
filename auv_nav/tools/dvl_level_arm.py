# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import math


def correct_lever_arm(linear_speeds, angular_speeds, dvl_pos_on_vehicle):
    """Correct DVL speeds when offset from AUV centre
    Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8729737

    Args:
        linear_speeds (tuple(float, float, float)):
            Raw DVL linear speeds
        angular_speeds (tuple(float, float, float)):
            Angular speed at the centre of the AUV (deg/s)
        dvl_pos_on_vehicle (tuple(float, float, float)):
            DVL offset to the centre of the AUV

    Returns:
        tuple(float, float, float): Corrected linear speeds
    """
    vx, vy, vz = linear_speeds
    wx, wy, wz = angular_speeds
    wx = math.radians(wx)
    wy = math.radians(wy)
    wz = math.radians(wz)
    x_offset, y_offset, z_offset = dvl_pos_on_vehicle

    vx += wy * z_offset - wz * y_offset
    vy += -wx * z_offset + wz * x_offset
    vz += wx * y_offset - wy * x_offset

    return vx, vy, vz


def compute_angular_speeds(orientation_vec, i):
    o1 = None
    o2 = None
    if i >= 1 and i < len(orientation_vec) - 1:
        o1 = orientation_vec[i - 1]
        o2 = orientation_vec[i + 1]
    elif i > 1 and i == len(orientation_vec) - 1:
        o1 = orientation_vec[i - 1]
        o2 = orientation_vec[i]
    elif i == 0:
        o1 = orientation_vec[i]
        o2 = orientation_vec[i + 1]
    else:
        print("ERROR: CONDITION NOT TAKEN INTO ACCOUNT")

    dt = o2.epoch_timestamp - o1.epoch_timestamp
    droll = o2.roll - o1.roll
    dpitch = o2.pitch - o1.pitch
    dyaw = o2.yaw - o1.yaw
    if dyaw > 180:
        dyaw -= 360
    elif dyaw < -180:
        dyaw += 360

    roll_speed = droll / dt
    pitch_speed = dpitch / dt
    yaw_speed = dyaw / dt

    return roll_speed, pitch_speed, yaw_speed

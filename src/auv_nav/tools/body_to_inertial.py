# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

# Scripts to rotate a coordinate frame (e.g. body to inertial taking angles
# in degrees as input)

# Author: Blair Thornton
# Date: 01/09/2017

import math

# http://www.json.org/
deg_to_rad = math.pi / 180  # 3.141592654/180


def body_to_inertial(roll, pitch, yaw, old_x, old_y, old_z):
    roll = roll * deg_to_rad
    pitch = pitch * deg_to_rad
    yaw = yaw * deg_to_rad

    new_x = (
        (math.cos(yaw) * math.cos(pitch)) * old_x
        + (
            -math.sin(yaw) * math.cos(roll)
            + math.cos(yaw) * math.sin(pitch) * math.sin(roll)
        )
        * old_y
        + (
            math.sin(yaw) * math.sin(roll)
            + (math.cos(yaw) * math.cos(roll) * math.sin(pitch))
        )
        * old_z
    )
    new_y = (
        (math.sin(yaw) * math.cos(pitch)) * old_x
        + (
            math.cos(yaw) * math.cos(roll)
            + math.sin(roll) * math.sin(pitch) * math.sin(yaw)
        )
        * old_y
        + (
            -math.cos(yaw) * math.sin(roll)
            + math.sin(yaw) * math.cos(roll) * math.sin(pitch)
        )
        * old_z
    )
    new_z = (
        (-math.sin(pitch) * old_x)
        + (math.cos(pitch) * math.sin(roll)) * old_y
        + (math.cos(pitch) * math.cos(roll)) * old_z
    )

    return new_x, new_y, new_z

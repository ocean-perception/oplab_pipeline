# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

# Function to transform a vector from the body frame to the inertial frame
# Based on the same structure as the body_to_inertial function
# Any given rotation matrix is - by definition - orthonormal. This means that
# the inverse of the rotation matrix is equal to the transpose of the rotation

# Author: Jose Cappelletto
# Date: 03/11/2022

import math

deg_to_rad = math.pi / 180


def inertial_to_body(roll, pitch, yaw, old_x, old_y, old_z):
    # input is assumed to be in degrees
    roll = roll * deg_to_rad
    pitch = pitch * deg_to_rad
    yaw = yaw * deg_to_rad
    # rotation matrix elements
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # rotation matrix
    #    a b c
    # R= d e f
    #    g h i

    # This is the original body_to_inertial function
    # new_x = ((cy*cp)            * old_x   # a
    #       + (-sy*cr + cy*sp*sr) * old_y   # b
    #       + ( sy*sr + cy*sp*cr) * old_z   # c
    # )

    # new_y = ((sy*cp)            * old_x   # d
    #       + ( cy*cr + sy*sp*sr) * old_y   # e
    #       + (-cy*sr + sy*sp*sr) * old_z   # f
    # )
    # new_z = ((-sp  )           * old_x    # g
    #       + ( cp*sr)           * old_y    # h
    #       + ( cp*cr)           * old_z    # i
    # )

    # rotation matrix - transpose -> same as inverse for orthonormal matrices
    #    a d g
    # R= b e h
    #    c f i

    new_x = (cy * cp) * old_x + (sy * cp) * old_y + (-sp) * old_z  # a  # d  # g

    new_y = (
        (-sy * cr + cy * sp * sr) * old_x  # b
        + (cy * cr + sy * sp * sr) * old_y  # e
        + (cp * sr) * old_z  # h
    )

    new_z = (
        (sy * sr + cy * sp * cr) * old_x  # c
        + (-cy * sr + sy * sp * sr) * old_y  # f
        + (cp * cr) * old_z  # i
    )

    return new_x, new_y, new_z

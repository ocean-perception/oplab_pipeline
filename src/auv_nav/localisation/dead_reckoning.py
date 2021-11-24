# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""


# Scripts to integrate velocities and return positions
def dead_reckoning(
    time_now,
    time_previous,
    north_velocity_now,
    north_velocity_previous,
    east_velocity_now,
    east_velocity_previous,
    northings_previous,
    eastings_previous,
):
    northings_now = (
        (time_now - time_previous) * (north_velocity_previous + north_velocity_now)
    ) / 2 + northings_previous
    eastings_now = (
        (time_now - time_previous) * (east_velocity_previous + east_velocity_now)
    ) / 2 + eastings_previous
    return northings_now, eastings_now

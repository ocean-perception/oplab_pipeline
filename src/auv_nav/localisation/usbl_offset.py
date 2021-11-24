# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

# Scripts to shift deadreackoning data to a single USBL position. This should
# be improved to match on average as minimum, but really needs to be turned
# into a kalman/particle filter

# Author: Blair Thornton
# Date: 13/02/2018

from auv_nav.tools.interpolate import interpolate
from oplab import Console


def usbl_offset(
    time_dead_reckoning,
    northings_dead_reckoning,
    eastings_dead_reckoning,
    time_usbl,
    northings_usbl,
    eastings_usbl,
):
    # ===============Average Offset============================================
    start_dead_reckoning = 0
    start_usbl = 0

    threshold = 20  # what to consider a big jump in time
    exit_flag = False

    # Find suitable start points
    if time_usbl[0] < time_dead_reckoning[0]:
        Console.info("USBL starts before dead_reckoning")
        while exit_flag is False:
            if start_usbl + 1 >= len(time_usbl):
                break
            if (
                time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl + 1]
                > 0
            ):
                start_usbl = start_usbl + 1
            else:
                if (
                    time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl]
                    < threshold
                    and time_usbl[start_usbl + 1] - time_usbl[start_usbl] < threshold
                ):
                    start_usbl += 1
                    exit_flag = True
                else:
                    start_dead_reckoning = start_dead_reckoning + 1
    else:
        # print('usbl starts after dead_reckoning')
        while exit_flag is False:
            if time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl] < 0:
                start_dead_reckoning = start_dead_reckoning + 1

            else:
                if (
                    time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl]
                    < threshold
                    and time_usbl[start_usbl + 1] - time_usbl[start_usbl] < threshold
                ):
                    start_usbl += 1
                    exit_flag = True
                else:  # if the jump is too big, ignore and try another fix
                    start_usbl = start_usbl + 1

    northings_dead_reckoning_interpolated = []
    eastings_dead_reckoning_interpolated = []

    for j_usbl in range(start_usbl, len(time_usbl)):
        if start_dead_reckoning + 1 >= len(time_dead_reckoning):
            break
        try:
            while time_dead_reckoning[start_dead_reckoning + 1] < time_usbl[j_usbl]:
                start_dead_reckoning += 1
        except IndexError:
            break
        northings_dead_reckoning_interpolated.append(
            interpolate(
                time_usbl[j_usbl],
                time_dead_reckoning[start_dead_reckoning],
                time_dead_reckoning[start_dead_reckoning + 1],
                northings_dead_reckoning[start_dead_reckoning],
                northings_dead_reckoning[start_dead_reckoning + 1],
            )
        )
        eastings_dead_reckoning_interpolated.append(
            interpolate(
                time_usbl[j_usbl],
                time_dead_reckoning[start_dead_reckoning],
                time_dead_reckoning[start_dead_reckoning + 1],
                eastings_dead_reckoning[start_dead_reckoning],
                eastings_dead_reckoning[start_dead_reckoning + 1],
            )
        )

    northings_offset = sum(
        northings_usbl[
            start_usbl : start_usbl + len(northings_dead_reckoning_interpolated)  # noqa
        ]
    ) / len(
        northings_usbl[
            start_usbl : (  # noqa
                start_usbl + len(northings_dead_reckoning_interpolated)
            )
        ]
    ) - sum(
        northings_dead_reckoning_interpolated
    ) / len(
        northings_dead_reckoning_interpolated
    )
    eastings_offset = sum(
        eastings_usbl[
            start_usbl : (  # noqa
                start_usbl + len(eastings_dead_reckoning_interpolated)
            )
        ]
    ) / len(
        eastings_usbl[
            start_usbl : start_usbl + len(eastings_dead_reckoning_interpolated)  # noqa
        ]
    ) - sum(
        eastings_dead_reckoning_interpolated
    ) / len(
        eastings_dead_reckoning_interpolated
    )

    return northings_offset, eastings_offset

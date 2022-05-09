# -*- coding: utf-8 -*-
"""
Copyright (c) 2021, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import pandas as pd

from auv_nav.sensors import Altitude, BodyVelocity, Category
from oplab import Console, get_raw_folder

header_list = [
    "time",
    "X_tp",
    "Y_tp",
    "Z_pg",
    "phi",
    "theta",
    "psi",
    "u_dvl",
    "v_dvl",
    "z_dvl",
    "p",
    "q",
    "r",
    "MT_DATA00",
    "MT_DATA01",
    "MT_DATA02",
    "MT_DATA03",
    "MT_DATA04",
    "MT_DATA05",
    "MT_DATA06",
    "MT_DATA07",
    "MT_DATA08",
    "MT_DATA09",
    "MT_DATA10",
    "MT_DATA11",
    "MT_DATA12",
    "MT_DATA13",
    "MT_DATA14",
    "MT_DATA15",
    "MT_DATA16",
    "MT_DATA17",
    "MT_DATA18",
    "MT_DATA19",
    "MT_DATA20",
    "MT_DATA21",
    "MT_DATA22",
    "MT_DATA23",
    "MT_DATA24",
    "MT_DATA25",
    "MT_DATA26",
    "MT_DATA27",
    "MT_DATA28",
    "MT_DATA29",
    "MT_DATA30",
    "MT_DATA31",
    "MT_DATA32",
    "MT_DATA33",
    "MT_DATA34",
    "MT_DATA35",
    "MT_DATA36",
    "MT_DATA37",
    "MT_DATA38",
    "MT_DATA39",
    "MT_DATA40",
    "MT_DATA41",
    "MT_DATA42",
    "dvl_alt0",
    "dvl_alt1",
    "dvl_alt2",
    "dvl_alt3",
    "ship_pos_x",
    "ship_pos_y",
    "ship_pos_psi",
    "pressure_kpa",
    "rov_str_z",
    "rov_str_psi",
    "rov_str_r",
]


def parse_ntnu_dvl(mission, vehicle, category, output_format, outpath):
    # Get your data from a file using mission paths, for example
    filepath = mission.velocity.filepath
    log_file_path = get_raw_folder(outpath / ".." / filepath)

    category_str = None
    if category == Category.VELOCITY:
        category_str = "velocity"
    elif category == Category.ALTITUDE:
        category_str = "altitude"

    velocity_std_factor = mission.velocity.std_factor
    velocity_std_offset = mission.velocity.std_offset
    heading_offset = vehicle.dvl.yaw
    body_velocity = BodyVelocity(
        velocity_std_factor,
        velocity_std_offset,
        heading_offset,
    )
    altitude = Altitude(altitude_std_factor=mission.altitude.std_factor)

    data_list = []

    num_entries = 0
    num_valid_entries = 0

    # For each log file
    for log_file in log_file_path.glob("*.log"):
        # Open the log file
        df = pd.read_csv(log_file, sep="\t", skiprows=(0, 1), names=header_list)

        if category == Category.VELOCITY:
            Console.info("... parsing velocity")
            # For each row in the file
            for index, row in df.iterrows():
                body_velocity.from_ntnu_dvl(str(log_file.name), row)
                num_entries += 1
                if body_velocity.valid():
                    # DVL provides -32 when no bottom lock
                    data = body_velocity.export(output_format)
                    num_valid_entries += 1
                    data_list.append(data)
        elif category == Category.ALTITUDE:
            Console.info("... parsing altitude")
            # For each row in the file
            for index, row in df.iterrows():
                num_entries += 1
                altitude.from_ntnu_dvl(str(log_file.name), row)
                if altitude.valid():
                    num_valid_entries += 1
                    data = altitude.export(output_format)
                    data_list.append(data)

        elif category == Category.ORIENTATION:
            Console.quit("NTNU DVL parser has no ORIENTATION available")
        elif category == Category.DEPTH:
            Console.quit("NTNU DVL parser has no DEPTH available")
        elif category == Category.USBL:
            Console.quit("NTNU DVL parser has no USBL available")
        else:
            Console.quit("NTNU DVL parser has no category", category, "available")

    Console.info(
        "... parsed",
        num_entries,
        "entries, of which",
        num_valid_entries,
        "are valid for category",
        category_str,
    )

    return data_list

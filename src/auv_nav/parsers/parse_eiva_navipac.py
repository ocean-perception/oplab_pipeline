# -*- coding: utf-8 -*-
"""
Copyright (c) 2021, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from auv_nav.sensors import Category, Depth, Orientation, Usbl
from oplab import Console, get_raw_folder


def parse_eiva_navipac(mission, vehicle, category, output_format, outpath):
    # Get your data from a file using mission paths, for example
    # Get your data from a file using mission paths, for example
    filepath = None
    category_str = None
    if category == Category.ORIENTATION:
        category_str = "orientation"
        filepath = mission.orientation.filepath
    elif category == Category.DEPTH:
        category_str = "depth"
        filepath = mission.depth.filepath
    elif category == Category.USBL:
        category_str = "usbl"
        filepath = mission.usbl.filepath

    log_file_path = get_raw_folder(outpath / ".." / filepath)

    depth_std_factor = mission.depth.std_factor
    orientation_std_offset = mission.orientation.std_offset
    heading_offset = vehicle.ins.yaw
    usbl_id = mission.usbl.label

    latitude_reference = float(mission.origin.latitude)
    longitude_reference = float(mission.origin.longitude)

    orientation = Orientation(heading_offset, orientation_std_offset)
    depth = Depth(depth_std_factor)
    usbl = Usbl(
        mission.usbl.std_factor,
        mission.usbl.std_offset,
        latitude_reference,
        longitude_reference,
    )

    data_list = []

    num_entries = 0
    num_valid_entries = 0

    # For each log file
    for log_file in log_file_path.glob("*_G.NPD"):
        # Open the log file
        with log_file.open("r", encoding="utf-8", errors="ignore") as filein:

            if category == Category.ORIENTATION:
                Console.info("... parsing orientation")
                # For each line in the file
                for line in filein.readlines():
                    if line.startswith("R132  4") and "$PRDID" in line:
                        orientation.from_eiva_navipac(line)
                        num_entries += 1
                        if orientation.roll is None:
                            continue
                        if orientation.valid():
                            num_valid_entries += 1
                            data = orientation.export(output_format)
                            data_list.append(data)
            elif category == Category.DEPTH:
                Console.info("... parsing depth")
                for line in filein.readlines():
                    if line[0:14] == "D     " + str(usbl_id) + "  19  1":
                        depth.from_eiva_navipac(line)
                        num_entries += 1
                        if depth.valid():
                            num_valid_entries += 1
                            data = depth.export(output_format)
                            data_list.append(data)
            elif category == Category.USBL:
                Console.info("... parsing USBL")
                for line in filein.readlines():
                    if line.startswith("P  D") and int(line[6:10]) == usbl_id:
                        num_entries += 1
                        usbl.from_eiva_navipac(line)
                        if usbl.valid():
                            num_valid_entries += 1
                            data = usbl.export(output_format)
                            data_list.append(data)
            elif category == Category.ALTITUDE:
                Console.quit("EIVA Navipac parser has no ALTITUDE available")
            elif category == Category.VELOCITY:
                Console.quit("EIVA Navipac parser has no VELOCITY available")
            else:
                Console.quit(
                    "EIVA Navipac parser has no category", category, "available"
                )
    Console.info(
        "Processed",
        num_entries,
        "entries , of which",
        num_valid_entries,
        "are valid for category",
        category_str,
    )

    return data_list

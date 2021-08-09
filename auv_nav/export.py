# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import json
from pathlib import Path

from auv_nav.converters import AcfrExporter
from auv_nav.parsers.parse_acfr_stereo_pose import AcfrStereoPoseFile
from auv_nav.sensors import (
    Altitude,
    BodyVelocity,
    Camera,
    Depth,
    InertialVelocity,
    Orientation,
    Usbl,
)
from auv_nav.tools.interpolate import interpolate_camera
from auv_nav.tools.time_conversions import (
    epoch_from_json,
    epoch_to_datetime,
    string_to_epoch,
)
from oplab import Console, Mission, Vehicle, get_processed_folder, valid_dive


def export(filepath, input_file, ftype, start_datetime, finish_datetime):
    Console.info("Requested data conversion to {}".format(ftype))

    filepath = Path(filepath).resolve()

    camera1_list = []
    camera2_list = []
    interpolate_laser = False

    if input_file is not None:
        input_file = Path(input_file).resolve()
        if input_file.suffix == ".data":
            # Process stereo_pose_est.data
            Console.info("Processing ACFR stereo pose estimation file...")
            s = AcfrStereoPoseFile(input_file)
            camera1_list, camera2_list = s.convert()
            file1 = Path(
                "csv/acfr/auv_acfr_Cam51707923.csv"
            )  # ToDo: use camera name as specified in mission.yaml.
            # Save in subfolder of json_renav folder.
            file2 = Path(
                "csv/acfr/auv_acfr_Cam51707925.csv"
            )  # ToDo: use camera name as specified in mission.yaml.
            # Save in subfolder of json_renav folder.
            file1.parent.mkdir(parents=True, exist_ok=True)
            fileout1 = file1.open("w")
            # ToDo:  Check if file exists and only overwrite if told ('-F').
            fileout2 = file2.open("w")
            # ToDo:  Check if file exists and only overwrite if told ('-F').
            fileout1.write(camera1_list[0].write_csv_header())
            fileout2.write(camera1_list[0].write_csv_header())
            for c1, c2 in zip(camera1_list, camera2_list):
                fileout1.write(c1.to_csv())
                fileout2.write(c2.to_csv())
            Console.info("Done! Two files converted:")
            Console.info(file1, file2)
            interpolate_laser = True

    if not valid_dive(filepath):
        return

    mission_file = filepath / "mission.yaml"
    vehicle_file = filepath / "vehicle.yaml"
    mission_file = get_processed_folder(mission_file)
    vehicle_file = get_processed_folder(vehicle_file)
    Console.info("Loading mission.yaml at {0}".format(mission_file))
    mission = Mission(mission_file)

    Console.info("Loading vehicle.yaml at {0}".format(vehicle_file))
    vehicle = Vehicle(vehicle_file)

    exporter = None
    if ftype == "acfr":
        exporter = AcfrExporter(mission, vehicle, filepath)
    else:
        Console.error("Exporter type {} not implemented.".format(ftype))

    nav_standard_file = filepath / "nav" / "nav_standard.json"
    nav_standard_file = get_processed_folder(nav_standard_file)
    Console.info("Loading json file {}".format(nav_standard_file))

    with nav_standard_file.open("r") as nav_standard:
        parsed_json_data = json.load(nav_standard)

    # setup start and finish date time
    if start_datetime == "":
        epoch_start_time = epoch_from_json(parsed_json_data[1])
        start_datetime = epoch_to_datetime(epoch_start_time)
    else:
        epoch_start_time = string_to_epoch(start_datetime)
    if finish_datetime == "":
        epoch_finish_time = epoch_from_json(parsed_json_data[-1])
        finish_datetime = epoch_to_datetime(epoch_finish_time)
    else:
        epoch_finish_time = string_to_epoch(finish_datetime)

    sensors_std = {
        "usbl": {"model": "json"},
        "dvl": {"model": "json"},
        "depth": {"model": "json"},
        "orientation": {"model": "json"},
    }

    if interpolate_laser:
        Console.info("Interpolating laser to ACFR stereo pose data...")
        file3 = Path(
            "csv/acfr/auv_acfr_LM165.csv"
        )  # ToDo: use camera name as specified in mission.yaml.
        # Save in subfolder of json_renav folder.
        file3.parent.mkdir(parents=True, exist_ok=True)
        fileout3 = file3.open(
            "w"
        )  # ToDo: Check if file exists and only overwrite if told ('-F')
        fileout3.write(camera1_list[0].write_csv_header())
        for i in range(len(parsed_json_data)):
            Console.progress(i, len(parsed_json_data))
            epoch_timestamp = parsed_json_data[i]["epoch_timestamp"]
            if (
                epoch_timestamp >= epoch_start_time
                and epoch_timestamp <= epoch_finish_time
            ):
                if "laser" in parsed_json_data[i]["category"]:
                    filename = parsed_json_data[i]["filename"]
                    c3_interp = interpolate_camera(
                        epoch_timestamp, camera1_list, filename
                    )
                    fileout3.write(c3_interp.to_csv())
        Console.info("Done! Laser file available at", str(file3))
        return

    # read in data from json file
    # i here is the number of the data packet
    for i in range(len(parsed_json_data)):
        Console.progress(i, len(parsed_json_data))
        epoch_timestamp = parsed_json_data[i]["epoch_timestamp"]
        if (
            epoch_timestamp >= epoch_start_time
            and epoch_timestamp <= epoch_finish_time
        ):
            if "velocity" in parsed_json_data[i]["category"]:
                if "body" in parsed_json_data[i]["frame"]:
                    # to check for corrupted data point which have inertial
                    # frame data values
                    if "epoch_timestamp_dvl" in parsed_json_data[i]:
                        # confirm time stamps of dvl are aligned with main
                        # clock (within a second)
                        if (
                            abs(
                                parsed_json_data[i]["epoch_timestamp"]
                                - parsed_json_data[i]["epoch_timestamp_dvl"]
                            )
                        ) < 1.0:
                            velocity_body = BodyVelocity()
                            velocity_body.from_json(
                                parsed_json_data[i], sensors_std["dvl"]
                            )
                            exporter.add(velocity_body)
                if "inertial" in parsed_json_data[i]["frame"]:
                    velocity_inertial = InertialVelocity()
                    velocity_inertial.from_json(parsed_json_data[i])
                    exporter.add(velocity_inertial)

            if "orientation" in parsed_json_data[i]["category"]:
                orientation = Orientation()
                orientation.from_json(
                    parsed_json_data[i], sensors_std["orientation"]
                )
                exporter.add(orientation)

            if "depth" in parsed_json_data[i]["category"]:
                depth = Depth()
                depth.from_json(parsed_json_data[i], sensors_std["depth"])
                exporter.add(depth)

            if "altitude" in parsed_json_data[i]["category"]:
                altitude = Altitude()
                altitude.from_json(parsed_json_data[i])
                exporter.add(altitude)

            if "usbl" in parsed_json_data[i]["category"]:
                usbl = Usbl()
                usbl.from_json(parsed_json_data[i], sensors_std["usbl"])
                exporter.add(usbl)

            if "image" in parsed_json_data[i]["category"]:
                camera1 = Camera()
                # LC
                camera1.from_json(parsed_json_data[i], "camera1")
                exporter.add(camera1)
                camera2 = Camera()
                camera2.from_json(parsed_json_data[i], "camera2")
                exporter.add(camera2)
    Console.info("Conversion to {} finished!".format(ftype))

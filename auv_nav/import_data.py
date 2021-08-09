# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
from pathlib import Path

from auv_nav.converters import HybisImporter
from auv_nav.tools.time_conversions import epoch_to_datetime

# fmt: off
from oplab import Console, Mission, Vehicle, get_processed_folder, get_raw_folder

# fmt: on


def import_data(filepath, ftype, force_overwite):
    Console.info("Requested data import from {}".format(ftype))

    filepath = Path(filepath)

    mission_file = filepath / "mission.yaml"
    mission_file = get_raw_folder(mission_file)
    vehicle_file = filepath / "vehicle.yaml"
    vehicle_file = get_raw_folder(vehicle_file)
    Console.info("Loading mission.yaml at {0}".format(mission_file))
    mission = Mission(mission_file)
    vehicle = Vehicle(vehicle_file)

    # copy mission.yaml and vehicle.yaml to processed folder for process step
    mission_processed = get_processed_folder(mission_file)
    vehicle_processed = get_processed_folder(vehicle_file)

    # Write mission with metadata (username, date and hostname)
    mission.write(mission_processed)
    # mission_file.copy(mission_processed)
    vehicle.write(vehicle_processed)

    importer = None
    if ftype == "hybis":
        importer = HybisImporter(mission, filepath)
    else:
        Console.error("Exporter type {} not implemented.".format(ftype))

    filepath = get_processed_folder(filepath)

    start_datetime = epoch_to_datetime(importer.start_epoch)
    finish_datetime = epoch_to_datetime(importer.finish_epoch)

    # make path for processed outputs
    json_filename = (
        "json_renav_"
        + start_datetime[0:8]
        + "_"
        + start_datetime[8:14]
        + "_"
        + finish_datetime[0:8]
        + "_"
        + finish_datetime[8:14]
    )
    renavpath = filepath / json_filename
    if not renavpath.is_dir():
        try:
            renavpath.mkdir()
        except Exception as e:
            print("Warning:", e)
    elif renavpath.is_dir() and not force_overwite:
        # Check if dataset has already been processed
        Console.error(
            "It looks like this dataset has already been processed for the",
            "specified time span.",
        )
        Console.error(
            "The following directory already exist: {}".format(renavpath)
        )
        Console.error(
            "To overwrite the contents of this directory rerun auv_nav with",
            "the flag -F.",
        )
        Console.error("Example:   auv_nav process -F PATH")
        Console.quit("auv_nav process would overwrite json_renav files")

    output_dr_centre_path = renavpath / "csv" / "dead_reckoning"
    if not output_dr_centre_path.exists():
        output_dr_centre_path.mkdir(parents=True)
    camera_name = mission.image.cameras[0].name
    output_dr_centre_path = output_dr_centre_path / (
        "auv_dr_" + camera_name + ".csv"
    )

    importer.write(output_dr_centre_path)

    Console.info("Finished")

import json
from pathlib import Path

import pandas as pd

# fmt: off
from auv_nav.parsers.acfr_combined_raw import AcfrCombinedRawWriter
from auv_nav.parsers.acfr_stereo_pose import AcfrStereoPoseParser, AcfrStereoPoseWriter
from auv_nav.parsers.acfr_vehicle_pose import (
    AcfrVehiclePoseParser,
    AcfrVehiclePoseWriter,
)
from auv_nav.parsers.hybis import HybisParser
from auv_nav.parsers.koyo20rov import RovParser
from auv_nav.sensors import (
    Altitude,
    BodyVelocity,
    Camera,
    Depth,
    InertialVelocity,
    Orientation,
    SyncedOrientationBodyVelocity,
    Usbl,
)
from auv_nav.tools.csv_tools import write_csv
from auv_nav.tools.interpolate import interpolate_camera
from auv_nav.tools.time_conversions import (
    epoch_from_json,
    epoch_to_datetime,
    string_to_epoch,
)
from oplab import Console, Mission, Vehicle, get_processed_folder

# fmt: on


def koyo20rov_to_oplab(args):
    if args.dive_path is None:
        Console.error("Please provide a dive path using --dive-path (-i)")
        Console.quit("Missing comandline arguments")
    # Do all necessary pathchecks on creating of class instance
    parser = RovParser(args.dive_path)
    parser.load_data()
    parser.check_for_outputs(args.force)
    # parser.filter_bad_measurements()
    parser.interpolate_to_images()
    parser.filter_nans()
    parser.add_northings_eastings()
    parser.add_lever_arms()
    parser.write_outputs()
    return


def hybis_to_oplab(args):
    if args.navigation_file is None:
        Console.error("Please provide a navigation file")
        Console.quit("Missing comandline arguments")
    if args.image_path is None:
        Console.error("Please provide an image path")
        Console.quit("Missing comandline arguments")
    if args.reference_lat is None:
        Console.error("Please provide a reference_lat")
        Console.quit("Missing comandline arguments")
    if args.reference_lon is None:
        Console.error("Please provide a reference_lon")
        Console.quit("Missing comandline arguments")
    if args.output_folder is None:
        Console.error("Please provide an output path")
        Console.quit("Missing comandline arguments")
    parser = HybisParser(
        args.navigation_file,
        args.image_path,
        args.reference_lat,
        args.reference_lon,
    )
    force_overwite = args.force
    filepath = get_processed_folder(args.output_folder)
    start_datetime = epoch_to_datetime(parser.start_epoch)
    finish_datetime = epoch_to_datetime(parser.finish_epoch)

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
        Console.error("The following directory already exist: {}".format(renavpath))
        Console.error(
            "To overwrite the contents of this directory rerun auv_nav with",
            "the flag -F.",
        )
        Console.error("Example:   auv_nav process -F PATH")
        Console.quit("auv_nav process would overwrite json_renav files")

    output_dr_centre_path = renavpath / "csv" / "dead_reckoning"
    if not output_dr_centre_path.exists():
        output_dr_centre_path.mkdir(parents=True)
    camera_name = "hybis_camera"
    output_dr_centre_path = output_dr_centre_path / ("auv_dr_" + camera_name + ".csv")
    parser.write(output_dr_centre_path)
    parser.write(output_dr_centre_path)
    Console.info("Output written to", output_dr_centre_path)


def acfr_to_oplab(args):
    csv_filepath = Path(args.output_folder)
    force_overwite = args.force
    if not args.vehicle_pose and not args.stereo_pose:
        Console.error("Please provide at least one file")
        Console.quit("Missing comandline arguments")
    if args.vehicle_pose:
        Console.info("Vehicle pose provided! Converting it to DR CSV...")
        vpp = AcfrVehiclePoseParser(args.vehicle_pose)
        dr = vpp.get_dead_reckoning()
        csv_filename = "auv_acfr_centre.csv"

        if (csv_filepath / csv_filename).exists() and not force_overwite:
            Console.error("The DR file already exists. Use -F to force overwrite it")
            Console.quit("Default behaviour: not to overwrite results")
        else:
            write_csv(csv_filepath, dr, csv_filename, csv_flag=True)
            Console.info("Output vehicle pose written to", csv_filepath / csv_filename)
    if args.stereo_pose:
        Console.info("Stereo pose provided! Converting it to camera CSVs...")
        spp = AcfrStereoPoseParser(args.stereo_pose)
        cam1, cam2 = spp.get_cameras()
        if (csv_filepath / "auv_dr_LC.csv").exists() and not force_overwite:
            Console.error(
                "The camera files already exists. Use -F to force overwrite it"
            )
            Console.quit("Default behaviour: not to overwrite results")
        else:
            write_csv(csv_filepath, cam1, "auv_acfr_LC", csv_flag=True)
            write_csv(csv_filepath, cam2, "auv_acfr_RC", csv_flag=True)
            Console.info("Output camera files written at", csv_filepath)

        if args.dive_folder is None:
            return

        # If the user provides a dive path, we can interpolate the ACFR navigation
        # to the laser timestamps
        path_processed = get_processed_folder(args.dive_folder)
        nav_standard_file = path_processed / "nav" / "nav_standard.json"
        nav_standard_file = get_processed_folder(nav_standard_file)
        Console.info("Loading json file {}".format(nav_standard_file))

        start_datetime = ""
        finish_datetime = ""

        file3 = csv_filepath / "auv_acfr_laser.csv"
        if file3.exists() and not force_overwite:
            Console.error("The laser file already exists. Use -F to force overwrite it")
            Console.quit("Default behaviour: not to overwrite results")

        with nav_standard_file.open("r") as nav_standard:
            parsed_json_data = json.load(nav_standard)

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

            Console.info("Interpolating laser to ACFR stereo pose data...")
            fileout3 = file3.open(
                "w"
            )  # ToDo: Check if file exists and only overwrite if told ('-F')
            fileout3.write(cam1[0].get_csv_header())
            for i in range(len(parsed_json_data)):
                Console.progress(i, len(parsed_json_data))
                epoch_timestamp = parsed_json_data[i]["epoch_timestamp"]
                if (
                    epoch_timestamp >= epoch_start_time
                    and epoch_timestamp <= epoch_finish_time
                ):
                    if "laser" in parsed_json_data[i]["category"]:
                        filename = parsed_json_data[i]["filename"]
                        c3_interp = interpolate_camera(epoch_timestamp, cam1, filename)
                        fileout3.write(c3_interp.to_csv_row())
            Console.info("Done! Laser file available at", str(file3))


def oplab_to_acfr(args):
    if not args.dive_folder:
        Console.error("No dive folder provided. Exiting...")
        Console.quit("Missing comandline arguments")
    if not args.output_folder:
        Console.error("No output folder provided. Exiting...")
        Console.quit("Missing comandline arguments")

    output_folder = get_processed_folder(args.output_folder)
    vf = output_folder / "vehicle_pose_est.data"
    sf = output_folder / "stereo_pose_est.data"
    if (vf.exists() or sf.exists()) and not args.force:
        Console.error("Output folder already exists. Use -F to overwrite. Exiting...")
        Console.quit("Default behaviour: not to overwrite results")

    Console.info("Loading mission.yaml")
    path_processed = get_processed_folder(args.dive_folder)
    mission_file = path_processed / "mission.yaml"
    mission = Mission(mission_file)

    vehicle_file = path_processed / "vehicle.yaml"
    vehicle = Vehicle(vehicle_file)

    origin_lat = mission.origin.latitude
    origin_lon = mission.origin.longitude

    json_list = list(path_processed.glob("json_*"))
    if len(json_list) == 0:
        Console.quit(
            "No navigation solution could be found. Please run ",
            "auv_nav parse and process first",
        )
    navigation_path = path_processed / json_list[0]
    # Try if ekf exists:
    full_navigation_path = navigation_path / "csv" / "ekf"

    vpw = AcfrVehiclePoseWriter(vf, origin_lat, origin_lon)
    vehicle_navigation_file = full_navigation_path / "auv_ekf_centre.csv"
    dataframe = pd.read_csv(vehicle_navigation_file)
    dr_list = []
    for _, row in dataframe.iterrows():
        sb = SyncedOrientationBodyVelocity()
        sb.from_df(row)
        dr_list.append(sb)
    vpw.write(dr_list)
    Console.info("Vehicle pose written to", vf)

    if not mission.image.cameras:
        Console.warn("No cameras found in the mission file.")
        return
    if len(mission.image.cameras) == 1:
        Console.error("Only one camera found in the mission file. Exiting...")
        Console.quit("ACFR stereo pose est requires two cameras.")

    camera0_name = mission.image.cameras[0].name
    camera1_name = mission.image.cameras[1].name

    camera0_nav_file = full_navigation_path / ("auv_ekf_" + camera0_name + ".csv")
    dataframe = pd.read_csv(camera0_nav_file)
    cam1_list = []
    for _, row in dataframe.iterrows():
        sb = Camera()
        sb.from_df(row)
        cam1_list.append(sb)

    camera1_nav_file = full_navigation_path / ("auv_ekf_" + camera1_name + ".csv")

    dataframe = pd.read_csv(camera1_nav_file)
    cam2_list = []
    for _, row in dataframe.iterrows():
        sb = Camera()
        sb.from_df(row)
        cam2_list.append(sb)

    spw = AcfrStereoPoseWriter(sf, origin_lat, origin_lon)

    spw.write(cam1_list, cam2_list)
    Console.info("Stereo pose written to", sf)

    Console.info("Generating combined.RAW")
    nav_standard_file = path_processed / "nav" / "nav_standard.json"
    nav_standard_file = get_processed_folder(nav_standard_file)
    Console.info("Loading json file {}".format(nav_standard_file))

    with nav_standard_file.open("r") as nav_standard:
        parsed_json_data = json.load(nav_standard)

    start_datetime = ""
    finish_datetime = ""

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
        "usbl": {"model": "sensor"},
        "dvl": {"model": "sensor"},
        "depth": {"model": "sensor"},
        "orientation": {"model": "sensor"},
    }

    exporter = AcfrCombinedRawWriter(mission, vehicle, output_folder)

    # read in data from json file
    # i here is the number of the data packet
    for i in range(len(parsed_json_data)):
        Console.progress(i, len(parsed_json_data))
        epoch_timestamp = parsed_json_data[i]["epoch_timestamp"]
        if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:
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
                orientation.from_json(parsed_json_data[i], sensors_std["orientation"])
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

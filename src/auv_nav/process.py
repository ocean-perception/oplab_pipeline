# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import copy
import json
import threading
import time
from pathlib import Path

import numpy as np

# Import libraries
import yaml

# fmt: off
from auv_nav.localisation.dead_reckoning import dead_reckoning
from auv_nav.localisation.ekf import (
    ExtendedKalmanFilter,
    save_ekf_to_list,
    update_camera_list,
)
from auv_nav.localisation.pf import run_particle_filter
from auv_nav.localisation.usbl_filter import usbl_filter
from auv_nav.localisation.usbl_offset import usbl_offset
from auv_nav.plot.plot_process_data import (
    plot_2d_deadreckoning,
    plot_cameras_vs_time,
    plot_deadreckoning_vs_time,
    plot_ekf_rejected_measurements,
    plot_ekf_states_and_std_vs_time,
    plot_orientation_vs_time,
    plot_pf_uncertainty,
    plot_sensor_uncertainty,
    plot_synced_states_and_ekf_list_and_std_from_ekf_vs_time,
    plot_velocity_vs_time,
)
from auv_nav.sensors import (
    Altitude,
    BodyVelocity,
    Camera,
    Depth,
    InertialVelocity,
    Orientation,
    Other,
    SyncedOrientationBodyVelocity,
    Usbl,
)
from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.csv_tools import load_states, spp_csv, write_csv, write_sidescan_csv
from auv_nav.tools.dvl_level_arm import compute_angular_speeds, correct_lever_arm
from auv_nav.tools.interpolate import interpolate, interpolate_sensor_list
from auv_nav.tools.latlon_wgs84 import metres_to_latlon
from auv_nav.tools.time_conversions import (
    epoch_from_json,
    epoch_to_datetime,
    string_to_epoch,
)
from oplab import (
    Console,
    Mission,
    Vehicle,
    get_config_folder,
    get_processed_folder,
    valid_dive,
)

# fmt: on

"""
Assumes filename_camera of 1, 2, and 3 contains the image number between the
last 11 and 4 characters for appropriate csv pose estimate files output.
e.g. 'Xviii/Cam51707923/0094853.raw' or 'LM165/001/image0001011.tif'

Scripts to extract data from nav_standard.json, and combined.auv.raw an save
csv files and, if plot is True, save plots
"""


def process(
    filepath,
    force_overwite,
    start_datetime,
    finish_datetime,
    compute_relative_pose_uncertainty=False,
    start_image_identifier=None,
    end_image_identifier=None,
):
    if compute_relative_pose_uncertainty and (
        start_image_identifier is None or end_image_identifier is None
    ):
        Console.quit(
            "start_image_identifier and end_image_identifier need to provided when compute_relative_pose_uncertainty "
            "is enabled."
        )

    # placeholders
    interpolate_remove_flag = False

    # selected start and finish time
    epoch_start_time = 0
    epoch_finish_time = 0

    # velocity body placeholders (DVL)
    velocity_body_list = []
    # velocity inertial placeholders
    velocity_inertial_list = []
    # orientation placeholders (INS)
    orientation_list = []
    # depth placeholders
    depth_list = []
    # altitude placeholders
    altitude_list = []
    # USBL placeholders
    usbl_list = []

    # camera1 placeholders
    camera1_list = []
    camera1_dr_list = []
    camera1_ekf_list = []
    camera1_pf_list = []
    # camera2 placeholders
    camera2_list = []
    camera2_dr_list = []
    camera2_ekf_list = []
    camera2_pf_list = []
    # camera3 placeholders
    camera3_list = []
    camera3_dr_list = []
    camera3_ekf_list = []
    camera3_pf_list = []

    ekf_list = []

    # placeholders for interpolated velocity body measurements based on
    # orientation and transformed coordinates
    dead_reckoning_centre_list = []
    dead_reckoning_dvl_list = []

    # placeholders for dvl_imu_data fused with usbl_data using particle filter
    pf_fusion_dvl_list = []
    pf_fusion_centre_list = []
    pf_usbl_datapoints = []
    pf_particles_list = []
    pf_northings_std = []
    pf_eastings_std = []
    pf_yaw_std = []

    # placeholders for chemical data
    chemical_list = []
    # chemical_ekf_list = []
    # chemical_pf_list = []

    # load auv_nav.yaml for particle filter and other setup
    filepath = Path(filepath).resolve()
    filepath = get_processed_folder(filepath)
    localisation_file = filepath / "auv_nav.yaml"
    localisation_file = get_config_folder(localisation_file)

    # check that it is a valid dive folder
    if not valid_dive(filepath):
        Console.error(
            "The dive folder supplied does not contain any mission or vehicle",
            "YAML files. Is the path correct?",
        )
        Console.quit("Invalid path")

    # check if auv_nav.yaml file exist, if not, generate one with default
    # settings
    if localisation_file.exists():
        Console.info("Loading existing auv_nav.yaml at {}".format(localisation_file))
    else:
        root = Path(__file__).parents[1]
        default_localisation = root / "auv_nav/default_yaml" / "auv_nav.yaml"
        Console.info("default_localisation: {}".format(default_localisation))
        Console.warn(
            "Cannot find {}, generating default from {}".format(
                localisation_file, default_localisation
            )
        )
        # save localisation yaml to processed directory
        if not localisation_file.parent.exists():
            localisation_file.parent.mkdir(parents=True)
        default_localisation.copy(localisation_file)

    # copy the configuration file
    localisation_file_processed = get_processed_folder(localisation_file)
    localisation_file.copy(localisation_file_processed)

    # Default to no EKF and PF and SPP
    particle_filter_activate = False
    ekf_activate = False
    activate_smoother = True  # Apply smoothing. Only has appliles of ekf is enabled
    mahalanobis_distance_threshold = 3.0
    spp_output_activate = False

    with localisation_file.open("r") as stream:
        load_localisation = yaml.safe_load(stream)
        if "usbl_filter" in load_localisation:
            usbl_filter_activate = load_localisation["usbl_filter"]["activate"]
            max_auv_speed = load_localisation["usbl_filter"]["max_auv_speed"]
            sigma_factor = load_localisation["usbl_filter"]["sigma_factor"]
        if "particle_filter" in load_localisation:
            particle_filter_activate = load_localisation["particle_filter"]["activate"]
            dvl_noise_sigma_factor = load_localisation["particle_filter"][
                "dvl_noise_sigma_factor"
            ]
            imu_noise_sigma_factor = load_localisation["particle_filter"][
                "imu_noise_sigma_factor"
            ]
            usbl_noise_sigma_factor = load_localisation["particle_filter"][
                "usbl_noise_sigma_factor"
            ]
            particles_number = load_localisation["particle_filter"]["particles_number"]
            particles_time_interval = load_localisation["particle_filter"][
                "particles_plot_time_interval"
            ]
        if "std" in load_localisation:
            sensors_std = load_localisation["std"]
            if "position_xy" not in sensors_std:
                sensors_std["position_xy"] = sensors_std["usbl"]
            if "position_z" not in sensors_std:
                sensors_std["position_z"] = sensors_std["depth"]
            if "speed" not in sensors_std:
                sensors_std["speed"] = sensors_std["dvl"]
        else:
            sensors_std = {}

        # Default to use JSON uncertainties
        if (
            "position_xy" not in sensors_std
            or "model" not in sensors_std["position_xy"]
        ):
            Console.warn(
                "No uncertainty model specified for position_xy, defaulting",
                "to sensor (JSON).",
            )
            if "position_xy" not in sensors_std:
                sensors_std["position_xy"] = {}
            sensors_std["position_xy"]["model"] = "sensor"
        if "speed" not in sensors_std or "model" not in sensors_std["speed"]:
            Console.warn(
                "No uncertainty model specified for speed, defaulting to",
                "sensor (JSON).",
            )
            if "speed" not in sensors_std:
                sensors_std["speed"] = {}
            sensors_std["speed"]["model"] = "sensor"
        if "position_z" not in sensors_std or "model" not in sensors_std["position_z"]:
            Console.warn(
                "No uncertainty model specified for Depth, defaulting to",
                "sensor (JSON).",
            )
            if "position_z" not in sensors_std:
                sensors_std["position_z"] = {}
            sensors_std["position_z"]["model"] = "sensor"
        if (
            "orientation" not in sensors_std
            or "model" not in sensors_std["orientation"]
        ):
            Console.warn(
                "No uncertainty model specified for Orientation, defaulting",
                "to sensor (JSON).",
            )
            if "orientation" not in sensors_std:
                sensors_std["orientation"] = {}
            sensors_std["orientation"]["model"] = "sensor"
        if "ekf" in load_localisation:
            ekf_activate = load_localisation["ekf"]["activate"]
            if "activate_smoother" in load_localisation["ekf"]:
                activate_smoother = load_localisation["ekf"]["activate_smoother"]
            if "mahalanobis_distance_threshold" in load_localisation["ekf"]:
                mahalanobis_distance_threshold = load_localisation["ekf"][
                    "mahalanobis_distance_threshold"
                ]
            ekf_process_noise_covariance = load_localisation["ekf"][
                "process_noise_covariance"
            ]
            ekf_initial_estimate_covariance = load_localisation["ekf"][
                "initial_estimate_covariance"
            ]
            if len(ekf_process_noise_covariance) != 144:
                d = np.asarray(ekf_process_noise_covariance).reshape((15, 15))
                ekf_process_noise_covariance = d[0:12, 0:12]
                d = np.asarray(ekf_initial_estimate_covariance).reshape((15, 15))
                ekf_initial_estimate_covariance = d[0:12, 0:12]
            else:
                ekf_process_noise_covariance = np.asarray(
                    ekf_process_noise_covariance
                ).reshape((12, 12))
                ekf_initial_estimate_covariance = np.asarray(
                    ekf_initial_estimate_covariance
                ).reshape((12, 12))
            ekf_initial_estimate_covariance = ekf_initial_estimate_covariance.astype(
                float
            )
            ekf_process_noise_covariance = ekf_process_noise_covariance.astype(float)
        if "csv_output" in load_localisation:
            # csv_active
            csv_output_activate = load_localisation["csv_output"]["activate"]
            csv_usbl = load_localisation["csv_output"]["usbl"]
            csv_dr_auv_centre = load_localisation["csv_output"]["dead_reckoning"][
                "auv_centre"
            ]
            csv_dr_auv_dvl = load_localisation["csv_output"]["dead_reckoning"][
                "auv_dvl"
            ]
            csv_dr_camera_1 = load_localisation["csv_output"]["dead_reckoning"][
                "camera_1"
            ]
            csv_dr_camera_2 = load_localisation["csv_output"]["dead_reckoning"][
                "camera_2"
            ]
            csv_dr_camera_3 = load_localisation["csv_output"]["dead_reckoning"][
                "camera_3"
            ]
            csv_dr_chemical = load_localisation["csv_output"]["dead_reckoning"][
                "chemical"
            ]

            csv_pf_auv_centre = load_localisation["csv_output"]["particle_filter"][
                "auv_centre"
            ]
            csv_pf_auv_dvl = load_localisation["csv_output"]["particle_filter"][
                "auv_dvl"
            ]
            csv_pf_camera_1 = load_localisation["csv_output"]["particle_filter"][
                "camera_1"
            ]
            csv_pf_camera_2 = load_localisation["csv_output"]["particle_filter"][
                "camera_2"
            ]
            csv_pf_camera_3 = load_localisation["csv_output"]["particle_filter"][
                "camera_3"
            ]
            csv_pf_chemical = load_localisation["csv_output"]["particle_filter"][
                "chemical"
            ]

            csv_ekf_auv_centre = load_localisation["csv_output"]["ekf"]["auv_centre"]
            csv_ekf_camera_1 = load_localisation["csv_output"]["ekf"]["camera_1"]
            csv_ekf_camera_2 = load_localisation["csv_output"]["ekf"]["camera_2"]
            csv_ekf_camera_3 = load_localisation["csv_output"]["ekf"]["camera_3"]
        else:
            csv_output_activate = False
            Console.warn(
                "csv output undefined in auv_nav.yaml. Has been"
                + ' set to "False". To activate, add as per'
                + " default auv_nav.yaml found within auv_nav"
                + ' file structure and set values to "True".'
            )

        if "spp_output" in load_localisation:
            # spp_active
            spp_output_activate = load_localisation["spp_output"]["activate"]
            spp_ekf_camera_1 = load_localisation["spp_output"]["ekf"]["camera_1"]
            spp_ekf_camera_2 = load_localisation["spp_output"]["ekf"]["camera_2"]
            spp_ekf_camera_3 = load_localisation["spp_output"]["ekf"]["camera_3"]

            if spp_output_activate and not ekf_activate:
                Console.warn(
                    "SLAM++ will be disabled due to EKF being disabled.",
                    "Enable EKF to make it work.",
                )

                spp_output_activate = False
        else:
            spp_output_activate = False
            Console.warn(
                "SLAM++ output undefined in auv_nav.yaml. Has been"
                + ' set to "False". To activate, add as per'
                + " default auv_nav.yaml found within auv_nav"
                + ' file structure and set values to "True".'
            )

        if "plot_output" in load_localisation:
            plot_output_activate = load_localisation["plot_output"]["activate"]
            # pdf_plot = load_localisation["plot_output"]["pdf_plot"]
            html_plot = load_localisation["plot_output"]["html_plot"]

    Console.info("Loading vehicle.yaml")
    vehicle_file = filepath / "vehicle.yaml"
    vehicle_file = get_processed_folder(vehicle_file)
    vehicle = Vehicle(vehicle_file)

    Console.info("Loading mission.yaml")
    mission_file = filepath / "mission.yaml"
    mission_file = get_processed_folder(mission_file)
    mission = Mission(mission_file)

    camera1_offsets = [
        vehicle.camera1.surge,
        vehicle.camera1.sway,
        vehicle.camera1.heave,
    ]
    camera2_offsets = [
        vehicle.camera2.surge,
        vehicle.camera2.sway,
        vehicle.camera2.heave,
    ]

    # For BioCam, camera 3 is grayscale camera recording laser
    # For SeaXerocks, camera 3 is a separate camera
    camera3_offsets = [
        vehicle.camera3.surge,
        vehicle.camera3.sway,
        vehicle.camera3.heave,
    ]

    if mission.image.format == "biocam":
        if mission.image.cameras[0].type == "grayscale":
            camera3_offsets = [
                vehicle.camera1.surge,
                vehicle.camera1.sway,
                vehicle.camera1.heave,
            ]
        elif mission.image.cameras[1].type == "grayscale":
            camera3_offsets = [
                vehicle.camera2.surge,
                vehicle.camera2.sway,
                vehicle.camera2.heave,
            ]
        else:
            Console.quit("BioCam format is expected to have a grayscale camera.")

    chemical_offset = [
        vehicle.chemical.surge,
        vehicle.chemical.sway,
        vehicle.chemical.heave,
    ]

    outpath = filepath / "nav"

    nav_standard_file = outpath / "nav_standard.json"
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

    # read in data from json file
    # i here is the number of the data packet
    for i in range(len(parsed_json_data)):
        if parsed_json_data[i] is None:
            continue
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
                                parsed_json_data[i], sensors_std["speed"]
                            )
                            velocity_body_list.append(velocity_body)
                if "inertial" in parsed_json_data[i]["frame"]:
                    velocity_inertial = InertialVelocity()
                    velocity_inertial.from_json(parsed_json_data[i])
                    velocity_inertial_list.append(velocity_inertial)

            if "orientation" in parsed_json_data[i]["category"]:
                orientation = Orientation()
                orientation.from_json(parsed_json_data[i], sensors_std["orientation"])
                orientation_list.append(orientation)

            if "depth" in parsed_json_data[i]["category"]:
                depth = Depth()
                depth.from_json(parsed_json_data[i], sensors_std["position_z"])
                depth_list.append(depth)

            if "altitude" in parsed_json_data[i]["category"]:
                altitude = Altitude()
                altitude.from_json(parsed_json_data[i])
                altitude_list.append(altitude)

            if "usbl" in parsed_json_data[i]["category"]:
                usbl = Usbl()
                usbl.from_json(parsed_json_data[i], sensors_std["position_xy"])
                usbl_list.append(usbl)

            if "image" in parsed_json_data[i]["category"]:
                camera1 = Camera()
                # LC
                camera1.from_json(parsed_json_data[i], "camera1")
                camera1_list.append(camera1)
                if len(mission.image.cameras) > 1:
                    camera2 = Camera()
                    camera2.from_json(parsed_json_data[i], "camera2")
                    camera2_list.append(camera2)

            if "laser" in parsed_json_data[i]["category"]:
                camera3 = Camera()
                camera3.from_json(parsed_json_data[i], "camera3")
                camera3_list.append(camera3)

            if "chemical" in parsed_json_data[i]["category"]:
                chemical = Other()
                chemical.from_json(parsed_json_data[i])
                chemical_list.append(chemical)

    camera1_dr_list = copy.deepcopy(camera1_list)
    camera2_dr_list = copy.deepcopy(camera2_list)
    camera3_dr_list = copy.deepcopy(camera3_list)

    if particle_filter_activate:
        camera1_pf_list = copy.deepcopy(camera1_list)
        camera2_pf_list = copy.deepcopy(camera2_list)
        camera3_pf_list = copy.deepcopy(camera3_list)
        # chemical_pf_list = copy.deepcopy(chemical_list)

    if ekf_activate:
        camera1_ekf_list = copy.deepcopy(camera1_list)
        camera2_ekf_list = copy.deepcopy(camera2_list)
        camera3_ekf_list = copy.deepcopy(camera3_list)
        camera3_ekf_list_at_dvl = copy.deepcopy(camera3_list)
        # chemical_ekf_list = copy.deepcopy(chemical_list)

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
    if renavpath.is_dir() is False:
        try:
            renavpath.mkdir()
        except Exception as e:
            print("Warning:", e)
    elif (
        renavpath.is_dir()
        and not force_overwite
        and not compute_relative_pose_uncertainty
    ):
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

    Console.info("Parsing has found:")
    Console.info("\t* Velocity_body: {} elements".format(len(velocity_body_list)))
    Console.info(
        "\t* Velocity_inertial: {} elements".format(len(velocity_inertial_list))
    )
    Console.info("\t* Orientation: {} elements".format(len(orientation_list)))
    Console.info("\t* Depth: {} elements".format(len(depth_list)))
    Console.info("\t* Altitude: {} elements".format(len(altitude_list)))
    Console.info("\t* Usbl: {} elements".format(len(usbl_list)))

    Console.info("Writing outputs to: {}".format(renavpath))
    raw_sensor_path = renavpath / "csv" / "sensor_values_and_uncertainties"

    threads = []
    mutex = threading.Lock()
    t = threading.Thread(
        target=write_csv,
        args=[raw_sensor_path, velocity_body_list, "velocity_body"],
        kwargs={"mutex": mutex},
    )
    t.start()
    threads.append(t)
    t = threading.Thread(
        target=write_csv,
        args=[raw_sensor_path, altitude_list, "altitude"],
        kwargs={"mutex": mutex},
    )
    t.start()
    threads.append(t)
    t = threading.Thread(
        target=write_csv,
        args=[raw_sensor_path, orientation_list, "orientation"],
        kwargs={"mutex": mutex},
    )
    t.start()
    threads.append(t)
    t = threading.Thread(
        target=write_csv,
        args=[raw_sensor_path, depth_list, "depth"],
        kwargs={"mutex": mutex},
    )
    t.start()
    threads.append(t)
    t = threading.Thread(
        target=write_csv,
        args=[raw_sensor_path, usbl_list, "usbl"],
        kwargs={"mutex": mutex},
    )
    t.start()
    threads.append(t)

    # interpolate to find the appropriate depth to compute seafloor depth for
    # each altitude measurement
    j = 0
    non_processed_altitude_index_list = []
    for i in range(len(altitude_list)):
        while (
            j < len(depth_list) - 1
            and depth_list[j].epoch_timestamp < altitude_list[i].epoch_timestamp
        ):
            j = j + 1

        if j > 0:
            altitude_list[i].seafloor_depth = (
                interpolate(
                    altitude_list[i].epoch_timestamp,
                    depth_list[j - 1].epoch_timestamp,
                    depth_list[j].epoch_timestamp,
                    depth_list[j - 1].depth,
                    depth_list[j].depth,
                )
                + altitude_list[i].altitude
            )
        else:
            non_processed_altitude_index_list.append(i)

    non_processed_altitude_index_list.reverse()
    for i in non_processed_altitude_index_list:
        altitude_list[i].seafloor_depth = altitude_list[i + 1].seafloor_depth

    if len(orientation_list) == 0 or len(velocity_body_list) == 0:
        Console.quit(
            "orientation_list and velocity_body_list must not be empty but at",
            "least one of them is empty (orientation_list contains",
            len(orientation_list),
            "elements and velocity_body_list",
            "conatains",
            len(velocity_body_list),
            "elements)",
        )

    # perform usbl_filter
    if usbl_filter_activate:
        usbl_list_no_dist_filter, usbl_list = usbl_filter(
            usbl_list, depth_list, sigma_factor, max_auv_speed
        )
        if len(usbl_list) == 0:
            Console.warn("Filtering USBL measurements lead to an empty list. ")
            Console.warn(" * Is USBL reliable?")
            Console.warn(" * Can you change filter parameters?")

    """
    Perform coordinate transformations and interpolations of state data
    to velocity_body time stamps with sensor position offset and perform
    dead reckoning. Assumes the first measurement of velocity_body is the
    beginning of mission. May not be robust to non-continuous measurements
    will any (sudden start and stop) affect it?
    """
    j = 0
    k = 0
    n = 0
    start_interpolate_index = 0

    while (
        orientation_list[start_interpolate_index].epoch_timestamp
        < velocity_body_list[0].epoch_timestamp
    ):
        start_interpolate_index += 1

    # if start_interpolate_index==0:
    # do something? because time_orientation may be way before
    # time_velocity_body

    if start_interpolate_index == 1:
        interpolate_remove_flag = True

    # time_velocity_body)):
    for i in range(start_interpolate_index, len(orientation_list)):

        # interpolate to find the appropriate dvl time for the orientation
        # measurements
        if orientation_list[i].epoch_timestamp > velocity_body_list[-1].epoch_timestamp:
            break

        while (
            j < len(velocity_body_list) - 1
            and orientation_list[i].epoch_timestamp
            > velocity_body_list[j + 1].epoch_timestamp
        ):
            j += 1

        dead_reckoning_dvl = SyncedOrientationBodyVelocity()
        dead_reckoning_dvl.epoch_timestamp = orientation_list[i].epoch_timestamp
        dead_reckoning_dvl.roll = orientation_list[i].roll
        dead_reckoning_dvl.pitch = orientation_list[i].pitch
        dead_reckoning_dvl.yaw = orientation_list[i].yaw
        dead_reckoning_dvl.roll_std = orientation_list[i].roll_std
        dead_reckoning_dvl.pitch_std = orientation_list[i].pitch_std
        dead_reckoning_dvl.yaw_std = orientation_list[i].yaw_std
        dead_reckoning_dvl.x_velocity = interpolate(
            orientation_list[i].epoch_timestamp,
            velocity_body_list[j].epoch_timestamp,
            velocity_body_list[j + 1].epoch_timestamp,
            velocity_body_list[j].x_velocity,
            velocity_body_list[j + 1].x_velocity,
        )
        dead_reckoning_dvl.y_velocity = interpolate(
            orientation_list[i].epoch_timestamp,
            velocity_body_list[j].epoch_timestamp,
            velocity_body_list[j + 1].epoch_timestamp,
            velocity_body_list[j].y_velocity,
            velocity_body_list[j + 1].y_velocity,
        )
        dead_reckoning_dvl.z_velocity = interpolate(
            orientation_list[i].epoch_timestamp,
            velocity_body_list[j].epoch_timestamp,
            velocity_body_list[j + 1].epoch_timestamp,
            velocity_body_list[j].z_velocity,
            velocity_body_list[j + 1].z_velocity,
        )
        dead_reckoning_dvl.x_velocity_std = interpolate(
            orientation_list[i].epoch_timestamp,
            velocity_body_list[j].epoch_timestamp,
            velocity_body_list[j + 1].epoch_timestamp,
            velocity_body_list[j].x_velocity_std,
            velocity_body_list[j + 1].x_velocity_std,
        )
        dead_reckoning_dvl.y_velocity_std = interpolate(
            orientation_list[i].epoch_timestamp,
            velocity_body_list[j].epoch_timestamp,
            velocity_body_list[j + 1].epoch_timestamp,
            velocity_body_list[j].y_velocity_std,
            velocity_body_list[j + 1].y_velocity_std,
        )
        dead_reckoning_dvl.z_velocity_std = interpolate(
            orientation_list[i].epoch_timestamp,
            velocity_body_list[j].epoch_timestamp,
            velocity_body_list[j + 1].epoch_timestamp,
            velocity_body_list[j].z_velocity_std,
            velocity_body_list[j + 1].z_velocity_std,
        )

        linear_speeds = [
            dead_reckoning_dvl.x_velocity,
            dead_reckoning_dvl.y_velocity,
            dead_reckoning_dvl.z_velocity,
        ]
        angular_speeds = compute_angular_speeds(orientation_list, i)
        dvl_pos_on_vehicle = [
            vehicle.dvl.surge,
            vehicle.dvl.sway,
            vehicle.dvl.heave,
        ]
        [vx, vy, vz] = correct_lever_arm(
            linear_speeds, angular_speeds, dvl_pos_on_vehicle
        )
        dead_reckoning_dvl.x_velocity = vx
        dead_reckoning_dvl.y_velocity = vy
        dead_reckoning_dvl.z_velocity = vz

        [north_velocity, east_velocity, down_velocity] = body_to_inertial(
            orientation_list[i].roll,
            orientation_list[i].pitch,
            orientation_list[i].yaw,
            dead_reckoning_dvl.x_velocity,
            dead_reckoning_dvl.y_velocity,
            dead_reckoning_dvl.z_velocity,
        )

        dead_reckoning_dvl.north_velocity = north_velocity
        dead_reckoning_dvl.east_velocity = east_velocity
        dead_reckoning_dvl.down_velocity = down_velocity

        [north_velocity_std, east_velocity_std, down_velocity_std] = body_to_inertial(
            orientation_list[i].roll,
            orientation_list[i].pitch,
            orientation_list[i].yaw,
            dead_reckoning_dvl.x_velocity_std,
            dead_reckoning_dvl.y_velocity_std,
            dead_reckoning_dvl.z_velocity_std,
        )

        dead_reckoning_dvl.north_velocity_std = north_velocity_std
        dead_reckoning_dvl.east_velocity_std = east_velocity_std
        dead_reckoning_dvl.down_velocity_std = down_velocity_std

        while (
            n < len(altitude_list) - 1
            and orientation_list[i].epoch_timestamp > altitude_list[n].epoch_timestamp
        ):
            n += 1
        dead_reckoning_dvl.altitude = interpolate(
            orientation_list[i].epoch_timestamp,
            altitude_list[n - 1].epoch_timestamp,
            altitude_list[n].epoch_timestamp,
            altitude_list[n - 1].altitude,
            altitude_list[n].altitude,
        )

        while (
            k < len(depth_list) - 1
            and depth_list[k].epoch_timestamp < orientation_list[i].epoch_timestamp
        ):
            k += 1
        # interpolate to find the appropriate depth for dead_reckoning
        dead_reckoning_dvl.depth = interpolate(
            orientation_list[i].epoch_timestamp,
            depth_list[k - 1].epoch_timestamp,
            depth_list[k].epoch_timestamp,
            depth_list[k - 1].depth,
            depth_list[k].depth,
        )
        dead_reckoning_dvl.depth_std = interpolate(
            orientation_list[i].epoch_timestamp,
            depth_list[k - 1].epoch_timestamp,
            depth_list[k].epoch_timestamp,
            depth_list[k - 1].depth_std,
            depth_list[k].depth_std,
        )
        dead_reckoning_dvl_list.append(dead_reckoning_dvl)

    # dead reckoning solution
    dead_reckoning_dvl_list[0].northings = 0.0
    dead_reckoning_dvl_list[0].eastings = 0.0
    for i in range(len(dead_reckoning_dvl_list)):
        # dead reckoning solution
        if i > 0:
            [
                dead_reckoning_dvl_list[i].northings,
                dead_reckoning_dvl_list[i].eastings,
            ] = dead_reckoning(
                dead_reckoning_dvl_list[i].epoch_timestamp,
                dead_reckoning_dvl_list[i - 1].epoch_timestamp,
                dead_reckoning_dvl_list[i].north_velocity,
                dead_reckoning_dvl_list[i - 1].north_velocity,
                dead_reckoning_dvl_list[i].east_velocity,
                dead_reckoning_dvl_list[i - 1].east_velocity,
                dead_reckoning_dvl_list[i - 1].northings,
                dead_reckoning_dvl_list[i - 1].eastings,
            )

    # offset sensor to plot origin/centre of vehicle
    dead_reckoning_centre_list = copy.deepcopy(dead_reckoning_dvl_list)  # [:] #.copy()
    for i in range(len(dead_reckoning_centre_list)):
        [x_offset, y_offset, a_offset] = body_to_inertial(
            dead_reckoning_centre_list[i].roll,
            dead_reckoning_centre_list[i].pitch,
            dead_reckoning_centre_list[i].yaw,
            vehicle.origin.surge - vehicle.dvl.surge,
            vehicle.origin.sway - vehicle.dvl.sway,
            vehicle.origin.heave - vehicle.dvl.heave,
        )
        [_, _, z_offset] = body_to_inertial(
            dead_reckoning_centre_list[i].roll,
            dead_reckoning_centre_list[i].pitch,
            dead_reckoning_centre_list[i].yaw,
            vehicle.origin.surge - vehicle.depth.surge,
            vehicle.origin.sway - vehicle.depth.sway,
            vehicle.origin.heave - vehicle.depth.heave,
        )
        dead_reckoning_centre_list[i].northings += x_offset
        dead_reckoning_centre_list[i].eastings += y_offset
        dead_reckoning_centre_list[i].altitude -= a_offset
        dead_reckoning_centre_list[i].depth += z_offset
    # correct for altitude and depth offset too!

    # remove first term if first time_orientation is < velocity_body time
    if interpolate_remove_flag:

        # del time_orientation[0]
        del dead_reckoning_centre_list[0]
        del dead_reckoning_dvl_list[0]
        interpolate_remove_flag = False  # reset flag
    Console.info(
        "Completed interpolation and coordinate transfomations for",
        "velocity_body",
    )

    # perform interpolations of state data to velocity_inertial time stamps
    # (without sensor offset and correct imu to dvl flipped interpolation)
    # and perform deadreckoning
    # initialise counters for interpolation
    if len(velocity_inertial_list) > 0:
        # dead_reckoning_built_in_values
        j = 0
        k = 0

        for i in range(len(velocity_inertial_list)):
            while (
                j < len(orientation_list) - 1
                and orientation_list[j].epoch_timestamp
                < velocity_inertial_list[i].epoch_timestamp
            ):
                j = j + 1

            if j == 1:
                interpolate_remove_flag = True
            else:
                velocity_inertial_list[i].roll = interpolate(
                    velocity_inertial_list[i].epoch_timestamp,
                    orientation_list[j - 1].epoch_timestamp,
                    orientation_list[j].epoch_timestamp,
                    orientation_list[j - 1].roll,
                    orientation_list[j].roll,
                )
                velocity_inertial_list[i].pitch = interpolate(
                    velocity_inertial_list[i].epoch_timestamp,
                    orientation_list[j - 1].epoch_timestamp,
                    orientation_list[j].epoch_timestamp,
                    orientation_list[j - 1].pitch,
                    orientation_list[j].pitch,
                )

                if abs(orientation_list[j].yaw - orientation_list[j - 1].yaw) > 180:
                    if orientation_list[j].yaw > orientation_list[j - 1].yaw:
                        velocity_inertial_list[i].yaw = interpolate(
                            velocity_inertial_list[i].epoch_timestamp,
                            orientation_list[j - 1].epoch_timestamp,
                            orientation_list[j].epoch_timestamp,
                            orientation_list[j - 1].yaw,
                            orientation_list[j].yaw - 360,
                        )
                    else:
                        velocity_inertial_list[i].yaw = interpolate(
                            velocity_inertial_list[i].epoch_timestamp,
                            orientation_list[j - 1].epoch_timestamp,
                            orientation_list[j].epoch_timestamp,
                            orientation_list[j - 1].yaw - 360,
                            orientation_list[j].yaw,
                        )

                    if velocity_inertial_list[i].yaw < 0:
                        velocity_inertial_list[i].yaw += 360

                    elif velocity_inertial_list[i].yaw > 360:
                        velocity_inertial_list[i].yaw -= 360
                else:
                    velocity_inertial_list[i].yaw = interpolate(
                        velocity_inertial_list[i].epoch_timestamp,
                        orientation_list[j - 1].epoch_timestamp,
                        orientation_list[j].epoch_timestamp,
                        orientation_list[j - 1].yaw,
                        orientation_list[j].yaw,
                    )

            while (
                k < len(depth_list) - 1
                and depth_list[k].epoch_timestamp
                < velocity_inertial_list[i].epoch_timestamp
            ):
                k = k + 1

            if k >= 1:
                velocity_inertial_list[i].depth = interpolate(
                    velocity_inertial_list[i].epoch_timestamp,
                    depth_list[k - 1].epoch_timestamp,
                    depth_list[k].epoch_timestamp,
                    depth_list[k - 1].depth,
                    depth_list[k].depth,
                )  # depth directly interpolated from depth sensor

        velocity_inertial_list[0].northings = 0.0
        velocity_inertial_list[0].eastings = 0.0
        for i in range(len(velocity_inertial_list)):
            if i > 0:
                [
                    velocity_inertial_list[i].northings,
                    velocity_inertial_list[i].eastings,
                ] = dead_reckoning(
                    velocity_inertial_list[i].epoch_timestamp,
                    velocity_inertial_list[i - 1].epoch_timestamp,
                    velocity_inertial_list[i].north_velocity,
                    velocity_inertial_list[i - 1].north_velocity,
                    velocity_inertial_list[i].east_velocity,
                    velocity_inertial_list[i - 1].east_velocity,
                    velocity_inertial_list[i - 1].northings,
                    velocity_inertial_list[i - 1].eastings,
                )

        if interpolate_remove_flag:
            del velocity_inertial_list[0]
            interpolate_remove_flag = False  # reset flag
        Console.info(
            "Completed interpolation and coordinate transfomations for",
            "velocity_inertial",
        )

    # offset velocity DR by average usbl estimate
    # offset velocity body DR by average usbl estimate
    if len(usbl_list) > 0:
        [northings_usbl_interpolated, eastings_usbl_interpolated] = usbl_offset(
            [i.epoch_timestamp for i in dead_reckoning_centre_list],
            [i.northings for i in dead_reckoning_centre_list],
            [i.eastings for i in dead_reckoning_centre_list],
            [i.epoch_timestamp for i in usbl_list],
            [i.northings for i in usbl_list],
            [i.eastings for i in usbl_list],
        )
        for i in range(len(dead_reckoning_centre_list)):
            dead_reckoning_centre_list[i].northings += northings_usbl_interpolated
            dead_reckoning_centre_list[i].eastings += eastings_usbl_interpolated
            (
                dead_reckoning_centre_list[i].latitude,
                dead_reckoning_centre_list[i].longitude,
            ) = metres_to_latlon(
                mission.origin.latitude,
                mission.origin.longitude,
                dead_reckoning_centre_list[i].eastings,
                dead_reckoning_centre_list[i].northings,
            )
        for i in range(len(dead_reckoning_dvl_list)):
            dead_reckoning_dvl_list[i].northings += northings_usbl_interpolated
            dead_reckoning_dvl_list[i].eastings += eastings_usbl_interpolated
            (
                dead_reckoning_dvl_list[i].latitude,
                dead_reckoning_dvl_list[i].longitude,
            ) = metres_to_latlon(
                mission.origin.latitude,
                mission.origin.longitude,
                dead_reckoning_dvl_list[i].eastings,
                dead_reckoning_dvl_list[i].northings,
            )

        # offset velocity inertial DR by average usbl estimate
        if len(velocity_inertial_list) > 0:
            [northings_usbl_interpolated, eastings_usbl_interpolated] = usbl_offset(
                [i.epoch_timestamp for i in velocity_inertial_list],
                [i.northings for i in velocity_inertial_list],
                [i.eastings for i in velocity_inertial_list],
                [i.epoch_timestamp for i in usbl_list],
                [i.northings for i in usbl_list],
                [i.eastings for i in usbl_list],
            )
            for i in range(len(velocity_inertial_list)):
                velocity_inertial_list[i].northings += northings_usbl_interpolated
                velocity_inertial_list[i].eastings += eastings_usbl_interpolated
                (
                    velocity_inertial_list[i].latitude,
                    velocity_inertial_list[i].longitude,
                ) = metres_to_latlon(
                    mission.origin.latitude,
                    mission.origin.longitude,
                    velocity_inertial_list[i].eastings,
                    velocity_inertial_list[i].northings,
                )
    else:
        Console.warn("There are no USBL measurements. Starting DR at origin...")

    # particle filter data fusion of usbl_data and dvl_imu_data
    if particle_filter_activate and len(usbl_list) > 0:
        Console.info("Running PF...")
        pf_start_time = time.time()
        [
            pf_fusion_dvl_list,
            pf_usbl_datapoints,
            pf_particles_list,
            pf_northings_std,
            pf_eastings_std,
            pf_yaw_std,
        ] = run_particle_filter(
            copy.deepcopy(usbl_list),
            copy.deepcopy(dead_reckoning_dvl_list),
            particles_number,
            sensors_std,
            dvl_noise_sigma_factor,
            imu_noise_sigma_factor,
            usbl_noise_sigma_factor,
            measurement_update_flag=True,
        )
        pf_end_time = time.time()
        pf_elapsed_time = pf_end_time - pf_start_time
        # maybe save this as text alongside plotly outputs
        Console.info(
            "PF with {} particles took {} mins".format(
                particles_number, pf_elapsed_time / 60
            )
        )
        pf_fusion_centre_list = copy.deepcopy(pf_fusion_dvl_list)
        for i in range(len(pf_fusion_centre_list)):
            pf_fusion_dvl_list[i].latitude,
            pf_fusion_dvl_list[i].longitude = metres_to_latlon(
                mission.origin.latitude,
                mission.origin.longitude,
                pf_fusion_dvl_list[i].eastings,
                pf_fusion_dvl_list[i].northings,
            )
            [x_offset, y_offset, z_offset] = body_to_inertial(
                pf_fusion_centre_list[i].roll,
                pf_fusion_centre_list[i].pitch,
                pf_fusion_centre_list[i].yaw,
                vehicle.origin.surge - vehicle.dvl.surge,
                vehicle.origin.sway - vehicle.dvl.sway,
                vehicle.origin.heave - vehicle.dvl.heave,
            )
            pf_fusion_centre_list[i].northings += x_offset
            pf_fusion_centre_list[i].eastings += y_offset
            lat, lon = metres_to_latlon(
                mission.origin.latitude,
                mission.origin.longitude,
                pf_fusion_centre_list[i].eastings,
                pf_fusion_centre_list[i].northings,
            )
            pf_fusion_centre_list[i].latitude = lat
            pf_fusion_centre_list[i].longitude = lon

    origin_offsets = [
        vehicle.origin.surge,
        vehicle.origin.sway,
        vehicle.origin.heave,
    ]
    latlon_reference = [mission.origin.latitude, mission.origin.longitude]

    if len(camera1_dr_list) > 1:
        interpolate_sensor_list(
            camera1_dr_list,
            mission.image.cameras[0].name,
            camera1_offsets,
            origin_offsets,
            latlon_reference,
            dead_reckoning_centre_list,
        )
    if len(camera2_dr_list) > 1:
        interpolate_sensor_list(
            camera2_dr_list,
            mission.image.cameras[1].name,
            camera2_offsets,
            origin_offsets,
            latlon_reference,
            dead_reckoning_centre_list,
        )
    if len(camera3_dr_list) > 1:
        if len(mission.image.cameras) > 2:
            interpolate_sensor_list(
                camera3_dr_list,
                mission.image.cameras[2].name,
                camera3_offsets,
                origin_offsets,
                latlon_reference,
                dead_reckoning_centre_list,
            )
        elif len(mission.image.cameras) == 2:  # Biocam
            interpolate_sensor_list(
                camera3_dr_list,
                mission.image.cameras[1].name + "_laser",
                camera3_offsets,
                origin_offsets,
                latlon_reference,
                dead_reckoning_centre_list,
            )

    if len(pf_fusion_centre_list) > 1:
        if len(camera1_pf_list) > 1:
            interpolate_sensor_list(
                camera1_pf_list,
                mission.image.cameras[0].name,
                camera1_offsets,
                origin_offsets,
                latlon_reference,
                pf_fusion_centre_list,
            )
        if len(camera2_pf_list) > 1:
            interpolate_sensor_list(
                camera2_pf_list,
                mission.image.cameras[1].name,
                camera2_offsets,
                origin_offsets,
                latlon_reference,
                pf_fusion_centre_list,
            )
        if len(camera3_pf_list) > 1:
            if len(mission.image.cameras) > 2:
                interpolate_sensor_list(
                    camera3_pf_list,
                    mission.image.cameras[2].name,
                    camera3_offsets,
                    origin_offsets,
                    latlon_reference,
                    pf_fusion_centre_list,
                )
            elif len(mission.image.cameras) == 2:  # Biocam
                interpolate_sensor_list(
                    camera3_pf_list,
                    mission.image.cameras[1].name + "_laser",
                    camera3_offsets,
                    origin_offsets,
                    latlon_reference,
                    pf_fusion_centre_list,
                )

    usbl_to_dvl = [
        vehicle.dvl.surge - vehicle.usbl.surge,
        vehicle.dvl.sway - vehicle.usbl.sway,
        vehicle.dvl.heave - vehicle.usbl.heave,
    ]
    depth_to_dvl = [
        vehicle.dvl.surge - vehicle.depth.surge,
        vehicle.dvl.sway - vehicle.depth.sway,
        vehicle.dvl.heave - vehicle.depth.heave,
    ]

    ekf_initial_state = copy.deepcopy(dead_reckoning_dvl_list[0])
    [_, _, z_offset] = body_to_inertial(
        ekf_initial_state.roll,
        ekf_initial_state.pitch,
        ekf_initial_state.yaw,
        depth_to_dvl[0],
        depth_to_dvl[1],
        depth_to_dvl[2],
    )
    ekf_initial_state.depth += z_offset

    ekf_end_time = dead_reckoning_dvl_list[-1].epoch_timestamp

    if compute_relative_pose_uncertainty:
        ekf_activate = True

        # Load previously computed states and covariances
        ekf_state_file_path = (
            renavpath
            / "csv"
            / "ekf"
            / ("auv_ekf_" + mission.image.cameras[2].name + "_at_dvl.csv")
        )
        laser_camera_at_dvl_states = load_states(
            ekf_state_file_path, start_image_identifier, end_image_identifier
        )

        if not laser_camera_at_dvl_states:
            Console.quit(
                "The pose of the indicated image was not found in the file provided. Please check your input."
            )

        # Initialise starting state for EKF
        ekf_initial_state = copy.deepcopy(laser_camera_at_dvl_states[0])
        # When reading the timestamp from the csv it sometimes is converted to a slightly different timestamp than the
        # timestamp read from the image file times file, due to the limited precision of floats.
        # If it is read as a time even a tiny bit later than in the images file times, the EKF solution will seemingly
        # start only after the image was taken and the program will crash when attempting to find the EKF position for
        # the first image. Subtracting 1 microsecond from the timesamp solves this.
        ekf_initial_state.epoch_timestamp -= 0.000001

        ekf_initial_estimate_covariance[
            0:6, 0:6
        ] = 0  # Initialise starting covariance for EKF (set it to 0)
        ekf_end_time = laser_camera_at_dvl_states[-1].epoch_timestamp

        # Initialise camera list (i.e. timestamps) for which the EKF will compute the states
        camera3_ekf_list_cropped = []
        use_camera = False
        for i in camera3_ekf_list:
            if i.filename == start_image_identifier:
                use_camera = True
            if use_camera:
                camera3_ekf_list_cropped.append(i)
            if i.filename == end_image_identifier:
                break
        if len(camera3_ekf_list_cropped) == 0:
            Console.quit(
                "camera3_ekf_list_cropped is empty. Check the",
                "start_image_identifier and start_image_identifier you ",
                "indicated",
            )

    if ekf_activate:
        # velocity_body_list, list of BodyVelocity()
        # orientation_list, list of Orientation()
        # depth_list, list of Depth()
        # usbl_list, list of Usbl()
        Console.info(
            "Running EKF ("
            + ("with" if activate_smoother else "without")
            + " smoother)..."
        )
        ekf_start_time = time.time()

        # Aggregate timestamps to run EKF only once
        ekf_timestamps = []
        if camera1_ekf_list:
            camera1_timestamp_list = [x.epoch_timestamp for x in camera1_ekf_list]
            ekf_timestamps += camera1_timestamp_list
        if camera2_ekf_list:
            camera2_timestamp_list = [x.epoch_timestamp for x in camera2_ekf_list]
            ekf_timestamps += camera2_timestamp_list
        if camera3_ekf_list:
            camera3_timestamp_list = [x.epoch_timestamp for x in camera3_ekf_list]
            ekf_timestamps += camera3_timestamp_list
        # Sort timestamps and remove duplicates in place
        ekf_timestamps = sorted(set(ekf_timestamps))

        ekf = ExtendedKalmanFilter(
            ekf_initial_state,
            ekf_end_time,
            ekf_initial_estimate_covariance,
            ekf_process_noise_covariance,
            sensors_std,
            usbl_list,
            depth_list,
            orientation_list,
            velocity_body_list,
            mahalanobis_distance_threshold,
            activate_smoother,
            usbl_to_dvl,
            depth_to_dvl,
        )
        ekf.run(ekf_timestamps)
        ekf_elapsed_time = time.time() - ekf_start_time
        Console.info("EKF took {} mins".format(ekf_elapsed_time / 60))
        ekf_states = ekf.get_smoothed_result()
        ekf_list = save_ekf_to_list(
            ekf_states, mission, vehicle, dead_reckoning_dvl_list
        )
        ekf_list_dvl = save_ekf_to_list(
            ekf_states, mission, vehicle, dead_reckoning_dvl_list, False
        )

    if compute_relative_pose_uncertainty:
        camera3_ekf_list_cropped = update_camera_list(
            camera3_ekf_list_cropped,
            ekf_list,
            [0, 0, 0],
            [0, 0, 0],
            latlon_reference,
        )
        assert len(camera3_ekf_list_cropped) == len(laser_camera_at_dvl_states)

        # Compute uncertainties with poses
        # Compute uncertainties by subtracting original states (-> uncertainties are summed)
        for i in range(1, len(laser_camera_at_dvl_states)):
            laser_camera_at_dvl_states[i].covariance += laser_camera_at_dvl_states[
                0
            ].covariance
        laser_camera_at_dvl_states[0].covariance[:, :] = 0

        # Plot uncertainties
        plots_folder = renavpath / "interactive_plots"

        # Plot uncertainties based on subtracting positions
        args = [
            laser_camera_at_dvl_states,
            plots_folder / "based_on_subtraction_at_dvl",
        ]
        t = threading.Thread(target=plot_cameras_vs_time, args=args)
        t.start()
        threads.append(t)

        # Plot uncertainties based on EKF propagation
        args = [
            laser_camera_at_dvl_states,
            camera3_ekf_list_cropped,
            plots_folder / "based_on_ekf_propagation_at_dvl",
        ]
        t = threading.Thread(
            target=plot_synced_states_and_ekf_list_and_std_from_ekf_vs_time, args=args
        )
        t.start()
        threads.append(t)

        # Write out poses with uncertainties to new file
        ekf_csv_folder = renavpath / "csv" / "ekf"
        if len(mission.image.cameras) > 2:
            filename_cov_from_ekf = (
                "auv_ekf_cov_based_on_ekf_propagation_"
                + mission.image.cameras[2].name
                + "_at_dvl"
            )
            filename_cov_from_subtract = (
                "auv_ekf_cov_based_on_subtraction_"
                + mission.image.cameras[2].name
                + "_at_dvl"
            )
        elif len(mission.image.cameras) == 2:
            filename_cov_from_ekf = (
                "auv_ekf_cov_based_on_ekf_propagation_"
                + mission.image.cameras[1].name
                + "_laser_at_dvl"
            )
            filename_cov_from_subtract = (
                "auv_ekf_cov_based_on_subtraction_"
                + mission.image.cameras[1].name
                + "_laser_at_dvl"
            )

        # Write out uncertainties based on subtracting positions
        args = [ekf_csv_folder, laser_camera_at_dvl_states, filename_cov_from_subtract]
        t = threading.Thread(target=write_csv, args=args)
        t.start()
        threads.append(t)

        # Write out uncertainties based on EKF propagation to csv file
        # Note:
        # Here we are using the the new poses, whereas we only want to use the new covariances but the original poses.
        # But when using the data in laser_bathymetry, the original pose file can be used witht this covariance file.
        args = [ekf_csv_folder, camera3_ekf_list_cropped, filename_cov_from_ekf]
        t = threading.Thread(target=write_csv, args=args)
        t.start()
        threads.append(t)

        Console.info("Waiting for all threads to finish")
        for t in threads:
            t.join()
        Console.info("DONE")
        Console.quit(
            "Finished writing out relative uncertainties and quitting (this is not a failure)"
        )

    if len(camera1_ekf_list) > 0:
        camera1_ekf_list = update_camera_list(
            camera1_ekf_list,
            ekf_list,
            origin_offsets,
            camera1_offsets,
            latlon_reference,
        )
    if len(camera2_ekf_list) > 0:
        camera2_ekf_list = update_camera_list(
            camera2_ekf_list,
            ekf_list,
            origin_offsets,
            camera2_offsets,
            latlon_reference,
        )
    if len(camera3_ekf_list) > 0:
        camera3_ekf_list = update_camera_list(
            camera3_ekf_list,
            ekf_list,
            origin_offsets,
            camera3_offsets,
            latlon_reference,
        )
        camera3_ekf_list_at_dvl = update_camera_list(
            camera3_ekf_list_at_dvl,
            ekf_list_dvl,
            [0, 0, 0],
            [0, 0, 0],
            latlon_reference,
        )

    # perform interpolations of state data to chemical time stamps for both
    # DR and PF
    if len(chemical_list) > 1:
        interpolate_sensor_list(
            chemical_list,
            "chemical",
            chemical_offset,
            origin_offsets,
            latlon_reference,
            dead_reckoning_centre_list,
        )
        if len(pf_fusion_centre_list) > 1:
            interpolate_sensor_list(
                chemical_list,
                "chemical",
                chemical_offset,
                origin_offsets,
                latlon_reference,
                pf_fusion_centre_list,
            )

    if plot_output_activate:
        # if pdf_plot:
        # pdf_plot()

        # plotly data in html
        if html_plot:
            plotlypath = renavpath / "interactive_plots"
            Console.info("Plotting plotly data at {}".format(plotlypath))
            if plotlypath.is_dir() == 0:
                try:
                    plotlypath.mkdir()
                except Exception as e:
                    print("Warning:", e)

            t = threading.Thread(
                target=plot_orientation_vs_time,
                args=[orientation_list, plotlypath],
            )
            t.start()
            threads.append(t)
            t = threading.Thread(
                target=plot_velocity_vs_time,
                args=[
                    dead_reckoning_dvl_list,
                    velocity_inertial_list,
                    dead_reckoning_centre_list,
                    mission.velocity.format,
                    plotlypath,
                ],
            )
            t.start()
            threads.append(t)
            t = threading.Thread(
                target=plot_deadreckoning_vs_time,
                args=[
                    dead_reckoning_dvl_list,
                    velocity_inertial_list,
                    usbl_list,
                    dead_reckoning_centre_list,
                    altitude_list,
                    depth_list,
                    mission.velocity.format,
                    plotlypath,
                ],
            )
            t.start()
            threads.append(t)
            t = threading.Thread(
                target=plot_sensor_uncertainty,
                args=[
                    orientation_list,
                    velocity_body_list,
                    depth_list,
                    usbl_list,
                    velocity_inertial_list,
                    mission.velocity.format,
                    plotlypath,
                ],
            )
            t.start()
            threads.append(t)
            if particle_filter_activate:
                t = threading.Thread(
                    target=plot_pf_uncertainty,
                    args=[
                        pf_fusion_dvl_list,
                        pf_northings_std,
                        pf_eastings_std,
                        pf_yaw_std,
                        plotlypath,
                    ],
                )
                t.start()
                threads.append(t)
            if ekf_activate and "ekf_states" in locals():
                t = threading.Thread(
                    target=plot_ekf_states_and_std_vs_time,
                    args=[ekf_states, plotlypath / "ekf"],
                )
                t.start()
                threads.append(t)
                t = threading.Thread(
                    target=plot_ekf_rejected_measurements,
                    args=[ekf.get_rejected_measurements(), plotlypath],
                )
                t.start()
                threads.append(t)
            t = threading.Thread(
                target=plot_2d_deadreckoning,
                args=[
                    camera1_dr_list,
                    camera1_ekf_list,
                    dead_reckoning_centre_list,
                    dead_reckoning_dvl_list,
                    pf_fusion_centre_list,
                    ekf_list,
                    camera1_pf_list,
                    pf_fusion_dvl_list,
                    particles_time_interval,
                    pf_particles_list,
                    usbl_list_no_dist_filter,
                    usbl_list,
                    plotlypath,
                ],
            )
            t.start()
            threads.append(t)

    csvpath = renavpath / "csv"
    drcsvpath = csvpath / "dead_reckoning"
    pfcsvpath = csvpath / "particle_filter"
    ekfcsvpath = csvpath / "ekf"

    if csv_output_activate:
        Console.info("Writing csv outputs to {}".format(csvpath))
        if csv_usbl:
            if len(usbl_list) > 1:
                if not csvpath.exists():
                    csvpath.mkdir()
                auv_usbl_file = csvpath / "auv_usbl.csv"
                with auv_usbl_file.open("w") as fileout:
                    fileout.write(
                        "Timestamp, Northing [m], Easting [m], Depth [m], \
                        Latitude [deg], Longitude [deg]\n"
                    )
                for i in range(len(usbl_list)):
                    with auv_usbl_file.open("a") as fileout:
                        try:
                            fileout.write(
                                str(usbl_list[i].epoch_timestamp)
                                + ","
                                + str(usbl_list[i].northings)
                                + ","
                                + str(usbl_list[i].eastings)
                                + ","
                                + str(usbl_list[i].depth)
                                + ","
                                + str(usbl_list[i].latitude)
                                + ","
                                + str(usbl_list[i].longitude)
                                + "\n"
                            )
                            fileout.close()
                        except IndexError:
                            break

        t = threading.Thread(
            target=write_csv,
            args=[
                drcsvpath,
                dead_reckoning_centre_list,
                "auv_dr_centre",
                csv_dr_auv_centre,
            ],
        )
        t.start()
        threads.append(t)

        t = threading.Thread(
            target=write_csv,
            args=[drcsvpath, dead_reckoning_dvl_list, "auv_dr_dvl", csv_dr_auv_dvl],
        )
        t.start()
        threads.append(t)

        t = threading.Thread(
            target=write_csv,
            args=[drcsvpath, chemical_list, "auv_dr_chemical", csv_dr_chemical],
        )
        t.start()
        threads.append(t)

        t = threading.Thread(
            target=write_csv,
            args=[
                pfcsvpath,
                pf_fusion_centre_list,
                "auv_pf_centre",
                csv_pf_auv_centre,
            ],
        )
        t.start()
        threads.append(t)

        t = threading.Thread(
            target=write_csv,
            args=[pfcsvpath, pf_fusion_dvl_list, "auv_pf_dvl", csv_pf_auv_dvl],
        )
        t.start()
        threads.append(t)

        t = threading.Thread(
            target=write_csv,
            args=[pfcsvpath, chemical_list, "auv_pf_chemical", csv_pf_chemical],
        )
        t.start()
        threads.append(t)

        t = threading.Thread(
            target=write_csv,
            args=[ekfcsvpath, ekf_list, "auv_ekf_centre", csv_ekf_auv_centre],
        )
        t.start()
        threads.append(t)

        t = threading.Thread(
            target=write_csv,
            args=[ekfcsvpath, chemical_list, "auv_ekf_chemical", csv_pf_chemical],
        )
        t.start()
        threads.append(t)

        if len(camera1_dr_list) > 0:
            t = threading.Thread(
                target=write_csv,
                args=[
                    drcsvpath,
                    camera1_dr_list,
                    "auv_dr_" + mission.image.cameras[0].name,
                    csv_dr_camera_1,
                ],
            )
            t.start()
            threads.append(t)
            if particle_filter_activate:
                t = threading.Thread(
                    target=write_csv,
                    args=[
                        pfcsvpath,
                        camera1_pf_list,
                        "auv_pf_" + mission.image.cameras[0].name,
                        csv_pf_camera_1,
                    ],
                )
                t.start()
                threads.append(t)
            if ekf_activate:
                t = threading.Thread(
                    target=write_csv,
                    args=[
                        ekfcsvpath,
                        camera1_ekf_list,
                        "auv_ekf_" + mission.image.cameras[0].name,
                        csv_ekf_camera_1,
                    ],
                )
                t.start()
                threads.append(t)
        if len(camera2_dr_list) > 1:
            t = threading.Thread(
                target=write_csv,
                args=[
                    drcsvpath,
                    camera2_dr_list,
                    "auv_dr_" + mission.image.cameras[1].name,
                    csv_dr_camera_2,
                ],
            )
            t.start()
            threads.append(t)
            if particle_filter_activate:
                t = threading.Thread(
                    target=write_csv,
                    args=[
                        pfcsvpath,
                        camera2_pf_list,
                        "auv_pf_" + mission.image.cameras[1].name,
                        csv_pf_camera_2,
                    ],
                )
                t.start()
                threads.append(t)
            if ekf_activate:
                t = threading.Thread(
                    target=write_csv,
                    args=[
                        ekfcsvpath,
                        camera2_ekf_list,
                        "auv_ekf_" + mission.image.cameras[1].name,
                        csv_ekf_camera_2,
                    ],
                )
                t.start()
                threads.append(t)
        if len(camera3_dr_list) > 1:
            if len(mission.image.cameras) > 2:
                t = threading.Thread(
                    target=write_csv,
                    args=[
                        drcsvpath,
                        camera3_dr_list,
                        "auv_dr_" + mission.image.cameras[2].name,
                        csv_dr_camera_3,
                    ],
                )
                t.start()
                threads.append(t)
                if particle_filter_activate:
                    t = threading.Thread(
                        target=write_csv,
                        args=[
                            pfcsvpath,
                            camera3_pf_list,
                            "auv_pf_" + mission.image.cameras[2].name,
                            csv_pf_camera_3,
                        ],
                    )
                    t.start()
                    threads.append(t)
                if ekf_activate:
                    t = threading.Thread(
                        target=write_csv,
                        args=[
                            ekfcsvpath,
                            camera3_ekf_list,
                            "auv_ekf_" + mission.image.cameras[2].name,
                            csv_ekf_camera_3,
                        ],
                    )
                    t.start()
                    threads.append(t)
                    t = threading.Thread(
                        target=write_csv,
                        args=[
                            ekfcsvpath,
                            camera3_ekf_list_at_dvl,
                            "auv_ekf_" + mission.image.cameras[2].name + "_at_dvl",
                            csv_ekf_camera_3,
                        ],
                    )
                    t.start()
                    threads.append(t)
            elif len(mission.image.cameras) == 2:
                t = threading.Thread(
                    target=write_csv,
                    args=[
                        drcsvpath,
                        camera3_dr_list,
                        "auv_dr_" + mission.image.cameras[1].name + "_laser",
                        csv_dr_camera_3,
                    ],
                )
                t.start()
                threads.append(t)
                if particle_filter_activate:
                    t = threading.Thread(
                        target=write_csv,
                        args=[
                            pfcsvpath,
                            camera3_pf_list,
                            "auv_pf_" + mission.image.cameras[1].name + "_laser",
                            csv_pf_camera_3,
                        ],
                    )
                    t.start()
                    threads.append(t)
                if ekf_activate:
                    t = threading.Thread(
                        target=write_csv,
                        args=[
                            ekfcsvpath,
                            camera3_ekf_list,
                            "auv_ekf_" + mission.image.cameras[1].name + "_laser",
                            csv_ekf_camera_3,
                        ],
                    )
                    t.start()
                    threads.append(t)
                    t = threading.Thread(
                        target=write_csv,
                        args=[
                            ekfcsvpath,
                            camera3_ekf_list_at_dvl,
                            "auv_ekf_"
                            + mission.image.cameras[1].name
                            + "_laser_at_dvl",
                            csv_ekf_camera_3,
                        ],
                    )
                    t.start()
                    threads.append(t)

        # Sidescan sonar outputs
        t = threading.Thread(
            target=write_sidescan_csv,
            args=[
                drcsvpath,
                dead_reckoning_centre_list,
                "auv_dr_centre_sss",
                csv_dr_auv_centre,
            ],
        )
        t.start()
        threads.append(t)
        if particle_filter_activate:
            t = threading.Thread(
                target=write_sidescan_csv,
                args=[
                    pfcsvpath,
                    pf_fusion_centre_list,
                    "auv_pf_centre_sss",
                    csv_pf_auv_centre,
                ],
            )
            t.start()
            threads.append(t)
        if ekf_activate:
            t = threading.Thread(
                target=write_sidescan_csv,
                args=[ekfcsvpath, ekf_list, "auv_ekf_centre_sss", csv_ekf_auv_centre],
            )
            t.start()
            threads.append(t)
    if spp_output_activate and ekf_activate:
        Console.info("Converting covariance matrices into information matrices...")
        for i in range(len(camera1_ekf_list)):
            camera1_ekf_list[i].get_info()
        for i in range(len(camera2_ekf_list)):
            camera2_ekf_list[i].get_info()
        if len(camera3_ekf_list) > 1:
            for i in range(len(camera3_ekf_list)):
                camera3_ekf_list[i].get_info()
        Console.info("Converting poses into sequential-relative poses...")
        for i in range(len(camera1_ekf_list) - 1):
            camera1_ekf_list[i].northings -= camera1_ekf_list[i + 1].northings
            camera1_ekf_list[i].eastings -= camera1_ekf_list[i + 1].eastings
            camera1_ekf_list[i].depth -= camera1_ekf_list[i + 1].depth
            camera1_ekf_list[i].roll -= camera1_ekf_list[i + 1].roll
            camera1_ekf_list[i].pitch -= camera1_ekf_list[i + 1].pitch
            camera1_ekf_list[i].yaw -= camera1_ekf_list[i + 1].yaw
            camera1_ekf_list[i].information = camera1_ekf_list[i + 1].information
        camera1_ekf_list = camera1_ekf_list[:-1]

        t = threading.Thread(
            target=spp_csv,
            args=[
                camera1_ekf_list,
                "auv_ekf_" + mission.image.cameras[0].name,
                ekfcsvpath,
                spp_ekf_camera_1,
            ],
        )
        t.start()
        threads.append(t)
        t = threading.Thread(
            target=spp_csv,
            args=[
                camera2_ekf_list,
                "auv_ekf_" + mission.image.cameras[1].name,
                ekfcsvpath,
                spp_ekf_camera_2,
            ],
        )
        t.start()
        threads.append(t)
        if len(camera3_ekf_list) > 1:
            if len(mission.image.cameras) > 2:
                t = threading.Thread(
                    target=spp_csv,
                    args=[
                        camera3_ekf_list,
                        "auv_ekf_" + mission.image.cameras[2].name,
                        ekfcsvpath,
                        spp_ekf_camera_3,
                    ],
                )
                t.start()
                threads.append(t)
            elif len(mission.image.cameras) == 2:
                t = threading.Thread(
                    target=spp_csv,
                    args=[
                        camera3_ekf_list,
                        "auv_ekf_" + mission.image.cameras[1].name + "_laser",
                        ekfcsvpath,
                        spp_ekf_camera_3,
                    ],
                )
                t.start()
                threads.append(t)
    Console.info("Waiting for all threads to finish")
    for t in threads:
        t.join()
    Console.info("DONE")

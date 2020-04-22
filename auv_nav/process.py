# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, University of Southampton
All rights reserved.
"""

from auv_nav.tools.interpolate import interpolate
from auv_nav.tools.interpolate import interpolate_sensor_list
from auv_nav.tools.interpolate import interpolate_altitude
from auv_nav.tools.time_conversions import string_to_epoch
from auv_nav.tools.time_conversions import epoch_from_json
from auv_nav.tools.time_conversions import epoch_to_datetime
from auv_nav.tools.latlon_wgs84 import metres_to_latlon
from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.csv_tools import write_csv, write_raw_sensor_csv, write_sidescan_csv
from auv_nav.tools.csv_tools import camera_csv
from auv_nav.tools.csv_tools import other_data_csv
from auv_nav.tools.csv_tools import spp_csv
from auv_nav.sensors import BodyVelocity, InertialVelocity
from auv_nav.sensors import Altitude, Depth, Usbl, Orientation
from auv_nav.sensors import Tide
from auv_nav.sensors import Other, Camera
from auv_nav.sensors import SyncedOrientationBodyVelocity
from auv_nav.localisation.dead_reckoning import dead_reckoning
from auv_nav.localisation.usbl_offset import usbl_offset
from auv_nav.localisation.pf import run_particle_filter
from auv_nav.localisation.ekf import ExtendedKalmanFilter, Index
from auv_nav.localisation.usbl_filter import usbl_filter
from auv_nav.plot.plot_process_data import plot_orientation_vs_time
from auv_nav.plot.plot_process_data import plot_velocity_vs_time
from auv_nav.plot.plot_process_data import plot_deadreckoning_vs_time
from auv_nav.plot.plot_process_data import plot_pf_uncertainty
from auv_nav.plot.plot_process_data import plot_uncertainty
from auv_nav.plot.plot_process_data import plot_2d_deadreckoning
from oplab import get_config_folder
from oplab import get_processed_folder
from oplab import valid_dive
from oplab import Vehicle
from oplab import Mission
from oplab import Console

# Import librarys
import yaml
import json
import time
import copy
import math
import threading

from pathlib import Path
import numpy as np


"""
Assumes filename_camera of 1, 2, and 3 contains the image number between the
last 11 and 4 characters for appropriate csv pose estimate files output.
e.g. 'Xviii/Cam51707923/0094853.raw' or 'LM165/001/image0001011.tif'

Scripts to extract data from nav_standard.json, and combined.auv.raw an save
csv files and, if plot is True, save plots
"""


def process(filepath, force_overwite, start_datetime, finish_datetime):
    if len(filepath) > 1:
        Console.error('Process only supports one folder as a target dive.')
        Console.quit('Wrong number of parameters speficied.')
    # Filepath is a list. Get the first element by default
    filepath = filepath[0]

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
    camera1_pf_list = []
    # camera2 placeholders
    camera2_list = []
    camera2_pf_list = []
    # camera3 placeholders
    camera3_list = []
    camera3_pf_list = []

    # tide placeholder
    tide_list = []
    
    # placeholders for interpolated velocity body measurements based on orientation and transformed coordinates
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
    chemical_pf_list = []

    # std factors and offsets defaults for EKF
    std_factor_usbl = 0.01
    std_offset_usbl = 10.
    std_factor_dvl = 0.001
    std_offset_dvl = 0.002
    std_factor_depth = 0
    std_offset_depth = 0.01
    std_factor_orientation = 0.
    std_offset_orientation = 0.003

    # load auv_nav.yaml for particle filter and other setup
    filepath = Path(filepath).resolve()
    filepath = get_processed_folder(filepath)
    localisation_file = filepath / 'auv_nav.yaml'
    localisation_file = get_config_folder(localisation_file)

    # check that it is a valid dive folder
    if not valid_dive(filepath):
        Console.error("The dive folder supplied does not contain any mission or vehicle YAML files. Is the path correct?")
        Console.quit("Invalid path")

    # check if auv_nav.yaml file exist, if not, generate one with default settings
    if localisation_file.exists():
        Console.info("Loading existing auv_nav.yaml at {}".format(localisation_file))
    else:
        root = Path(__file__).parents[1]
        default_localisation = root / 'auv_nav/default_yaml' / 'auv_nav.yaml'
        Console.info("default_localisation: {}".format(default_localisation))
        Console.warn("Cannot find {}, generating default from {}".format(
            localisation_file, default_localisation))
        # save localisation yaml to processed directory
        if not localisation_file.parent.exists():
            localisation_file.parent.mkdir(parents=True)
        default_localisation.copy(localisation_file)

    # Default to no EKF and PF and SPP
    particle_filter_activate = False
    ekf_activate = False
    spp_output_activate = False

    with localisation_file.open('r') as stream:
        load_localisation = yaml.safe_load(stream)
        if 'usbl_filter' in load_localisation:
            usbl_filter_activate = load_localisation['usbl_filter']['activate']
            max_auv_speed = load_localisation['usbl_filter']['max_auv_speed']
            sigma_factor = load_localisation['usbl_filter']['sigma_factor']
        if 'particle_filter' in load_localisation:
            particle_filter_activate = load_localisation['particle_filter']['activate']
            dvl_noise_sigma_factor = load_localisation['particle_filter']['dvl_noise_sigma_factor']
            imu_noise_sigma_factor = load_localisation['particle_filter']['imu_noise_sigma_factor']
            usbl_noise_sigma_factor = load_localisation['particle_filter']['usbl_noise_sigma_factor']
            particles_number = load_localisation['particle_filter']['particles_number']
            particles_time_interval = load_localisation['particle_filter']['particles_plot_time_interval']
        if 'std' in load_localisation:
            sensors_std = load_localisation['std']
        else:
            sensors_std = {
                'usbl': {
                    'factor': std_factor_usbl,
                    'offset': std_offset_usbl
                },
                'dvl': {
                    'factor': std_factor_dvl,
                    'offset': std_offset_dvl
                },
                'depth': {
                    'factor': std_factor_depth,
                    'offset': std_offset_depth
                },
                'orientation': {
                    'factor': std_factor_orientation,
                    'offset': std_offset_orientation
                },
            }
            if 'model' not in sensors_std['usbl']:
                sensors_std['usbl']['model'] = 'json'
            if 'model' not in sensors_std['dvl']:
                sensors_std['dvl']['model'] = 'json'
            if 'model' not in sensors_std['depth']:
                sensors_std['depth']['model'] = 'json'
            if 'model' not in sensors_std['orientation']:
                sensors_std['orientation']['model'] = 'json'
        if 'ekf' in load_localisation:
            ekf_activate = load_localisation['ekf']['activate']
            ekf_process_noise_covariance = load_localisation['ekf']['process_noise_covariance']
            ekf_initial_estimate_covariance = load_localisation['ekf']['initial_estimate_covariance']
            if len(ekf_process_noise_covariance) != 144:
                d = np.asarray(
                    ekf_process_noise_covariance).reshape((15, 15))
                ekf_process_noise_covariance = d[0:12, 0:12]
                d = np.asarray(
                    ekf_initial_estimate_covariance).reshape((15, 15))
                ekf_initial_estimate_covariance = d[0:12, 0:12]
            else:
                ekf_process_noise_covariance = np.asarray(
                    ekf_process_noise_covariance).reshape((12, 12))
                ekf_initial_estimate_covariance = np.asarray(
                    ekf_initial_estimate_covariance).reshape((12, 12))
        if 'csv_output' in load_localisation:
            # csv_active
            csv_output_activate = load_localisation['csv_output']['activate']
            csv_usbl = load_localisation['csv_output']['usbl']
            csv_dr_auv_centre = load_localisation['csv_output']['dead_reckoning']['auv_centre']
            csv_dr_auv_dvl = load_localisation['csv_output']['dead_reckoning']['auv_dvl']
            csv_dr_camera_1 = load_localisation['csv_output']['dead_reckoning']['camera_1']
            csv_dr_camera_2 = load_localisation['csv_output']['dead_reckoning']['camera_2']
            csv_dr_camera_3 = load_localisation['csv_output']['dead_reckoning']['camera_3']
            csv_dr_chemical = load_localisation['csv_output']['dead_reckoning']['chemical']

            csv_pf_auv_centre = load_localisation['csv_output']['particle_filter']['auv_centre']
            csv_pf_auv_dvl = load_localisation['csv_output']['particle_filter']['auv_dvl']
            csv_pf_camera_1 = load_localisation['csv_output']['particle_filter']['camera_1']
            csv_pf_camera_2 = load_localisation['csv_output']['particle_filter']['camera_2']
            csv_pf_camera_3 = load_localisation['csv_output']['particle_filter']['camera_3']
            csv_pf_chemical = load_localisation['csv_output']['particle_filter']['chemical']

            csv_ekf_auv_centre = load_localisation['csv_output']['ekf']['auv_centre']
            csv_ekf_camera_1 = load_localisation['csv_output']['ekf']['camera_1']
            csv_ekf_camera_2 = load_localisation['csv_output']['ekf']['camera_2']
            csv_ekf_camera_3 = load_localisation['csv_output']['ekf']['camera_3']
            csv_ekf_chemical = load_localisation['csv_output']['ekf']['chemical']
        else:
            csv_output_activate = False
            Console.warn('csv output undefined in auv_nav.yaml. Has been'
                         + ' set to "False". To activate, add as per'
                         + ' default auv_nav.yaml found within auv_nav'
                         + ' file structure and set values to "True".')

        if 'spp_output' in load_localisation:
            # spp_active
            spp_output_activate = load_localisation['spp_output']['activate']
            spp_ekf_auv_centre = load_localisation['spp_output']['ekf']['auv_centre']
            spp_ekf_camera_1 = load_localisation['spp_output']['ekf']['camera_1']
            spp_ekf_camera_2 = load_localisation['spp_output']['ekf']['camera_2']
            spp_ekf_camera_3 = load_localisation['spp_output']['ekf']['camera_3']
            spp_ekf_chemical = load_localisation['spp_output']['ekf']['chemical']

            if spp_output_activate and not ekf_activate:
                Console.warn('SLAM++ will be disabled due to EKF being disabled. Enable EKF to make it work.')

                spp_output_activate = False
        else:
            spp_output_activate = False
            Console.warn('SLAM++ output undefined in auv_nav.yaml. Has been'
                         + ' set to "False". To activate, add as per'
                         + ' default auv_nav.yaml found within auv_nav'
                         + ' file structure and set values to "True".')

        if 'plot_output' in load_localisation:
            plot_output_activate = load_localisation['plot_output']['activate']
            pdf_plot = load_localisation['plot_output']['pdf_plot']
            html_plot = load_localisation['plot_output']['html_plot']

    Console.info('Loading vehicle.yaml')
    vehicle_file = filepath / 'vehicle.yaml'
    vehicle_file = get_processed_folder(vehicle_file)
    vehicle = Vehicle(vehicle_file)

    Console.info('Loading mission.yaml')
    mission_file = filepath / 'mission.yaml'
    mission_file = get_processed_folder(mission_file)
    mission = Mission(mission_file)

    camera1_offsets = [vehicle.camera1.surge,
                       vehicle.camera1.sway,
                       vehicle.camera1.heave]
    camera2_offsets = [vehicle.camera2.surge,
                       vehicle.camera2.sway,
                       vehicle.camera2.heave]
    
    # For BioCam, camera 3 is grayscale camera recording laser
    # For SeaXerocks, camera 3 is a separate camera
    camera3_offsets = [vehicle.camera3.surge,
                       vehicle.camera3.sway,
                       vehicle.camera3.heave]

    if mission.image.format == 'biocam':
        if mission.image.cameras[0].type == 'grayscale':
            camera3_offsets = [vehicle.camera1.surge,
                               vehicle.camera1.sway,
                               vehicle.camera1.heave]
        elif mission.image.cameras[1].type == 'grayscale':
            camera3_offsets = [vehicle.camera2.surge,
                               vehicle.camera2.sway,
                               vehicle.camera2.heave]
        else:
            Console.quit('BioCam format is expected to have a grayscale camera.')

    chemical_offset = [vehicle.chemical.surge,
                       vehicle.chemical.sway,
                       vehicle.chemical.heave]

    outpath = filepath / 'nav'

    nav_standard_file = outpath / 'nav_standard.json'
    nav_standard_file = get_processed_folder(nav_standard_file)
    Console.info('Loading json file {}'.format(nav_standard_file))
    with nav_standard_file.open('r') as nav_standard:
        parsed_json_data = json.load(nav_standard)

    # setup start and finish date time
    if start_datetime == '':
        epoch_start_time = epoch_from_json(parsed_json_data[1])
        start_datetime = epoch_to_datetime(epoch_start_time)
    else:
        epoch_start_time = string_to_epoch(start_datetime)
    if finish_datetime == '':
        epoch_finish_time = epoch_from_json(parsed_json_data[-1])
        finish_datetime = epoch_to_datetime(epoch_finish_time)
    else:
        epoch_finish_time = string_to_epoch(finish_datetime)

    # read in data from json file
    # i here is the number of the data packet
    for i in range(len(parsed_json_data)):
        epoch_timestamp = parsed_json_data[i]['epoch_timestamp']
        if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:
            if 'velocity' in parsed_json_data[i]['category']:
                if 'body' in parsed_json_data[i]['frame']:
                    # to check for corrupted data point which have inertial frame data values
                    if 'epoch_timestamp_dvl' in parsed_json_data[i]:
                        # confirm time stamps of dvl are aligned with main clock (within a second)
                        if (abs(parsed_json_data[i]['epoch_timestamp']
                           - parsed_json_data[i]['epoch_timestamp_dvl'])) < 1.0:
                            velocity_body = BodyVelocity()
                            velocity_body.from_json(parsed_json_data[i], sensors_std['dvl'])
                            velocity_body_list.append(velocity_body)
                if 'inertial' in parsed_json_data[i]['frame']:
                    velocity_inertial = InertialVelocity()
                    velocity_inertial.from_json(parsed_json_data[i])
                    velocity_inertial_list.append(velocity_inertial)

            if 'orientation' in parsed_json_data[i]['category']:
                orientation = Orientation()
                orientation.from_json(parsed_json_data[i], sensors_std['orientation'])
                orientation_list.append(orientation)

            if 'depth' in parsed_json_data[i]['category']:
                depth = Depth()
                depth.from_json(parsed_json_data[i], sensors_std['depth'])
                depth_list.append(depth)

            if 'altitude' in parsed_json_data[i]['category']:
                altitude = Altitude()
                altitude.from_json(parsed_json_data[i])
                altitude_list.append(altitude)

            if 'usbl' in parsed_json_data[i]['category']:
                usbl = Usbl()
                usbl.from_json(parsed_json_data[i], sensors_std['usbl'])
                usbl_list.append(usbl)

            if 'image' in parsed_json_data[i]['category']:
                camera1 = Camera()
                # LC
                camera1.from_json(parsed_json_data[i], 'camera1')
                camera1_list.append(camera1)
                camera2 = Camera()
                camera2.from_json(parsed_json_data[i], 'camera2')
                camera2_list.append(camera2)

            if 'laser' in parsed_json_data[i]['category']:
                camera3 = Camera()
                camera3.from_json(parsed_json_data[i], 'camera3')
                camera3_list.append(camera3)

            if 'chemical' in parsed_json_data[i]['category']:
                chemical = Other()
                chemical.from_json(parsed_json_data[i])
                chemical_list.append(chemical)

    if particle_filter_activate:
        camera1_pf_list = copy.deepcopy(camera1_list)
        camera2_pf_list = copy.deepcopy(camera2_list)
        camera3_pf_list = copy.deepcopy(camera3_list)
        chemical_pf_list = copy.deepcopy(chemical_list)

    if ekf_activate:
        camera1_ekf_list = copy.deepcopy(camera1_list)
        camera2_ekf_list = copy.deepcopy(camera2_list)
        camera3_ekf_list = copy.deepcopy(camera3_list)
        chemical_ekf_list = copy.deepcopy(chemical_list)

    # make path for processed outputs
    json_filename = ('json_renav_'
                     + start_datetime[0:8]
                     + '_'
                     + start_datetime[8:14]
                     + '_'
                     + finish_datetime[0:8]
                     + '_'
                     + finish_datetime[8:14])
    renavpath = filepath / json_filename
    if renavpath.is_dir() is False:
        try:
            renavpath.mkdir()
        except Exception as e:
            print("Warning:", e)
    elif renavpath.is_dir() and not force_overwite:
        # Check if dataset has already been processed
        Console.error('It looks like this dataset has already been processed for the specified time span.')
        Console.error('The following directory already exist: {}'.format(renavpath))
        Console.error('To overwrite the contents of this directory rerun auv_nav with the flag -F.')
        Console.error('Example:   auv_nav process -F PATH')
        Console.quit('auv_nav process would overwrite json_renav files')

    Console.info("Parsing has found:")
    Console.info("\t* Velocity_body: {} elements".format(len(velocity_body_list)))
    Console.info("\t* Velocity_inertial: {} elements".format(len(velocity_inertial_list)))
    Console.info("\t* Orientation: {} elements".format(len(orientation_list)))
    Console.info("\t* Depth: {} elements".format(len(depth_list)))
    Console.info("\t* Altitude: {} elements".format(len(altitude_list)))
    Console.info("\t* Usbl: {} elements".format(len(usbl_list)))

    Console.info('Writing outputs to: {}'.format(renavpath))
    raw_sensor_path = renavpath / 'csv' / 'sensors'

    threads = []
    mutex = threading.Lock()
    t = threading.Thread(target=write_raw_sensor_csv,
                         args=[raw_sensor_path, velocity_body_list, 'velocity_body_raw', mutex])
    t.start()
    threads.append(t)
    t = threading.Thread(target=write_raw_sensor_csv,
                         args=[raw_sensor_path, altitude_list, 'altitude_raw', mutex])
    t.start()
    threads.append(t)
    t = threading.Thread(target=write_raw_sensor_csv,
                         args=[raw_sensor_path, orientation_list, 'orientation_raw', mutex])
    t.start()
    threads.append(t)
    t = threading.Thread(target=write_raw_sensor_csv,
                         args=[raw_sensor_path, depth_list, 'depth_raw', mutex])
    t.start()
    threads.append(t)
    t = threading.Thread(target=write_raw_sensor_csv,
                         args=[raw_sensor_path, usbl_list, 'usbl_raw', mutex])
    t.start()
    threads.append(t)
    if len(camera3_list) > 0:
        t = threading.Thread(target=write_raw_sensor_csv,
                             args=[raw_sensor_path, camera3_list, 'camera3_raw', mutex])
        t.start()
        threads.append(t)

    # interpolate to find the appropriate depth to compute seafloor depth for each altitude measurement
    j = 0
    for i in range(len(altitude_list)):
        while j < len(depth_list)-1 and depth_list[j].epoch_timestamp < altitude_list[i].epoch_timestamp:
            j = j + 1

        if j >= 1:
            altitude_list[i].seafloor_depth = interpolate(
                altitude_list[i].epoch_timestamp,
                depth_list[j-1].epoch_timestamp,
                depth_list[j].epoch_timestamp,
                depth_list[j-1].depth,
                depth_list[j].depth)+altitude_list[i].altitude

    # perform usbl_filter
    if usbl_filter_activate:
        usbl_list = usbl_filter(
            usbl_list, depth_list, sigma_factor, max_auv_speed)
        if len(usbl_list) == 0:
            Console.warn('Filtering USBL measurements lead to an empty list. ')
            Console.warn(' * Is USBL reliable?')
            Console.warn(' * Can you change filter parameters?')

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

    while orientation_list[start_interpolate_index].epoch_timestamp < velocity_body_list[0].epoch_timestamp:
        start_interpolate_index += 1

    # if start_interpolate_index==0:
    # do something? because time_orientation may be way before time_velocity_body

    if start_interpolate_index == 1:
        interpolate_remove_flag = True

    # time_velocity_body)):
    for i in range(start_interpolate_index, len(orientation_list)):

        # interpolate to find the appropriate dvl time for the orientation measurements
        if orientation_list[i].epoch_timestamp > velocity_body_list[-1].epoch_timestamp:
            break

        while j < len(velocity_body_list)-1 and orientation_list[i].epoch_timestamp > velocity_body_list[j+1].epoch_timestamp:
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
            velocity_body_list[j+1].epoch_timestamp,
            velocity_body_list[j].x_velocity,
            velocity_body_list[j+1].x_velocity)
        dead_reckoning_dvl.y_velocity = interpolate(
            orientation_list[i].epoch_timestamp,
            velocity_body_list[j].epoch_timestamp,
            velocity_body_list[j+1].epoch_timestamp,
            velocity_body_list[j].y_velocity,
            velocity_body_list[j+1].y_velocity)
        dead_reckoning_dvl.z_velocity = interpolate(
            orientation_list[i].epoch_timestamp,
            velocity_body_list[j].epoch_timestamp,
            velocity_body_list[j+1].epoch_timestamp,
            velocity_body_list[j].z_velocity,
            velocity_body_list[j+1].z_velocity)
        dead_reckoning_dvl.x_velocity_std = interpolate(
            orientation_list[i].epoch_timestamp,
            velocity_body_list[j].epoch_timestamp,
            velocity_body_list[j+1].epoch_timestamp,
            velocity_body_list[j].x_velocity_std,
            velocity_body_list[j+1].x_velocity_std)
        dead_reckoning_dvl.y_velocity_std = interpolate(
            orientation_list[i].epoch_timestamp,
            velocity_body_list[j].epoch_timestamp,
            velocity_body_list[j+1].epoch_timestamp,
            velocity_body_list[j].y_velocity_std,
            velocity_body_list[j+1].y_velocity_std)
        dead_reckoning_dvl.z_velocity_std = interpolate(
            orientation_list[i].epoch_timestamp,
            velocity_body_list[j].epoch_timestamp,
            velocity_body_list[j+1].epoch_timestamp,
            velocity_body_list[j].z_velocity_std,
            velocity_body_list[j+1].z_velocity_std)

        [x_offset, y_offset, z_offset] = body_to_inertial(
            orientation_list[i].roll,
            orientation_list[i].pitch,
            orientation_list[i].yaw,
            dead_reckoning_dvl.x_velocity,
            dead_reckoning_dvl.y_velocity,
            dead_reckoning_dvl.z_velocity)

        dead_reckoning_dvl.north_velocity = x_offset
        dead_reckoning_dvl.east_velocity = y_offset
        dead_reckoning_dvl.down_velocity = z_offset

        [x_offset, y_offset, z_offset] = body_to_inertial(
            orientation_list[i].roll,
            orientation_list[i].pitch,
            orientation_list[i].yaw,
            dead_reckoning_dvl.x_velocity_std,
            dead_reckoning_dvl.y_velocity_std,
            dead_reckoning_dvl.z_velocity_std)

        dead_reckoning_dvl.north_velocity_std = x_offset
        dead_reckoning_dvl.east_velocity_std = y_offset
        dead_reckoning_dvl.down_velocity_std = z_offset

        while n < len(altitude_list)-1 and orientation_list[i].epoch_timestamp > altitude_list[n].epoch_timestamp:
            n += 1
        dead_reckoning_dvl.altitude = interpolate(
            orientation_list[i].epoch_timestamp,
            altitude_list[n-1].epoch_timestamp,
            altitude_list[n].epoch_timestamp,
            altitude_list[n-1].altitude,
            altitude_list[n].altitude)

        while k < len(depth_list)-1 and depth_list[k].epoch_timestamp < orientation_list[i].epoch_timestamp:
            k += 1
        # interpolate to find the appropriate depth for dead_reckoning
        dead_reckoning_dvl.depth = interpolate(
            orientation_list[i].epoch_timestamp,
            depth_list[k-1].epoch_timestamp,
            depth_list[k].epoch_timestamp,
            depth_list[k-1].depth,
            depth_list[k].depth)
        dead_reckoning_dvl.depth_std = interpolate(
            orientation_list[i].epoch_timestamp,
            depth_list[k-1].epoch_timestamp,
            depth_list[k].epoch_timestamp,
            depth_list[k-1].depth_std,
            depth_list[k].depth_std)
        dead_reckoning_dvl_list.append(dead_reckoning_dvl)

    # dead reckoning solution
    for i in range(len(dead_reckoning_dvl_list)):
        # dead reckoning solution
        if i >= 1:
            [dead_reckoning_dvl_list[i].northings, dead_reckoning_dvl_list[i].eastings] = dead_reckoning(
                dead_reckoning_dvl_list[i].epoch_timestamp,
                dead_reckoning_dvl_list[i-1].epoch_timestamp,
                dead_reckoning_dvl_list[i].north_velocity,
                dead_reckoning_dvl_list[i-1].north_velocity,
                dead_reckoning_dvl_list[i].east_velocity,
                dead_reckoning_dvl_list[i-1].east_velocity,
                dead_reckoning_dvl_list[i-1].northings,
                dead_reckoning_dvl_list[i-1].eastings)

    # offset sensor to plot origin/centre of vehicle
    dead_reckoning_centre_list = copy.deepcopy(
        dead_reckoning_dvl_list)  # [:] #.copy()
    for i in range(len(dead_reckoning_centre_list)):
        [x_offset, y_offset, a_offset] = body_to_inertial(
            dead_reckoning_centre_list[i].roll,
            dead_reckoning_centre_list[i].pitch,
            dead_reckoning_centre_list[i].yaw,
            vehicle.origin.surge - vehicle.dvl.surge,
            vehicle.origin.sway - vehicle.dvl.sway,
            vehicle.origin.heave - vehicle.dvl.heave)
        [_, _, z_offset] = body_to_inertial(
            dead_reckoning_centre_list[i].roll,
            dead_reckoning_centre_list[i].pitch,
            dead_reckoning_centre_list[i].yaw,
            vehicle.origin.surge - vehicle.depth.surge,
            vehicle.origin.sway - vehicle.depth.sway,
            vehicle.origin.heave - vehicle.depth.heave)
        dead_reckoning_centre_list[i].northings += x_offset
        dead_reckoning_centre_list[i].eastings += y_offset
        dead_reckoning_centre_list[i].altitude += a_offset
        dead_reckoning_centre_list[i].depth += z_offset
    # correct for altitude and depth offset too!

    # remove first term if first time_orientation is < velocity_body time
    if interpolate_remove_flag:

        # del time_orientation[0]
        del dead_reckoning_centre_list[0]
        del dead_reckoning_dvl_list[0]
        interpolate_remove_flag = False  # reset flag
    Console.info('Complete interpolation and coordinate transfomations for velocity_body')

    # perform interpolations of state data to velocity_inertial time stamps
    # (without sensor offset and correct imu to dvl flipped interpolation)
    # and perform deadreckoning
    # initialise counters for interpolation
    if len(velocity_inertial_list) > 0:
        # dead_reckoning_built_in_values
        j = 0
        k = 0

        for i in range(len(velocity_inertial_list)):

            while j < len(orientation_list)-1 and orientation_list[j].epoch_timestamp < velocity_inertial_list[i].epoch_timestamp:
                j = j+1

            if j == 1:
                interpolate_remove_flag = True
            else:
                velocity_inertial_list[i].roll = interpolate(
                    velocity_inertial_list[i].epoch_timestamp,
                    orientation_list[j-1].epoch_timestamp,
                    orientation_list[j].epoch_timestamp,
                    orientation_list[j-1].roll,
                    orientation_list[j].roll)
                velocity_inertial_list[i].pitch = interpolate(
                    velocity_inertial_list[i].epoch_timestamp,
                    orientation_list[j-1].epoch_timestamp,
                    orientation_list[j].epoch_timestamp,
                    orientation_list[j-1].pitch,
                    orientation_list[j].pitch)

                if abs(orientation_list[j].yaw-orientation_list[j-1].yaw) > 180:
                    if orientation_list[j].yaw > orientation_list[j-1].yaw:
                        velocity_inertial_list[i].yaw = interpolate(
                            velocity_inertial_list[i].epoch_timestamp,
                            orientation_list[j-1].epoch_timestamp,
                            orientation_list[j].epoch_timestamp,
                            orientation_list[j-1].yaw,
                            orientation_list[j].yaw-360)
                    else:
                        velocity_inertial_list[i].yaw = interpolate(
                            velocity_inertial_list[i].epoch_timestamp,
                            orientation_list[j-1].epoch_timestamp,
                            orientation_list[j].epoch_timestamp,
                            orientation_list[j-1].yaw-360,
                            orientation_list[j].yaw)

                    if velocity_inertial_list[i].yaw < 0:
                        velocity_inertial_list[i].yaw += 360

                    elif velocity_inertial_list[i].yaw > 360:
                        velocity_inertial_list[i].yaw -= 360
                else:
                    velocity_inertial_list[i].yaw = interpolate(
                        velocity_inertial_list[i].epoch_timestamp,
                        orientation_list[j-1].epoch_timestamp,
                        orientation_list[j].epoch_timestamp,
                        orientation_list[j-1].yaw,
                        orientation_list[j].yaw)

            while k < len(depth_list)-1 and depth_list[k].epoch_timestamp < velocity_inertial_list[i].epoch_timestamp:
                k = k+1

            if k >= 1:
                velocity_inertial_list[i].depth = interpolate(
                    velocity_inertial_list[i].epoch_timestamp,
                    depth_list[k-1].epoch_timestamp,
                    depth_list[k].epoch_timestamp,
                    depth_list[k-1].depth,
                    depth_list[k].depth)  # depth directly interpolated from depth sensor

        for i in range(len(velocity_inertial_list)):
            if i >= 1:
                [velocity_inertial_list[i].northings, velocity_inertial_list[i].eastings] = dead_reckoning(
                    velocity_inertial_list[i].epoch_timestamp,
                    velocity_inertial_list[i-1].epoch_timestamp,
                    velocity_inertial_list[i].north_velocity,
                    velocity_inertial_list[i-1].north_velocity,
                    velocity_inertial_list[i].east_velocity,
                    velocity_inertial_list[i-1].east_velocity,
                    velocity_inertial_list[i-1].northings,
                    velocity_inertial_list[i-1].eastings)

        if interpolate_remove_flag:
            del velocity_inertial_list[0]
            interpolate_remove_flag = False  # reset flag
        Console.info('Complete interpolation and coordinate transfomations for velocity_inertial')

    # offset velocity DR by average usbl estimate
    # offset velocity body DR by average usbl estimate
    if len(usbl_list) > 0:
        [northings_usbl_interpolated, eastings_usbl_interpolated] = usbl_offset(
            [i.epoch_timestamp for i in dead_reckoning_centre_list],
            [i.northings for i in dead_reckoning_centre_list],
            [i.eastings for i in dead_reckoning_centre_list],
            [i.epoch_timestamp for i in usbl_list],
            [i.northings for i in usbl_list],
            [i.eastings for i in usbl_list])
        for i in range(len(dead_reckoning_centre_list)):
            dead_reckoning_centre_list[i].northings += northings_usbl_interpolated
            dead_reckoning_centre_list[i].eastings += eastings_usbl_interpolated
            dead_reckoning_centre_list[i].latitude, dead_reckoning_centre_list[i].longitude = metres_to_latlon(
                mission.origin.latitude,
                mission.origin.longitude,
                dead_reckoning_centre_list[i].eastings,
                dead_reckoning_centre_list[i].northings)
        for i in range(len(dead_reckoning_dvl_list)):
            dead_reckoning_dvl_list[i].northings += northings_usbl_interpolated
            dead_reckoning_dvl_list[i].eastings += eastings_usbl_interpolated
            dead_reckoning_dvl_list[i].latitude, dead_reckoning_dvl_list[i].longitude = metres_to_latlon(
                mission.origin.latitude,
                mission.origin.longitude,
                dead_reckoning_dvl_list[i].eastings,
                dead_reckoning_dvl_list[i].northings)

        # offset velocity inertial DR by average usbl estimate
        if len(velocity_inertial_list) > 0:
            [northings_usbl_interpolated, eastings_usbl_interpolated] = usbl_offset(
                [i.epoch_timestamp for i in velocity_inertial_list],
                [i.northings for i in velocity_inertial_list],
                [i.eastings for i in velocity_inertial_list],
                [i.epoch_timestamp for i in usbl_list],
                [i.northings for i in usbl_list],
                [i.eastings for i in usbl_list])
            for i in range(len(velocity_inertial_list)):
                velocity_inertial_list[i].northings += northings_usbl_interpolated
                velocity_inertial_list[i].eastings += eastings_usbl_interpolated
                velocity_inertial_list[i].latitude, velocity_inertial_list[i].longitude = metres_to_latlon(
                    mission.origin.latitude,
                    mission.origin.longitude,
                    velocity_inertial_list[i].eastings,
                    velocity_inertial_list[i].northings)
    else:
        Console.warn("There are no USBL measurements. Starting DR at origin...")

# particle filter data fusion of usbl_data and dvl_imu_data
    if particle_filter_activate and len(usbl_list) > 0:
        Console.info("Running PF...")
        pf_start_time = time.time()
        [pf_fusion_dvl_list,
         pf_usbl_datapoints,
         pf_particles_list,
         pf_northings_std,
         pf_eastings_std,
         pf_yaw_std] = run_particle_filter(
            copy.deepcopy(usbl_list),
            copy.deepcopy(dead_reckoning_dvl_list),
            particles_number,
            sensors_std,
            dvl_noise_sigma_factor,
            imu_noise_sigma_factor,
            usbl_noise_sigma_factor,
            measurement_update_flag=True)
        pf_end_time = time.time()
        pf_elapsed_time = pf_end_time - pf_start_time
        # maybe save this as text alongside plotly outputs
        Console.info("PF with {} particles took {} mins".format(
            particles_number, pf_elapsed_time/60))
        pf_fusion_centre_list = copy.deepcopy(pf_fusion_dvl_list)
        for i in range(len(pf_fusion_centre_list)):
            pf_fusion_dvl_list[i].latitude,
            pf_fusion_dvl_list[i].longitude = metres_to_latlon(
                mission.origin.latitude, mission.origin.longitude,
                pf_fusion_dvl_list[i].eastings,
                pf_fusion_dvl_list[i].northings)
            [x_offset, y_offset, z_offset] = body_to_inertial(
                pf_fusion_centre_list[i].roll,
                pf_fusion_centre_list[i].pitch,
                pf_fusion_centre_list[i].yaw,
                vehicle.origin.surge - vehicle.dvl.surge,
                vehicle.origin.sway - vehicle.dvl.sway,
                vehicle.origin.heave - vehicle.dvl.heave)
            pf_fusion_centre_list[i].northings += x_offset
            pf_fusion_centre_list[i].eastings += y_offset
            lat, lon = metres_to_latlon(
                mission.origin.latitude, mission.origin.longitude,
                pf_fusion_centre_list[i].eastings,
                pf_fusion_centre_list[i].northings)
            pf_fusion_centre_list[i].latitude = lat
            pf_fusion_centre_list[i].longitude = lon

    ekf_list = []
    if ekf_activate and len(usbl_list) > 0:
        ekf_start_time = time.time()
        # velocity_body_list, list of BodyVelocity()
        # orientation_list, list of Orientation()
        # depth_list, list of Depth()
        # usbl_list, list of Usbl()
        Console.info("Running EKF...")
        ekf = ExtendedKalmanFilter(ekf_initial_estimate_covariance,
                                   ekf_process_noise_covariance,
                                   sensors_std,
                                   dead_reckoning_dvl_list,
                                   usbl_list)
        ekf_states = ekf.get_smoothed_result()
        ekf_end_time = time.time()
        ekf_elapsed_time = ekf_end_time - ekf_start_time
        Console.info("EKF took {} mins".format(ekf_elapsed_time/60))
        # TODO: convert from EKF states in meters to lat lon
        dr_idx = 1
        for s in ekf_states:
            b = SyncedOrientationBodyVelocity()
            b.epoch_timestamp = s.time
            b.northings = s.state[Index.X, 0]
            b.eastings = s.state[Index.Y, 0]
            b.depth = s.state[Index.Z, 0]
            b.roll = s.state[Index.ROLL, 0]*180.0/math.pi
            b.pitch = s.state[Index.PITCH, 0]*180.0/math.pi
            b.yaw = s.state[Index.YAW, 0]*180.0/math.pi
            b.roll_std = s.covariance[Index.ROLL, Index.ROLL]*180.0/math.pi
            b.pitch_std = s.covariance[Index.PITCH, Index.PITCH]*180.0/math.pi
            b.yaw_std = s.covariance[Index.YAW, Index.YAW]*180.0/math.pi
            b.x_velocity = s.state[Index.VX, 0]
            b.y_velocity = s.state[Index.VY, 0]
            b.z_velocity = s.state[Index.VZ, 0]
            b.x_velocity_std = s.covariance[Index.VX, Index.VX]
            b.y_velocity_std = s.covariance[Index.VY, Index.VY]
            b.z_velocity_std = s.covariance[Index.VZ, Index.VZ]
            [x_offset, y_offset, z_offset] = body_to_inertial(
                b.roll, b.pitch, b.yaw,
                vehicle.origin.surge - vehicle.dvl.surge,
                vehicle.origin.sway - vehicle.dvl.sway,
                vehicle.origin.heave - vehicle.dvl.heave)
            b.northings += x_offset
            b.eastings += y_offset
            b.depth += z_offset
            b.latitude, b.longitude = metres_to_latlon(
                mission.origin.latitude, mission.origin.longitude,
                b.eastings, b.northings)
            b.covariance = s.covariance
            while dr_idx < len(dead_reckoning_dvl_list) and dead_reckoning_dvl_list[dr_idx].epoch_timestamp < b.epoch_timestamp:
                dr_idx += 1
            b.altitude = interpolate(b.epoch_timestamp,
                                     dead_reckoning_dvl_list[dr_idx-1].epoch_timestamp,
                                     dead_reckoning_dvl_list[dr_idx].epoch_timestamp,
                                     dead_reckoning_dvl_list[dr_idx-1].altitude,
                                     dead_reckoning_dvl_list[dr_idx].altitude)
            ekf_list.append(b)

    origin_offsets = [vehicle.origin.surge, vehicle.origin.sway, vehicle.origin.heave]
    latlon_reference = [mission.origin.latitude, mission.origin.longitude]

    # perform interpolations of state data to camera{1/2/3} time stamps
    # for both DR and PF
    if len(camera1_list) > 1:
        interpolate_sensor_list(
            camera1_list,
            mission.image.cameras[0].name,
            camera1_offsets,
            origin_offsets,
            latlon_reference,
            dead_reckoning_centre_list)
    if len(camera2_list) > 1:
        interpolate_sensor_list(
            camera2_list,
            mission.image.cameras[1].name,
            camera2_offsets,
            origin_offsets,
            latlon_reference,
            dead_reckoning_centre_list)
    if len(camera3_list) > 1:
        if len(mission.image.cameras) > 2:
            interpolate_sensor_list(
                camera3_list,
                mission.image.cameras[2].name,
                camera3_offsets,
                origin_offsets,
                latlon_reference,
                dead_reckoning_centre_list)
        elif len(mission.image.cameras) == 2:  # Biocam
            interpolate_sensor_list(
                camera3_list,
                mission.image.cameras[1].name + '_laser',
                camera3_offsets,
                origin_offsets,
                latlon_reference,
                dead_reckoning_centre_list)

    if len(pf_fusion_centre_list) > 1:
        if len(camera1_pf_list) > 1:
            interpolate_sensor_list(
                camera1_pf_list,
                mission.image.cameras[0].name,
                camera1_offsets,
                origin_offsets,
                latlon_reference,
                pf_fusion_centre_list)
        if len(camera2_pf_list) > 1:
            interpolate_sensor_list(
                camera2_pf_list,
                mission.image.cameras[1].name,
                camera2_offsets,
                origin_offsets,
                latlon_reference,
                pf_fusion_centre_list)
        if len(camera3_pf_list) > 1:
            if len(mission.image.cameras) > 2:
                interpolate_sensor_list(
                    camera3_pf_list,
                    mission.image.cameras[2].name,
                    camera3_offsets,
                    origin_offsets,
                    latlon_reference,
                    pf_fusion_centre_list)
            elif len(mission.image.cameras) == 2:  # Biocam
                interpolate_sensor_list(
                    camera3_pf_list,
                    mission.image.cameras[1].name + '_laser',
                    camera3_offsets,
                    origin_offsets,
                    latlon_reference,
                    pf_fusion_centre_list)
    if len(ekf_list) > 1:
        if len(camera1_ekf_list) > 1:
            interpolate_sensor_list(
                camera1_ekf_list,
                mission.image.cameras[0].name,
                camera1_offsets,
                origin_offsets,
                latlon_reference,
                ekf_list)
        if len(camera2_ekf_list) > 1:
            interpolate_sensor_list(
                camera2_ekf_list,
                mission.image.cameras[1].name,
                camera2_offsets,
                origin_offsets,
                latlon_reference,
                ekf_list)
        if len(camera3_ekf_list) > 1:
            if len(mission.image.cameras) > 2:
                interpolate_sensor_list(
                    camera3_ekf_list,
                    mission.image.cameras[2].name,
                    camera3_offsets,
                    origin_offsets,
                    latlon_reference,
                    ekf_list)
            elif len(mission.image.cameras) == 2:  # Biocam
                interpolate_sensor_list(
                    camera3_ekf_list,
                    mission.image.cameras[1].name + '_laser',
                    camera3_offsets,
                    origin_offsets,
                    latlon_reference,
                    ekf_list)

    # perform interpolations of state data to chemical time stamps for both
    # DR and PF
    if len(chemical_list) > 1:
        interpolate_sensor_list(
            chemical_list,
            'chemical',
            chemical_offset,
            origin_offsets,
            latlon_reference,
            dead_reckoning_centre_list)
        if len(pf_fusion_centre_list) > 1:
            interpolate_sensor_list(
                chemical_list,
                'chemical',
                chemical_offset,
                origin_offsets,
                latlon_reference,
                pf_fusion_centre_list)

    if plot_output_activate:
        # if pdf_plot:
            # pdf_plot()

        # plotly data in html
        if html_plot:
            plotlypath = renavpath / 'interactive_plots'
            Console.info('Plotting plotly data at {} ...'.format(plotlypath))
            if plotlypath.is_dir() == 0:
                try:
                    plotlypath.mkdir()
                except Exception as e:
                    print("Warning:", e)

            t = threading.Thread(target=plot_orientation_vs_time,
                                 args=[orientation_list,
                                       plotlypath])
            t.start()
            threads.append(t)
            t = threading.Thread(target=plot_velocity_vs_time,
                                 args=[dead_reckoning_dvl_list,
                                       velocity_inertial_list,
                                       dead_reckoning_centre_list,
                                       mission.velocity.format,
                                       plotlypath])
            t.start()
            threads.append(t)
            t = threading.Thread(target=plot_deadreckoning_vs_time,
                                 args=[dead_reckoning_dvl_list,
                                       velocity_inertial_list,
                                       usbl_list,
                                       dead_reckoning_centre_list,
                                       altitude_list,
                                       depth_list,
                                       mission.velocity.format,
                                       plotlypath])
            t.start()
            threads.append(t)
            t = threading.Thread(target=plot_uncertainty,
                                 args=[orientation_list,
                                       velocity_body_list,
                                       depth_list,
                                       usbl_list,
                                       velocity_inertial_list,
                                       mission.velocity.format,
                                       plotlypath])
            t.start()
            threads.append(t)
            if particle_filter_activate:
                t = threading.Thread(target=plot_pf_uncertainty,
                                     args=[pf_fusion_dvl_list,
                                           pf_northings_std,
                                           pf_eastings_std,
                                           pf_yaw_std,
                                           plotlypath])
                t.start()
                threads.append(t)
            t = threading.Thread(target=plot_2d_deadreckoning,
                                 args=[camera1_list,
                                       dead_reckoning_centre_list,
                                       dead_reckoning_dvl_list,
                                       pf_fusion_centre_list,
                                       ekf_list,
                                       camera1_pf_list,
                                       pf_fusion_dvl_list,
                                       particles_time_interval,
                                       pf_particles_list,
                                       usbl_list,
                                       plotlypath])
            t.start()
            threads.append(t)

    csvpath = renavpath / 'csv'
    drcsvpath = csvpath / 'dead_reckoning'
    pfcsvpath = csvpath / 'particle_filter'
    ekfcsvpath = csvpath / 'ekf'

    if csv_output_activate:
        Console.info('Writing csv outputs to {} ...'.format(csvpath))
        if csv_usbl:
            if len(usbl_list) > 1:
                if not csvpath.exists():
                    csvpath.mkdir()
                auv_usbl_file = csvpath / 'auv_usbl.csv'
                with auv_usbl_file.open('w') as fileout:
                    fileout.write(
                        'Timestamp, Northing [m], Easting [m], Depth [m], \
                        Latitude [deg], Longitude [deg]\n')
                for i in range(len(usbl_list)):
                    with auv_usbl_file.open('a') as fileout:
                        try:
                            fileout.write(
                                str(usbl_list[i].epoch_timestamp) + ','
                                + str(usbl_list[i].northings) + ','
                                + str(usbl_list[i].eastings) + ','
                                + str(usbl_list[i].depth) + ','
                                + str(usbl_list[i].latitude) + ','
                                + str(usbl_list[i].longitude)+'\n')
                            fileout.close()
                        except IndexError:
                            break

        t = threading.Thread(target=write_csv,
                             args=[drcsvpath,
                                   dead_reckoning_centre_list,
                                   'auv_dr_centre',
                                   csv_dr_auv_centre])
        t.start()
        threads.append(t)

        t = threading.Thread(target=write_csv,
                             args=[drcsvpath,
                                   dead_reckoning_dvl_list,
                                   'auv_dr_dvl',
                                   csv_dr_auv_dvl])
        t.start()
        threads.append(t)

        t = threading.Thread(target=other_data_csv,
                             args=[chemical_list,
                                   'auv_dr_chemical',
                                   drcsvpath,
                                   csv_dr_chemical])
        t.start()
        threads.append(t)

        t = threading.Thread(target=write_csv,
                             args=[pfcsvpath,
                                   pf_fusion_centre_list,
                                   'auv_pf_centre',
                                   csv_pf_auv_centre])
        t.start()
        threads.append(t)

        t = threading.Thread(target=write_csv,
                             args=[pfcsvpath,
                                   pf_fusion_dvl_list,
                                   'auv_pf_dvl',
                                   csv_pf_auv_dvl])
        t.start()
        threads.append(t)

        t = threading.Thread(target=other_data_csv,
                             args=[chemical_list,
                                   'auv_pf_chemical',
                                   pfcsvpath,
                                   csv_pf_chemical])
        t.start()
        threads.append(t)

        t = threading.Thread(target=write_csv,
                             args=[ekfcsvpath,
                                   ekf_list,
                                   'auv_ekf_centre',
                                   csv_ekf_auv_centre])
        t.start()
        threads.append(t)

        t = threading.Thread(target=other_data_csv,
                             args=[chemical_list,
                                   'auv_ekf_chemical',
                                   ekfcsvpath,
                                   csv_pf_chemical])
        t.start()
        threads.append(t)

        if len(camera1_list) > 0:
            t = threading.Thread(target=camera_csv,
                                 args=[camera1_list,
                                       'auv_dr_' + mission.image.cameras[0].name,
                                       drcsvpath,
                                       csv_dr_camera_1])
            t.start()
            threads.append(t)
            if particle_filter_activate:
                t = threading.Thread(target=camera_csv,
                                    args=[camera1_pf_list,
                                        'auv_pf_' + mission.image.cameras[0].name,
                                        pfcsvpath,
                                        csv_pf_camera_1])
                t.start()
                threads.append(t)
            if ekf_activate:
                t = threading.Thread(target=camera_csv,
                                    args=[camera1_ekf_list,
                                        'auv_ekf_' + mission.image.cameras[0].name,
                                        ekfcsvpath,
                                        csv_ekf_camera_1])
                t.start()
                threads.append(t)
        if len(camera2_list) > 1:
            t = threading.Thread(target=camera_csv,
                                 args=[camera2_list,
                                       'auv_dr_' + mission.image.cameras[1].name,
                                       drcsvpath,
                                       csv_dr_camera_2])
            t.start()
            threads.append(t)
            if particle_filter_activate:
                t = threading.Thread(target=camera_csv,
                                    args=[camera2_pf_list,
                                        'auv_pf_' + mission.image.cameras[1].name,
                                        pfcsvpath,
                                        csv_pf_camera_2])
                t.start()
                threads.append(t)
            if ekf_activate:
                t = threading.Thread(target=camera_csv,
                                    args=[camera2_ekf_list,
                                        'auv_ekf_' + mission.image.cameras[1].name,
                                        ekfcsvpath,
                                        csv_ekf_camera_2])
                t.start()
                threads.append(t)
        if len(camera3_list) > 1:
            if len(mission.image.cameras) > 2:
                t = threading.Thread(target=camera_csv,
                                     args=[camera3_list,
                                           'auv_dr_' + mission.image.cameras[2].name,
                                           drcsvpath,
                                           csv_dr_camera_3])
                t.start()
                threads.append(t)
                if particle_filter_activate:
                    t = threading.Thread(target=camera_csv,
                                        args=[camera3_pf_list,
                                            'auv_pf_' + mission.image.cameras[2].name,
                                            pfcsvpath,
                                                csv_pf_camera_3])
                    t.start()
                    threads.append(t)
                if ekf_activate:
                    t = threading.Thread(target=camera_csv,
                                        args=[camera3_ekf_list,
                                            'auv_ekf_' + mission.image.cameras[2].name,
                                            ekfcsvpath,
                                            csv_ekf_camera_3])
                    t.start()
                    threads.append(t)
            elif len(mission.image.cameras) == 2:
                t = threading.Thread(target=camera_csv,
                                     args=[camera3_list,
                                           'auv_dr_' + mission.image.cameras[1].name + '_laser',
                                           drcsvpath,
                                           csv_dr_camera_3])
                t.start()
                threads.append(t)
                if particle_filter_activate:
                    t = threading.Thread(target=camera_csv,
                                        args=[camera3_pf_list,
                                            'auv_pf_' + mission.image.cameras[1].name + '_laser',
                                            pfcsvpath,
                                            csv_pf_camera_3])
                    t.start()
                    threads.append(t)
                if ekf_activate:
                    t = threading.Thread(target=camera_csv,
                                        args=[camera3_ekf_list,
                                            'auv_ekf_' + mission.image.cameras[1].name + '_laser',
                                            ekfcsvpath,
                                            csv_ekf_camera_3])
                    t.start()
                    threads.append(t)

        # Sidescan sonar outputs
        t = threading.Thread(target=write_sidescan_csv,
                             args=[drcsvpath,
                                   dead_reckoning_centre_list,
                                   'auv_dr_centre_sss',
                                   csv_dr_auv_centre])
        t.start()
        threads.append(t)
        if particle_filter_activate:
            t = threading.Thread(target=write_sidescan_csv,
                                args=[pfcsvpath,
                                    pf_fusion_centre_list,
                                    'auv_pf_centre_sss',
                                    csv_pf_auv_centre])
            t.start()
            threads.append(t)
        if ekf_activate:
            t = threading.Thread(target=write_sidescan_csv,
                                args=[ekfcsvpath,
                                    ekf_list,
                                    'auv_ekf_centre_sss',
                                    csv_ekf_auv_centre])
            t.start()
            threads.append(t)
    if spp_output_activate and ekf_activate:
        Console.info("Converting covariance matrices into information matrices...")
        for i in range(len(camera1_ekf_list)):
            camera1_ekf_list[i].get_info()
        for i in range(len(camera2_ekf_list)):
            camera2_ekf_list[i].get_info()
        if len(camera3_list) > 1:
            for i in range(len(camera3_ekf_list)):
                camera3_ekf_list[i].get_info()
        Console.info("Converting poses into sequential-relative poses...")
        for i in range(len(camera1_ekf_list) - 1):
            camera1_ekf_list[i].northings -= camera1_ekf_list[i+1].northings
            camera1_ekf_list[i].eastings -= camera1_ekf_list[i+1].eastings
            camera1_ekf_list[i].depth -= camera1_ekf_list[i+1].depth
            camera1_ekf_list[i].roll -= camera1_ekf_list[i+1].roll
            camera1_ekf_list[i].pitch -= camera1_ekf_list[i+1].pitch
            camera1_ekf_list[i].yaw -= camera1_ekf_list[i+1].yaw
            camera1_ekf_list[i].information = camera1_ekf_list[i+1].information
        camera1_ekf_list = camera1_ekf_list[:-1]

        t = threading.Thread(
            target=spp_csv,
            args=[camera1_ekf_list, 'auv_ekf_' +
                  mission.image.cameras[0].name,
                  ekfcsvpath,
                  spp_ekf_camera_1])
        t.start()
        threads.append(t)
        t = threading.Thread(
            target=spp_csv,
            args=[camera2_ekf_list, 'auv_ekf_' +
                  mission.image.cameras[1].name,
                  ekfcsvpath,
                  spp_ekf_camera_2])
        t.start()
        threads.append(t)
        if len(camera3_list) > 1:
            if len(mission.image.cameras) > 2:
                t = threading.Thread(
                    target=spp_csv,
                    args=[camera3_ekf_list, 'auv_ekf_' +
                          mission.image.cameras[2].name,
                          ekfcsvpath,
                          spp_ekf_camera_3])
                t.start()
                threads.append(t)
            elif len(mission.image.cameras) == 2:
                t = threading.Thread(
                    target=spp_csv,
                    args=[camera3_ekf_list, 'auv_ekf_' +
                          mission.image.cameras[1].name + '_laser',
                          ekfcsvpath,
                          spp_ekf_camera_3])
                t.start()
                threads.append(t)
    Console.info("Waiting for all threads to finish")
    for t in threads:
        t.join()
    Console.info("DONE")

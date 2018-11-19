# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""

from auv_nav.tools.interpolate import interpolate
from auv_nav.tools.interpolate import interpolate_sensor_list
from auv_nav.tools.time_conversions import string_to_epoch
from auv_nav.tools.time_conversions import epoch_from_json
from auv_nav.tools.time_conversions import epoch_to_datetime
from auv_nav.tools.latlon_wgs84 import metres_to_latlon
from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.csv_tools import write_csv
from auv_nav.tools.csv_tools import camera_csv
from auv_nav.tools.csv_tools import other_data_csv
from auv_nav.sensors import BodyVelocity, InertialVelocity
from auv_nav.sensors import Altitude, Depth, Usbl, Orientation
from auv_nav.sensors import Other, Camera
from auv_nav.sensors import SyncedOrientationBodyVelocity
from auv_nav.localisation.dead_reckoning import dead_reckoning
from auv_nav.localisation.usbl_offset import usbl_offset
from auv_nav.localisation.particle_filter import ParticleFilter
from auv_nav.localisation.ekf import ExtendedKalmanFilter, Index
from auv_nav.localisation.usbl_filter import usbl_filter
from auv_nav.plot.plot_process_data import plot_orientation_vs_time
from auv_nav.plot.plot_process_data import plot_velocity_vs_time
from auv_nav.plot.plot_process_data import plot_deadreckoning_vs_time
from auv_nav.plot.plot_process_data import plot_pf_uncertainty
from auv_nav.plot.plot_process_data import plot_2d_deadreckoning
from auv_nav.plot.plot_process_data import plot_2d_localisation


# Import librarys
import os
import yaml
import json
import shutil
import time
import codecs
import pathlib
import copy
import math

from pathlib import Path
import numpy as np


"""
Assumes filename_camera of 1, 2, and 3 contains the image number between the
last 11 and 4 characters for appropriate csv pose estimate files output.
e.g. 'Xviii/Cam51707923/0094853.raw' or 'LM165/001/image0001011.tif'

Scripts to extract data from nav_standard.json, and combined.auv.raw an save
csv files and, if plot is True, save plots
"""


def process_data(filepath, ftype, start_datetime, finish_datetime):
    # placeholders
    interpolate_remove_flag = False

    # selected start and finish time
    epoch_start_time = 0
    epoch_finish_time = 0

    # velocity body placeholders (DVL)
    velocity_body_list = []
    velocity_body_sensor_name = 'vel_body'
    # velocity inertial placeholders
    velocity_inertial_list = []
    velocity_inertial_sensor_name = 'vel_inertial'
    # orientation placeholders (INS)
    orientation_list = []
    orientation_sensor_name = 'ins'
    # depth placeholders
    depth_list = []
    depth_sensor_name = 'depth'
    # altitude placeholders
    altitude_list = []
    altitude_sensor_name = 'alt'
    # USBL placeholders
    usbl_list = []
    usbl_sensor_name = 'usbl'

    # camera1 placeholders
    camera1_list = []
    camera1_pf_list = []
    camera1_sensor_name = 'cam1'  # original serial_camera1
    # camera2 placeholders
    camera2_list = []
    camera2_pf_list = []
    camera2_sensor_name = 'cam2'
    # camera3 placeholders
    camera3_list = []
    camera3_pf_list = []
    camera3_sensor_name = 'cam3'

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

    # std factors and offsets defaults
    std_factor_usbl = 0.01
    std_offset_usbl = 10.
    std_factor_dvl = 0.001
    std_offset_dvl = 0.002
    std_factor_depth = 0.
    std_offset_depth = 0.
    std_factor_orientation = 0.
    std_offset_orientation = 0.003

# load localisaion.yaml for particle filter and other setup
    print('Loading localisation.yaml')
    localisation = os.path.join(filepath, 'localisation.yaml')
    localisation_file = Path(localisation)

    # check if localisation.yaml file exist, if not, generate one with default settings
    if localisation_file.exists():
        print("Loading existing localisation.yaml at {}".format(localisation))
    else:
        print(
            str(os.path.join(str(pathlib.Path(__file__).parents[1]), 'default_yaml')))
        default_localisation = os.path.join(
            str(pathlib.Path(__file__).parents[1]), 'default_yaml', 'localisation.yaml')
        print("default_localisation: " + default_localisation)
        print("Cannot find {}, generating default from {}".format(
            localisation, default_localisation))
        # save localisation yaml to processed directory
        shutil.copy2(default_localisation, filepath)

    with open(localisation, 'r') as stream:
        load_localisation = yaml.load(stream)
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
            std_factor_usbl = load_localisation['std']['usbl']['factor']
            std_offset_usbl = load_localisation['std']['usbl']['offset']
            std_factor_dvl = load_localisation['std']['dvl']['factor']
            std_offset_dvl = load_localisation['std']['dvl']['offset']
            std_factor_depth = load_localisation['std']['depth']['factor']
            std_offset_depth = load_localisation['std']['depth']['offset']
            std_factor_orientation = load_localisation['std']['orientation']['factor']
            std_offset_orientation = load_localisation['std']['orientation']['offset']
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
        if 'ekf' in load_localisation:
            ekf_activate = load_localisation['ekf']['activate']
            ekf_process_noise_covariance = load_localisation['ekf']['process_noise_covariance']
            ekf_process_noise_covariance = np.asarray(
                ekf_process_noise_covariance).reshape((15, 15))
            ekf_initial_estimate_covariance = load_localisation[
                'ekf']['initial_estimate_covariance']
            ekf_initial_estimate_covariance = np.asarray(
                ekf_initial_estimate_covariance).reshape((15, 15))
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
        if 'plot_output' in load_localisation:
            plot_output_activate = load_localisation['plot_output']['activate']
            pdf_plot = load_localisation['plot_output']['pdf_plot']
            html_plot = load_localisation['plot_output']['html_plot']

# get information of sensor position offset from origin/centre reference point from vehicle.yaml
    # origin_x_offset = origin_y_offset = origin_z_offset = 0
    # camera1_x_offset = camera1_y_offset = camera1_z_offset = 0
    # camera2_x_offset = camera2_y_offset = camera2_z_offset = 0
    # camera3_x_offset = camera3_y_offset = camera3_z_offset = 0
    # usbl_x_offset = usbl_y_offset = usbl_z_offset = 0
    # dvl_x_offset = dvl_y_offset = dvl_z_offset = 0
    # depth_x_offset = depth_y_offset = depth_z_offset = 0
    # ins_x_offset = ins_y_offset = ins_z_offset = 0
    # chemical_x_offset = chemical_y_offset = chemical_z_offset = 0
    print('Loading vehicle.yaml')
    vehicle = os.path.join(filepath, 'vehicle.yaml')
    # if os.path.isdir(vehicle):
    with open(vehicle, 'r') as stream:
        vehicle_data = yaml.load(stream)
    if 'origin' in vehicle_data:
        origin_x_offset = vehicle_data['origin']['x_offset']
        origin_y_offset = vehicle_data['origin']['y_offset']
        origin_z_offset = vehicle_data['origin']['z_offset']
    if 'camera1' in vehicle_data:
        camera1_x_offset = vehicle_data['camera1']['x_offset']
        camera1_y_offset = vehicle_data['camera1']['y_offset']
        camera1_z_offset = vehicle_data['camera1']['z_offset']
    if 'camera2' in vehicle_data:
        camera2_x_offset = vehicle_data['camera2']['x_offset']
        camera2_y_offset = vehicle_data['camera2']['y_offset']
        camera2_z_offset = vehicle_data['camera2']['z_offset']
    if 'camera3' in vehicle_data:
        camera3_x_offset = vehicle_data['camera3']['x_offset']
        camera3_y_offset = vehicle_data['camera3']['y_offset']
        camera3_z_offset = vehicle_data['camera3']['z_offset']
    if 'usbl' in vehicle_data:
        usbl_x_offset = vehicle_data['usbl']['x_offset']
        usbl_y_offset = vehicle_data['usbl']['y_offset']
        usbl_z_offset = vehicle_data['usbl']['z_offset']
    if 'dvl' in vehicle_data:
        dvl_x_offset = vehicle_data['dvl']['x_offset']
        dvl_y_offset = vehicle_data['dvl']['y_offset']
        dvl_z_offset = vehicle_data['dvl']['z_offset']
    if 'depth' in vehicle_data:
        depth_x_offset = vehicle_data['depth']['x_offset']
        depth_y_offset = vehicle_data['depth']['y_offset']
        depth_z_offset = vehicle_data['depth']['z_offset']
    if 'ins' in vehicle_data:
        ins_x_offset = vehicle_data['ins']['x_offset']
        ins_y_offset = vehicle_data['ins']['y_offset']
        ins_z_offset = vehicle_data['ins']['z_offset']
    if 'chemical' in vehicle_data:
        chemical_x_offset = vehicle_data['chemical']['x_offset']
        chemical_y_offset = vehicle_data['chemical']['y_offset']
        chemical_z_offset = vehicle_data['chemical']['z_offset']

# OPLAB mode
    if ftype == 'oplab':  # or (ftype is not 'acfr'):
        outpath = os.path.join(filepath, 'nav')

        filename = 'nav_standard.json'
        print('Loading json file ' + os.path.join(outpath, filename))
        with open(os.path.join(outpath, filename)) as nav_standard:
            parsed_json_data = json.load(nav_standard)

        print('Loading mission.yaml')
        mission = os.path.join(filepath, 'mission.yaml')
        with open(mission, 'r') as stream:
            mission_data = yaml.load(stream)
        # assigns sensor names from mission.yaml instead of json data packet (instead of looking at json data as TunaSand don't have serial yet)
        if 'origin' in mission_data:
            origin_flag = 1
            latitude_reference = mission_data['origin']['latitude']
            longitude_reference = mission_data['origin']['longitude']
            coordinate_reference = mission_data['origin']['coordinate_reference_system']
            date = mission_data['origin']['date']
        if 'velocity' in mission_data:
            velocity_body_sensor_name = mission_data['velocity']['format']
            velocity_inertial_sensor_name = mission_data['velocity']['format']
        if 'orientation' in mission_data:
            orientation_sensor_name = mission_data['orientation']['format']
        if 'depth' in mission_data:
            depth_sensor_name = mission_data['depth']['format']
        if 'altitude' in mission_data:
            altitude_sensor_name = mission_data['altitude']['format']
        if 'usbl' in mission_data:
            usbl_sensor_name = mission_data['usbl']['format']
        if 'image' in mission_data:
            if 'camera1' in mission_data['image']:
                camera1_sensor_name = '_'.join(
                    mission_data['image']['camera1'].split('/'))
            if 'camera2' in mission_data['image']:
                camera2_sensor_name = '_'.join(
                    mission_data['image']['camera2'].split('/'))
            if 'camera3' in mission_data['image']:
                camera3_sensor_name = '_'.join(
                    mission_data['image']['camera3'].split('/'))

        # setup start and finish date time
        if start_datetime == '':
            epoch_start_time = epoch_from_json(parsed_json_data[0])
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
                                velocity_body.from_json(parsed_json_data[i])
                                velocity_body_list.append(velocity_body)
                    if 'inertial' in parsed_json_data[i]['frame']:
                        velocity_inertial = InertialVelocity()
                        velocity_inertial.from_json(parsed_json_data[i])
                        velocity_inertial_list.append(velocity_inertial)

                if 'orientation' in parsed_json_data[i]['category']:
                    orientation = Orientation()
                    orientation.from_json(parsed_json_data[i])
                    orientation_list.append(orientation)

                if 'depth' in parsed_json_data[i]['category']:
                    depth = Depth()
                    depth.from_json(parsed_json_data[i])
                    depth_list.append(depth)

                if 'altitude' in parsed_json_data[i]['category']:
                    altitude = Altitude()
                    altitude.from_json(parsed_json_data[i])
                    altitude_list.append(altitude)

                if 'usbl' in parsed_json_data[i]['category']:
                    usbl = Usbl()
                    usbl.from_json(parsed_json_data[i])
                    usbl_list.append(usbl)

                if 'image' in parsed_json_data[i]['category']:
                    camera1 = Camera()
                    # LC
                    camera1.from_json(parsed_json_data[1], 'camera1')
                    camera1_list.append(camera1)
                    camera2 = Camera()
                    camera2.from_json(parsed_json_data[1], 'camera2')
                    camera2_list.append(camera2)

                if 'laser' in parsed_json_data[i]['category']:
                    camera3 = Camera()
                    camera3.from_json(parsed_json_data[1], 'camera3')
                    camera3_list.append(camera3)

                if 'chemical' in parsed_json_data[i]['category']:
                    chemical = Other()
                    chemical.from_json(parsed_json_data[i])
                    chemical_list.append(chemical)

        camera1_pf_list = copy.deepcopy(camera1_list)
        camera2_pf_list = copy.deepcopy(camera2_list)
        camera3_pf_list = copy.deepcopy(camera3_list)
        chemical_pf_list = copy.deepcopy(chemical_list)

        camera1_ekf_list = copy.deepcopy(camera1_list)
        camera2_ekf_list = copy.deepcopy(camera2_list)
        camera3_ekf_list = copy.deepcopy(camera3_list)
        chemical_ekf_list = copy.deepcopy(chemical_list)

    # make path for processed outputs
        renavpath = os.path.join(
            filepath, ('json_renav_' + start_datetime[0:8] + '_' + start_datetime[8:14] + '_' + finish_datetime[0:8] + '_' + finish_datetime[8:14]))
        if os.path.isdir(renavpath) == 0:
            try:
                os.mkdir(renavpath)
            except Exception as e:
                print("Warning:", e)

        print('Complete parse of:' + os.path.join(outpath, filename))
        print('Writing outputs to: ' + renavpath)
        # copy to renav folder os.path.join(renavpath, 'localisation.yaml')
        # shutil.copy2(localisation, renavpath) # save mission yaml to processed directory

# ACFR mode
    if ftype == 'acfr':
        # extract_acfr()
        print('Loading mission.cfg')
        mission = os.path.join(filepath, 'mission.cfg')
        with codecs.open(mission, 'r', encoding='utf-8', errors='ignore') as filein:
            for line in filein.readlines():
                line_split = line.strip().split(' ')
                if str(line_split[0]) == 'MAG_VAR_LAT':
                    latitude_reference = float(line_split[1])
                if str(line_split[0]) == 'MAG_VAR_LNG':
                    longitude_reference = float(line_split[1])
                if str(line_split[0]) == 'MAG_VAR_DATE':
                    date = str(line_split[1])

        outpath = os.path.join(filepath, 'dRAWLOGS_cv')

        filename = 'combined.RAW.auv'
        print('Loading acfr standard RAW.auv file ' +
              os.path.join(outpath, filename))

        with codecs.open(os.path.join(outpath, filename), 'r',
                         encoding='utf-8', errors='ignore') as filein:
            # setup the time window
            parsed_acfr_data = filein.readlines()
            if start_datetime == '':
                start_epoch_timestamp = float(
                    parsed_acfr_data[0].split(' ')[1])
                start_datetime = time.strftime(
                    '%Y%m%d%H%M%S', time.localtime(start_epoch_timestamp))
            epoch_start_time = string_to_epoch(start_datetime)
            if finish_datetime == '':
                finish_epoch_timestamp = float(
                    parsed_acfr_data[-1].split(' ')[1])
                finish_datetime = time.strftime(
                    '%Y%m%d%H%M%S', time.localtime(finish_epoch_timestamp))
            epoch_finish_time = string_to_epoch(finish_datetime)

            for line in parsed_acfr_data:
                line_split = line.split(' ')
                if str(line_split[0]) == 'RDI:':
                    epoch_timestamp = float(line_split[1])
                    if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:
                        velocity_body = BodyVelocity()
                        velocity_body.epoch_timestamp = epoch_timestamp
                        altitude = Altitude()
                        altitude.epoch_timestamp = epoch_timestamp
                        for i in range(len(line_split)):
                            value = line_split[i].split(':')
                            if value[0] == 'alt':
                                altitude.altitude = float(value[1])
                            if value[0] == 'vx':
                                velocity_body.x_velocity = float(value[1])
                            if value[0] == 'vy':
                                velocity_body.y_velocity = float(value[1])
                            if value[0] == 'vz':
                                velocity_body.z_velocity = float(value[1])
                        velocity_body_list.append(velocity_body)
                        altitude_list.append(altitude)
                if str(line_split[0]) == 'PHINS_COMPASS:':
                    epoch_timestamp = float(line_split[1])
                    if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:
                        velocity_inertial = InertialVelocity()
                        velocity_inertial.epoch_timestamp = epoch_timestamp
                        orientation = Orientation()
                        orientation.epoch_timestamp = epoch_timestamp
                        for i in range(len(line_split)-1):
                            if line_split[i] == 'r:':
                                orientation.roll = float(line_split[i+1])
                            if line_split[i] == 'p:':
                                orientation.pitch = float(line_split[i+1])
                            if line_split[i] == 'h:':
                                orientation.yaw = float(line_split[i+1])
                        velocity_inertial_list.append(velocity_inertial)
                        orientation_list.append(orientation)
                if str(line_split[0]) == 'PAROSCI:':
                    epoch_timestamp = float(line_split[1])
                    if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:
                        depth = Depth()
                        depth.epoch_timestamp = epoch_timestamp
                        depth.depth = float(line_split[2])
                        depth_list.append(depth)
                if str(line_split[0]) == 'SSBL_FIX:':
                    epoch_timestamp = float(line_split[1])
                    if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:
                        usbl = Usbl()
                        usbl.epoch_timestamp = epoch_timestamp
                        for i in range(len(line_split)-1):
                            if line_split[i] == 'target_x:':
                                usbl.northings = float(line_split[i+1])
                            if line_split[i] == 'target_y:':
                                usbl.eastings = float(line_split[i+1])
                            if line_split[i] == 'target_z:':
                                usbl.depth = float(line_split[i+1])
                        usbl_list.append(usbl)
                if str(line_split[0]) == 'VIS:':
                    if 'FC' or 'LC' in line_split[3]:
                        camera1 = Camera()
                        camera1.epoch_timestamp = float(line_split[1])
                        camera1.filename = line_split[3]
                        camera1_list.append(camera1)
                    if 'AC' or 'RC' in line_split[3]:
                        camera2 = Camera()
                        camera2.epoch_timestamp = float(line_split[1])
                        camera2.filename = line_split[3]
                        camera2_list.append(camera2)
        camera1_pf_list = copy.deepcopy(camera1_list)
        camera2_pf_list = copy.deepcopy(camera2_list)
        # camera3_pf_list = copy.deepcopy(camera3_list)
        # chemical_pf_list = copy.deepcopy(chemical_list)

        # make folder to store csv and plots
        renavpath = os.path.join(
            filepath, ('acfr_renav_'
                       + start_datetime[0:8]
                       + '_'
                       + start_datetime[8:14]
                       + '_'
                       + finish_datetime[0:8]
                       + '_'
                       + finish_datetime[8:14]))
        if os.path.isdir(renavpath) == 0:
            try:
                os.mkdir(renavpath)
            except Exception as e:
                print("Warning:", e)

        print('Complete parse of:' + os.path.join(outpath, filename))
        print('Writing outputs to: ' + renavpath)

    # interpolate to find the appropriate depth to compute seafloor depth for each altitude measurement
    j = 0
    for i in range(len(altitude_list)):
        while j < len(depth_list)-1 and depth_list[j].epoch_timestamp < altitude_list[i].epoch_timestamp:
            j = j + 1

        if j >= 1:
            altitude_list[i].seafloor_depth = interpolate(altitude_list[i].epoch_timestamp, depth_list[j-1].epoch_timestamp,
                                                          depth_list[j].epoch_timestamp, depth_list[j-1].depth, depth_list[j].depth)+altitude_list[i].altitude

    # perform usbl_filter
    if usbl_filter_activate:
        usbl_list = usbl_filter(
            usbl_list, depth_list, sigma_factor, max_auv_speed, ftype)

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
        dead_reckoning_dvl.x_velocity = interpolate(orientation_list[i].epoch_timestamp, velocity_body_list[j].epoch_timestamp,
                                                    velocity_body_list[j+1].epoch_timestamp, velocity_body_list[j].x_velocity, velocity_body_list[j+1].x_velocity)
        dead_reckoning_dvl.y_velocity = interpolate(orientation_list[i].epoch_timestamp, velocity_body_list[j].epoch_timestamp,
                                                    velocity_body_list[j+1].epoch_timestamp, velocity_body_list[j].y_velocity, velocity_body_list[j+1].y_velocity)
        dead_reckoning_dvl.z_velocity = interpolate(orientation_list[i].epoch_timestamp, velocity_body_list[j].epoch_timestamp,
                                                    velocity_body_list[j+1].epoch_timestamp, velocity_body_list[j].z_velocity, velocity_body_list[j+1].z_velocity)

        [x_offset, y_offset, z_offset] = body_to_inertial(orientation_list[i].roll, orientation_list[i].pitch, orientation_list[i].yaw,
                                                          dead_reckoning_dvl.x_velocity, dead_reckoning_dvl.y_velocity, dead_reckoning_dvl.z_velocity)

        dead_reckoning_dvl.north_velocity = x_offset
        dead_reckoning_dvl.east_velocity = y_offset
        dead_reckoning_dvl.down_velocity = z_offset

        # double check this step, i.e. what if velocity_body_list timestamps not = altitude timestamps
        while n < len(altitude_list)-1 and n < len(velocity_body_list)-1 and orientation_list[i].epoch_timestamp > altitude_list[n+1].epoch_timestamp and orientation_list[i].epoch_timestamp > velocity_body_list[n+1].epoch_timestamp:
            n += 1
        dead_reckoning_dvl.altitude = interpolate(orientation_list[i].epoch_timestamp, velocity_body_list[n].epoch_timestamp,
                                                  velocity_body_list[n+1].epoch_timestamp, altitude_list[n].altitude, altitude_list[n+1].altitude)

        while k < len(depth_list)-1 and depth_list[k].epoch_timestamp < orientation_list[i].epoch_timestamp:
            k += 1
        # interpolate to find the appropriate depth for dead_reckoning
        dead_reckoning_dvl.depth = interpolate(
            orientation_list[i].epoch_timestamp, depth_list[k-1].epoch_timestamp, depth_list[k].epoch_timestamp, depth_list[k-1].depth, depth_list[k].depth)
        dead_reckoning_dvl.depth_std = interpolate(
            orientation_list[i].epoch_timestamp, depth_list[k-1].epoch_timestamp, depth_list[k].epoch_timestamp, depth_list[k-1].depth_std, depth_list[k].depth_std)
        dead_reckoning_dvl_list.append(dead_reckoning_dvl)

    # dead reckoning solution
    for i in range(len(dead_reckoning_dvl_list)):
        # dead reckoning solution
        if i >= 1:
            [dead_reckoning_dvl_list[i].northings, dead_reckoning_dvl_list[i].eastings] = dead_reckoning(dead_reckoning_dvl_list[i].epoch_timestamp, dead_reckoning_dvl_list[i-1].epoch_timestamp, dead_reckoning_dvl_list[i].north_velocity,
                                                                                                         dead_reckoning_dvl_list[i-1].north_velocity, dead_reckoning_dvl_list[i].east_velocity, dead_reckoning_dvl_list[i-1].east_velocity, dead_reckoning_dvl_list[i-1].northings, dead_reckoning_dvl_list[i-1].eastings)

    # offset sensor to plot origin/centre of vehicle
    dead_reckoning_centre_list = copy.deepcopy(
        dead_reckoning_dvl_list)  # [:] #.copy()
    for i in range(len(dead_reckoning_centre_list)):
        [x_offset, y_offset, z_offset] = body_to_inertial(dead_reckoning_centre_list[i].roll, dead_reckoning_centre_list[i].pitch,
                                                          dead_reckoning_centre_list[i].yaw, origin_x_offset - dvl_x_offset, origin_y_offset - dvl_y_offset, origin_z_offset - dvl_z_offset)
        dead_reckoning_centre_list[i].northings += x_offset
        dead_reckoning_centre_list[i].eastings += y_offset
        # dead_reckoning_centre_list[i].depth += z_offset
    # correct for altitude and depth offset too!

    # remove first term if first time_orientation is < velocity_body time
    if interpolate_remove_flag == True:

        # del time_orientation[0]
        del dead_reckoning_centre_list[0]
        del dead_reckoning_dvl_list[0]
        interpolate_remove_flag = False  # reset flag
    print('Complete interpolation and coordinate transfomations for velocity_body')

# perform interpolations of state data to velocity_inertial time stamps (without sensor offset and correct imu to dvl flipped interpolation) and perform deadreckoning
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
                velocity_inertial_list[i].roll = interpolate(velocity_inertial_list[i].epoch_timestamp, orientation_list[j -
                                                                                                                         1].epoch_timestamp, orientation_list[j].epoch_timestamp, orientation_list[j-1].roll, orientation_list[j].roll)
                velocity_inertial_list[i].pitch = interpolate(velocity_inertial_list[i].epoch_timestamp, orientation_list[j -
                                                                                                                          1].epoch_timestamp, orientation_list[j].epoch_timestamp, orientation_list[j-1].pitch, orientation_list[j].pitch)

                if abs(orientation_list[j].yaw-orientation_list[j-1].yaw) > 180:
                    if orientation_list[j].yaw > orientation_list[j-1].yaw:
                        velocity_inertial_list[i].yaw = interpolate(velocity_inertial_list[i].epoch_timestamp, orientation_list[j-1].epoch_timestamp,
                                                                    orientation_list[j].epoch_timestamp, orientation_list[j-1].yaw, orientation_list[j].yaw-360)

                    else:
                        velocity_inertial_list[i].yaw = interpolate(velocity_inertial_list[i].epoch_timestamp, orientation_list[j-1].epoch_timestamp,
                                                                    orientation_list[j].epoch_timestamp, orientation_list[j-1].yaw-360, orientation_list[j].yaw)

                    if velocity_inertial_list[i].yaw < 0:
                        velocity_inertial_list[i].yaw += 360

                    elif velocity_inertial_list[i].yaw > 360:
                        velocity_inertial_list[i].yaw -= 360

                else:
                    velocity_inertial_list[i].yaw = interpolate(velocity_inertial_list[i].epoch_timestamp, orientation_list[j -
                                                                                                                            1].epoch_timestamp, orientation_list[j].epoch_timestamp, orientation_list[j-1].yaw, orientation_list[j].yaw)

            while k < len(depth_list)-1 and depth_list[k].epoch_timestamp < velocity_inertial_list[i].epoch_timestamp:
                k = k+1

            if k >= 1:
                velocity_inertial_list[i].depth = interpolate(velocity_inertial_list[i].epoch_timestamp, depth_list[k-1].epoch_timestamp,
                                                              depth_list[k].epoch_timestamp, depth_list[k-1].depth, depth_list[k].depth)  # depth directly interpolated from depth sensor

        for i in range(len(velocity_inertial_list)):
            if i >= 1:
                [velocity_inertial_list[i].northings, velocity_inertial_list[i].eastings] = dead_reckoning(velocity_inertial_list[i].epoch_timestamp, velocity_inertial_list[i-1].epoch_timestamp, velocity_inertial_list[i].north_velocity,
                                                                                                           velocity_inertial_list[i-1].north_velocity, velocity_inertial_list[i].east_velocity, velocity_inertial_list[i-1].east_velocity, velocity_inertial_list[i-1].northings, velocity_inertial_list[i-1].eastings)

        if interpolate_remove_flag == True:
            del velocity_inertial_list[0]
            interpolate_remove_flag = False  # reset flag
        print(
            'Complete interpolation and coordinate transfomations for velocity_inertial')

# offset velocity DR by average usbl estimate
    # offset velocity body DR by average usbl estimate
    [northings_usbl_interpolated, eastings_usbl_interpolated] = usbl_offset([i.epoch_timestamp for i in dead_reckoning_centre_list], [i.northings for i in dead_reckoning_centre_list], [
                                                                            i.eastings for i in dead_reckoning_centre_list], [i.epoch_timestamp for i in usbl_list], [i.northings for i in usbl_list], [i.eastings for i in usbl_list])
    for i in range(len(dead_reckoning_centre_list)):
        dead_reckoning_centre_list[i].northings += northings_usbl_interpolated
        dead_reckoning_centre_list[i].eastings += eastings_usbl_interpolated
        dead_reckoning_centre_list[i].latitude, dead_reckoning_centre_list[i].longitude = metres_to_latlon(
            latitude_reference, longitude_reference, dead_reckoning_centre_list[i].eastings, dead_reckoning_centre_list[i].northings)
    for i in range(len(dead_reckoning_dvl_list)):
        dead_reckoning_dvl_list[i].northings += northings_usbl_interpolated
        dead_reckoning_dvl_list[i].eastings += eastings_usbl_interpolated
        dead_reckoning_dvl_list[i].latitude, dead_reckoning_dvl_list[i].longitude = metres_to_latlon(
            latitude_reference, longitude_reference, dead_reckoning_dvl_list[i].eastings, dead_reckoning_dvl_list[i].northings)

    # offset velocity inertial DR by average usbl estimate
    if len(velocity_inertial_list) > 0:
        [northings_usbl_interpolated, eastings_usbl_interpolated] = usbl_offset([i.epoch_timestamp for i in velocity_inertial_list], [i.northings for i in velocity_inertial_list], [
                                                                                i.eastings for i in velocity_inertial_list], [i.epoch_timestamp for i in usbl_list], [i.northings for i in usbl_list], [i.eastings for i in usbl_list])
        for i in range(len(velocity_inertial_list)):
            velocity_inertial_list[i].northings += northings_usbl_interpolated
            velocity_inertial_list[i].eastings += eastings_usbl_interpolated
            velocity_inertial_list[i].latitude, velocity_inertial_list[i].longitude = metres_to_latlon(
                latitude_reference, longitude_reference, velocity_inertial_list[i].eastings, velocity_inertial_list[i].northings)

# particle filter data fusion of usbl_data and dvl_imu_data
    if particle_filter_activate:
        pf_start_time = time.time()
        [pf_fusion_dvl_list,
         pf_usbl_datapoints,
         pf_particles_list,
         pf_northings_std,
         pf_eastings_std,
         pf_yaw_std] = ParticleFilter(
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
        print("particle filter with {} particles took {} seconds".format(
            particles_number, pf_elapsed_time))
        pf_fusion_centre_list = copy.deepcopy(pf_fusion_dvl_list)
        for i in range(len(pf_fusion_centre_list)):
            pf_fusion_dvl_list[i].latitude,
            pf_fusion_dvl_list[i].longitude = metres_to_latlon(
                latitude_reference, longitude_reference,
                pf_fusion_dvl_list[i].eastings,
                pf_fusion_dvl_list[i].northings)
            [x_offset, y_offset, z_offset] = body_to_inertial(
                pf_fusion_centre_list[i].roll,
                pf_fusion_centre_list[i].pitch,
                pf_fusion_centre_list[i].yaw,
                origin_x_offset - dvl_x_offset,
                origin_y_offset - dvl_y_offset,
                origin_z_offset - dvl_z_offset)
            pf_fusion_centre_list[i].northings += x_offset
            pf_fusion_centre_list[i].eastings += y_offset
            lat, lon = metres_to_latlon(
                latitude_reference, longitude_reference,
                pf_fusion_centre_list[i].eastings,
                pf_fusion_centre_list[i].northings)
            pf_fusion_centre_list[i].latitude = lat
            pf_fusion_centre_list[i].longitude = lon

    if ekf_activate:
        ekf_start_time = time.time()
        # velocity_body_list, list of BodyVelocity()
        # orientation_list, list of Orientation()
        # depth_list, list of Depth()
        # usbl_list, list of Usbl()
        ekf = ExtendedKalmanFilter(ekf_initial_estimate_covariance,
                                   ekf_process_noise_covariance,
                                   sensors_std,
                                   dead_reckoning_dvl_list,
                                   usbl_list)
        ekf_states = ekf.get_smoothed_result()
        ekf_end_time = time.time()
        ekf_elapsed_time = ekf_end_time - ekf_start_time
        print("EKF took {} seconds".format(ekf_elapsed_time))
        # TODO: convert from EKF states in meters to lat lon
        ekf_list = []
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
                origin_x_offset - dvl_x_offset,
                origin_y_offset - dvl_y_offset,
                origin_z_offset - dvl_z_offset)
            b.northings += x_offset
            b.eastings += y_offset
            b.depth += z_offset
            b.latitude, b.longitude = metres_to_latlon(
                latitude_reference, longitude_reference,
                b.eastings, b.northings)
            ekf_list.append(b)

    origin_offsets = [origin_x_offset, origin_y_offset, origin_z_offset]
    latlon_reference = [latitude_reference, longitude_reference]

    # perform interpolations of state data to camera{1/2/3} time stamps
    # for both DR and PF
    if len(camera1_list) > 1:
        interpolate_sensor_list(
            camera1_list,
            camera1_sensor_name,
            [camera1_x_offset, camera1_y_offset, camera1_z_offset],
            origin_offsets,
            latlon_reference,
            dead_reckoning_centre_list)
    if len(camera2_list) > 1:
        interpolate_sensor_list(
            camera2_list,
            camera2_sensor_name,
            [camera2_x_offset, camera2_y_offset, camera2_z_offset],
            origin_offsets,
            latlon_reference,
            dead_reckoning_centre_list)
    if len(camera3_list) > 1:
        interpolate_sensor_list(
            camera3_list,
            camera3_sensor_name,
            [camera3_x_offset, camera3_y_offset, camera3_z_offset],
            origin_offsets,
            latlon_reference,
            dead_reckoning_centre_list)
    if len(pf_fusion_centre_list) > 1:
        if len(camera1_pf_list) > 1:
            interpolate_sensor_list(
                camera1_pf_list,
                camera1_sensor_name,
                [camera1_x_offset, camera1_y_offset, camera1_z_offset],
                origin_offsets,
                latlon_reference,
                pf_fusion_centre_list)
        if len(camera2_pf_list) > 1:
            interpolate_sensor_list(
                camera2_pf_list,
                camera2_sensor_name,
                [camera1_x_offset, camera1_y_offset, camera1_z_offset],
                origin_offsets,
                latlon_reference,
                pf_fusion_centre_list)
        if len(camera3_pf_list) > 1:
            interpolate_sensor_list(
                camera3_pf_list,
                camera3_sensor_name,
                [camera1_x_offset, camera1_y_offset, camera1_z_offset],
                origin_offsets,
                latlon_reference,
                pf_fusion_centre_list)
    if len(ekf_list) > 1:
        if len(camera1_ekf_list) > 1:
            interpolate_sensor_list(
                camera1_ekf_list,
                camera1_sensor_name,
                [camera1_x_offset, camera1_y_offset, camera1_z_offset],
                origin_offsets,
                latlon_reference,
                ekf_list)
        if len(camera2_ekf_list) > 1:
            interpolate_sensor_list(
                camera2_ekf_list,
                camera2_sensor_name,
                [camera1_x_offset, camera1_y_offset, camera1_z_offset],
                origin_offsets,
                latlon_reference,
                ekf_list)
        if len(camera3_ekf_list) > 1:
            interpolate_sensor_list(
                camera3_ekf_list,
                camera3_sensor_name,
                [camera1_x_offset, camera1_y_offset, camera1_z_offset],
                origin_offsets,
                latlon_reference,
                ekf_list)

    # perform interpolations of state data to chemical time stamps for both
    # DR and PF
    if len(chemical_list) > 1:
        interpolate_sensor_list(
            chemical_list,
            'chemical',
            [chemical_x_offset, chemical_y_offset, chemical_z_offset],
            origin_offsets,
            latlon_reference,
            dead_reckoning_centre_list)
        if len(pf_fusion_centre_list) > 1:
            interpolate_sensor_list(
                chemical_list,
                'chemical',
                [chemical_x_offset, chemical_y_offset, chemical_z_offset],
                origin_offsets,
                latlon_reference,
                pf_fusion_centre_list)

    if plot_output_activate:
        # if pdf_plot:
            # pdf_plot()

        # plotly data in html
        if html_plot is True:
            print('Plotting plotly data ...')
            plotlypath = os.path.join(renavpath, 'interactive_plots')
            if os.path.isdir(plotlypath) == 0:
                try:
                    os.mkdir(plotlypath)
                except Exception as e:
                    print("Warning:", e)

            plot_orientation_vs_time(orientation_list,
                                     plotlypath)
            plot_velocity_vs_time(dead_reckoning_dvl_list,
                                  velocity_inertial_list,
                                  dead_reckoning_centre_list,
                                  velocity_inertial_sensor_name,
                                  plotlypath)
            plot_deadreckoning_vs_time(dead_reckoning_dvl_list,
                                       velocity_inertial_list,
                                       usbl_list,
                                       dead_reckoning_centre_list,
                                       altitude_list,
                                       depth_list,
                                       velocity_inertial_sensor_name,
                                       plotlypath)
            if particle_filter_activate:
                plot_pf_uncertainty(pf_fusion_dvl_list,
                                    pf_northings_std,
                                    pf_eastings_std,
                                    pf_yaw_std,
                                    orientation_list,
                                    velocity_body_list,
                                    depth_list,
                                    usbl_list,
                                    velocity_inertial_list,
                                    velocity_inertial_sensor_name,
                                    plotlypath)
                plot_2d_deadreckoning(camera1_list,
                                      dead_reckoning_centre_list,
                                      dead_reckoning_dvl_list,
                                      pf_fusion_centre_list,
                                      ekf_list,
                                      camera1_pf_list,
                                      pf_fusion_dvl_list,
                                      particles_time_interval,
                                      pf_particles_list,
                                      usbl_list,
                                      plotlypath)
            print('Complete plot data: ', plotlypath)

# write values out to a csv file
    # create a directory with the time stamp

    # if csv_write is True:

    csvpath = os.path.join(renavpath, 'csv')
    drcsvpath = os.path.join(csvpath, 'dead_reckoning')
    pfcsvpath = os.path.join(csvpath, 'particle_filter')
    ekfcsvpath = os.path.join(csvpath, 'ekf')

    if csv_output_activate is True:
        if csv_usbl is True:
            if len(usbl_list) > 1:
                csv_file = Path(csvpath)
                if csv_file.exists() is False:
                    os.mkdir(csvpath)

                print("Writing outputs to auv_usbl.csv ...")
                with open(os.path.join(csvpath, 'auv_usbl.csv'),
                          'w') as fileout:
                    fileout.write(
                        'Timestamp, Northing [m], Easting [m], Depth [m], \
                        Latitude [deg], Longitude [deg]\n')
                for i in range(len(usbl_list)):
                    with open(os.path.join(csvpath, 'auv_usbl.csv'),
                              'a') as fileout:
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

        write_csv(drcsvpath, dead_reckoning_centre_list,
                  'auv_dr_centre', csv_dr_auv_centre)
        write_csv(drcsvpath, dead_reckoning_dvl_list,
                  'auv_dr_dvl', csv_dr_auv_dvl)
        camera_csv(camera1_list, 'auv_dr_' +
                   camera1_sensor_name, drcsvpath, csv_dr_camera_1)
        camera_csv(camera2_list, 'auv_dr_' +
                   camera2_sensor_name, drcsvpath, csv_dr_camera_2)
        camera_csv(camera3_list, 'auv_dr_' +
                   camera3_sensor_name, drcsvpath, csv_dr_camera_3)
        other_data_csv(chemical_list, 'auv_dr_chemical',
                       drcsvpath, csv_dr_chemical)
        write_csv(pfcsvpath, pf_fusion_centre_list,
                  'auv_pf_centre', csv_pf_auv_centre)
        write_csv(pfcsvpath, pf_fusion_dvl_list,
                  'auv_pf_dvl', csv_pf_auv_dvl)
        camera_csv(camera1_pf_list, 'auv_pf_' +
                   camera1_sensor_name, pfcsvpath, csv_pf_camera_1)
        camera_csv(camera2_pf_list, 'auv_pf_' +
                   camera2_sensor_name, pfcsvpath, csv_pf_camera_2)
        camera_csv(camera3_pf_list, 'auv_pf_' +
                   camera3_sensor_name, pfcsvpath, csv_pf_camera_3)
        other_data_csv(chemical_list, 'auv_pf_chemical',
                       pfcsvpath, csv_pf_chemical)
        write_csv(ekfcsvpath, ekf_list,
                  'auv_ekf_centre', csv_ekf_auv_centre)
        camera_csv(camera1_ekf_list, 'auv_ekf_' +
                   camera1_sensor_name, ekfcsvpath, csv_ekf_camera_1)
        camera_csv(camera2_ekf_list, 'auv_ekf_' +
                   camera2_sensor_name, ekfcsvpath, csv_ekf_camera_2)
        camera_csv(camera3_ekf_list, 'auv_ekf_' +
                   camera3_sensor_name, ekfcsvpath, csv_ekf_camera_3)
        other_data_csv(chemical_list, 'auv_ekf_chemical',
                       ekfcsvpath, csv_pf_chemical)

    print('Complete extraction of data: ', csvpath)

    print('Completed data extraction: ', renavpath)

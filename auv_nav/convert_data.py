# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""

from auv_nav.tools.time_conversions import string_to_epoch
from auv_nav.tools.time_conversions import epoch_from_json
from auv_nav.tools.time_conversions import epoch_to_datetime
from auv_nav.tools.latlon_wgs84 import metres_to_latlon
from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.csv_tools import write_csv, write_raw_sensor_csv
from auv_nav.tools.csv_tools import camera_csv
from auv_nav.tools.csv_tools import other_data_csv
from auv_nav.sensors import BodyVelocity, InertialVelocity
from auv_nav.sensors import Altitude, Depth, Usbl, Orientation
from auv_nav.sensors import Other, Camera
from auv_nav.sensors import SyncedOrientationBodyVelocity
from auv_nav.tools.folder_structure import get_config_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.parsers.vehicle import Vehicle
from auv_nav.parsers.mission import Mission
from auv_nav.tools.console import Console


# Import librarys
import yaml
import json
import time
import copy
import math

from pathlib import Path
import numpy as np


class AcfrConverter():
    """
    This output format has four main sensor types:
    * RDI: mixed body velocity, orientation and altitude measurement
    * PHINS_COMPASS: for orientation
    * PAROSCI: for pressure sensor
    * VIS: for cameras
    * SSBL_FIX: for USBL or SSBL global localization
    """
    def __init__(self, mission, vehicle, filepath):
        self.mission = mission
        self.vehicle = vehicle
        self.filepath = filepath
        self.rdi_altitude = None
        self.rdi_orientation = None
        self.data = ''

        outpath = get_processed_folder(self.filepath)
        config_filename = outpath / 'mission.cfg'

        outpath = outpath / 'dRAWLOGS_cv'

        if not outpath.exists():
            outpath.mkdir(parents=True)

        self.nav_file = outpath / 'combined.RAW.auv'

        with config_filename.open('w') as f:
            data = ('MAG_VAR_LAT ' + str(float(self.mission.origin.latitude))
                    + '\nMAG_VAR_LNG ' + str(float(self.mission.origin.longitude))
                    + '\nMAG_VAR_DATE "' + str(self.mission.origin.date) + '"'
                    + '\nMAGNETIC_VAR_DEG ' + str(float(0)))
            f.write(data)
        # keep the file opened
        self.f = self.nav_file.open('w')

    def rdi_ready(self):
        if (self.rdi_altitude is not None
                and self.rdi_orientation is not None):
            return True
        else:
            return False

    def add(self, measurement):
        data = None
        if type(measurement) is BodyVelocity:
            if self.rdi_ready():
                data = measurement.to_acfr(self.rdi_altitude,
                                           self.rdi_orientation)
                self.rdi_orientation = None
                self.rdi_altitude = None
        elif type(measurement) is InertialVelocity:
            pass
        elif type(measurement) is Altitude:
            self.rdi_altitude = measurement
        elif type(measurement) is Depth:
            data = measurement.to_acfr()
        elif type(measurement) is Usbl:
            data = measurement.to_acfr()
        elif type(measurement) is Orientation:
            data = measurement.to_acfr()
            self.rdi_orientation = measurement
        elif type(measurement) is Other:
            pass
        elif type(measurement) is Camera:
            # Get rid of laser images
            if 'xxx' in measurement.filename:
                pass
            else:
                data = measurement.to_acfr()
        else:
            Console.error('AcfrConverter type {} not supported'.format(type(measurement)))
        if data is not None:
            self.f.write(data)


def convert_data(filepath, ftype, start_datetime, finish_datetime):
    Console.info('Requested data conversion to {}'.format(ftype))

    filepath = Path(filepath).resolve()
    mission_file = filepath / 'mission.yaml'
    vehicle_file = filepath / 'vehicle.yaml'
    mission_file = get_processed_folder(mission_file)
    vehicle_file = get_processed_folder(vehicle_file)
    Console.info('Loading mission.yaml at {0}'.format(mission_file))
    mission = Mission(mission_file)

    Console.info('Loading vehicle.yaml at {0}'.format(vehicle_file))
    vehicle = Vehicle(vehicle_file)

    converter = None
    if ftype == 'acfr':
        converter = AcfrConverter(mission, vehicle, filepath)
    else:
        Console.error('Converter type {} not implemented.'.format(ftype))

    nav_standard_file = filepath / 'nav' / 'nav_standard.json'
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

    sensors_std = {
        'usbl': {
            'model': 'json'
        },
        'dvl': {
            'model': 'json'
        },
        'depth': {
            'model': 'json'
        },
        'orientation': {
            'model': 'json'
        }
    }

    # read in data from json file
    # i here is the number of the data packet
    for i in range(len(parsed_json_data)):
        Console.progress(i, len(parsed_json_data))
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
                            converter.add(velocity_body)
                if 'inertial' in parsed_json_data[i]['frame']:
                    velocity_inertial = InertialVelocity()
                    velocity_inertial.from_json(parsed_json_data[i])
                    converter.add(velocity_inertial)

            if 'orientation' in parsed_json_data[i]['category']:
                orientation = Orientation()
                orientation.from_json(parsed_json_data[i], sensors_std['orientation'])
                converter.add(orientation)

            if 'depth' in parsed_json_data[i]['category']:
                depth = Depth()
                depth.from_json(parsed_json_data[i], sensors_std['depth'])
                converter.add(depth)

            if 'altitude' in parsed_json_data[i]['category']:
                altitude = Altitude()
                altitude.from_json(parsed_json_data[i])
                converter.add(altitude)

            if 'usbl' in parsed_json_data[i]['category']:
                usbl = Usbl()
                usbl.from_json(parsed_json_data[i], sensors_std['usbl'])
                converter.add(usbl)

            if 'image' in parsed_json_data[i]['category']:
                camera1 = Camera()
                # LC
                camera1.from_json(parsed_json_data[i], 'camera1')
                converter.add(camera1)
                camera2 = Camera()
                camera2.from_json(parsed_json_data[i], 'camera2')
                converter.add(camera2)

            # if 'laser' in parsed_json_data[i]['category']:
            #     camera3 = Camera()
            #     camera3.from_json(parsed_json_data[i], 'camera3')
            #     converter.add(camera3)

            # if 'chemical' in parsed_json_data[i]['category']:
            #     chemical = Other()
            #     chemical.from_json(parsed_json_data[i])
            #     converter.add(chemical)
    Console.info('Conversion to {} finished!'.format(ftype))

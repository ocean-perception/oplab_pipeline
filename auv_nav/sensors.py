# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.time_conversions import date_time_to_epoch
from auv_nav.tools.time_conversions import read_timezone
from auv_nav.tools.latlon_wgs84 import latlon_to_metres
from auv_nav.tools.latlon_wgs84 import metres_to_latlon
from oplab import Console
from math import sqrt, atan2, pi, sin, cos
import json as js
import numpy as np


class PhinsHeaders():
    # define headers used in phins
    START = '$PIXSE'
    TIME = 'TIME__'
    HEADING = '$HEHDT'
    ATTITUDE = 'ATITUD'
    ATTITUDE_STD = 'STDHRP'
    DVL = 'LOGIN_'
    VEL = 'SPEED_'
    VEL_STD = 'STDSPD'
    DEPTH = 'DEPIN_'
    ALTITUDE = 'LOGDVL'


class Category():
    POSITION = 'position'
    ORIENTATION = 'orientation'
    VELOCITY = 'velocity'
    DEPTH = 'depth'
    ALTITUDE = 'altitude'
    USBL = 'usbl'
    TIDE = 'tide'


class OutputFormat():
    """
    Metaclass to contain virtual methods for sensors and
    their output formats
    """
    OPLAB = 'oplab'
    ACFR = 'acfr'

    def __init__(self):
        # Nothing to do
        self.epoch_timestamp = 0

    def __lt__(self, o):
        return self.epoch_timestamp < o.epoch_timestamp

    def _to_json(self):
        # To implement in child class
        raise NotImplementedError()

    def _to_acfr(self):
        # To implement in child class
        raise NotImplementedError()

    def clear(self):
        """
        To implement in child class.

        This method must clear all class member variables
        """
        raise NotImplementedError()

    def valid(self):
        """
        To implement in child class. This method must validate that
        all fields are valid and that the data is ready to be written
        """
        raise NotImplementedError

    def export(self, format):
        # Main output method
        data = None
        if self.valid():
            if format == self.OPLAB:
                data = self._to_json()
            elif format == self.ACFR:
                data = self._to_acfr()
            self.clear()
        return data


class BodyVelocity(OutputFormat):
    def __init__(self, velocity_std_factor=0.001,
                 velocity_std_offset=0.2, heading_offset=0,
                 timestamp=None):
        self.epoch_timestamp = None
        self.timestamp = timestamp
        self.velocity_std_factor = velocity_std_factor
        self.velocity_std_offset = velocity_std_offset
        self.yaw_offset = heading_offset
        self.sensor_string = 'unknown'
        self.clear()

    def clear(self):
        self.epoch_timestamp_dvl = None
        self.x_velocity = None
        self.y_velocity = None
        self.z_velocity = None
        self.x_velocity_std = None
        self.y_velocity_std = None
        self.z_velocity_std = None

    def valid(self):
        # The class is populated with just one line message.
        if self.x_velocity is not None and self.epoch_timestamp is not None and self.epoch_timestamp_dvl is not None:
            return True
        else:
            return False

    def get_std(self, value):
        return abs(value)*self.velocity_std_factor + self.velocity_std_offset

    def from_autosub(self, data, i):
        self.epoch_timestamp = data['eTime'][i]
        self.epoch_timestamp_dvl = data['eTime'][i]
        self.x_velocity = - data['Vnorth0'][i] * 0.001  # Relative to seabed
        self.y_velocity = - data['Veast0'][i] * 0.001
        self.z_velocity = - data['Vdown0'][i] * 0.001
        self.x_velocity_std = data['Verr0'][i] * 0.001
        self.y_velocity_std = data['Verr0'][i] * 0.001
        self.z_velocity_std = data['Verr0'][i] * 0.001

    def from_phins(self, line):
        self.sensor_string = 'phins'
        vx = float(line[2])  # DVL convention is +ve aft to forward
        vy = float(line[3])  # DVL convention is +ve port to starboard
        vz = float(line[4])  # DVL convention is bottom to top +ve
        # account for sensor rotational offset
        [vx, vy, vz] = body_to_inertial(0, 0, self.yaw_offset, vx, vy, vz)
        vy = -1*vy
        vz = -1*vz

        self.x_velocity = vx
        self.y_velocity = vy
        self.z_velocity = vz
        self.x_velocity_std = self.get_std(vx)
        self.y_velocity_std = self.get_std(vy)
        self.z_velocity_std = self.get_std(vz)
        self.epoch_timestamp_dvl = self.parse_dvl_time(line)

    def parse_dvl_time(self, line):
        epoch_time_dvl = None
        velocity_time = str(line[6])
        hour_dvl = int(velocity_time[0:2])
        mins_dvl = int(velocity_time[2:4])
        try:
            secs_dvl = int(velocity_time[4:6])
            # phins sometimes returns 60s...
            if secs_dvl < 60:
                msec_dvl = int(velocity_time[7:10])
                epoch_time_dvl = self.timestamp.get(hour_dvl,
                                                    mins_dvl,
                                                    secs_dvl,
                                                    msec_dvl)
        except Exception as exc:
            Console.warn('Warning: Badly formatted packet (PHINS TIME): '
                         + line[6] + ' Exception: ' + str(exc))
        return epoch_time_dvl

    def from_json(self, json, sensor_std):
        self.epoch_timestamp = json['epoch_timestamp_dvl']
        self.x_velocity = json['data'][0]['x_velocity']
        self.y_velocity = json['data'][1]['y_velocity']
        self.z_velocity = json['data'][2]['z_velocity']

        if sensor_std['model'] == 'sensor':
            self.x_velocity_std = json['data'][0]['x_velocity_std']
            self.y_velocity_std = json['data'][1]['y_velocity_std']
            self.z_velocity_std = json['data'][2]['z_velocity_std']
        elif sensor_std['model'] == 'linear':
            self.x_velocity_std = sensor_std['offset'] + sensor_std['factor']*self.x_velocity
            self.y_velocity_std = sensor_std['offset'] + sensor_std['factor']*self.y_velocity
            self.z_velocity_std = sensor_std['offset'] + sensor_std['factor']*self.z_velocity
        else:
            Console.error('The STD model you entered for DVL is not supported.')
            Console.quit('STD model not supported.')

    def write_csv_header(self):
        return ('epoch_timestamp,'
                + 'x_velocity,'
                + 'y_velocity,'
                + 'z_velocity,'
                + 'x_velocity_std,'
                + 'y_velocity_std,'
                + 'z_velocity_std\n')

    def to_csv(self):
        return (str(self.epoch_timestamp) + ','
                + str(self.x_velocity) + ','
                + str(self.y_velocity) + ','
                + str(self.z_velocity) + ','
                + str(self.x_velocity_std) + ','
                + str(self.y_velocity_std) + ','
                + str(self.z_velocity_std) + '\n')

    def _to_json(self):
        data = {
            'epoch_timestamp': float(self.epoch_timestamp),
            'epoch_timestamp_dvl': float(self.epoch_timestamp_dvl),
            'class': 'measurement',
            'sensor': self.sensor_string,
            'frame': 'body',
            'category': Category.VELOCITY,
            'data': [
                {
                    'x_velocity': float(self.x_velocity),
                    'x_velocity_std': float(self.x_velocity_std)
                }, {
                    'y_velocity': float(self.y_velocity),
                    'y_velocity_std': float(self.y_velocity_std)
                }, {
                    'z_velocity': float(self.z_velocity),
                    'z_velocity_std': float(self.z_velocity_std)
                }]
            }
        return data

    def to_acfr(self, altitude, orientation):
        sound_velocity = -9999
        data = ('RDI: ' + str(float(self.epoch_timestamp))
                + ' alt:' + str(float(altitude.altitude))
                + ' r1:0 r2:0 r3:0 r4:0 '
                + ' h:' + str(float(orientation.yaw))
                + ' p:' + str(float(orientation.pitch))
                + ' r:' + str(float(orientation.roll))
                + ' vx:' + str(float(self.x_velocity))
                + ' vy:' + str(float(self.y_velocity))
                + ' vz:' + str(float(self.z_velocity))
                + ' nx:0 ny:0 nz:0 COG:0 SOG:0 bt_status:0 '
                + ' h_true:0 p_gimbal:0 '
                + ' sv: ' + str(float(sound_velocity))
                + '\n')
        return data


class InertialVelocity(OutputFormat):
    def __init__(self):
        self.epoch_timestamp = None
        self.sensor_string = 'unknown'
        self.clear()

    def clear(self):
        self.north_velocity = None
        self.east_velocity = None
        self.down_velocity = None

        self.north_velocity_std = None
        self.east_velocity_std = None
        self.down_velocity_std = None

        # interpolated data.
        # maybe separate below to synced_velocity_inertial_orientation_...?
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        self.northings = 0
        self.eastings = 0
        self.depth = 0

        self.latitude = 0
        self.longitude = 0

    def valid(self):
        if (self.north_velocity is not None
                and self.north_velocity_std is not None
                and self.epoch_timestamp is not None):
            return True
        else:
            return False

    def from_phins(self, line):
        self.sensor_string = 'phins'
        if line[1] == PhinsHeaders.VEL:
            # phins convention is west +ve so a minus should be necessary
            # phins convention is up +ve
            self.east_velocity = float(line[2])
            self.north_velocity = float(line[3])
            self.down_velocity = -1*float(line[4])
        elif line[1] == PhinsHeaders.VEL_STD:
            # phins convention is west +ve
            # phins convention is up +ve
            self.east_velocity_std = float(line[2])
            self.north_velocity_std = float(line[3])
            self.down_velocity_std = -1*float(line[4])

    def from_json(self, json):
        self.epoch_timestamp = json['epoch_timestamp']
        self.north_velocity = json['data'][0]['north_velocity']
        self.east_velocity = json['data'][1]['east_velocity']
        self.down_velocity = json['data'][2]['down_velocity']
        self.north_velocity_std = json['data'][0]['north_velocity_std']
        self.east_velocity_std = json['data'][1]['east_velocity_std']
        self.down_velocity_std = json['data'][2]['down_velocity_std']

    def _to_json(self):
        data = {
            'epoch_timestamp': float(self.epoch_timestamp),
            'class': 'measurement',
            'sensor': self.sensor_string,
            'frame': 'inertial',
            'category': Category.VELOCITY,
            'data': [
                {
                    'north_velocity': float(self.north_velocity),
                    'north_velocity_std': float(self.north_velocity_std)
                }, {
                    'east_velocity': float(self.east_velocity),
                    'east_velocity_std': float(self.east_velocity_std)
                }, {
                    'down_velocity': float(self.down_velocity),
                    'down_velocity_std': float(self.down_velocity_std)
                }]
            }
        return data

    def to_acfr(self):
        pass


class Orientation(OutputFormat):
    def __init__(self, heading_offset=0.0, orientation_std_offset=None):
        self.epoch_timestamp = None
        self.yaw_offset = heading_offset
        self.sensor_string = 'unknown'
        self.clear()
        self.roll_std = None
        self.pitch_std = None
        self.yaw_std = None
        if orientation_std_offset is not None:
            self.roll_std = orientation_std_offset
            self.pitch_std = orientation_std_offset
            self.yaw_std = orientation_std_offset

    def clear(self):
        self.roll = None
        self.pitch = None
        self.yaw = None

    def valid(self):
        return (self.roll is not None
                and self.roll_std is not None
                and self.yaw is not None
                and self.epoch_timestamp is not None)

    def from_autosub(self, data, i):
        self.epoch_timestamp = data['eTime'][i]
        self.roll = data['Roll'][i] * 180.0 / pi
        self.pitch = data['Pitch'][i] * 180.0 / pi
        self.yaw = data['Heading'][i] * 180.0 / pi

    def from_phins(self, line):
        self.sensor_string = 'phins'
        if line[0] == PhinsHeaders.HEADING:
            # phins +ve clockwise so no need to change
            self.yaw = float(line[1])

        if line[1] == PhinsHeaders.ATTITUDE:
            self.roll = -1*float(line[2])
            # phins +ve nose up so no need to change
            self.pitch = -1*float(line[3])
            if self.yaw is not None:
                [self.roll, self.pitch, self.yaw] = body_to_inertial(
                    0, 0, self.yaw_offset,
                    self.roll, self.pitch, self.yaw)
                if self.yaw > 360:
                    self.yaw = self.yaw - 360
                if self.yaw < 0:
                    self.yaw = self.yaw + 360

        if line[1] == PhinsHeaders.ATTITUDE_STD:
            self.yaw_std = float(line[2])
            self.roll_std = float(line[3])
            self.pitch_std = float(line[4])

            # account for sensor rotational offset
            [roll_std, pitch_std, heading_std] = body_to_inertial(
                0, 0, self.yaw_offset,
                self.roll_std, self.pitch_std, self.yaw_std)

    def from_json(self, json, sensor_std):
        self.epoch_timestamp = json['epoch_timestamp']
        self.roll = json['data'][1]['roll']
        self.pitch = json['data'][2]['pitch']
        self.yaw = json['data'][0]['heading']

        if sensor_std['model'] == 'sensor':
            self.roll_std = json['data'][1]['roll_std']
            self.pitch_std = json['data'][2]['pitch_std']
            self.yaw_std = json['data'][0]['heading_std']
        elif sensor_std['model'] == 'linear':
            self.roll_std = sensor_std['offset'] + sensor_std['factor']*self.roll
            self.pitch_std = sensor_std['offset'] + sensor_std['factor']*self.pitch
            self.yaw_std = sensor_std['offset'] + sensor_std['factor']*self.yaw
        else:
            Console.error('The STD model you entered for USBL is not supported.')
            Console.quit('STD model not supported.')

    def write_csv_header(self):
        return ('epoch_timestamp,'
                + 'roll,'
                + 'pitch,'
                + 'yaw,'
                + 'roll_std,'
                + 'pitch_std,'
                + 'yaw_std\n')

    def to_csv(self):
        return (str(self.epoch_timestamp) + ','
                + str(self.roll) + ','
                + str(self.pitch) + ','
                + str(self.yaw) + ','
                + str(self.roll_std) + ','
                + str(self.pitch_std) + ','
                + str(self.yaw_std) + '\n')

    def _to_json(self):
        data = {
            'epoch_timestamp': float(self.epoch_timestamp),
            'class': 'measurement',
            'sensor': self.sensor_string,
            'frame': 'body',
            'category': Category.ORIENTATION,
            'data': [
                {
                    'heading': float(self.yaw),
                    'heading_std': float(self.yaw_std)
                }, {
                    'roll': float(self.roll),
                    'roll_std': float(self.roll_std)
                }, {
                    'pitch': float(self.pitch),
                    'pitch_std': float(self.pitch_std)
                }]
            }
        return data

    def to_acfr(self):
        data = ('PHINS_COMPASS: ' + str(float(self.epoch_timestamp))
                + ' r: ' + str(float(self.roll))
                + ' p: ' + str(float(self.pitch))
                + ' h: ' + str(float(self.yaw))
                + ' std_r: ' + str(float(self.roll_std))
                + ' std_p: ' + str(float(self.pitch_std))
                + ' std_h: ' + str(float(self.yaw_std)) + '\n')
        return data


class Depth(OutputFormat):
    def __init__(self, depth_std_factor=0.0001, ts=None):
        self.epoch_timestamp = None
        self.ts = ts
        self.depth_std_factor = depth_std_factor
        self.sensor_string = 'unknown'
        self.clear()

    def clear(self):
        self.depth = None
        self.depth_std = None
        self.depth_timestamp = None

    def valid(self):
        return (self.depth is not None
                and self.epoch_timestamp is not None
                and self.depth_timestamp is not None)

    def from_autosub(self, data, i):
        self.epoch_timestamp = data['eTime'][i]
        self.depth_timestamp = self.epoch_timestamp
        self.depth = data['DepCtldepth'][i]
        self.depth_std = self.depth*self.depth_std_factor

    def from_phins(self, line):
        self.sensor_string = 'phins'
        self.depth = float(line[2])
        self.depth_std = self.depth*self.depth_std_factor
        time_string = str(line[3])
        hour = int(time_string[0:2])
        mins = int(time_string[2:4])

        try:
            secs = int(time_string[4:6])
            # phins sometimes returns 60s...
            if secs < 60:
                msec = int(time_string[7:10])
                self.depth_timestamp = self.ts.get(
                    hour, mins, secs, msec)
        except Exception as exc:
            Console.warn('Badly formatted packet (DEPTH TIME): '
                         + time_string + ' Exception: ' + str(exc))

    def from_json(self, json, sensor_std):
        self.epoch_timestamp = json['epoch_timestamp']
        self.depth_timestamp = json['epoch_timestamp_depth']
        self.depth = json['data'][0]['depth']
        if sensor_std['model'] == 'sensor':
            self.depth_std = json['data'][0]['depth_std']
        elif sensor_std['model'] == 'linear':
            self.depth_std = sensor_std['offset'] + sensor_std['factor']*self.depth
        else:
            Console.error('The STD model you entered for USBL is not supported.')
            Console.quit('STD model not supported.')

    def write_csv_header(self):
        return ('epoch_timestamp,'
                + 'depth,'
                + 'depth_std\n')

    def to_csv(self):
        return (str(self.epoch_timestamp) + ','
                + str(self.depth) + ','
                + str(self.depth_std) + '\n')

    def _to_json(self):
        data = {
            'epoch_timestamp': float(self.epoch_timestamp),
            'epoch_timestamp_depth': float(self.depth_timestamp),
            'class': 'measurement',
            'sensor': self.sensor_string,
            'frame': 'inertial',
            'category': Category.DEPTH,
            'data': [
                {
                    'depth': float(self.depth),
                    'depth_std': float(self.depth_std)
                }]
            }
        return data

    def to_acfr(self):
        data = ('PAROSCI: ' + str(self.depth_timestamp)
                + ' ' + str(self.depth) + '\n')
        return data


class Altitude(OutputFormat):
    def __init__(self, altitude_std_factor=0.01):
        self.epoch_timestamp = None
        self.altitude_std_factor = altitude_std_factor
        self.sensor_string = 'unknown'
        self.clear()

    def clear(self):
        self.altitude = None
        self.altitude_std = None
        self.altitude_timestamp = None
        # interpolate depth and add altitude for every altitude measurement
        self.seafloor_depth = 0
        self.sound_velocity = 0
        self.sound_velocity_correction = 0

    def valid(self):
        return (self.altitude is not None
                and self.epoch_timestamp is not None
                and self.altitude_timestamp is not None)

    def from_autosub(self, data, i):
        self.epoch_timestamp = data['eTime'][i]
        self.altitude_timestamp = self.epoch_timestamp
        self.altitude = data['ADCPAvAlt'][i]
        self.altitude_std = self.altitude*self.altitude_std_factor

    def from_phins(self, line, altitude_timestamp):
        self.sensor_string = 'phins'
        self.altitude_timestamp = altitude_timestamp
        self.sound_velocity = float(line[2])
        self.sound_velocity_correction = float(line[3])
        self.altitude = float(line[4])
        self.altitude_std = self.altitude*self.altitude_std_factor

    def from_json(self, json):
        self.epoch_timestamp = json['epoch_timestamp']
        self.altitude = json['data'][0]['altitude']

    def write_csv_header(self):
        return ('epoch_timestamp,'
                + 'altitude\n')

    def to_csv(self):
        return (str(self.epoch_timestamp) + ','
                + str(self.altitude) + '\n')

    def _to_json(self):
        data = {
            'epoch_timestamp': float(self.epoch_timestamp),
            'epoch_timestamp_dvl': float(self.altitude_timestamp),
            'class': 'measurement',
            'sensor': self.sensor_string,
            'frame': 'body',
            'category': Category.ALTITUDE,
            'data': [
                {
                    'altitude': float(self.altitude),
                    'altitude_std': float(self.altitude_std)
                }, {
                    'sound_velocity': float(self.sound_velocity),
                    'sound_velocity_correction': float(
                        self.sound_velocity_correction)
                }]
            }
        return data

    def to_acfr(self):
        pass


class Usbl(OutputFormat):
    def __init__(self, std_factor=None, std_offset=None,
                 latitude_reference=None, longitude_reference=None):
        self.epoch_timestamp = None
        self.sensor_string = 'unknown'

        self.latitude = None
        self.longitude = None
        self.latitude_std = 0
        self.longitude_std = 0

        self.northings = None
        self.eastings = None
        self.northings_std = 0
        self.eastings_std = 0

        self.depth = 0
        self.depth_std = 0

        self.distance_to_ship = 0

        self.latitude_ship = 0
        self.longitude_ship = 0
        self.northings_ship = 0
        self.eastings_ship = 0
        self.heading_ship = 0

        # temporary solution for fk180731 cruise
        # self.epoch_timestamp = 0 timestamp
        self.northings_ship = 0
        self.eastings_ship = 0
        # self.northings_target = 0 northings
        # self.eastings_target = 0 eastings
        # self.depth = 0 depth
        self.lateral_distace = 0
        self.distance = 0
        self.bearing = 0

        if std_factor is not None:
            self.std_factor = std_factor
        if std_offset is not None:
            self.std_offset = std_offset
        if latitude_reference is not None:
            self.latitude_reference = latitude_reference
        if longitude_reference is not None:
            self.longitude_reference = longitude_reference

    def clear(self):
        self.latitude = None
        self.longitude = None
        self.eastings = None
        self.northings = None
        self.epoch_timestamp = None

    def valid(self):
        return (self.latitude is not None
                and self.longitude is not None
                and self.eastings is not None
                and self.northings is not None
                and self.epoch_timestamp is not None)

    def from_nmea(self, msg):
        self.epoch_timestamp = msg.timestamp
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        self.depth = - msg.altitude

        # calculate in meters from reference
        lateral_distance, bearing = latlon_to_metres(
            self.latitude, self.longitude,
            self.latitude_reference, self.longitude_reference)
        self.distance_to_ship = -1.0
        self.eastings = sin(
            bearing*pi/180.0)*lateral_distance
        self.northings = cos(
            bearing*pi/180.0)*lateral_distance
        self.eastings_std = self.std_factor*self.depth + self.std_offset
        self.northings_std = self.std_factor*self.depth + self.std_offset
        self.depth_std = self.std_factor*self.depth + self.std_offset
        # If your displacements aren't too great (less than a few kilometers)
        # and you're not right at the poles, use the quick and dirty estimate
        # that 111,111 meters (111.111 km) in the y direction is 1 degree (of
        # latitude) and 111,111 * cos(latitude) meters in the x direction is
        # 1 degree (of longitude).
        self.latitude_std = self.depth / 111.111e3
        self.longitude_std = self.latitude_std * cos(self.latitude*pi/180.0)

    def from_json(self, json, sensor_std):
        self.epoch_timestamp = json['epoch_timestamp']
        self.latitude = json['data_target'][0]['latitude']
        self.longitude = json['data_target'][1]['longitude']
        self.northings = json['data_target'][2]['northings']
        self.eastings = json['data_target'][3]['eastings']
        self.depth = json['data_target'][4]['depth']
        self.depth_std = json['data_target'][4]['depth_std']
        self.distance_to_ship = json['data_target'][5]['distance_to_ship']

        if sensor_std['model'] == 'sensor':
            self.latitude_std = json['data_target'][0]['latitude_std']
            self.longitude_std = json['data_target'][1]['longitude_std']
            self.northings_std = json['data_target'][2]['northings_std']
            self.eastings_std = json['data_target'][3]['eastings_std']
        elif sensor_std['model'] == 'linear':
            self.northings_std = sensor_std['offset'] + sensor_std['factor']*self.distance_to_ship
            self.eastings_std = sensor_std['offset'] + sensor_std['factor']*self.distance_to_ship
            self.latitude_std = self.eastings_std / 111.111e3
            self.longitude_std = self.northings_std / 111.111e3

        else:
            Console.error('The STD model you entered for USBL is not supported.')
            Console.quit('STD model not supported.')
        try:
            self.latitude_ship = json['data_ship'][0]['latitude']
            self.longitude_ship = json['data_ship'][0]['longitude']
            self.northings_ship = json['data_ship'][1]['northings']
            self.eastings_ship = json['data_ship'][1]['eastings']
            self.heading_ship = json['data_ship'][2]['heading']
        except Exception as exc:
            Console.warn('Please parse again this dataset.')

    def write_csv_header(self):
        return ('epoch_timestamp,'
                + 'latitude,'
                + 'longitude,'
                + 'northings,'
                + 'eastings,'
                + 'depth,'
                + 'distance_to_ship,'
                + 'latitude_std,'
                + 'longitude_std,'
                + 'northings_std,'
                + 'eastings_std,'
                + 'depth_std\n')

    def to_csv(self):
        return (str(self.epoch_timestamp) + ','
                + str(self.latitude) + ','
                + str(self.longitude) + ','
                + str(self.northings) + ','
                + str(self.eastings) + ','
                + str(self.depth) + ','
                + str(self.distance_to_ship) + ','
                + str(self.latitude_std) + ','
                + str(self.longitude_std) + ','
                + str(self.northings_std) + ','
                + str(self.eastings_std) + ','
                + str(self.depth_std) + '\n')

    def _to_json(self):
        data = {
            'epoch_timestamp': float(self.epoch_timestamp),
            'class': 'measurement',
            'sensor': self.sensor_string,
            'frame': 'inertial',
            'category': Category.USBL,
            'data_ship': [{
                'latitude': float(self.latitude_ship),
                'longitude': float(self.longitude_ship)
                }, {
                'northings': float(self.northings_ship),
                'eastings': float(self.eastings_ship)
                }, {
                'heading': float(self.heading_ship)}],
            'data_target': [{
                'latitude': float(self.latitude),
                'latitude_std': float(self.latitude_std)
                }, {
                'longitude': float(self.longitude),
                'longitude_std': float(self.longitude_std)
                }, {
                'northings': float(self.northings),
                'northings_std': float(self.northings_std)
                }, {
                'eastings': float(self.eastings),
                'eastings_std': float(self.eastings_std)
                }, {
                'depth': float(self.depth),
                'depth_std': float(self.depth_std)
                }, {
                'distance_to_ship': float(self.distance_to_ship)}]}
        return data

    def to_acfr(self):
        distance_range = -1.0
        if self.distance_to_ship > self.depth and self.distance_to_ship > 0:
            try:
                distance_range = sqrt(self.distance_to_ship**2 - self.depth**2)
            except ValueError:
                print('Value error:')
                print('Value distance_to_ship: ' + str(self.distance_to_ship))
                print('Value depth:            ' + str(self.depth))
        bearing = atan2(self.eastings, self.northings)*180/pi
        data = ('SSBL_FIX: ' + str(float(self.epoch_timestamp))
                + ' ship_x: ' + str(float(self.northings_ship))
                + ' ship_y: ' + str(float(self.eastings_ship))
                + ' target_x: ' + str(float(self.northings))
                + ' target_y: ' + str(float(self.eastings))
                + ' target_z: ' + str(float(self.depth))
                + ' target_hr: ' + str(float(distance_range))
                + ' target_sr: ' + str(float(self.distance_to_ship))
                + ' target_bearing: ' + str(float(bearing)) + '\n')
        return data


class Camera():
    def __init__(self, timestamp=None):
        self.epoch_timestamp = None
        self.filename = ''
        #
        self.northings = 0
        self.eastings = 0
        self.depth = 0

        self.latitude = 0
        self.longitude = 0

        # interpolated data
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.x_velocity = 0
        self.y_velocity = 0
        self.z_velocity = 0

        self.altitude = 0
        self.covariance = None
        self.information = None

    def get_info(self):
        try:
            self.information = np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError as err:
            Console.error('Failed to invert covariance matrix: '
                             + ' Error: ' + str(err))

    def from_json(self, json, cam_name):
        if cam_name in json:
            self.epoch_timestamp = json[cam_name][0]['epoch_timestamp']
            self.filename = json[cam_name][0]['filename']
        else:
            self.epoch_timestamp = json['epoch_timestamp']
            self.filename = json['filename']

    def to_acfr(self):
        data = ('VIS: ' + str(self.epoch_timestamp)
                + ' [' + str(self.epoch_timestamp)
                + '] ' + str(self.filename) + ' exp: 0\n')
        return data

    def write_csv_header(self):
        return 'Imagenumber,Northing [m],Easting [m],Depth [m],'\
               'Roll [deg],Pitch [deg],Heading [deg],Altitude '\
               '[m],Timestamp,Latitude [deg],Longitude [deg]'\
               ',x_velocity,y_velocity,z_velocity\n'

    def write_csv_header_cov(self):
        str_to_write_cov = ''
        cov = ['x', 'y', 'z',
               'roll', 'pitch', 'yaw',
               'vx', 'vy', 'vz',
               'vroll', 'vpitch', 'vyaw']
        for a in cov:
            for b in cov:
                str_to_write_cov += ', cov_'+a+'_'+b
        str_to_write_cov += '\n'
        return str_to_write_cov

    def to_csv(self):
        return (str(self.filename) + ','
                + str(self.northings) + ','
                + str(self.eastings) + ','
                + str(self.depth) + ','
                + str(self.roll) + ','
                + str(self.pitch) + ','
                + str(self.yaw) + ','
                + str(self.altitude) + ','
                + str(self.epoch_timestamp) + ','
                + str(self.latitude) + ','
                + str(self.longitude) + ','
                + str(self.x_velocity) + ','
                + str(self.y_velocity) + ','
                + str(self.z_velocity) + '\n')

    def to_csv_cov(self):
        if self.covariance is not None:
            cov = self.covariance.flatten().tolist()
            cov = [item for sublist in cov for item in sublist]
            str_to_write_cov = str(self.filename)
            for c in cov:
                str_to_write_cov += ', {:.6f}'.format(c)
            str_to_write_cov += '\n'


class Tide(OutputFormat):
    def __init__(self, height_std_factor=0.0001, ts=None):
        self.epoch_timestamp = None
        self.ts = ts
        self.height_std_factor = height_std_factor
        self.sensor_string = 'unknown'
        self.clear()

    def clear(self):
        self.height = None
        self.height_std = None

    def valid(self):
        return (self.height is not None
                and self.epoch_timestamp is not None)

    def from_json(self, json, sensor_std):
        self.epoch_timestamp = json['epoch_timestamp']
#        self.tide_timestamp = json['epoch_timestamp_tide']
        self.height = json['data'][0]['tide']

        if sensor_std['model'] == 'sensor':
            self.height_std = json['data'][0]['tide_std']
        elif sensor_std['model'] == 'linear':
            self.height_std = sensor_std['offset'] + sensor_std['factor']*self.height
        else:
            Console.error('The STD model you entered for TIDE is not supported.')
            Console.quit('STD model not supported.')

    def write_csv_header(self):
        return ('epoch_timestamp,'
                + 'height,'
                + 'height_std\n')

    def to_csv(self):
        return (str(self.epoch_timestamp) + ','
                + str(self.height) + ','
                + str(self.height_std) + '\n')

    def _to_json(self):
        data = {
            'epoch_timestamp': float(self.epoch_timestamp),
            'class': 'measurement',
            'sensor': self.sensor_string,
            'frame': 'inertial',
            'category': Category.TIDE,
            'data': [
                {
                    'height': float(self.height),
                    'height_std': float(self.height_std)
                }]
            }
        return data

class Other():
    def __init__(self, timestamp=None):
        self.epoch_timestamp = None
        self.data = []

        self.northings = 0
        self.eastings = 0
        self.depth = 0

        self.latitude = 0
        self.longitude = 0

        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        self.altitude = 0
        self.covariance = None

    def from_json(self, json):
        self.epoch_timestamp = json['epoch_timestamp']
        self.data = json['data']


class SyncedOrientationBodyVelocity():
    def __init__(self, timestamp=None):
        self.epoch_timestamp = None
        # from orientation
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.roll_std = 0
        self.pitch_std = 0
        self.yaw_std = 0
        # interpolated
        self.x_velocity = 0
        self.y_velocity = 0
        self.z_velocity = 0
        self.x_velocity_std = 0
        self.y_velocity_std = 0
        self.z_velocity_std = 0
        # transformed
        self.north_velocity = 0
        self.east_velocity = 0
        self.down_velocity = 0
        self.north_velocity_std = 0
        self.east_velocity_std = 0
        self.down_velocity_std = 0
        # interpolated
        self.altitude = 0
        # calculated
        self.northings = 0
        self.eastings = 0
        self.depth = 0  # from interpolation of depth, not dr
        self.depth_std = 0

        self.latitude = 0
        self.longitude = 0
        self.covariance = None

    def __lt__(self, o):
        return self.epoch_timestamp < o.epoch_timestamp

# class synced_velocity_inertial_orientation:
#   def __init__(self):
#       self.epoch_timestamp = 0

# maybe do one synchronised orientation_bodyVelocity, and then one class of dead_reckoning.
# and separate these steps in extract_data?

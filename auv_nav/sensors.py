# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""

from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.time_conversions import date_time_to_epoch
from auv_nav.tools.time_conversions import read_timezone
import json as js


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


class Timestamp():
    def __init__(self, date, timezone, offset):
        self.epoch_timestamp_from_zone_offset(date, timezone, offset)

    def epoch_timestamp_from_zone_offset(self, date, timezone, offset):
        self.year, self.month, self.day = date
        self.tz_offset = read_timezone(timezone)
        self.offset = offset

    def get(self, hour, mins, secs, msec):
        epoch_time = date_time_to_epoch(
            self.year, self.month, self.day,
            hour, mins, secs, self.tz_offset)
        return epoch_time + msec/1000+self.offset

    def epoch_timestamp_from_phins(self, line):
        epoch_timestamp = None
        time_string = str(line[2])
        if len(time_string) == 10:
            hour = int(time_string[0:2])
            mins = int(time_string[2:4])
            try:
                secs = int(time_string[4:6])
                # phins sometimes returns 60s...
                if secs < 60:
                    msec = int(time_string[7:10])
                    epoch_timestamp = self.get(hour, mins, secs, msec)
            except Exception as exc:
                print('Warning: Badly formatted packet (PHINS TIME): '
                      + time_string + ' Exception: ' + str(exc))
        else:
            print('Warning: Badly formatted packet (PHINS TIME): ' + str(line))
        return epoch_timestamp


class Category():
    POSITION = 'position'
    ORIENTATION = 'orientation'
    VELOCITY = 'velocity'
    DEPTH = 'depth'
    ALTITUDE = 'altitude'
    USBL = 'usbl'


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
        secs_dvl = int(velocity_time[4:6])
        try:
            secs_dvl = int(velocity_time[4:6])
            # phins sometimes returns 60s...
            if secs_dvl < 60:
               msec_dvl = int(velocity_time[7:10])
               epoch_time_dvl = self.timestamp.get( hour_dvl, mins_dvl, secs_dvl, msec_dvl)
        except Exception as exc:
            print('Warning: Badly formatted packet (PHINS TIME): '
                  + line[6] + ' Exception: ' + str(exc))
        return epoch_time_dvl

    def from_json(self, json):
        self.epoch_timestamp = json['epoch_timestamp_dvl']
        self.x_velocity = json['data'][0]['x_velocity']
        self.y_velocity = json['data'][1]['y_velocity']
        self.z_velocity = json['data'][2]['z_velocity']
        self.x_velocity_std = json['data'][0]['x_velocity_std']
        self.y_velocity_std = json['data'][1]['y_velocity_std']
        self.z_velocity_std = json['data'][2]['z_velocity_std']

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

    def _to_acfr(self):
        # This function has to be called when altitude and orientation
        # are available. Moved to PhinsParse class.
        pass


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

        # interpolated data. mabye separate below to synced_velocity_inertial_orientation_...?
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

    def _to_acfr(self):
        pass


class Orientation(OutputFormat):
    def __init__(self, heading_offset=0.0):
        self.epoch_timestamp = None
        self.epoch_timestamp = None
        self.yaw_offset = heading_offset
        self.sensor_string = 'unknown'
        self.clear()

    def clear(self):
        self.roll = None
        self.pitch = None
        self.yaw = None

        self.roll_std = None
        self.pitch_std = None
        self.yaw_std = None

    def valid(self):
        return (self.roll is not None
                and self.roll_std is not None
                and self.yaw is not None
                and self.epoch_timestamp is not None)

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
                # heading=heading+headingoffset
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

    def from_json(self, json):
        self.epoch_timestamp = json['epoch_timestamp']
        self.roll = json['data'][1]['roll']
        self.pitch = json['data'][2]['pitch']
        self.yaw = json['data'][0]['heading']
        self.roll_std = json['data'][1]['roll_std']
        self.pitch_std = json['data'][2]['pitch_std']
        self.yaw_std = json['data'][0]['heading_std']

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

    def _to_acfr(self):
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
            print('Warning: Badly formatted packet (DEPTH TIME): '
                  + time_string + ' Exception: ' + str(exc))

    def from_json(self, json):
        self.epoch_timestamp = json['epoch_timestamp_depth']
        self.depth = json['data'][0]['depth']
        self.depth_std = json['data'][0]['depth_std']

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

    def _to_acfr(self):
        data = ('PAROSCI: ' + str(float(self.depth_timestamp))
                + ' ' + str(float(self.depth)) + '\n')
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
        self.seafloor_depth = None
        self.sound_velocity = None
        self.sound_velocity_correction = None

    def valid(self):
        return (self.altitude is not None
                and self.epoch_timestamp is not None
                and self.altitude_timestamp is not None)

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

    def _to_acfr(self):
        pass


class Usbl(OutputFormat):
    def __init__(self):
        self.epoch_timestamp = None

        self.latitude = 0
        self.longitude = 0
        self.latitude_std = 0
        self.longitude_std = 0

        self.northings = 0
        self.eastings = 0
        self.northings_std = 0
        self.eastings_std = 0

        self.depth = 0
        self.depth_std = 0

        self.distance_to_ship = 0

        ### temporary solution for fk180731 cruise
        # self.epoch_timestamp = 0 timestamp
        self.northings_ship = 0
        self.eastings_ship = 0
        # self.northings_target = 0 northings
        # self.eastings_target = 0 eastings
        # self.depth = 0 depth
        self.lateral_distace = 0
        self.distance = 0  #distance
        self.bearing = 0

    def from_json(self, json):
        self.epoch_timestamp = json['epoch_timestamp']
        self.latitude = json['data_target'][0]['latitude']
        self.latitude_std = json['data_target'][0]['latitude_std']
        self.longitude = json['data_target'][1]['longitude']
        self.longitude_std = json['data_target'][1]['longitude_std']
        self.northings = json['data_target'][2]['northings']
        self.northings_std = json['data_target'][2]['northings_std']
        self.eastings = json['data_target'][3]['eastings']
        self.eastings_std = json['data_target'][3]['eastings_std']
        self.depth = json['data_target'][4]['depth']
        self.depth_std = json['data_target'][4]['depth_std']
        self.distance_to_ship = json['data_target'][5]['distance_to_ship']

    def _to_json(self):
        data = {
            'epoch_timestamp': float(self.epoch_timestamp),
            'category': Category.USBL,
            'data_target': [
                {
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
                    'distance_to_ship': float(self.distance_to_ship),
                }]
            }
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

        self.altitude = 0
        self.covariance = None

    def from_json(self, json, cam_name):
        if cam_name in json:
            self.epoch_timestamp = json[cam_name][0]['epoch_timestamp']
            self.filename = json[cam_name][0]['filename']
        else:
            self.epoch_timestamp = json['epoch_timestamp']
            self.filename = json['filename']


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
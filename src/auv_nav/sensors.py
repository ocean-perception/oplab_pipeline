# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import datetime
import time
from math import atan2, cos, pi, sin, sqrt
from pathlib import Path

import numpy as np
import pynmea2

from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.latlon_wgs84 import latlon_to_metres, metres_to_latlon
from auv_nav.tools.transformations import euler_from_quaternion
from oplab import Console


def ros_stamp_to_epoch(stamp):
    """Converts a ROS timestamp to an epoch time.

    Parameters
    ----------
    stamp : dict
        ROS timestamp.

    Returns
    -------
    float
        Epoch time
    """
    secs = float(stamp.secs)
    nsecs = float(stamp.nsecs)
    return secs + nsecs * 1e-9


def float_or_none(val):
    if val is not None:
        return float(val)
    else:
        return None


class PhinsHeaders:
    # define headers used in phins
    START = "$PIXSE"
    TIME = "TIME__"
    HEADING = "$HEHDT"
    ATTITUDE = "ATITUD"
    ATTITUDE_STD = "STDHRP"
    DVL = "LOGIN_"
    VEL = "SPEED_"
    VEL_STD = "STDSPD"
    DEPTH = "DEPIN_"
    ALTITUDE = "LOGDVL"


class Category:
    POSITION = "position"
    ORIENTATION = "orientation"
    VELOCITY = "velocity"
    DEPTH = "depth"
    ALTITUDE = "altitude"
    USBL = "usbl"
    TIDE = "tide"
    IMAGES = "images"


class OutputFormat:
    """
    Metaclass to contain virtual methods for sensors and
    their output formats
    """

    OPLAB = "oplab"
    ACFR = "acfr"

    def __init__(self):
        # Nothing to do
        self.epoch_timestamp = None
        self.tz_offset_s = 0.0
        self.category = "unknown"

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
        if data is None:
            Console.warn("WARNING: exporting non valid data from", self.category, "!")
        return data


class BodyVelocity(OutputFormat):
    def __init__(
        self,
        velocity_std_factor=0.001,
        velocity_std_offset=0.2,
        heading_offset=0,
        timestamp=None,
    ):
        self.epoch_timestamp = None
        self.epoch_timestamp_dvl = None
        self.x_velocity = None
        self.y_velocity = None
        self.z_velocity = None
        self.x_velocity_std = None
        self.y_velocity_std = None
        self.z_velocity_std = None
        self.timestamp = timestamp
        self.velocity_std_factor = velocity_std_factor
        self.velocity_std_offset = velocity_std_offset
        self.yaw_offset = heading_offset
        self.sensor_string = "unknown"
        self.category = Category.VELOCITY

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
        return (
            self.x_velocity is not None
            and self.y_velocity is not None
            and self.epoch_timestamp is not None
            and self.epoch_timestamp_dvl is not None
        )

    def get_std(self, value):
        return abs(value) * self.velocity_std_factor + self.velocity_std_offset

    def from_alr(self, t, vx, vy, vz):
        self.epoch_timestamp = t
        self.epoch_timestamp_dvl = t
        self.x_velocity = vx
        self.y_velocity = vy
        self.z_velocity = vz
        self.x_velocity_std = self.get_std(self.x_velocity)
        self.y_velocity_std = self.get_std(self.y_velocity)
        self.z_velocity_std = self.get_std(self.z_velocity)

    def from_autosub(self, data, i):
        self.epoch_timestamp = data["eTime"][i]
        self.epoch_timestamp_dvl = data["eTime"][i]
        self.x_velocity = -data["Vnorth0"][i] * 0.001  # Relative to seabed
        self.y_velocity = -data["Veast0"][i] * 0.001
        self.z_velocity = -data["Vdown0"][i] * 0.001
        self.x_velocity_std = data["Verr0"][i] * 0.001
        self.y_velocity_std = data["Verr0"][i] * 0.001
        self.z_velocity_std = data["Verr0"][i] * 0.001

    def from_phins(self, line):
        self.sensor_string = "phins"
        vx = float(line[2])  # DVL convention is +ve aft to forward
        vy = float(line[3])  # DVL convention is +ve port to starboard
        vz = float(line[4])  # DVL convention is bottom to top +ve
        # account for sensor rotational offset
        [vx, vy, vz] = body_to_inertial(0, 0, self.yaw_offset, vx, vy, vz)
        vy = -1 * vy
        vz = -1 * vz

        self.x_velocity = vx
        self.y_velocity = vy
        self.z_velocity = vz
        self.x_velocity_std = self.get_std(vx)
        self.y_velocity_std = self.get_std(vy)
        self.z_velocity_std = self.get_std(vz)
        self.epoch_timestamp_dvl = self.parse_dvl_time(line)

    def from_ntnu_dvl(self, filename, line):
        self.sensor_string = "ntnu_dvl"
        date_obj = datetime.datetime.strptime(filename[0:8], "%Y%m%d")
        time_obj = datetime.datetime.strptime(line["time"], "%H:%M:%S.%f")
        date_time_obj = datetime.datetime.combine(date_obj, time_obj.time())
        self.epoch_timestamp = date_time_obj.timestamp()
        self.epoch_timestamp_dvl = self.epoch_timestamp
        vx = float(line["u_dvl"])
        vy = float(line["v_dvl"])
        vz = float(line["z_dvl"])
        [vx, vy, vz] = body_to_inertial(0, 0, self.yaw_offset, vx, vy, vz)
        self.x_velocity = vx
        self.y_velocity = vy
        self.z_velocity = vz
        self.x_velocity_std = self.get_std(self.x_velocity)
        self.y_velocity_std = self.get_std(self.y_velocity)
        self.z_velocity_std = self.get_std(self.z_velocity)

        if (
            (self.x_velocity > 32 or self.x_velocity < -32)
            or (self.y_velocity > 32 or self.y_velocity < -32)
            or (self.z_velocity > 32 or self.z_velocity < -32)
        ):
            self.x_velocity = None
            self.y_velocity = None
            self.z_velocity = None

    def apply_offset(self, with_std=True):
        if not self.valid():
            return
        [self.x_velocity, self.y_velocity, self.z_velocity] = body_to_inertial(
            0,
            0,
            self.yaw_offset,
            self.x_velocity,
            self.y_velocity,
            self.z_velocity,
        )
        [
            self.x_velocity_std,
            self.y_velocity_std,
            self.z_velocity_std,
        ] = body_to_inertial(
            0,
            0,
            self.yaw_offset,
            self.x_velocity_std,
            self.y_velocity_std,
            self.z_velocity_std,
        )

    def from_ros(self, msg, msg_type, output_dir):
        """Parse ROS topic / from DVL data"""
        if msg_type == "cola2_msgs/DVL":
            # Check that the average speed is less than 2.5 m/s
            if msg.velocity_covariance[0] <= 0:
                return
            if msg.altitude < 0:
                # No bottom lock, no good.
                return
            if np.sqrt(msg.velocity.x**2 + msg.velocity.y**2) < 2.5:
                self.x_velocity = msg.velocity.x
                self.y_velocity = msg.velocity.y
                self.z_velocity = msg.velocity.z
                self.x_velocity_std = np.sqrt(msg.velocity_covariance[0])
                self.y_velocity_std = np.sqrt(msg.velocity_covariance[4])
                self.z_velocity_std = np.sqrt(msg.velocity_covariance[8])
        elif msg_type == "teledyne_explorer_dvl/TeledyneExplorerDVLData":
            vx = None
            vy = None
            vz = None
            verr = None

            if msg.bi_status == "A":
                vx = msg.bi_x_axis_mms * 0.001
                vy = msg.bi_y_axis_mms * 0.001
                vz = msg.bi_z_axis_mms * 0.001
                verr = abs(msg.bi_error_mms * 0.001)
            elif msg.wi_status == "A":
                vx = msg.wi_x_axis_mms * 0.001
                vy = msg.wi_y_axis_mms * 0.001
                vz = msg.wi_z_axis_mms * 0.001
                verr = abs(msg.wi_error_mms * 0.001)
            else:
                return
            self.x_velocity = vx
            self.y_velocity = vy
            self.z_velocity = vz
            self.x_velocity_std = verr
            self.y_velocity_std = verr
            self.z_velocity_std = verr

            # Check if data is valid
            bottom_valid = (
                msg.bi_x_axis_mms > -32768
                and msg.bi_y_axis_mms > -32768
                and msg.bi_z_axis_mms > -32768
            )
            water_valid = (
                msg.wi_x_axis_mms > -32768
                and msg.wi_y_axis_mms > -32768
                and msg.wi_z_axis_mms > -32768
            )
            if not (bottom_valid or water_valid):
                self.x_velocity = None
                self.y_velocity = None
                self.z_velocity = None
        else:
            Console.quit("BodyVelocity ROS parser for", msg_type, "not supported.")
        self.epoch_timestamp = ros_stamp_to_epoch(msg.header.stamp) - self.tz_offset_s
        self.epoch_timestamp_dvl = self.epoch_timestamp
        self.apply_offset(with_std=False)

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
                epoch_time_dvl = self.timestamp.get(
                    hour_dvl, mins_dvl, secs_dvl, msec_dvl
                )
        except Exception as exc:
            Console.warn(
                "Warning: Badly formatted packet (PHINS TIME): "
                + line[6]
                + " Exception: "
                + str(exc)
            )
        return epoch_time_dvl

    def from_json(self, json, sensor_std):
        self.epoch_timestamp = json["epoch_timestamp_dvl"]
        self.x_velocity = json["data"][0]["x_velocity"]
        self.y_velocity = json["data"][1]["y_velocity"]
        self.z_velocity = json["data"][2]["z_velocity"]

        if sensor_std["model"] == "sensor":
            self.x_velocity_std = json["data"][0]["x_velocity_std"]
            self.y_velocity_std = json["data"][1]["y_velocity_std"]
            self.z_velocity_std = json["data"][2]["z_velocity_std"]
        elif sensor_std["model"] == "linear":
            if "offset_x" in sensor_std:
                self.x_velocity_std = (
                    sensor_std["offset_x"] + sensor_std["factor_x"] * self.x_velocity
                )
                self.y_velocity_std = (
                    sensor_std["offset_y"] + sensor_std["factor_y"] * self.y_velocity
                )
                self.z_velocity_std = (
                    sensor_std["offset_z"] + sensor_std["factor_z"] * self.z_velocity
                )
            else:
                self.x_velocity_std = (
                    sensor_std["offset"] + sensor_std["factor"] * self.x_velocity
                )
                self.y_velocity_std = (
                    sensor_std["offset"] + sensor_std["factor"] * self.y_velocity
                )
                self.z_velocity_std = (
                    sensor_std["offset"] + sensor_std["factor"] * self.z_velocity
                )
        else:
            Console.error("The STD model you entered for DVL is not supported.")
            Console.quit("STD model not supported.")

    def get_csv_header(self):
        return (
            "epoch_timestamp,"
            + "x_velocity,"
            + "y_velocity,"
            + "z_velocity,"
            + "x_velocity_std,"
            + "y_velocity_std,"
            + "z_velocity_std\n"
        )

    def to_csv_row(self):
        return (
            str(self.epoch_timestamp)
            + ","
            + str(self.x_velocity)
            + ","
            + str(self.y_velocity)
            + ","
            + str(self.z_velocity)
            + ","
            + str(self.x_velocity_std)
            + ","
            + str(self.y_velocity_std)
            + ","
            + str(self.z_velocity_std)
            + "\n"
        )

    def _to_json(self):
        data = {
            "epoch_timestamp": float_or_none(self.epoch_timestamp),
            "epoch_timestamp_dvl": float_or_none(self.epoch_timestamp_dvl),
            "class": "measurement",
            "sensor": self.sensor_string,
            "frame": "body",
            "category": Category.VELOCITY,
            "data": [
                {
                    "x_velocity": float_or_none(self.x_velocity),
                    "x_velocity_std": float_or_none(self.x_velocity_std),
                },
                {
                    "y_velocity": float_or_none(self.y_velocity),
                    "y_velocity_std": float_or_none(self.y_velocity_std),
                },
                {
                    "z_velocity": float_or_none(self.z_velocity),
                    "z_velocity_std": float_or_none(self.z_velocity_std),
                },
            ],
        }
        return data

    def to_acfr(self, altitude, orientation):
        sound_velocity = -9999
        data = (
            "RDI: "
            + str(float(self.epoch_timestamp))
            + " alt:"
            + str(float(altitude.altitude))
            + " r1:0 r2:0 r3:0 r4:0 "
            + " h:"
            + str(float(orientation.yaw))
            + " p:"
            + str(float(orientation.pitch))
            + " r:"
            + str(float(orientation.roll))
            + " vx:"
            + str(float(self.x_velocity))
            + " vy:"
            + str(float(self.y_velocity))
            + " vz:"
            + str(float(self.z_velocity))
            + " nx:0 ny:0 nz:0 COG:0 SOG:0 bt_status:0 "
            + " h_true:0 p_gimbal:0 "
            + " sv: "
            + str(float(sound_velocity))
            + "\n"
        )
        return data


class InertialVelocity(OutputFormat):
    def __init__(self):
        self.epoch_timestamp = None
        self.sensor_string = "unknown"
        self.category = Category.VELOCITY
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
        self.roll = None
        self.pitch = None
        self.yaw = None

        self.northings = None
        self.eastings = None
        self.depth = None

        self.latitude = None
        self.longitude = None

    def valid(self):
        return (
            self.north_velocity is not None
            and self.north_velocity_std is not None
            and self.epoch_timestamp is not None
        )

    def from_phins(self, line):
        self.sensor_string = "phins"
        if line[1] == PhinsHeaders.VEL:
            # phins convention is west +ve so a minus should be necessary
            # phins convention is up +ve
            self.east_velocity = float(line[2])
            self.north_velocity = float(line[3])
            self.down_velocity = -1 * float(line[4])
        elif line[1] == PhinsHeaders.VEL_STD:
            # phins convention is west +ve
            # phins convention is up +ve
            self.east_velocity_std = float(line[2])
            self.north_velocity_std = float(line[3])
            self.down_velocity_std = -1 * float(line[4])

    def from_json(self, json):
        self.epoch_timestamp = json["epoch_timestamp"]
        self.north_velocity = json["data"][0]["north_velocity"]
        self.east_velocity = json["data"][1]["east_velocity"]
        self.down_velocity = json["data"][2]["down_velocity"]
        self.north_velocity_std = json["data"][0]["north_velocity_std"]
        self.east_velocity_std = json["data"][1]["east_velocity_std"]
        self.down_velocity_std = json["data"][2]["down_velocity_std"]

    def _to_json(self):
        data = {
            "epoch_timestamp": float(self.epoch_timestamp),
            "class": "measurement",
            "sensor": self.sensor_string,
            "frame": "inertial",
            "category": Category.VELOCITY,
            "data": [
                {
                    "north_velocity": float_or_none(self.north_velocity),
                    "north_velocity_std": float_or_none(self.north_velocity_std),
                },
                {
                    "east_velocity": float_or_none(self.east_velocity),
                    "east_velocity_std": float_or_none(self.east_velocity_std),
                },
                {
                    "down_velocity": float_or_none(self.down_velocity),
                    "down_velocity_std": float_or_none(self.down_velocity_std),
                },
            ],
        }
        return data

    def to_acfr(self):
        pass


class Orientation(OutputFormat):
    def __init__(self, heading_offset=0.0, orientation_std_offset=None):
        self.epoch_timestamp = None
        self.yaw_offset = heading_offset
        self.sensor_string = "unknown"
        self.clear()
        self.roll_std = None
        self.pitch_std = None
        self.yaw_std = None
        self.category = Category.ORIENTATION
        if orientation_std_offset is not None:
            self.roll_std = orientation_std_offset
            self.pitch_std = orientation_std_offset
            self.yaw_std = orientation_std_offset

    def clear(self):
        self.roll = None
        self.pitch = None
        self.yaw = None

    def valid(self):
        return (
            self.roll is not None
            and self.roll_std is not None
            and self.yaw is not None
            and self.epoch_timestamp is not None
        )

    def apply_offset(self):
        if self.yaw is not None:
            [self.roll, self.pitch, self.yaw] = body_to_inertial(
                0, 0, self.yaw_offset, self.roll, self.pitch, self.yaw
            )
            if self.yaw > 360:
                self.yaw = self.yaw - 360
            if self.yaw < 0:
                self.yaw = self.yaw + 360

    def apply_std_offset(self):
        # account for sensor rotational offset
        [self.roll_std, self.pitch_std, self.heading_std] = body_to_inertial(
            0,
            0,
            self.yaw_offset,
            self.roll_std,
            self.pitch_std,
            self.yaw_std,
        )

    def from_eiva_navipac(self, line):
        self.sensor_string = "eiva_navipac"
        parts = line.split()
        date_time_obj = datetime.datetime.strptime(parts[2], "%Y:%m:%d:%H:%M:%S.%f")
        self.epoch_timestamp = date_time_obj.timestamp()
        if self.epoch_timestamp == 0:
            # Invalid timestamp
            self.epoch_timestamp = None
            return

        nmea_string = parts[3].replace("\n", "")
        msg = pynmea2.parse(nmea_string)

        if msg.roll is None or msg.pitch is None or msg.heading is None:
            Console.warn("Dropping EIVA Navipac message", nmea_string)
            return

        try:
            self.roll = float(msg.roll)  # / 180.0 * pi
            self.pitch = float(msg.pitch)  # / 180.0 * pi
            self.yaw = float(msg.heading)  # / 180.0 * pi
            self.apply_offset()
        except (ValueError, TypeError):
            Console.warn("Invalid NMEA sentence from EIVA Navipac", nmea_string)
            self.roll = None
            self.pitch = None
            self.yaw = None

    def from_alr(self, t, roll_rad, pitch_rad, yaw_rad):
        self.epoch_timestamp = t
        self.roll = roll_rad * 180.0 / pi
        self.pitch = pitch_rad * 180.0 / pi
        self.yaw = yaw_rad * 180.0 / pi

    def from_autosub(self, data, i):
        self.epoch_timestamp = data["eTime"][i]
        self.roll = data["Roll"][i] * 180.0 / pi
        self.pitch = data["Pitch"][i] * 180.0 / pi
        self.yaw = data["Heading"][i] * 180.0 / pi

    def from_phins(self, line):
        self.sensor_string = "phins"
        if line[0] == PhinsHeaders.HEADING:
            # phins +ve clockwise so no need to change
            self.yaw = float(line[1])

        if line[1] == PhinsHeaders.ATTITUDE:
            self.roll = -1 * float(line[2])
            # phins +ve nose up so no need to change
            self.pitch = -1 * float(line[3])
            self.apply_offset()

        if line[1] == PhinsHeaders.ATTITUDE_STD:
            self.yaw_std = float(line[2])
            self.roll_std = float(line[3])
            self.pitch_std = float(line[4])
            self.apply_std_offset()

    def from_ros(self, msg, msg_type, output_dir):
        """Parse ROS message

        Parameters
        ----------
        msg : dict
            Message dict
        """
        if msg_type == "sensor_msgs/Imu":
            q = [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
            r, p, y = euler_from_quaternion(q)
            self.epoch_timestamp = (
                ros_stamp_to_epoch(msg.header.stamp) - self.tz_offset_s
            )
            self.roll = np.degrees(r)
            self.pitch = np.degrees(p)
            self.yaw = np.degrees(y)
            self.roll_std = np.degrees(np.sqrt(msg.orientation_covariance[0]))
            self.pitch_std = np.degrees(np.sqrt(msg.orientation_covariance[4]))
            self.yaw_std = np.degrees(np.sqrt(msg.orientation_covariance[8]))
        else:
            Console.quit("Orientation ROS parser for", msg_type, "not supported.")
        self.apply_offset()
        self.apply_std_offset()

    def from_json(self, json, sensor_std):
        self.epoch_timestamp = json["epoch_timestamp"]
        self.roll = json["data"][1]["roll"]
        self.pitch = json["data"][2]["pitch"]
        self.yaw = json["data"][0]["heading"]

        if sensor_std["model"] == "sensor":
            self.roll_std = json["data"][1]["roll_std"]
            self.pitch_std = json["data"][2]["pitch_std"]
            self.yaw_std = json["data"][0]["heading_std"]
        elif sensor_std["model"] == "linear":
            self.roll_std = sensor_std["offset"] + sensor_std["factor"] * self.roll
            self.pitch_std = sensor_std["offset"] + sensor_std["factor"] * self.pitch
            self.yaw_std = sensor_std["offset"] + sensor_std["factor"] * self.yaw
        else:
            Console.error("The STD model you entered for Orientation is not supported.")
            Console.quit("STD model not supported.")

    def get_csv_header(self):
        return (
            "epoch_timestamp,"
            + "roll,"
            + "pitch,"
            + "yaw,"
            + "roll_std,"
            + "pitch_std,"
            + "yaw_std\n"
        )

    def to_csv_row(self):
        return (
            str(self.epoch_timestamp)
            + ","
            + str(self.roll)
            + ","
            + str(self.pitch)
            + ","
            + str(self.yaw)
            + ","
            + str(self.roll_std)
            + ","
            + str(self.pitch_std)
            + ","
            + str(self.yaw_std)
            + "\n"
        )

    def _to_json(self):
        data = {
            "epoch_timestamp": float_or_none(self.epoch_timestamp),
            "class": "measurement",
            "sensor": self.sensor_string,
            "frame": "body",
            "category": Category.ORIENTATION,
            "data": [
                {
                    "heading": float_or_none(self.yaw),
                    "heading_std": float_or_none(self.yaw_std),
                },
                {
                    "roll": float_or_none(self.roll),
                    "roll_std": float_or_none(self.roll_std),
                },
                {
                    "pitch": float_or_none(self.pitch),
                    "pitch_std": float_or_none(self.pitch_std),
                },
            ],
        }
        return data

    def to_acfr(self):
        data = (
            "PHINS_COMPASS: "
            + str(float(self.epoch_timestamp))
            + " r: "
            + str(float(self.roll))
            + " p: "
            + str(float(self.pitch))
            + " h: "
            + str(float(self.yaw))
            + " std_r: "
            + str(float(self.roll_std))
            + " std_p: "
            + str(float(self.pitch_std))
            + " std_h: "
            + str(float(self.yaw_std))
            + "\n"
        )
        return data


class Depth(OutputFormat):
    def __init__(self, depth_std_factor=0.0001, ts=None):
        self.epoch_timestamp = None
        self.ts = ts
        self.depth_std_factor = depth_std_factor
        self.sensor_string = "unknown"
        self.category = Category.DEPTH
        self.clear()

    def clear(self):
        self.depth = None
        self.depth_std = None
        self.depth_timestamp = None

    def valid(self):
        return (
            self.depth is not None
            and self.epoch_timestamp is not None
            and self.depth_timestamp is not None
        )

    def from_ros(self, msg, msg_type, output_dir):
        if msg_type == "sensor_msgs/FluidPressure":
            # TODO: is this an expected density value?
            self.depth = msg.fluid_pressure / (1030 * 9.80665)
            self.depth_std = np.sqrt(msg.variance / (1030 * 9.80665))
        elif msg_type == "geometry_msgs/PoseWithCovarianceStamped":
            self.depth = msg.pose.pose.position.z
            self.depth_std = np.sqrt(msg.pose.covariance[14])
        else:
            Console.quit("Depth ROS parser for", msg_type, "not supported.")
        self.epoch_timestamp = ros_stamp_to_epoch(msg.header.stamp) - self.tz_offset_s
        self.depth_timestamp = self.epoch_timestamp
        if self.depth_std == 0:
            self.depth_std = self.depth * self.depth_std_factor

    def from_eiva_navipac(self, line):
        self.sensor_string = "eiva_navipac"
        parts = line.split()
        date_time_obj = datetime.datetime.strptime(parts[4], "%Y:%m:%d:%H:%M:%S.%f")
        self.epoch_timestamp = date_time_obj.timestamp()
        self.depth_timestamp = self.epoch_timestamp
        if self.epoch_timestamp == 0:
            # Invalid timestamp
            self.epoch_timestamp = None
            return
        self.depth = float(parts[5])
        self.depth_std = self.depth * self.depth_std_factor

    def from_alr(self, t, d):
        self.epoch_timestamp = t
        self.depth_timestamp = self.epoch_timestamp
        self.depth = d
        self.depth_std = self.depth * self.depth_std_factor

    def from_autosub(self, data, i):
        self.epoch_timestamp = data["eTime"][i]
        self.depth_timestamp = self.epoch_timestamp
        self.depth = data["DepCtldepth"][i]
        self.depth_std = self.depth * self.depth_std_factor

    def from_phins(self, line):
        self.sensor_string = "phins"
        self.depth = float(line[2])
        self.depth_std = self.depth * self.depth_std_factor
        time_string = str(line[3])
        hour = int(time_string[0:2])
        mins = int(time_string[2:4])

        try:
            secs = int(time_string[4:6])
            # phins sometimes returns 60s...
            if secs < 60:
                msec = int(time_string[7:10])
                self.depth_timestamp = self.ts.get(hour, mins, secs, msec)
        except Exception as exc:
            Console.warn(
                "Badly formatted packet (DEPTH TIME): "
                + time_string
                + " Exception: "
                + str(exc)
            )

    def from_json(self, json, sensor_std):
        self.epoch_timestamp = json["epoch_timestamp"]
        self.depth_timestamp = json["epoch_timestamp_depth"]
        self.depth = json["data"][0]["depth"]
        if sensor_std["model"] == "sensor":
            self.depth_std = json["data"][0]["depth_std"]
        elif sensor_std["model"] == "linear":
            self.depth_std = sensor_std["offset"] + sensor_std["factor"] * self.depth
        else:
            Console.error("The STD model you entered for Depth is not supported.")
            Console.quit("STD model not supported.")

    def get_csv_header(self):
        return "epoch_timestamp," + "depth," + "depth_std\n"

    def to_csv_row(self):
        return (
            str(self.epoch_timestamp)
            + ","
            + str(self.depth)
            + ","
            + str(self.depth_std)
            + "\n"
        )

    def _to_json(self):
        data = {
            "epoch_timestamp": float_or_none(self.epoch_timestamp),
            "epoch_timestamp_depth": float_or_none(self.depth_timestamp),
            "class": "measurement",
            "sensor": self.sensor_string,
            "frame": "inertial",
            "category": Category.DEPTH,
            "data": [
                {
                    "depth": float_or_none(self.depth),
                    "depth_std": float_or_none(self.depth_std),
                }
            ],
        }
        return data

    def to_acfr(self):
        data = "PAROSCI: " + str(self.depth_timestamp) + " " + str(self.depth) + "\n"
        return data


class Altitude(OutputFormat):
    def __init__(self, altitude_std_factor=0.01):
        self.epoch_timestamp = None
        self.altitude_std_factor = altitude_std_factor
        self.sensor_string = "unknown"
        self.category = Category.ALTITUDE
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
        return (
            self.altitude is not None
            and self.epoch_timestamp is not None
            and self.altitude_timestamp is not None
        )

    def from_ros(self, msg, msg_type, output_dir):
        """Parse ROS topic / from DVL data in Turbot AUV"""
        if msg_type == "cola2_msgs/DVL":
            if msg.altitude < 0:
                return
            self.altitude = msg.altitude
        elif msg_type == "sensor_msgs/Range":
            if msg.range < 0:
                return
            self.altitude = msg.range
        elif msg_type == "teledyne_explorer_dvl/TeledyneExplorerDVLData":
            if msg.bi_status == "V":
                # If status is "V" is NOT valid
                return
            self.altitude = msg.bd_range
        else:
            Console.quit("Altitude ROS parser for", msg_type, "not supported.")
        self.epoch_timestamp = ros_stamp_to_epoch(msg.header.stamp) - self.tz_offset_s
        self.altitude_timestamp = self.epoch_timestamp
        self.altitude_std = self.altitude * self.altitude_std_factor

    def from_ntnu_dvl(self, filename, row):
        self.sensor_string = "ntnu_dvl"
        date_obj = datetime.datetime.strptime(filename[0:8], "%Y%m%d")
        time_obj = datetime.datetime.strptime(row["time"], "%H:%M:%S.%f")
        date_time_obj = datetime.datetime.combine(date_obj, time_obj.time())
        self.epoch_timestamp = date_time_obj.timestamp()
        self.altitude_timestamp = self.epoch_timestamp
        # alt0 = float(row["dvl_alt0"])  # fore
        alt1 = float(row["dvl_alt1"])  # left
        # alt2 = float(row["dvl_alt2"])  # right
        alt3 = float(row["dvl_alt3"])  # aft
        self.altitude = (alt1 + alt3) / 2.0
        self.altitude_std = self.altitude * self.altitude_std_factor

    def from_alr(self, t, a):
        self.epoch_timestamp = t
        self.altitude_timestamp = self.epoch_timestamp
        self.altitude = a
        self.altitude_std = self.altitude * self.altitude_std_factor

    def from_autosub(self, data, i):
        self.epoch_timestamp = data["eTime"][i]
        self.altitude_timestamp = self.epoch_timestamp
        self.altitude = data["ADCPAvAlt"][i]
        self.altitude_std = self.altitude * self.altitude_std_factor

    def from_phins(self, line, altitude_timestamp):
        self.sensor_string = "phins"
        self.altitude_timestamp = altitude_timestamp
        self.sound_velocity = float(line[2])
        self.sound_velocity_correction = float(line[3])
        self.altitude = float(line[4])
        self.altitude_std = self.altitude * self.altitude_std_factor

    def from_json(self, json):
        self.epoch_timestamp = json["epoch_timestamp"]
        self.altitude = json["data"][0]["altitude"]
        self.altitude_std = json["data"][0]["altitude_std"]

    def get_csv_header(self):
        return "epoch_timestamp," + "altitude\n"

    def to_csv_row(self):
        return str(self.epoch_timestamp) + "," + str(self.altitude) + "\n"

    def _to_json(self):
        data = {
            "epoch_timestamp": float_or_none(self.epoch_timestamp),
            "epoch_timestamp_dvl": float_or_none(self.altitude_timestamp),
            "class": "measurement",
            "sensor": self.sensor_string,
            "frame": "body",
            "category": Category.ALTITUDE,
            "data": [
                {
                    "altitude": float_or_none(self.altitude),
                    "altitude_std": float_or_none(self.altitude_std),
                },
                {
                    "sound_velocity": float_or_none(self.sound_velocity),
                    "sound_velocity_correction": float_or_none(
                        self.sound_velocity_correction
                    ),
                },
            ],
        }
        return data

    def to_acfr(self):
        pass


class Usbl(OutputFormat):
    def __init__(
        self,
        std_factor=None,
        std_offset=None,
        latitude_reference=None,
        longitude_reference=None,
    ):
        self.epoch_timestamp = None
        self.sensor_string = "unknown"

        self.latitude = None
        self.longitude = None
        self.latitude_std = None
        self.longitude_std = None

        self.northings = None
        self.eastings = None
        self.northings_std = None
        self.eastings_std = None

        self.depth = None
        self.depth_std = None

        self.distance_to_ship = None

        self.latitude_ship = None
        self.longitude_ship = None
        self.northings_ship = None
        self.eastings_ship = None
        self.heading_ship = None

        # temporary solution for fk180731 cruise
        # self.epoch_timestamp = None timestamp
        self.northings_ship = None
        self.eastings_ship = None
        # self.northings_target = None northings
        # self.eastings_target = None eastings
        # self.depth = None depth
        self.lateral_distace = None
        self.distance = None
        self.bearing = None
        self.category = Category.USBL
        self.m_declination = None

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
        return (
            self.latitude is not None
            and self.longitude is not None
            and self.eastings is not None
            and self.northings is not None
            and self.epoch_timestamp is not None
        )

    def set_magnetic_declination(self, m_declination):
        self.m_declination = m_declination

    def apply_declination(self):
        n = self.northings * np.cos(self.m_declination) - self.eastings * np.sin(
            self.m_declination
        )
        e = self.northings * np.sin(self.m_declination) + self.eastings * np.cos(
            self.m_declination
        )
        self.northings = n
        self.eastings = e
        # Convert to lat lon from the reference
        self.latitude, self.longitude = metres_to_latlon(
            self.latitude_reference,
            self.longitude_reference,
            self.eastings,
            self.northings,
        )

    def from_ros(self, msg, msg_type, output_dir):
        if msg_type == "geometry_msgs/PoseWithCovarianceStamped":
            self.northings = msg.pose.pose.position.x
            self.eastings = msg.pose.pose.position.y
            self.depth = msg.pose.pose.position.z
            self.northings_std = np.sqrt(msg.pose.covariance[0])
            self.eastings_std = np.sqrt(msg.pose.covariance[7])
            self.depth_std = np.sqrt(msg.pose.covariance[14])

            if self.eastings_std == 0:
                self.eastings_std = self.std_factor * self.depth + self.std_offset
            if self.northings_std == 0:
                self.northings_std = self.std_factor * self.depth + self.std_offset
            if self.depth_std == 0:
                self.depth_std = self.std_factor * self.depth + self.std_offset

            # Convert to lat lon from the reference
            self.latitude, self.longitude = metres_to_latlon(
                self.latitude_reference,
                self.longitude_reference,
                self.eastings,
                self.northings,
            )
            # Transform STD in meters to degrees
            self.latitude_std, self.longitude_std = metres_to_latlon(
                self.latitude_reference,
                self.longitude_reference,
                self.eastings_std,
                self.northings_std,
            )

            if msg.header.frame_id == "sparus2/modem":
                # Handle special case for UdG, message is (lat, lon, -z)
                self.latitude = msg.pose.pose.position.x
                self.longitude = msg.pose.pose.position.y
                self.depth = -msg.pose.pose.position.z
                self.fill_from_lat_lon_depth()

        elif msg_type == "sensor_msgs/NavSatFix":
            if msg.status.status == 0:
                self.latitude = msg.latitude
                self.longitude = msg.longitude
                self.depth = 0.0  # GPS measurement

                self.fill_from_lat_lon_depth()

                self.eastings_std = msg.position_covariance[0]
                self.northings_std = msg.position_covariance[4]
                self.depth_std = msg.position_covariance[8]
        elif msg_type == "evologics_ros_sync/EvologicsUsbllong":
            self.northings = msg.N
            self.eastings = msg.E
            self.depth = msg.D

            # Convert to lat lon from the reference
            self.latitude, self.longitude = metres_to_latlon(
                self.latitude_reference,
                self.longitude_reference,
                self.eastings,
                self.northings,
            )

        else:
            Console.quit("USBL ROS parser for", msg_type, "not supported.")
        self.epoch_timestamp = ros_stamp_to_epoch(msg.header.stamp) - self.tz_offset_s

    def from_eiva_navipac(self, line):
        self.sensor_string = "eiva_navipac"
        parts = line.split()
        date_time_obj = datetime.datetime.strptime(parts[3], "%Y:%m:%d:%H:%M:%S.%f")
        self.epoch_timestamp = date_time_obj.timestamp()
        if self.epoch_timestamp == 0:
            # Invalid timestamp
            self.epoch_timestamp = None
            return
        self.latitude = float(parts[7])
        self.longitude = float(parts[8])
        self.depth = -float(parts[6])
        self.fill_from_lat_lon_depth()

    def fill_from_lat_lon_depth(self):
        # calculate in meters from reference
        lateral_distance, bearing = latlon_to_metres(
            self.latitude,
            self.longitude,
            self.latitude_reference,
            self.longitude_reference,
        )
        self.distance_to_ship = -1.0
        self.eastings = sin(bearing * pi / 180.0) * lateral_distance
        self.northings = cos(bearing * pi / 180.0) * lateral_distance
        self.eastings_std = self.std_factor * self.depth + self.std_offset
        self.northings_std = self.std_factor * self.depth + self.std_offset
        self.depth_std = self.std_factor * self.depth + self.std_offset
        # If your displacements aren't too great (less than a few kilometers)
        # and you're not right at the poles, use the quick and dirty estimate
        # that 111,111 meters (111.111 km) in the y direction is 1 degree (of
        # latitude) and 111,111 * cos(latitude) meters in the x direction is
        # 1 degree (of longitude).
        self.latitude_std = self.eastings_std / 111.111e3
        self.longitude_std = self.northings_std / (
            111.111e3 * cos(self.latitude * pi / 180.0)
        )

    def from_nmea(self, msg):
        self.epoch_timestamp = msg.timestamp
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        self.depth = -msg.altitude

        # calculate in meters from reference
        lateral_distance, bearing = latlon_to_metres(
            self.latitude,
            self.longitude,
            self.latitude_reference,
            self.longitude_reference,
        )
        self.distance_to_ship = -1.0
        self.eastings = sin(bearing * pi / 180.0) * lateral_distance
        self.northings = cos(bearing * pi / 180.0) * lateral_distance
        self.eastings_std = self.std_factor * self.depth + self.std_offset
        self.northings_std = self.std_factor * self.depth + self.std_offset
        self.depth_std = self.std_factor * self.depth + self.std_offset
        # If your displacements aren't too great (less than a few kilometers)
        # and you're not right at the poles, use the quick and dirty estimate
        # that 111,111 meters (111.111 km) in the y direction is 1 degree (of
        # latitude) and 111,111 * cos(latitude) meters in the x direction is
        # 1 degree (of longitude).
        self.latitude_std = self.depth / 111.111e3
        self.longitude_std = self.latitude_std * cos(self.latitude * pi / 180.0)

    def from_json(self, json, sensor_std):
        self.epoch_timestamp = json["epoch_timestamp"]
        self.latitude = json["data_target"][0]["latitude"]
        self.longitude = json["data_target"][1]["longitude"]
        self.northings = json["data_target"][2]["northings"]
        self.eastings = json["data_target"][3]["eastings"]
        self.depth = json["data_target"][4]["depth"]
        self.depth_std = json["data_target"][4]["depth_std"]
        self.distance_to_ship = json["data_target"][5]["distance_to_ship"]

        if sensor_std["model"] == "sensor":
            self.latitude_std = json["data_target"][0]["latitude_std"]
            self.longitude_std = json["data_target"][1]["longitude_std"]
            self.northings_std = json["data_target"][2]["northings_std"]
            self.eastings_std = json["data_target"][3]["eastings_std"]
        elif sensor_std["model"] == "linear":
            self.northings_std = (
                sensor_std["offset"] + sensor_std["factor"] * self.distance_to_ship
            )
            self.eastings_std = (
                sensor_std["offset"] + sensor_std["factor"] * self.distance_to_ship
            )

            # determine range to input to uncertainty model
            distance = sqrt(self.distance_to_ship**2 + self.depth**2)
            distance_std = sensor_std["factor"] * distance + sensor_std["offset"]

            # determine uncertainty in terms of latitude and longitude
            latitude_offset, longitude_offset = metres_to_latlon(
                abs(self.latitude),
                abs(self.longitude),
                distance_std,
                distance_std,
            )
            self.latitude_std = abs(abs(self.latitude) - latitude_offset)
            self.longitude_std = abs(abs(self.longitude) - longitude_offset)
        else:
            Console.error("The STD model you entered for USBL is not supported.")
            Console.quit("STD model not supported.")
        try:
            self.latitude_ship = json["data_ship"][0]["latitude"]
            self.longitude_ship = json["data_ship"][0]["longitude"]
            self.northings_ship = json["data_ship"][1]["northings"]
            self.eastings_ship = json["data_ship"][1]["eastings"]
            self.heading_ship = json["data_ship"][2]["heading"]
        except Exception as exc:
            Console.warn("Please parse again this dataset.", exc)

    def get_csv_header(self):
        return (
            "epoch_timestamp,"
            + "latitude,"
            + "longitude,"
            + "northings,"
            + "eastings,"
            + "depth,"
            + "distance_to_ship,"
            + "latitude_std,"
            + "longitude_std,"
            + "northings_std,"
            + "eastings_std,"
            + "depth_std\n"
        )

    def to_csv_row(self):
        return (
            str(self.epoch_timestamp)
            + ","
            + str(self.latitude)
            + ","
            + str(self.longitude)
            + ","
            + str(self.northings)
            + ","
            + str(self.eastings)
            + ","
            + str(self.depth)
            + ","
            + str(self.distance_to_ship)
            + ","
            + str(self.latitude_std)
            + ","
            + str(self.longitude_std)
            + ","
            + str(self.northings_std)
            + ","
            + str(self.eastings_std)
            + ","
            + str(self.depth_std)
            + "\n"
        )

    def _to_json(self):
        data = {
            "epoch_timestamp": float_or_none(self.epoch_timestamp),
            "class": "measurement",
            "sensor": self.sensor_string,
            "frame": "inertial",
            "category": Category.USBL,
            "data_ship": [
                {
                    "latitude": float_or_none(self.latitude_ship),
                    "longitude": float_or_none(self.longitude_ship),
                },
                {
                    "northings": float_or_none(self.northings_ship),
                    "eastings": float_or_none(self.eastings_ship),
                },
                {"heading": float_or_none(self.heading_ship)},
            ],
            "data_target": [
                {
                    "latitude": float_or_none(self.latitude),
                    "latitude_std": float_or_none(self.latitude_std),
                },
                {
                    "longitude": float_or_none(self.longitude),
                    "longitude_std": float_or_none(self.longitude_std),
                },
                {
                    "northings": float_or_none(self.northings),
                    "northings_std": float_or_none(self.northings_std),
                },
                {
                    "eastings": float_or_none(self.eastings),
                    "eastings_std": float_or_none(self.eastings_std),
                },
                {
                    "depth": float_or_none(self.depth),
                    "depth_std": float_or_none(self.depth_std),
                },
                {"distance_to_ship": float_or_none(self.distance_to_ship)},
            ],
        }
        return data

    def to_acfr(self):
        distance_range = -1.0
        if self.distance_to_ship is not None:
            if self.distance_to_ship > self.depth and self.distance_to_ship > 0:
                try:
                    distance_range = sqrt(self.distance_to_ship**2 - self.depth**2)
                except ValueError:
                    print("Value error:")
                    print("Value distance_to_ship: " + str(self.distance_to_ship))
                    print("Value depth:            " + str(self.depth))
        bearing = atan2(self.eastings, self.northings) * 180 / pi
        data = (
            "SSBL_FIX: "
            + str(float(self.epoch_timestamp))
            + " ship_x: "
            + str(float_or_none(self.northings_ship))
            + " ship_y: "
            + str(float_or_none(self.eastings_ship))
            + " target_x: "
            + str(float(self.northings))
            + " target_y: "
            + str(float(self.eastings))
            + " target_z: "
            + str(float(self.depth))
            + " target_hr: "
            + str(float_or_none(distance_range))
            + " target_sr: "
            + str(float_or_none(self.distance_to_ship))
            + " target_bearing: "
            + str(float_or_none(bearing))
            + "\n"
        )
        return data


class Tide(OutputFormat):
    def __init__(self, height_std_factor=0.0001, ts=None):
        self.epoch_timestamp = None
        self.ts = ts
        self.height_std_factor = height_std_factor
        self.sensor_string = "unknown"
        self.clear()

    def clear(self):
        self.height = None
        self.height_std = None

    def valid(self):
        return self.height is not None and self.epoch_timestamp is not None

    def from_json(self, json, sensor_std):
        self.epoch_timestamp = json["epoch_timestamp"]
        #        self.tide_timestamp = json['epoch_timestamp_tide']
        self.height = json["data"][0]["tide"]

        if sensor_std["model"] == "sensor":
            self.height_std = json["data"][0]["tide_std"]
        elif sensor_std["model"] == "linear":
            self.height_std = sensor_std["offset"] + sensor_std["factor"] * self.height
        else:
            Console.error("The STD model you entered for TIDE is not supported.")
            Console.quit("STD model not supported.")

    def get_csv_header(self):
        return "epoch_timestamp," + "height," + "height_std\n"

    def to_csv_row(self):
        return (
            str(self.epoch_timestamp)
            + ","
            + str(self.height)
            + ","
            + str(self.height_std)
            + "\n"
        )

    def _to_json(self):
        data = {
            "epoch_timestamp": float_or_none(self.epoch_timestamp),
            "class": "measurement",
            "sensor": self.sensor_string,
            "frame": "inertial",
            "category": Category.TIDE,
            "data": [
                {
                    "height": float_or_none(self.height),
                    "height_std": float_or_none(self.height_std),
                }
            ],
        }
        return data


class Other:
    def __init__(self, timestamp=None):
        self.epoch_timestamp = None
        self.data = []

        self.northings = None
        self.eastings = None
        self.depth = None

        self.latitude = None
        self.longitude = None

        self.roll = None
        self.pitch = None
        self.yaw = None

        self.altitude = None
        self.covariance = None

    def from_json(self, json):
        self.epoch_timestamp = json["epoch_timestamp"]
        self.data = json["data"]

    def get_csv_header(self):
        str_to_write = (
            "timestamp,northing [m],easting [m],depth [m],"
            "roll [deg],pitch [deg],heading [deg],altitude "
            "[m],latitude [deg],longitude [deg]"
            ",data\n"
        )
        return str_to_write

    def to_csv_row(self):
        str_to_write = (
            str(self.epoch_timestamp)
            + ","
            + str(self.northings)
            + ","
            + str(self.eastings)
            + ","
            + str(self.depth)
            + ","
            + str(self.roll)
            + ","
            + str(self.pitch)
            + ","
            + str(self.yaw)
            + ","
            + str(self.altitude)
            + ","
            + str(self.latitude)
            + ","
            + str(self.longitude)
            + ","
            + str(self.data)
            + "\n"
        )
        return str_to_write


class SyncedOrientationBodyVelocity(OutputFormat):
    def __init__(self):
        self.epoch_timestamp = None
        # from orientation
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.roll_std = None
        self.pitch_std = None
        self.yaw_std = None
        self.vroll = None
        self.vpitch = None
        self.vyaw = None
        self.vroll_std = None
        self.vpitch_std = None
        self.vyaw_std = None
        # interpolated
        self.x_velocity = None
        self.y_velocity = None
        self.z_velocity = None
        self.x_velocity_std = None
        self.y_velocity_std = None
        self.z_velocity_std = None
        # transformed
        self.north_velocity = None
        self.east_velocity = None
        self.down_velocity = None
        self.north_velocity_std = None
        self.east_velocity_std = None
        self.down_velocity_std = None
        # interpolated
        self.altitude = None
        # calculated
        self.northings = None
        self.eastings = None
        self.northings_std = None
        self.eastings_std = None
        self.depth = None  # from interpolation of depth, not dr
        self.depth_std = None

        self.latitude = None
        self.longitude = None
        self.covariance = None

    def __str__(self):
        msg = "SyncedOrientationBodyVelocity object:\n"
        msg += "\tTimestamp: " + str(self.epoch_timestamp) + "\n"
        msg += (
            "\tPosition: ("
            + str(self.northings)
            + ", "
            + str(self.eastings)
            + ", "
            + str(self.depth)
            + ")\n"
        )
        msg += (
            "\tPos. std: ("
            + str(self.northings_std)
            + ", "
            + str(self.eastings_std)
            + ", "
            + str(self.depth_std)
            + ")\n"
        )
        msg += (
            "\tOrientation: ("
            + str(self.roll)
            + ", "
            + str(self.pitch)
            + ", "
            + str(self.yaw)
            + ")\n"
        )
        msg += (
            "\tOrient. std: ("
            + str(self.roll_std)
            + ", "
            + str(self.pitch_std)
            + ", "
            + str(self.yaw_std)
            + ")\n"
        )
        msg += (
            "\tSpeeds: ("
            + str(self.x_velocity)
            + ", "
            + str(self.y_velocity)
            + ", "
            + str(self.z_velocity)
            + ")\n"
        )
        msg += (
            "\tS. std: ("
            + str(self.x_velocity_std)
            + ", "
            + str(self.y_velocity_std)
            + ", "
            + str(self.z_velocity_std)
            + ")"
        )
        return msg

    def __lt__(self, o):
        return self.epoch_timestamp < o.epoch_timestamp

    def northing_std_from_cov(self) -> float:
        return sqrt(max(0, self.covariance[0, 0]))

    def easting_std_from_cov(self) -> float:
        return sqrt(max(0, self.covariance[1, 1]))

    def depth_std_from_cov(self) -> float:
        return sqrt(max(0, self.covariance[2, 2]))

    def roll_std_from_cov_deg(self) -> float:
        return 180 / pi * sqrt(max(0, self.covariance[3, 3]))

    def pitch_std_from_cov_deg(self) -> float:
        return 180 / pi * sqrt(max(0, self.covariance[4, 4]))

    def yaw_std_from_cov_deg(self) -> float:
        return 180 / pi * sqrt(max(0, self.covariance[5, 5]))

    def from_df(self, df_row):
        self.epoch_timestamp = df_row["timestamp"]
        self.northings = df_row["northing [m]"]
        self.eastings = df_row["easting [m]"]
        self.depth = df_row["depth [m]"]
        self.roll = df_row["roll [deg]"]
        self.pitch = df_row["pitch [deg]"]
        self.yaw = df_row["heading [deg]"]
        self.altitude = df_row["altitude [m]"]
        self.latitude = df_row["latitude [deg]"]
        self.longitude = df_row["longitude [deg]"]
        self.northings_std = df_row["vehicle_std_x [m]"]
        self.eastings_std = df_row["vehicle_std_y [m]"]
        self.depth_std = df_row["vehicle_std_z [m]"]
        self.roll_std = df_row["vehicle_std_roll [deg]"]
        self.pitch_std = df_row["vehicle_std_pitch [deg]"]
        self.yaw_std = df_row["vehicle_std_yaw [deg]"]
        self.x_velocity_std = df_row["vehicle_std_vx [m/s]"]
        self.y_velocity_std = df_row["vehicle_std_vy [m/s]"]
        self.z_velocity_std = df_row["vehicle_std_vz [m/s]"]
        self.vroll_std = df_row["vehicle_std_vroll [deg/s]"]
        self.vpitch_std = df_row["vehicle_std_vpitch [deg/s]"]
        self.vyaw_std = df_row["vehicle_std_vyaw [deg/s]"]

    def get_csv_header(self):
        str_to_write = (
            "timestamp,northing [m],easting [m],depth [m],"
            "roll [deg],pitch [deg],heading [deg],altitude "
            "[m],latitude [deg],longitude [deg]"
            ",vehicle_std_x [m],vehicle_std_y [m],vehicle_std_z [m],"
            "vehicle_std_roll [deg],vehicle_std_pitch [deg],"
            "vehicle_std_yaw [deg],vehicle_std_vx [m/s],vehicle_std_vy [m/s],"
            "vehicle_std_vz [m/s],vehicle_std_vroll [deg/s],"
            "vehicle_std_vpitch [deg/s],vehicle_std_vyaw [deg/s]\n"
        )
        return str_to_write

    def get_sidescan_header(self):
        str_to_write = "#Mission Date Time NorthDeg EastDeg HeadingDeg \
            RollDeg PitchDeg Altitude Depth Speed\n"
        return str_to_write

    def to_sidescan_row(self):
        datetime_str = time.strftime(
            "%Y%m%d %H%M%S",
            time.gmtime(self.epoch_timestamp),
        )
        lat = self.latitude if self.latitude is not None else 0.0
        lon = self.longitude if self.latitude is not None else 0.0
        str_to_write = (
            "M150 "
            + datetime_str
            + " "
            + "{:.6f}".format(lat)
            + " "
            + "{:.6f}".format(lon)
            + " "
            + "{:.3f}".format(self.yaw)
            + " "
            + "{:.3f}".format(self.roll)
            + " "
            + "{:.3f}".format(self.pitch)
            + " "
            + "{:.3f}".format(self.altitude)
            + " "
            + "{:.3f}".format(self.depth)
            + " "
            + "{:.3f}".format(self.x_velocity)
            + "\n"
        )
        return str_to_write

    def to_csv_row(self):
        str_to_write = (
            str(self.epoch_timestamp)
            + ","
            + str(self.northings)
            + ","
            + str(self.eastings)
            + ","
            + str(self.depth)
            + ","
            + str(self.roll)
            + ","
            + str(self.pitch)
            + ","
            + str(self.yaw)
            + ","
            + str(self.altitude)
            + ","
            + str(self.latitude)
            + ","
            + str(self.longitude)
            + ","
            + str(self.northings_std)
            + ","
            + str(self.eastings_std)
            + ","
            + str(self.depth_std)
            + ","
            + str(self.roll_std)
            + ","
            + str(self.pitch_std)
            + ","
            + str(self.yaw_std)
            + ","
            + str(self.x_velocity_std)
            + ","
            + str(self.y_velocity_std)
            + ","
            + str(self.z_velocity_std)
            + ","
            + str(self.vroll_std)
            + ","
            + str(self.vpitch_std)
            + ","
            + str(self.vyaw_std)
            + "\n"
        )
        return str_to_write


class Camera(SyncedOrientationBodyVelocity):
    def __init__(self):
        super().__init__()
        self.filename = ""
        self.information = None
        self.updated = False

    def clear(self):
        self.filename = ""
        self.epoch_timestamp = None

    def valid(self):
        return self.filename != "" and self.epoch_timestamp is not None

    def fromSyncedBodyVelocity(
        self, other, origin_offsets, sensor_offsets, latlon_reference
    ):
        [x_offset, y_offset, z_offset] = body_to_inertial(
            other.roll,
            other.pitch,
            other.yaw,
            origin_offsets[0] - sensor_offsets[0],
            origin_offsets[1] - sensor_offsets[1],
            origin_offsets[2] - sensor_offsets[2],
        )
        self.sensor_string = "unknown"

        self.epoch_timestamp = other.epoch_timestamp
        self.roll = other.roll
        self.pitch = other.pitch
        self.yaw = other.yaw
        self.roll_std = other.roll_std
        self.pitch_std = other.pitch_std
        self.yaw_std = other.yaw_std
        self.vroll = other.vroll
        self.vpitch = other.vpitch
        self.vyaw = other.vyaw
        self.vroll_std = other.vroll_std
        self.vpitch_std = other.vpitch_std
        self.vyaw_std = other.vyaw_std
        self.x_velocity = other.x_velocity
        self.y_velocity = other.y_velocity
        self.z_velocity = other.z_velocity
        self.x_velocity_std = other.x_velocity_std
        self.y_velocity_std = other.y_velocity_std
        self.z_velocity_std = other.z_velocity_std
        self.north_velocity = other.north_velocity
        self.east_velocity = other.east_velocity
        self.down_velocity = other.down_velocity
        self.north_velocity_std = other.north_velocity_std
        self.east_velocity_std = other.east_velocity_std
        self.down_velocity_std = other.down_velocity_std
        self.altitude = other.altitude + z_offset
        self.northings = other.northings - x_offset
        self.eastings = other.eastings - y_offset
        self.northings_std = other.northings_std
        self.eastings_std = other.eastings_std
        self.depth = other.depth - z_offset
        self.depth_std = other.depth_std
        self.covariance = other.covariance

        latitude_reference, longitude_reference = latlon_reference

        [self.latitude, self.longitude] = metres_to_latlon(
            latitude_reference,
            longitude_reference,
            other.eastings,
            other.northings,
        )

        self.updated = True

    def get_info(self):
        try:
            self.information = np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError as err:
            Console.error(
                "Failed to invert covariance matrix: " + " Error: " + str(err)
            )

    def from_json(self, json, cam_name):
        if cam_name in json:
            self.epoch_timestamp = json[cam_name][0]["epoch_timestamp"]
            self.filename = json[cam_name][0]["filename"]
        elif "filename" in json:
            self.epoch_timestamp = json["epoch_timestamp"]
            self.filename = json["filename"]

    def to_acfr(self):
        data = (
            "VIS: "
            + str(self.epoch_timestamp)
            + " ["
            + str(self.epoch_timestamp)
            + "] "
            + str(self.filename)
            + " exp: 0\n"
        )
        return data

    def get_csv_header(self):
        return (
            "relative_path,northing [m],easting [m],depth [m],"
            "roll [deg],pitch [deg],heading [deg],altitude "
            "[m],timestamp [s],latitude [deg],longitude [deg]"
            ",x_velocity [m/s],y_velocity [m/s],z_velocity [m/s]"
            ",vehicle_std_x [m],vehicle_std_y [m],vehicle_std_z [m],"
            "vehicle_std_roll [deg],vehicle_std_pitch [deg],"
            "vehicle_std_yaw [deg],vehicle_std_vx [m/s],vehicle_std_vy [m/s],"
            "vehicle_std_vz [m/s],vehicle_std_vroll [deg/s],"
            "vehicle_std_vpitch [deg/s],vehicle_std_vyaw [deg/s]\n"
        )

    def get_csv_header_cov(self):
        str_to_write_cov = "relative_path"
        cov = ["x", "y", "z", "roll", "pitch", "yaw"]
        for a in cov:
            for b in cov:
                str_to_write_cov += ", cov_" + a + "_" + b
        str_to_write_cov += "\n"
        return str_to_write_cov

    def from_df(self, df_row):
        self.filename = df_row["relative_path"]
        self.northings = df_row["northing [m]"]
        self.eastings = df_row["easting [m]"]
        self.depth = df_row["depth [m]"]
        self.roll = df_row["roll [deg]"]
        self.pitch = df_row["pitch [deg]"]
        self.yaw = df_row["heading [deg]"]
        self.altitude = df_row["altitude [m]"]
        self.epoch_timestamp = df_row["timestamp [s]"]
        self.latitude = df_row["latitude [deg]"]
        self.longitude = df_row["longitude [deg]"]
        self.x_velocity = df_row["x_velocity [m/s]"]
        self.y_velocity = df_row["y_velocity [m/s]"]
        self.z_velocity = df_row["z_velocity [m/s]"]
        self.northings_std = df_row["vehicle_std_x [m]"]
        self.eastings_std = df_row["vehicle_std_y [m]"]
        self.depth_std = df_row["vehicle_std_z [m]"]
        self.roll_std = df_row["vehicle_std_roll [deg]"]
        self.pitch_std = df_row["vehicle_std_pitch [deg]"]
        self.yaw_std = df_row["vehicle_std_yaw [deg]"]
        self.x_velocity_std = df_row["vehicle_std_vx [m/s]"]
        self.y_velocity_std = df_row["vehicle_std_vy [m/s]"]
        self.z_velocity_std = df_row["vehicle_std_vz [m/s]"]
        self.vroll_std = df_row["vehicle_std_vroll [deg/s]"]
        self.vpitch_std = df_row["vehicle_std_vpitch [deg/s]"]
        self.vyaw_std = df_row["vehicle_std_vyaw [deg/s]"]

    def to_csv_row(self):
        str_to_write = (
            str(self.filename)
            + ","
            + str(self.northings)
            + ","
            + str(self.eastings)
            + ","
            + str(self.depth)
            + ","
            + str(self.roll)
            + ","
            + str(self.pitch)
            + ","
            + str(self.yaw)
            + ","
            + str(self.altitude)
            + ","
            + str(self.epoch_timestamp)
            + ","
            + str(self.latitude)
            + ","
            + str(self.longitude)
            + ","
            + str(self.x_velocity)
            + ","
            + str(self.y_velocity)
            + ","
            + str(self.z_velocity)
            + ","
            + str(self.northings_std)
            + ","
            + str(self.eastings_std)
            + ","
            + str(self.depth_std)
            + ","
            + str(self.roll_std)
            + ","
            + str(self.pitch_std)
            + ","
            + str(self.yaw_std)
            + ","
            + str(self.x_velocity_std)
            + ","
            + str(self.y_velocity_std)
            + ","
            + str(self.z_velocity_std)
            + ","
            + str(self.vroll_std)
            + ","
            + str(self.vpitch_std)
            + ","
            + str(self.vyaw_std)
            + "\n"
        )
        return str_to_write

    def to_csv_cov_row(self):
        if self.covariance is not None:
            str_to_write_cov = str(self.filename)
            for k1 in range(6):
                for k2 in range(6):
                    c = self.covariance[k1, k2]
                    str_to_write_cov += "," + str(c)
            str_to_write_cov += "\n"
            return str_to_write_cov
        else:
            return ""

    def _to_json(self):
        data = {
            "epoch_timestamp": float(self.epoch_timestamp),
            "class": "measurement",
            "sensor": self.sensor_string,
            "frame": "body",
            "category": "image",
            "camera1": [
                {
                    "epoch_timestamp": float(self.epoch_timestamp),
                    "filename": self.filename,
                }
            ],
        }
        return data

    def from_ros(self, msg, msg_type, output_dir):
        """Parse ROS image topic"""

        # Check image message type
        """
        bridge = CvBridge()
        cv_img = None
        if msg_type == "sensor_msgs/CompressedImage":
            cv_img = bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        elif msg_type == "sensor_msgs/Image":
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        else:
            Console.quit("auv_nav does not support ROS image topic type", msg_type)
        """
        self.epoch_timestamp = ros_stamp_to_epoch(msg.header.stamp) - self.tz_offset_s
        stamp = str(self.epoch_timestamp)
        output_dir = Path(output_dir) / ("image/raw/" + self.sensor_string)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        self.filename = str(output_dir / (stamp + ".png"))

        # cv2.imwrite(self.filename, cv_img)


class StereoCamera:
    def __init__(self):
        self.left = Camera()
        self.right = Camera()
        self.sensor_string = "unknown"

    def _to_json(self):
        data = {
            "epoch_timestamp": float(self.left.epoch_timestamp),
            "class": "measurement",
            "sensor": self.sensor_string,
            "frame": "body",
            "category": "image",
            "camera1": [
                {
                    "epoch_timestamp": float(self.left.epoch_timestamp),
                    "filename": self.left.filename,
                }
            ],
            "camera2": [
                {
                    "epoch_timestamp": float(self.right.epoch_timestamp),
                    "filename": self.right.filename,
                }
            ],
        }
        return data

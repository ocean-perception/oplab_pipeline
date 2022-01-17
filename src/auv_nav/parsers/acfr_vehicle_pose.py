# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import math
from pathlib import Path

from auv_nav import SyncedOrientationBodyVelocity
from oplab import Console

header = """
% VEHICLE_POSE_FILE VERSION 3
%
% Produced by oplab_pipeline
%
% SLAM statistics:
%    Number of augmented poses: 0
%    State vector size        : 0
% Loop closure statistics:
%    Number of hypotheses   : 0
%    Number of loop closures: 0
%
% Each line of this file describes the pose of the vehicle relative to the local
% navigation frame. The vehicle poses may have been estimated since they are the
% locations at which stereo images or multibeam sonar data were acquired.
%
% If a pose was estimated because it was the location images were acquired,
% additional information for that pose can be found in the file
% stereo_pose_est.data. The pose identifier can be used to locate matching
% poses.
%
% The X and Y coordinates are produced using a local transverse Mercator
% projection using the WGS84 ellipsoid and a central meridian at the origin
% latitude. You will probably want to use the provided latitude and longitude to
% produce coordinates in what map projection you require.
%
% The first two lines of the data contain the latitude and longitude of the
% origin.
%
% Each line contains the following items describing the pose of the vehicle:
%
% 1) Pose identifier                   - integer value
% 2) Timestamp                         - in seconds
% 3) Latitude                          - in degrees
% 4) Longitude                         - in degrees
% 5) X position (Northing)             - in meters, relative to local nav frame
% 6) Y position (Easting)              - in meters, relative to local nav frame
% 7) Z position (Depth)                - in meters, relative to local nav frame
% 8) X-axis Euler angle (Roll)         - in radians, relative to local nav frame
% 9) Y-axis Euler angle (Pitch)        - in radians, relative to local nav frame
% 10) Z-axis Euler angle (Yaw/Heading) - in radians, relative to local nav frame
% 11) Altitude                         - in meters. (0 when unknown)
%
"""


class AcfrVehiclePose:
    """ACFR Vehicle pose class"""

    def __init__(self, line=None):
        self.id = None
        self.stamp = None
        self.latitude = None
        self.longitude = None
        self.x_north = None
        self.y_east = None
        self.z_depth = None
        self.x_euler_angle = None
        self.y_euler_angle = None
        self.z_euler_angle = None
        self.altitude = None

        if line is not None:
            self.parse(line)

    def parse(self, line):
        """Parses a line of the ACFR stereo pose data file

        Parameters
        ----------
        line : a string that contains a line of the document
            The string should contain 11 items separated by spaces.
            According to ACFR format, the items should be:
            1) Pose identifier      - integer value
            2) Timestamp            - in seconds
            3) Latitude             - in degrees
            4) Longitude            - in degrees
            5) X position (North)   - in meters, relative to local nav frame
            6) Y position (East)    - in meters, relative to local nav frame
            7) Z position (Depth)   - in meters, relative to local nav frame
            8) X-axis Euler angle   - in radians, relative to local nav frame
            9) Y-axis Euler angle   - in radians, relative to local nav frame
            10) Z-axis Euler angle  - in radians, relative to local nav frame
            11) Vehicle altitude    - in meters
        """
        parts = line.split()
        if len(parts) != 11:
            Console.error("The line passed to ACFR stereo pose parser is malformed.")
        self.id = int(parts[0])
        self.stamp = float(parts[1])
        self.latitude = float(parts[2])
        self.longitude = float(parts[3])
        self.x_north = float(parts[4])
        self.y_east = float(parts[5])
        self.z_depth = float(parts[6])
        self.x_euler_angle = math.degrees(float(parts[7]))
        self.y_euler_angle = math.degrees(float(parts[8]))
        self.z_euler_angle = math.degrees(float(parts[9]))
        self.altitude = float(parts[10])

    def __repr__(self):
        msg = self.__str__()
        return "AcfrStereoPose with " + msg

    def __str__(self):
        msg = [
            "id: ",
            self.id,
            ", stamp: ",
            self.stamp,
            ", latitude: ",
            self.latitude,
            "longitude: ",
            self.longitude,
            ", x_north: ",
            self.x_north,
            ", y_east: ",
            self.y_east,
            ", z_depth: ",
            self.z_depth,
            ", x_euler_angle: ",
            self.x_euler_angle,
            "y_euler_angle: ",
            self.y_euler_angle,
            ", z_euler_angle: ",
            self.z_euler_angle,
            ", altitude: ",
            self.altitude,
        ]
        return "".join(str(e) for e in msg)


class AcfrVehiclePoseParser:
    """Parse an ACFR stereo pose file"""

    def __init__(self, filename=None):
        self._entries = []
        self.origin_latitude = None
        self.origin_longitude = None

        if filename is not None:
            self.parse(filename)

    def parse(self, filename):
        f = Path(filename)
        stream = f.open("r")

        for i, line in enumerate(stream):
            # Read origins
            # Line 56: ORIGIN_LATITUDE  59.8136000000000010
            # Line 57: ORIGIN_LONGITUDE -7.3532999999999999
            if i == 44:
                self.origin_latitude = float(line.split()[1])
            if i == 45:
                self.origin_longitude = float(line.split()[1])
            if i > 45:
                self._entries.append(AcfrVehiclePose(line))

    def __call__(self, index):
        return self._entries[index]

    def get_dead_reckoning(self):
        """Converts the parsed ACFR stereo file to a DR list for auv_nav"""
        dr_list = []
        for entry in self._entries:
            dr = SyncedOrientationBodyVelocity()
            dr.epoch_timestamp = entry.stamp
            dr.northings = entry.x_north
            dr.eastings = entry.y_east
            dr.depth = entry.z_depth
            dr.latitude = entry.latitude
            dr.longitude = entry.longitude
            dr.roll = entry.y_euler_angle
            dr.pitch = -entry.x_euler_angle
            dr.yaw = entry.z_euler_angle
            dr.altitude = entry.altitude

            dr.x_velocity = 0
            dr.y_velocity = 0
            dr.z_velocity = 0

        return dr_list


class AcfrVehiclePoseWriter:
    def __init__(self, filename, origin_latitude, origin_longitude):
        self.filename = Path(filename)
        self.origin_latitude = origin_latitude
        self.origin_longitude = origin_longitude

    def write(self, dr_list):
        """Writes a DR list to an ACFR vehicle pose file"""
        if not self.filename.exists():
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            self.filename.touch()
        with self.filename.open("w") as f:
            f.write(header)
            f.write("ORIGIN_LATITUDE  " + str(self.origin_latitude) + "\n")
            f.write("ORIGIN_LONGITUDE " + str(self.origin_longitude) + "\n")
            for i, dr in enumerate(dr_list):
                f.write(
                    str(i)
                    + " \t"
                    + str(dr.epoch_timestamp)
                    + " \t"
                    + str(dr.latitude)
                    + " \t"
                    + str(dr.longitude)
                    + " \t"
                    + str(dr.northings)
                    + " \t"
                    + str(dr.eastings)
                    + " \t"
                    + str(dr.depth)
                    + " \t"
                    + str(dr.roll)
                    + " \t"
                    + str(dr.pitch)
                    + " \t"
                    + str(dr.yaw)
                    + " \t"
                    + str(dr.altitude)
                    + "\n"
                )

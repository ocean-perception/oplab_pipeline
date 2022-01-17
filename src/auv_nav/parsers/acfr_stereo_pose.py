# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import math
from pathlib import Path

from auv_nav import Camera
from oplab import Console

header = """
% STEREO_POSE_FILE VERSION 2
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
% Each line of this file describes the pose of the stereo-vision system relative
% to the local navigation frame at the time a pair of stereo images were
% acquired. The reference frame of the stereo-vision system is defined to be
% coincident with the left camera.
%
% The X and Y coordinates are produced using a local transverse Mercator
% projection using the WGS84 ellipsoid and a central meridian at the origin
% latitude. You will probably want to use the provided latitude and longitude to
% produce coordinates in what map projection you require.
%
% The first two lines of the data contain the latitude and longitude of the
% origin.
%
% Each line contains the following items describing the pose of the stereo rig:
%
% 1) Pose identifier                   - integer value
% 2) Timestamp                         - in seconds
% 3) Latitude                          - in degrees
% 4) Longitude                         - in degrees
% 5) X position (North)                - in meters, relative to local nav frame
% 6) Y position (East)                 - in meters, relative to local nav frame
% 7) Z position (Depth)                - in meters, relative to local nav frame
% 8) X-axis Euler angle                - in radians, relative to local nav frame
% 9) Y-axis Euler angle                - in radians, relative to local nav frame
% 10) Z-axis Euler angle               - in radians, relative to local nav frame
% 11) Left image name
% 12) Right image name
% 13) Vehicle altitude                   - in meters
% 14) Approx. bounding image radius      - in meters
% 15) Likely trajectory cross-over point - 1 for true, 0 for false
%
% Data items 14 and 15 are used within our 3D mesh building software, and can
% safely be ignored in other applications.
%
% Note: The Euler angles correspond to the orientation of the stereo-rig, and
% do not correspond to the roll, pitch and heading of the vehicle. The stereo-
% frame is defined such that the positive Z-axis is along the principal ray of
% the camera (in the direction the camera is pointed), and the X and Y axes are
% aligned with the image axes. The positive X axis is pointing towards the
% right of the image, while the positive Y axis points to the bottom of the
% image. The Euler angles specify the sequence of rotations in XYZ order, that
% align the navigation frame axes (North, East, Down) with the stereo frame.
%
"""


class AcfrStereoPose:
    """ACFR Stereo pose class"""

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
        self.left_image_name = None
        self.right_image_name = None
        self.altitude = None
        self.bounding_image_radius = None
        self.crossover_likelihood = None

        if line is not None:
            self.parse(line)

    def parse(self, line):
        """Parses a line of the ACFR stereo pose data file

        Parameters
        ----------
        line : a string that contains a line of the document
            The string should contain 15 items separated by spaces. According to ACFR format, # noqa
            the items should be:
            1) Pose identifier                   - integer value
            2) Timestamp                         - in seconds
            3) Latitude                          - in degrees
            4) Longitude                         - in degrees
            5) X position (North)                - in meters, relative to local nav frame # noqa
            6) Y position (East)                 - in meters, relative to local nav frame # noqa
            7) Z position (Depth)                - in meters, relative to local nav frame # noqa
            8) X-axis Euler angle                - in radians, relative to local nav frame # noqa
            9) Y-axis Euler angle                - in radians, relative to local nav frame # noqa
            10) Z-axis Euler angle               - in radians, relative to local nav frame # noqa
            11) Left image name
            12) Right image name
            13) Vehicle altitude                   - in meters
            14) Approx. bounding image radius      - in meters
            15) Likely trajectory cross-over point - 1 for true, 0 for false
        """
        parts = line.split()
        if len(parts) != 15:
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
        self.left_image_name = str(parts[10])
        self.right_image_name = str(parts[11])
        self.altitude = float(parts[12])
        self.bounding_image_radius = float(parts[13])
        self.crossover_likelihood = int(parts[14])

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
            "left_image_name: ",
            self.left_image_name,
            ", right_image_name: ",
            self.right_image_name,
            ", altitude: ",
            self.altitude,
        ]
        return "".join(str(e) for e in msg)


class AcfrStereoPoseParser:
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
            if i == 55:
                self.origin_latitude = float(line.split()[1])
            if i == 56:
                self.origin_longitude = float(line.split()[1])
            if i > 56:
                self._entries.append(AcfrStereoPose(line))

    def __call__(self, index):
        return self._entries[index]

    def get_cameras(self):
        """Converts the parsed ACFR stereo file to a Camera list for auv_nav"""
        camera1_list = []
        camera2_list = []
        for entry in self._entries:
            c1 = Camera()
            c1.epoch_timestamp = entry.stamp
            c1.filename = entry.left_image_name
            c1.northings = entry.x_north
            c1.eastings = entry.y_east
            c1.depth = entry.z_depth
            c1.latitude = entry.latitude
            c1.longitude = entry.longitude
            """
            Note: The Euler angles correspond to the orientation of the stereo-
            rig, and do not correspond to the roll, pitch and heading of the
            vehicle. The stereo-frame is defined such that the positive Z-axis
            is along the principal ray of the camera (in the direction the
            camera is pointed), and the X and Y axes are aligned with the image
            axes. The positive X axis is pointing towards the right of the
            image, while the positive Y axis points to the bottom of the image.
            The Euler angles specify the sequence of rotations in XYZ order,
            that align the navigation frame axes (North, East, Down) with the
            stereo frame.
            """

            # Top-down stereo facing backwards, as in SeaXerocks3 on AE2000
            c1.roll = entry.y_euler_angle
            c1.pitch = -entry.x_euler_angle
            c1.yaw = entry.z_euler_angle + 90

            c1.x_velocity = 0
            c1.y_velocity = 0
            c1.z_velocity = 0
            c1.altitude = entry.altitude

            c2 = Camera()
            c2.epoch_timestamp = c1.epoch_timestamp
            c2.filename = entry.right_image_name
            c2.northings = c1.northings
            c2.eastings = c1.eastings
            c2.depth = c1.depth
            c2.latitude = c1.latitude
            c2.longitude = c1.longitude
            c2.roll = c1.roll
            c2.pitch = c1.pitch
            c2.yaw = c1.yaw
            c2.x_velocity = c1.x_velocity
            c2.y_velocity = c1.y_velocity
            c2.z_velocity = c1.z_velocity
            c2.altitude = c1.altitude

            camera1_list.append(c1)
            camera2_list.append(c2)

        return camera1_list, camera2_list


class AcfrStereoPoseWriter:
    def __init__(self, filename, origin_latitude, origin_longitude):
        self.filename = filename
        self.origin_latitude = origin_latitude
        self.origin_longitude = origin_longitude

    def write(self, cam1_list, cam2_list):
        """Writes a DR list to an ACFR vehicle pose file"""
        with open(self.filename, "w") as f:
            f.write(header)
            f.write("ORIGIN_LATITUDE  " + str(self.origin_latitude) + "\n")
            f.write("ORIGIN_LONGITUDE " + str(self.origin_longitude) + "\n")
            for i, (c1, c2) in enumerate(zip(cam1_list, cam2_list)):
                c1.filename = Path(c1.filename).name
                c2.filename = Path(c2.filename).name
                f.write(
                    str(i)
                    + " \t"
                    + str(c1.epoch_timestamp)
                    + " \t"
                    + str(c1.latitude)
                    + " \t"
                    + str(c1.longitude)
                    + " \t"
                    + str(c1.northings)
                    + " \t"
                    + str(c1.eastings)
                    + " \t"
                    + str(c1.depth)
                    + " \t"
                    + str(c1.roll)
                    + " \t"
                    + str(c1.pitch)
                    + " \t"
                    + str(c1.yaw)
                    + " \t"
                    + str(c1.filename)
                    + " \t"
                    + str(c2.filename)
                    + " \t"
                    + str(c1.altitude)
                    + " \t 0 \t 0\n"
                )

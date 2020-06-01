# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import unittest
import tempfile
import os
import math
from pathlib import Path
from auv_nav.parsers.parse_phins import PhinsTimestamp
from auv_nav.parsers.parse_acfr_stereo_pose import AcfrStereoPoseFile


class TestPhinsTimestamp(unittest.TestCase):
    def setUp(self):
        date = [2018, 11, 19]
        timezone = 10.0  # in hours
        timeoffset = 1.0  # in seconds
        self.t = PhinsTimestamp(date, timezone, timeoffset)

    def test_epoch_from_offset(self):
        hour = 8
        mins = 30
        secs = 15
        msec = 23
        epoch = self.t.get(hour, mins, secs, msec)
        self.assertEqual(epoch, 1542580216.023, "Time conversion is wrong")

    def test_timestamp_from_phins(self):
        line = ["TIME__", " ", "083015.023"]
        epoch = self.t.epoch_timestamp_from_phins(line)
        self.assertEqual(epoch, 1542580216.023, "Time conversion is wrong")


class TestAcfrStereoPose(unittest.TestCase):
    def setUp(self):
        data = (
            "\n% STEREO_POSE_FILE VERSION 2 \n"
            "% \n"
            "% Produced by seabed_slam\n"
            "% \n"
            "% SLAM statistics: \n"
            "%    Number of augmented poses: 22392\n"
            "%    State vector size        : 268716\n"
            "% Loop closure statistics: \n"
            "%    Number of hypotheses   : 0\n"
            "%    Number of loop closures: 5082\n"
            "% \n"
            "% Each line of this file describes the pose of the stereo-vision system relative\n"
            "% to the local navigation frame at the time a pair of stereo images were\n"
            "% acquired. The reference frame of the stereo-vision system is defined to be\n"
            "% coincident with the left camera.\n"
            "% \n"
            "% The X and Y coordinates are produced using a local transverse Mercator \n"
            "% projection using the WGS84 ellipsoid and a central meridian at the origin\n"
            "% latitude. You will probably want to use the provided latitude and longitude to\n"
            "% produce coordinates in what map projection you require.\n"
            "% \n"
            "% The first two lines of the data contain the latitude and longitude of the\n"
            "% origin.\n"
            "% \n"
            "% Each line contains the following items describing the pose of the stereo rig:\n"
            "% \n"
            "% 1) Pose identifier                   - integer value\n"
            "% 2) Timestamp                         - in seconds\n"
            "% 3) Latitude                          - in degrees\n"
            "% 4) Longitude                         - in degrees\n"
            "% 5) X position (North)                - in meters, relative to local nav frame\n"
            "% 6) Y position (East)                 - in meters, relative to local nav frame\n"
            "% 7) Z position (Depth)                - in meters, relative to local nav frame\n"
            "% 8) X-axis Euler angle                - in radians, relative to local nav frame\n"
            "% 9) Y-axis Euler angle                - in radians, relative to local nav frame\n"
            "% 10) Z-axis Euler angle               - in radians, relative to local nav frame\n"
            "% 11) Left image name\n"
            "% 12) Right image name\n"
            "% 13) Vehicle altitude                   - in meters\n"
            "% 14) Approx. bounding image radius      - in meters\n"
            "% 15) Likely trajectory cross-over point - 1 for true, 0 for false\n"
            "% \n"
            "% Data items 14 and 15 are used within our 3D mesh building software, and can\n"
            "% safely be ignored in other applications.\n"
            "% \n"
            "% Note: The Euler angles correspond to the orientation of the stereo-rig, and\n"
            "% do not correspond to the roll, pitch and heading of the vehicle. The stereo-\n"
            "% frame is defined such that the positive Z-axis is along the principal ray of\n"
            "% the camera (in the direction the camera is pointed), and the X and Y axes are\n"
            "% aligned with the image axes. The positive X axis is pointing towards the\n"
            "% right of the image, while the positive Y axis points to the bottom of the\n"
            "% image. The Euler angles specify the sequence of rotations in XYZ order, that \n"
            "% align the navigation frame axes (North, East, Down) with the stereo frame.\n"
            "% \n"
            "ORIGIN_LATITUDE  59.8136000000000010\n"
            "ORIGIN_LONGITUDE -7.3532999999999999\n"
            "1141 	1568624706.9503738880157471 	59.8140319507500209 	"
            "-7.3547900343418879 	48.1042130295795403 	-83.5766631892393974"
            " 	962.3138844564567762 	-0.0116629178991342 	0.0592197626619286"
            " 	-0.0286128901495461 	PCO_190916090506950374_FC.png 	"
            "PCO_190916090506948457_AC.png 	4.3129998664855957 	3.2025993712939327 	1\n"
            "1142 	1568624709.9565351009368896 	59.8140319331347854 	"
            "-7.3548417967423685 	48.1023177169713634 	-86.4800384045786785"
            " 	962.2779424463255964 	-0.0106603074717418 	0.0516333356444368"
            " 	-0.0255026417779395 	PCO_190916090509954416_FC.png 	"
            "PCO_190916090509956535_AC.png 	4.2704998474121094 	3.1732652035609017 	1\n"
            "1143 	1568624712.9632830619812012 	59.8140313132289805 	"
            "-7.3548932724292806 	48.0333504624794756 	-89.3673333486288186"
            " 	962.3690214898499562 	-0.0069141654804990 	0.0493079909303522"
            " 	-0.0038653766339864 	PCO_190916090512963283_FC.png 	"
            "PCO_190916090512963081_AC.png 	4.2279998283386231 	3.1439310358278716 	1"
        )
        fd, filename = tempfile.mkstemp(suffix="test_acfr_stereo_pose")
        # Close opened file
        os.close(fd)

        p = Path(filename)
        p.open("w").write(data)
        self.filepath = filename

    def test_acfr_stereo_pose(self):
        s = AcfrStereoPoseFile(self.filepath)
        self.assertAlmostEqual(s.origin_latitude, 59.8136000000000010)
        self.assertAlmostEqual(s.origin_longitude, -7.3532999999999999)
        self.assertEqual(s._entries[0].id, 1141)
        self.assertAlmostEqual(s._entries[0].stamp, 1568624706.9503738880157471)
        self.assertAlmostEqual(s._entries[0].latitude, 59.8140319507500209)
        self.assertAlmostEqual(s._entries[0].longitude, -7.3547900343418879)
        self.assertAlmostEqual(s._entries[0].x_north, 48.1042130295795403)
        self.assertAlmostEqual(s._entries[0].y_east, -83.5766631892393974)
        self.assertAlmostEqual(s._entries[0].z_depth, 962.3138844564567762)
        self.assertAlmostEqual(
            s._entries[0].x_euler_angle, math.degrees(-0.0116629178991342)
        )
        self.assertAlmostEqual(
            s._entries[0].y_euler_angle, math.degrees(0.0592197626619286)
        )
        self.assertAlmostEqual(
            s._entries[0].z_euler_angle, math.degrees(-0.0286128901495461)
        )
        self.assertEqual(s._entries[0].left_image_name, "PCO_190916090506950374_FC.png")
        self.assertEqual(
            s._entries[0].right_image_name, "PCO_190916090506948457_AC.png"
        )
        self.assertAlmostEqual(s._entries[0].altitude, 4.3129998664855957)
        self.assertAlmostEqual(s._entries[0].bounding_image_radius, 3.2025993712939327)
        self.assertEqual(s._entries[0].crossover_likelihood, 1)
        self.assertEqual(len(s._entries), 3)


if __name__ == "__main__":
    unittest.main()

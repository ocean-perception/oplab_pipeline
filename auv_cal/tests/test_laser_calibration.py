# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import unittest
import numpy as np
from auv_cal.laser_calibrator import build_plane
from auv_cal.laser_calibrator import opencv_to_ned
from auv_cal.laser_calibrator import get_angles
from auv_cal.ransac import fit_plane


class TestLaserCalibration(unittest.TestCase):
    def test_fit_plane(self):
        xyz = np.array([[1, 0, 0], [1, 1, 1], [1, -1, 1]])
        m = fit_plane(xyz)
        sqrt2_2 = 0.7071067811865476
        self.assertAlmostEqual(m[0], sqrt2_2)
        self.assertAlmostEqual(m[1], 0.0)
        self.assertAlmostEqual(m[2], 0.0)
        self.assertAlmostEqual(m[3], -sqrt2_2)

    def test_opencv_to_ned(self):
        a = np.array([1, 2, 3])
        b = opencv_to_ned(a)
        self.assertAlmostEqual(float(b[0]), -2.0)
        self.assertAlmostEqual(float(b[1]), 1.0)
        self.assertAlmostEqual(float(b[2]), 3.0)

    def assert_plane_normal(
        self, pitch, yaw, expected_normal, centroid=[0, 0, 0]
    ):
        centroid = np.array(centroid)
        plane, normal, offset = build_plane(pitch, yaw, centroid)

        expected_normal = np.array(expected_normal)
        expected_normal = expected_normal / np.linalg.norm(expected_normal)
        if expected_normal[0] < 0:
            # The function will always return positive X values
            expected_normal = expected_normal * (-1)

        print(normal, expected_normal)

        self.assertAlmostEqual(normal[0], expected_normal[0])
        self.assertAlmostEqual(normal[1], expected_normal[1])
        self.assertAlmostEqual(normal[2], expected_normal[2])

    def test_build_plane(self):
        centroid = np.array([1, 0, 0])
        # Square root of 2 divided by 2
        sqrt2_2 = 0.7071067811865476

        self.assert_plane_normal(0, 0, [1, 0, 0], centroid)
        self.assert_plane_normal(0, 45.0, [sqrt2_2, sqrt2_2, 0], centroid)
        self.assert_plane_normal(0, -45.0, [sqrt2_2, -sqrt2_2, 0], centroid)
        self.assert_plane_normal(45.0, 0, [sqrt2_2, 0, sqrt2_2], centroid)
        self.assert_plane_normal(-45.0, 0, [sqrt2_2, 0, -sqrt2_2], centroid)
        centroid = np.array([1.5, 0, 10.0])
        self.assert_plane_normal(0, 0, [1, 0, 0], centroid)

    def assert_normal(self, vector, expected_pitch, expected_yaw):
        plane_angle, pitch_angle, yaw_angle = get_angles(vector)
        self.assertAlmostEqual(pitch_angle, expected_pitch)
        self.assertAlmostEqual(yaw_angle, expected_yaw)
        self.assert_plane_normal(pitch_angle, yaw_angle, vector)

    def test_get_angles(self):
        vector = [1, 0, 0]
        self.assert_normal(vector, 0.0, 0.0)

        vector = [-1, 0, 0]
        self.assert_normal(vector, 0.0, 0.0)

        vector = [1, 1, 0]
        self.assert_normal(vector, 0.0, 45.0)

        vector = [-1, -1, 0]
        self.assert_normal(vector, 0.0, 45.0)

        vector = [1, -1, 0]
        self.assert_normal(vector, 0.0, -45.0)

        vector = [1, 0, 1]
        self.assert_normal(vector, 45.0, 0.0)

        vector = [1, 0, -1]
        self.assert_normal(vector, -45.0, 0.0)

        angle = 6.34019174590991
        vector = [0.9, 0, 0.1]
        self.assert_normal(vector, angle, 0.0)

        vector = [0.9, 0.1, 0]
        self.assert_normal(vector, 0.0, angle)

        vector = [0.9, 0.1, 0.1]
        self.assert_normal(vector, angle, angle)

        vector = [0.9, 0, -0.1]
        self.assert_normal(vector, -angle, 0.0)

        vector = [0.9, -0.1, 0]
        self.assert_normal(vector, 0.0, -angle)

        vector = [0.9, -0.1, 0.1]
        self.assert_normal(vector, angle, -angle)

        vector = [0.9, 0.1, -0.1]
        self.assert_normal(vector, -angle, angle)

        vector = [0.9, -0.1, -0.1]
        self.assert_normal(vector, -angle, -angle)


if __name__ == "__main__":
    unittest.main()

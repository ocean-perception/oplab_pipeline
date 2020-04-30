import unittest
import os
import math
import numpy as np
from auv_cal.laser_calibrator import build_plane
from auv_cal.laser_calibrator import opencv_to_ned
from auv_cal.laser_calibrator import get_angles
from auv_cal.laser_calibrator import fit_plane


class TestLaserCalibration(unittest.TestCase):
    def test_fit_plane(self):
        xyz = np.array([[1, 0, 0],
                        [1, 1, 1],
                        [1, -1, 1]])
        a, b, c, d = fit_plane(xyz)
        self.assertEqual(a, 1.0)
        self.assertEqual(b, 0.0)
        self.assertEqual(c, 0.0)
        self.assertEqual(d, -1.0)

    def test_opencv_to_ned(self):
        a = np.array([1, 2, 3])
        b = opencv_to_ned(a)
        self.assertEqual(b[0], 2.0)
        self.assertEqual(b[1], -1.)
        self.assertEqual(b[2], 3.)

    def test_build_plane(self):
        pitch = 0
        yaw = 0
        centroid = np.array([1, 0, 0])
        plane, normal, offset = build_plane(pitch, yaw, centroid)
        self.assertEqual(plane[0], 1.)
        self.assertEqual(plane[1], 0.)
        self.assertEqual(plane[2], 0.)
        self.assertEqual(plane[3], -1.)
        self.assertEqual(offset, 1.)

        plane, normal, offset = build_plane(0, 45.0, centroid)
        self.assertAlmostEqual(plane[0], 0.7071067811865476)
        self.assertAlmostEqual(plane[1], 0.7071067811865476)
        self.assertAlmostEqual(plane[2], 0.)

        plane, normal, offset = build_plane(0, -45.0, centroid)
        self.assertAlmostEqual(plane[0], 0.7071067811865476)
        self.assertAlmostEqual(plane[1], -0.7071067811865476)
        self.assertAlmostEqual(plane[2], 0.)

        plane, normal, offset = build_plane(45.0, 0, centroid)
        self.assertAlmostEqual(plane[0], 0.7071067811865476)
        self.assertAlmostEqual(plane[1], 0.)
        self.assertAlmostEqual(plane[2], 0.7071067811865476)

        plane, normal, offset = build_plane(-45.0, 0, centroid)
        self.assertAlmostEqual(plane[0], 0.7071067811865476)
        self.assertAlmostEqual(plane[1], 0.)
        self.assertAlmostEqual(plane[2], -0.7071067811865476)

    def test_get_angles(self):
        vector = [1, 0, 0]
        plane_angle, pitch_angle, yaw_angle = get_angles(vector)
        self.assertEqual(plane_angle, 0.0)
        self.assertEqual(pitch_angle, 0.0)
        self.assertEqual(yaw_angle, 0.0)

        vector = [1, 1, 0]
        plane_angle, pitch_angle, yaw_angle = get_angles(vector)
        self.assertEqual(pitch_angle, 0.)
        self.assertEqual(yaw_angle, 45.0)
        self.assertEqual(plane_angle, 45.0)

        vector = [1, -1, 0]
        plane_angle, pitch_angle, yaw_angle = get_angles(vector)
        self.assertEqual(pitch_angle, 0.0)
        self.assertEqual(yaw_angle, -45.0)
        self.assertEqual(plane_angle, 45.0)

        vector = [1, 0, 1]
        plane_angle, pitch_angle, yaw_angle = get_angles(vector)
        self.assertEqual(pitch_angle, 45.0)
        self.assertEqual(yaw_angle, 0.0)
        self.assertEqual(plane_angle, 45.0)

        vector = [1, 0, -1]
        plane_angle, pitch_angle, yaw_angle = get_angles(vector)
        self.assertEqual(pitch_angle, -45.0)
        self.assertEqual(yaw_angle, 0.0)
        self.assertEqual(plane_angle, 45.0)

if __name__ == '__main__':
    unittest.main()






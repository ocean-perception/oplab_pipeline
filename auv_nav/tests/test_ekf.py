# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import copy
import unittest

import numpy as np

from auv_nav.localisation.ekf import ExtendedKalmanFilter
from auv_nav.sensors import SyncedOrientationBodyVelocity


class TestEkf(unittest.TestCase):
    def setUp(self):
        self.initial_estimate_covariance = np.array(
            [
                [1e-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1e-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1e-6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1e-6, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1e-6, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1e-6, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1e-6, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1e-6, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1e-6, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-6, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-6, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-6],
            ]
        )
        self.process_noise_covariance = np.array(
            [
                [0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.06, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.03, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.03, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0.06, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0.025, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0.025, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0.04, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02],
            ]
        )
        self.sensors_std = {
            "position_xy": {"factor": 1.0, "offset": 0.0},
            "speed": {"factor": 1.0, "offset": 0.0},
            "position_z": {"factor": 1.0, "offset": 0.0},
            "orientation": {"factor": 1.0, "offset": 0.0},
        }

    def test_ekf(self):

        c = SyncedOrientationBodyVelocity()
        c.northings = 0.0
        c.eastings = 0.0
        c.depth = 0.0
        c.roll = 0.0
        c.pitch = 0.0
        c.yaw = 0.0
        c.x_velocity = 0.1
        c.y_velocity = 0.1
        c.z_velocity = 0.0
        c.roll_std = 0.5  # degrees
        c.pitch_std = 0.5  # degrees
        c.yaw_std = 0.5  # degrees
        c.depth_std = 1.0  # meters
        c.x_velocity_std = 0.05  # m/s
        c.y_velocity_std = 0.05  # m/s
        c.z_velocity_std = 0.05  # m/s

        dr_list = []
        t = 0.0
        dt = 0.1
        t_limit = 60.1
        while t < t_limit:
            m = copy.deepcopy(c)

            m.depth = np.random.normal(c.depth, c.depth_std)
            m.roll = np.random.normal(c.roll, c.roll_std)
            m.pitch = np.random.normal(c.pitch, c.pitch_std)
            m.yaw = np.random.normal(c.yaw, c.yaw_std)
            m.x_velocity = np.random.normal(c.x_velocity, c.x_velocity_std)
            m.y_velocity = np.random.normal(c.y_velocity, c.y_velocity_std)
            m.z_velocity = np.random.normal(c.z_velocity, c.z_velocity_std)

            m.epoch_timestamp = t
            dr_list.append(m)
            t += dt

        usbl_list = []
        mahalanobis_distance_threshold = 3.0

        ekf = ExtendedKalmanFilter(
            self.initial_estimate_covariance,
            self.process_noise_covariance,
            self.sensors_std,
            dr_list,
            usbl_list,
            mahalanobis_distance_threshold,
        )
        ekf.run()
        ekf_states = ekf.get_result()

        m = [s.toSyncedOrientationBodyVelocity() for s in ekf_states]
        ls: SyncedOrientationBodyVelocity = m[-1]

        std_th = 9.0  # STD threshold

        self.assertGreater(
            ls.northings, t_limit * c.x_velocity - std_th * c.x_velocity_std
        )
        self.assertLess(
            ls.northings, t_limit * c.x_velocity + std_th * c.x_velocity_std
        )
        self.assertGreater(
            ls.eastings, t_limit * c.y_velocity - std_th * c.y_velocity_std
        )
        self.assertLess(
            ls.eastings, t_limit * c.y_velocity + std_th * c.y_velocity_std
        )
        self.assertGreater(ls.depth, -c.depth_std * std_th)
        self.assertLess(ls.depth, c.depth_std * std_th)

        self.assertGreater(
            ls.x_velocity, c.x_velocity - std_th * c.x_velocity_std
        )
        self.assertLess(
            ls.x_velocity, c.x_velocity + std_th * c.x_velocity_std
        )
        self.assertGreater(ls.y_velocity, -c.y_velocity_std * std_th)
        self.assertLess(ls.y_velocity, c.y_velocity_std * std_th)
        self.assertGreater(ls.z_velocity, -c.z_velocity_std * std_th)
        self.assertLess(ls.z_velocity, c.z_velocity_std * std_th)

        self.assertGreater(ls.roll, -c.roll_std * std_th)
        self.assertLess(ls.roll, c.roll_std * std_th)
        self.assertGreater(ls.pitch, -c.pitch_std * std_th)
        self.assertLess(ls.pitch, c.pitch_std * std_th)
        self.assertGreater(ls.yaw, -c.yaw_std * std_th)
        self.assertLess(ls.yaw, c.yaw_std * std_th)


if __name__ == "__main__":
    unittest.main()
    # t = TestEkf()
    # t.setUp()
    # t.test_ekf()

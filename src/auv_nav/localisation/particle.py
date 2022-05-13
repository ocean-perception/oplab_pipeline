# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import math


class Particle:
    def __init__(self):  # origin_location):
        # self.dvl_noise = 0.0 # x_noise, y_noise
        # self.imu_noise    = 0.0 # yaw_noise (don;t)
        # self.usbl_noise   = 0.0 # = USBL std

        self.parentID = ""  # '0-0'/'0-1'/'0-2'/... so can string split and determine encoded location # noqa
        self.childIDList = []  # ['1-0','1-1','1'3']

        # self.weight = 1 # [] 0
        self.averaged_weight = 0
        self.averaged_error = 0
        self.error = []  # 0

        # self.trajectoryNrEsTs = [] # [[easting, northing, timestamp], [easting, northing, timestamp], ...] # write a function to plot nice graphs for visualization purposes (for you and publishing) # noqa
        self.eastings = []
        self.northings = []
        self.timestamps = []

        # from dvl_imu_data with noise
        self.x_velocity = []
        self.y_velocity = []
        self.z_velocity = []
        self.roll = []
        self.pitch = []
        self.yaw = []
        # from dvl_imu_data
        self.altitude = []
        self.depth = []

    def set(
        self,
        new_easting,
        new_northing,
        new_timestamp,
        new_x,
        new_y,
        new_z,
        new_roll,
        new_pitch,
        new_yaw,
        new_altitude,
        new_depth,
    ):  # , timestamp):
        self.eastings.append(new_easting)
        self.northings.append(new_northing)
        self.timestamps.append(new_timestamp)
        self.x_velocity.append(new_x)
        self.y_velocity.append(new_y)
        self.z_velocity.append(new_z)
        self.roll.append(new_roll)
        self.pitch.append(new_pitch)
        self.yaw.append(new_yaw)
        self.altitude.append(new_altitude)
        self.depth.append(new_depth)

    # def set_error(self, measurement):
    #     self.error = math.sqrt((self.eastings[0] - measurement.eastings) ** 2 + (self.northings[0] - measurement.northings) ** 2) # noqa

    def set_weight(self, new_weight):
        self.weight = new_weight

    def Gaussian(self, mu, sigma, x):
        # calculates the probability of x for 1-dim Gaussian with mean mu and sigma (standard deviation) # noqa
        return math.exp(-((mu - x) ** 2) / (sigma**2) / 2.0) / math.sqrt(
            2.0 * math.pi * (sigma**2)
        )

    def measurement_prob(self, measurement, measurement_error):
        # calculates how likely a measurement should be
        prob = 1.0
        dist = math.sqrt(
            (self.eastings[-1] - measurement.eastings) ** 2
            + (self.northings[-1] - measurement.northings) ** 2
        )
        self.error.append(dist)
        # prob *= self.Gaussian(0, measurement.northings_std, dist) # it should be there (mu = dist = 0), but measurement says its there (x = dist = particle - measurement), with some uncertainty (std = sense_noise) # noqa
        prob *= self.Gaussian(0, measurement_error, dist)
        #        for i in range(len(landmarks)):
        #            dist = math.sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2) # noqa
        #            prob *= self.Gaussian(dist, self.sense_noise, measurement[i]) # noqa
        return prob

    def __repr__(self):
        return "[x=%.6s y=%.6s orient=%.6s]" % (
            str(self.x),
            str(self.y),
            str(self.orientation),
        )

# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""


import numpy as np
from enum import IntEnum, unique


@unique
class Measure(IntEnum):
    X = 0
    Y = 1
    Z = 2
    ROLL = 3
    PITCH = 4
    YAW = 5
    VX = 6
    VY = 7
    VZ = 8
    VROLL = 9
    VPITCH = 10
    VYAW = 11
    AX = 12
    AY = 13
    AZ = 14


class Vector3():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class Orientation():
    def __init__(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0


class Pose():
    def __init__(self):
        self.position = Vector3()
        self.orientation = Orientation()
        self.covariance = np.zeros((6, 6), dtype=float)


class Velocity():
    def __init__(self):
        self.linear = Vector3()
        self.angular = Orientation()
        self.covariance = np.zeros((6, 6), dtype=float)


class Acceleration():
    def __init__(self):
        self.linear = Vector3()
        self.covariance = np.zeros((3, 3), dtype=float)


class SensorMeasurement():

    def __init__(self):
        self.stamp = 0.0
        self.origin = Pose()
        self.pose = Pose()
        self.velocity = Velocity()
        self.acceleration = Acceleration()
        self.valid = [False, False, False,  # x, y, z
                      False, False, False,  # r, p, y
                      False, False, False,  # vx, vy, vz
                      False, False, False,  # vr, vp, vy
                      False, False, False]  # ax, ay, az

    def from_dvl(self, stamp, vx, vy, vz, cov_vx, cov_vy, cov_vz):
        self.stamp = stamp
        self.valid[Measure.VX] = True
        self.valid[Measure.VY] = True
        self.valid[Measure.VZ] = True
        self.velocity.linear.x = vx
        self.velocity.linear.y = vy
        self.velocity.linear.z = vz
        self.velocity.covariance[Measure.X, Measure.X] = cov_vx
        self.velocity.covariance[Measure.Y, Measure.Y] = cov_vy
        self.velocity.covariance[Measure.Z, Measure.Z] = cov_vz

    def from_ins(self, stamp, roll, pitch, yaw,
                 cov_roll, cov_pitch, cov_yaw,
                 vroll=False, vpitch=False, vyaw=False,
                 cov_vroll=False, cov_vpitch=False, cov_vyaw=False,
                 ax=False, ay=False, az=False,
                 cov_ax=False, cov_ay=False, cov_az=False):
        self.stamp = stamp
        self.valid[Measure.ROLL] = True
        self.valid[Measure.PITCH] = True
        self.valid[Measure.YAW] = True
        self.pose.orientation.roll = roll
        self.pose.orientation.pitch = pitch
        self.pose.orientation.yaw = yaw
        self.pose.covariance[Measure.ROLL, Measure.ROLL] = cov_roll
        self.pose.covariance[Measure.PITCH, Measure.PITCH] = cov_pitch
        self.pose.covariance[Measure.YAW, Measure.YAW] = cov_yaw
        if vroll and vpitch and vyaw:
            self.valid[Measure.VROLL] = True
            self.valid[Measure.VPITCH] = True
            self.valid[Measure.VYAW] = True
            self.velocity.angular.roll = vroll
            self.velocity.angular.pitch = vpitch
            self.velocity.angular.yaw = vyaw
            self.velocity.covariance[Measure.ROLL, Measure.ROLL] = cov_vroll
            self.velocity.covariance[Measure.PITCH, Measure.PITCH] = cov_vpitch
            self.velocity.covariance[Measure.YAW, Measure.YAW] = cov_vyaw
        if ax and ay and az:
            self.valid[Measure.AX] = True
            self.valid[Measure.AY] = True
            self.valid[Measure.AZ] = True
            self.acceleration.x = ax
            self.acceleration.y = ay
            self.acceleration.z = az
            self.acceleration.covariance[Measure.X, Measure.X] = cov_ax
            self.acceleration.covariance[Measure.Y, Measure.Y] = cov_ay
            self.acceleration.covariance[Measure.Z, Measure.Z] = cov_az

    def from_depth(self, stamp, z, cov_z):
        self.stamp = stamp
        self.valid[Measure.Z] = True
        self.pose.position.z = z
        self.pose.covariance[Measure.Z, Measure.Z] = cov_z

    def from_usbl(self, stamp, x, y, cov_x, cov_y):
        self.stamp = stamp
        self.valid[Measure.X] = True
        self.valid[Measure.Y] = True
        self.pose.position.x = x
        self.pose.position.y = y
        self.pose.covariance[Measure.X, Measure.X] = cov_x
        self.pose.covariance[Measure.Y, Measure.Y] = cov_y
        # TODO who handles the offsets?

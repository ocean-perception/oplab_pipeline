# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import copy
import math
from typing import List

import numpy as np
from auv_nav.sensors import Camera, SyncedOrientationBodyVelocity
from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.interpolate import interpolate
from auv_nav.tools.latlon_wgs84 import metres_to_latlon
from oplab import Console


class Index:
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
    DIM = 12


class MeasurementReport:
    __slots__ = ["valid", "dropped", "total"]

    def __init__(self):
        self.valid = 0
        self.dropped = 0

    def add(self, valid):
        if valid:
            self.valid += 1
        else:
            self.dropped += 1
        self.total = self.valid + self.dropped


class EkfState(object):
    __slots__ = ["time", "state", "covariance"]

    def __init__(self, time, state, covariance):
        # The real-valued time, in seconds, since some epoch
        self.time = time

        # A 12 dimensional state vector
        self.state = state

        # A 12 x 12 covariance matrix
        self.covariance = covariance

    def set(self, state, covariance):
        self.state = state
        self.covariance = covariance

    def get(self):
        return self.state, self.covariance

    def get_time(self):
        return self.time

    def fromSyncedOrientationBodyVelocity(self, b):
        self.time = b.epoch_timestamp
        self.state[Index.X, 0] = b.northings
        self.state[Index.Y, 0] = b.eastings
        self.state[Index.Z, 0] = b.depth
        self.state[Index.ROLL, 0] = b.roll * math.pi / 180.0
        self.state[Index.PITCH, 0] = b.pitch * math.pi / 180.0
        self.state[Index.YAW, 0] = b.yaw * math.pi / 180.0
        self.state[Index.VROLL, 0] = b.vroll * math.pi / 180.0
        self.state[Index.VPITCH, 0] = b.vpitch * math.pi / 180.0
        self.state[Index.VYAW, 0] = b.vyaw * math.pi / 180.0
        self.state[Index.VX, 0] = b.x_velocity
        self.state[Index.VY, 0] = b.y_velocity
        self.state[Index.VZ, 0] = b.z_velocity
        self.covariance = b.covariance

    def toSyncedOrientationBodyVelocity(self):
        b = SyncedOrientationBodyVelocity()
        b.epoch_timestamp = self.time
        b.northings = self.state[Index.X, 0]
        b.eastings = self.state[Index.Y, 0]
        b.depth = self.state[Index.Z, 0]
        b.northings_std = np.sqrt(abs(self.covariance[Index.X, Index.X]))
        b.eastings_std = np.sqrt(abs(self.covariance[Index.Y, Index.Y]))
        b.depth_std = np.sqrt(abs(self.covariance[Index.Z, Index.Z]))
        b.roll = self.state[Index.ROLL, 0] * 180.0 / math.pi
        b.pitch = self.state[Index.PITCH, 0] * 180.0 / math.pi
        b.yaw = self.state[Index.YAW, 0] * 180.0 / math.pi
        b.roll_std = np.sqrt(
            abs(self.covariance[Index.ROLL, Index.ROLL]) * 180.0 / math.pi
        )
        b.pitch_std = np.sqrt(
            abs(self.covariance[Index.PITCH, Index.PITCH]) * 180.0 / math.pi
        )
        b.yaw_std = np.sqrt(
            abs(self.covariance[Index.YAW, Index.YAW]) * 180.0 / math.pi
        )
        b.vroll = self.state[Index.VROLL, 0] * 180.0 / math.pi
        b.vpitch = self.state[Index.VPITCH, 0] * 180.0 / math.pi
        b.vyaw = self.state[Index.VYAW, 0] * 180.0 / math.pi
        b.vroll_std = np.sqrt(
            abs(self.covariance[Index.VROLL, Index.VROLL]) * 180.0 / math.pi
        )
        b.vpitch_std = np.sqrt(
            abs(self.covariance[Index.VPITCH, Index.VPITCH]) * 180.0 / math.pi
        )
        b.vyaw_std = np.sqrt(
            abs(self.covariance[Index.VYAW, Index.VYAW]) * 180.0 / math.pi
        )
        b.x_velocity = self.state[Index.VX, 0]
        b.y_velocity = self.state[Index.VY, 0]
        b.z_velocity = self.state[Index.VZ, 0]
        b.x_velocity_std = np.sqrt(abs(self.covariance[Index.VX, Index.VX]))
        b.y_velocity_std = np.sqrt(abs(self.covariance[Index.VY, Index.VY]))
        b.z_velocity_std = np.sqrt(abs(self.covariance[Index.VZ, Index.VZ]))

        b.covariance = self.covariance
        return b


def warn_if_zero(val, name):
    if val == 0:
        Console.warn("The value for", name, "is zero. Is this expected?")


class Measurement(object):
    __slots__ = [
        "measurement",
        "covariance",
        "sensors_std",
        "update_vector",
        "time",
        "type",
        "mahalanobis_threshold",
    ]

    def __init__(self, sensors_std):
        # The measurement and its associated covariance
        self.measurement = np.zeros(Index.DIM, dtype=np.float64)
        self.covariance = np.zeros((Index.DIM, Index.DIM), dtype=np.float64)
        self.sensors_std = sensors_std

        # This defines which variables within this measurement
        # actually get passed into the filter.
        self.update_vector = np.zeros(Index.DIM, dtype=int)

        # The real-valued time, in seconds, since some epoch
        self.time = 0.0

        # Measurement type string
        self.type = ""

        # The Mahalanobis distance threshold in number of sigmas
        self.mahalanobis_threshold = 8.0

    def __str__(self):
        msg = (
            "Measurement: "
            + str(self.measurement)
            + "\n Covariance: "
            + str(self.covariance)
        )
        return msg

    def from_depth(self, value):
        depth_std_factor = self.sensors_std["position_z"]["factor"]
        depth_std_offset = self.sensors_std["position_z"]["offset"]
        self.time = value.epoch_timestamp
        self.measurement[Index.Z] = value.depth
        if value.depth_std > 0:
            self.covariance[Index.Z, Index.Z] = value.depth_std ** 2
        else:
            self.covariance[Index.Z, Index.Z] = (
                value.depth * depth_std_factor + depth_std_offset
            ) ** 2
        warn_if_zero(self.covariance[Index.Z, Index.Z], "Z Covariance")
        # print('Depth cov:', self.covariance[Index.Z, Index.Z])
        self.update_vector[Index.Z] = 1

    def from_dvl(self, value):
        # Vinnay's dvl_noise model:
        # velocity_std = (-0.0125*((velocity)**2)+0.2*(velocity)+0.2125)/100)
        # assuming noise of x_velocity = y_velocity = z_velocity
        velocity_std_factor = self.sensors_std["speed"]["factor"]
        velocity_std_offset = self.sensors_std["speed"]["offset"]

        self.time = value.epoch_timestamp
        self.measurement[Index.VX] = value.x_velocity
        self.measurement[Index.VY] = value.y_velocity
        self.measurement[Index.VZ] = value.z_velocity

        if value.x_velocity_std > 0:
            self.covariance[Index.VX, Index.VX] = value.x_velocity_std ** 2
            self.covariance[Index.VY, Index.VY] = value.y_velocity_std ** 2
            self.covariance[Index.VZ, Index.VZ] = value.z_velocity_std ** 2
        else:
            self.covariance[Index.VX, Index.VX] = (
                abs(value.x_velocity) * velocity_std_factor
                + velocity_std_offset
            ) ** 2
            self.covariance[Index.VY, Index.VY] = (
                abs(value.y_velocity) * velocity_std_factor
                + velocity_std_offset
            ) ** 2
            self.covariance[Index.VZ, Index.VZ] = (
                abs(value.z_velocity) * velocity_std_factor
                + velocity_std_offset  # hack here
            ) ** 2
        warn_if_zero(self.covariance[Index.VX, Index.VX], "VX Covariance")
        warn_if_zero(self.covariance[Index.VY, Index.VY], "VY Covariance")
        warn_if_zero(self.covariance[Index.VZ, Index.VZ], "VZ Covariance")
        # print('DVL cov:', self.covariance[Index.VX, Index.VX])
        self.update_vector[Index.VX] = 1
        self.update_vector[Index.VY] = 1
        self.update_vector[Index.VZ] = 1

    def from_usbl(self, value):
        usbl_noise_std_offset = self.sensors_std["position_xy"]["offset"]
        usbl_noise_std_factor = self.sensors_std["position_xy"]["factor"]
        distance = math.sqrt(
            value.northings ** 2 + value.eastings ** 2 + value.depth ** 2
        )
        error = usbl_noise_std_offset + usbl_noise_std_factor * distance

        self.time = value.epoch_timestamp
        self.measurement[Index.X] = value.northings
        self.measurement[Index.Y] = value.eastings

        if value.northings_std > 0:
            self.covariance[Index.X, Index.X] = value.northings_std ** 2
            self.covariance[Index.Y, Index.Y] = value.eastings_std ** 2
        else:
            self.covariance[Index.X, Index.X] = error ** 2
            self.covariance[Index.Y, Index.Y] = error ** 2
        warn_if_zero(self.covariance[Index.X, Index.X], "X Covariance")
        warn_if_zero(self.covariance[Index.Y, Index.Y], "Y Covariance")
        # print('USBL cov:', self.covariance[Index.X, Index.X])
        self.update_vector[Index.X] = 1
        self.update_vector[Index.Y] = 1
        self.type = "USBL"

    def from_orientation(self, value):
        imu_noise_std_offset = self.sensors_std["orientation"]["offset"]
        imu_noise_std_factor = self.sensors_std["orientation"]["factor"]

        self.time = value.epoch_timestamp
        self.measurement[Index.ROLL] = value.roll * math.pi / 180.0
        self.measurement[Index.PITCH] = value.pitch * math.pi / 180.0
        self.measurement[Index.YAW] = value.yaw * math.pi / 180.0

        if value.roll_std > 0:
            self.covariance[Index.ROLL, Index.ROLL] = (
                value.roll_std * math.pi / 180.0
            ) ** 2
            self.covariance[Index.PITCH, Index.PITCH] = (
                value.pitch_std * math.pi / 180.0
            ) ** 2
            self.covariance[Index.YAW, Index.YAW] = (
                value.yaw_std * math.pi / 180.0
            ) ** 2
        else:
            self.covariance[Index.ROLL, Index.ROLL] = (
                (imu_noise_std_offset + value.roll * imu_noise_std_factor)
                * math.pi
                / 180.0
            ) ** 2
            self.covariance[Index.PITCH, Index.PITCH] = (
                (imu_noise_std_offset + value.pitch * imu_noise_std_factor)
                * math.pi
                / 180.0
            ) ** 2
            self.covariance[Index.YAW, Index.YAW] = (
                (imu_noise_std_offset + value.yaw * imu_noise_std_factor)
                * math.pi
                / 180.0
            ) ** 2
        warn_if_zero(
            self.covariance[Index.ROLL, Index.ROLL], "ROLL Covariance"
        )
        warn_if_zero(
            self.covariance[Index.PITCH, Index.PITCH], "PITCH Covariance"
        )
        warn_if_zero(self.covariance[Index.YAW, Index.YAW], "YAW Covariance")
        # print('Ori cov:', self.covariance[Index.YAW, Index.YAW])
        self.update_vector[Index.ROLL] = 1
        self.update_vector[Index.PITCH] = 1
        self.update_vector[Index.YAW] = 1

    def from_synced_orientation_body_velocity(self, value):
        self.from_orientation(value)
        self.from_dvl(value)
        self.from_depth(value)
        self.type = "DR"


class EkfImpl(object):
    __slots__ = [
        "covariance",
        "state",
        "initialized",
        "last_update_time",
        "predicted_state",
        "process_noise_covariance",
        "transfer_function",
        "transfer_function_jacobian",
        "states_vector",
        "smoothed_states_vector",
        "measurements",
    ]

    def __init__(self):
        self.covariance = np.array([])
        self.state = np.array([])
        self.initialized = False
        self.last_update_time = 0.0
        self.predicted_state = np.array([])
        self.process_noise_covariance = np.array([])
        self.transfer_function = np.array([])
        self.transfer_function_jacobian = np.array([])
        self.states_vector = []
        self.smoothed_states_vector = []
        self.measurements = {}

    def get_states(self):
        return self.states_vector

    def get_smoothed_states(self):
        return self.smoothed_states_vector

    def get_last_update_time(self):
        return self.last_update_time

    def set_state(self, state):
        self.state = copy.deepcopy(state.astype(float))

    def get_state(self):
        return self.state

    def set_process_noise_covariance(self, pnc):
        self.process_noise_covariance = copy.deepcopy(
            np.mat(pnc).astype(float)
        )

    def set_last_update_time(self, time):
        self.last_update_time = time

    def set_covariance(self, cov):
        self.covariance = copy.deepcopy(np.mat(cov).astype(float))

    def wrap_state_angles(self):
        self.state[Index.ROLL] = self.clamp_rotation(self.state[Index.ROLL])
        self.state[Index.PITCH] = self.clamp_rotation(self.state[Index.PITCH])
        self.state[Index.YAW] = self.clamp_rotation(self.state[Index.YAW])

    def clamp_rotation(self, rotation):
        # rotation = (rotation % 2*math.pi)
        while rotation > math.pi:
            rotation -= 2 * math.pi
        while rotation < -math.pi:
            rotation += 2 * math.pi
        return rotation

    def check_mahalanobis_distance(self, innovation, innovation_cov, nsigmas):
        # print('innovation:', innovation.shape)
        # print('innovation:', innovation)
        # print('innovation_cov:', innovation_cov.shape)
        # print('innovation_cov:', innovation_cov)
        sq_mahalanobis = np.dot(innovation.T, innovation_cov @ innovation)
        threshold = nsigmas * nsigmas
        if sq_mahalanobis >= threshold:
            # print("Mahalanobis distance too large ("
            #      + str(float(np.sqrt(sq_mahalanobis)))
            #      + "). Correction step will not be applied.")
            return False
        else:
            return True

    def predict(self, timestamp, delta, save_state=True):
        f = self.compute_transfer_function(delta, self.state)
        A = self.compute_transfer_function_jacobian(delta, self.state, f)
        # (1) Project the state forward: x = Ax + Bu (really, x = f(x, u))
        self.state = f @ self.state

        # (2) Project the error forward: P = J * P * J' + Q
        # print('0:', delta, self.covariance[Index.X, Index.X])
        self.covariance = A @ self.covariance @ A.T
        # print('1:', delta, self.covariance[Index.X, Index.X])
        self.covariance += abs(delta) * self.process_noise_covariance
        # print('2:', timestamp, self.covariance[Index.X, Index.X])

        # print('Prediction {0}'.format(str(self.state.T)))
        self.last_update_time = timestamp
        # Wrap state angles
        self.wrap_state_angles()

        # (3) Save the state for posterior smoothing
        s = EkfState(timestamp, self.state, self.covariance)
        if save_state:
            self.states_vector.append(s)
        return s

    def correct(self, measurement):
        # First, determine how many state vector values we're updating
        update_indices = []
        for i in range(len(measurement.update_vector)):
            nans_around = np.isnan(measurement.measurement[i]).any()
            update_value = measurement.update_vector[i]
            if update_value == 1 and not nans_around:
                update_indices.append(i)

        # Now build the sub-matrices from the full-sized matrices
        update_size = len(update_indices)
        state_subset = np.zeros((update_size, 1), dtype=np.float64)
        meas_subset = np.zeros((update_size, 1), dtype=np.float64)
        meas_cov_subset = np.zeros(
            (update_size, update_size), dtype=np.float64
        )
        state_to_meas_subset = np.zeros(
            (update_size, Index.DIM), dtype=np.float64
        )
        kalman_gain_subset = np.zeros(
            (update_size, update_size), dtype=np.float64
        )
        innovation_subset = np.zeros((update_size, 1), dtype=np.float64)

        for i, upd_i in enumerate(update_indices):
            meas_subset[i] = measurement.measurement[upd_i]
            state_subset[i] = self.state[upd_i]
            for j, upd_j in enumerate(update_indices):
                meas_cov_subset[i, j] = measurement.covariance[upd_i, upd_j]
            """
            Handle negative (read: bad) covariances in the measurement. Rather
            than exclude the measurement or make up a covariance, just take
            the absolute value.
            """
            if meas_cov_subset[i, i] < 0.0:
                meas_cov_subset[i, i] = abs(meas_cov_subset[i, i])
            """
            If the measurement variance for a given variable is very
            near 0 (as in e-50 or so) and the variance for that
            variable in the covariance matrix is also near zero, then
            the Kalman gain computation will blow up. Really, no
            measurement can be completely without error, so add a small
            amount in that case.
            """
            if meas_cov_subset[i, i] < 1e-9:
                meas_cov_subset[i, i] = 1e-9
            """
            The state-to-measurement function, h, will now be a
            measurement_size x full_state_size matrix, with ones in the (i, i)
            locations of the values to be updated
            """
            state_to_meas_subset[i, upd_i] = 1

        # print('update_indices:\n', update_indices)
        # print('H:', state_to_meas_subset)
        # print('z:', meas_subset)
        # print('R:', meas_cov_subset)
        # print('R2:', measurement.covariance)
        # print('covariance:\n', self.covariance)

        # (1) Compute the Kalman gain: K = (PH') / (HPH' + R)
        pht = self.covariance @ state_to_meas_subset.T
        # print('pht:\n',pht)
        hphr_inv = np.linalg.inv(state_to_meas_subset @ pht + meas_cov_subset)
        kalman_gain_subset = pht @ hphr_inv
        innovation_subset = meas_subset - state_subset

        # print('K:', kalman_gain_subset)
        # print('Y:', innovation_subset)

        # Wrap angles of the innovation_subset
        for i, idx in enumerate(update_indices):
            if idx == Index.ROLL or idx == Index.PITCH or idx == Index.YAW:
                while innovation_subset[i] < -math.pi:
                    innovation_subset[i] += 2 * math.pi
                while innovation_subset[i] > math.pi:
                    innovation_subset[i] -= 2 * math.pi

        # (2) Check mahalanobis distance
        valid = self.check_mahalanobis_distance(
            innovation_subset, hphr_inv, measurement.mahalanobis_threshold
        )
        if valid:
            # (3) Apply the gain
            self.state += kalman_gain_subset @ innovation_subset
            # (4) Update the estimated covariance (Joseph form)
            gain_residual = np.eye(Index.DIM, dtype=np.float64)
            gain_residual -= kalman_gain_subset @ state_to_meas_subset
            self.covariance = gain_residual @ self.covariance @ gain_residual.T
            self.covariance += (
                kalman_gain_subset @ meas_cov_subset @ kalman_gain_subset.T
            )
            # Wrap state angles
            self.wrap_state_angles()
            self.last_update_time = measurement.time

            # (5) Update the state for posterior smoothing
            if len(self.states_vector) > 0:
                self.states_vector[-1].set(self.state, self.covariance)

        if measurement.type not in self.measurements:
            self.measurements[measurement.type] = MeasurementReport()
        self.measurements[measurement.type].add(valid)

    def smooth(self, enable=True):
        if len(self.states_vector) < 2:
            return
        ns = len(self.states_vector)
        self.smoothed_states_vector = copy.deepcopy(self.states_vector)
        if enable:
            for i in range(ns):
                Console.progress(ns + i, 2 * ns)
                sf = self.smoothed_states_vector[ns - 1 - i]
                s = self.states_vector[ns - 2 - i]
                x_prior, p_prior = s.get()
                x_smoothed, p_smoothed = sf.get()

                delta = sf.get_time() - s.get_time()

                f = self.compute_transfer_function(delta, x_prior)
                A = self.compute_transfer_function_jacobian(delta, x_prior, f)

                p_prior_pred = (
                    A @ p_prior @ A.T + self.process_noise_covariance * delta
                )
                J = p_prior * A.T * np.linalg.inv(p_prior_pred)

                innovation = x_smoothed - f @ x_prior
                # Wrap angles of the innovation_subset
                for idx in range(Index.DIM):
                    if (
                        idx == Index.ROLL
                        or idx == Index.PITCH
                        or idx == Index.YAW
                    ):
                        innovation[idx] = np.arctan2(
                            np.sin(innovation[idx]), np.cos(innovation[idx])
                        )

                x_prior_smoothed = x_prior + J @ innovation
                p_prior_smoothed = (
                    p_prior + J @ (p_smoothed - p_prior_pred) @ J.T
                )
                self.smoothed_states_vector[ns - 2 - i].set(
                    x_prior_smoothed, p_prior_smoothed
                )

    def compute_transfer_function(self, delta, state):
        roll = state[Index.ROLL]
        pitch = state[Index.PITCH]
        yaw = state[Index.YAW]

        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cpi = 1.0 / cp
        tp = sp * cpi

        f = np.eye(Index.DIM, dtype=np.float64)
        f[Index.X, Index.VX] = cy * cp * delta
        f[Index.X, Index.VY] = (cy * sp * sr - sy * cr) * delta
        f[Index.X, Index.VZ] = (cy * sp * cr + sy * sr) * delta
        f[Index.Y, Index.VX] = sy * cp * delta
        f[Index.Y, Index.VY] = (sy * sp * sr + cy * cr) * delta
        f[Index.Y, Index.VZ] = (sy * sp * cr - cy * sr) * delta
        f[Index.Z, Index.VX] = -sp * delta
        f[Index.Z, Index.VY] = cp * sr * delta
        f[Index.Z, Index.VZ] = cp * cr * delta
        f[Index.ROLL, Index.VROLL] = delta
        f[Index.ROLL, Index.VPITCH] = sr * tp * delta
        f[Index.ROLL, Index.VYAW] = cr * tp * delta
        f[Index.PITCH, Index.VPITCH] = cr * delta
        f[Index.PITCH, Index.VYAW] = -sr * delta
        f[Index.YAW, Index.VPITCH] = sr * cpi * delta
        f[Index.YAW, Index.VYAW] = cr * cpi * delta
        return f

    def compute_transfer_function_jacobian(self, delta, state, f):
        roll = state[Index.ROLL]
        pitch = state[Index.PITCH]
        yaw = state[Index.YAW]
        vx = state[Index.VX]
        vy = state[Index.VY]
        vz = state[Index.VZ]
        vpitch = state[Index.VPITCH]
        vyaw = state[Index.VYAW]

        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cpi = 1.0 / cp
        tp = sp * cpi

        x_coeff = 0.0
        y_coeff = 0.0
        z_coeff = 0.0

        y_coeff = cy * sp * cr + sy * sr
        z_coeff = -cy * sp * sr + sy * cr
        dFx_dR = (y_coeff * vy + z_coeff * vz) * delta
        dFR_dR = 1.0 + (cr * tp * vpitch - sr * tp * vyaw) * delta

        x_coeff = -cy * sp
        y_coeff = cy * cp * sr
        z_coeff = cy * cp * cr
        dFx_dP = (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta
        dFR_dP = (cpi * cpi * sr * vpitch + cpi * cpi * cr * vyaw) * delta

        x_coeff = -sy * cp
        y_coeff = -sy * sp * sr - cy * cr
        z_coeff = -sy * sp * cr + cy * sr
        dFx_dY = (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta

        y_coeff = sy * sp * cr - cy * sr
        z_coeff = -sy * sp * sr - cy * cr
        dFy_dR = (y_coeff * vy + z_coeff * vz) * delta
        dFP_dR = (-sr * vpitch - cr * vyaw) * delta

        x_coeff = -sy * sp
        y_coeff = sy * cp * sr
        z_coeff = sy * cp * cr
        dFy_dP = (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta

        x_coeff = cy * cp
        y_coeff = cy * sp * sr - sy * cr
        z_coeff = cy * sp * cr + sy * sr
        dFy_dY = (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta

        y_coeff = cp * cr
        z_coeff = -cp * sr
        dFz_dR = (y_coeff * vy + z_coeff * vz) * delta
        dFY_dR = (cr * cpi * vpitch - sr * cpi * vyaw) * delta

        x_coeff = -cp
        y_coeff = -sp * sr
        z_coeff = -sp * cr
        dFz_dP = (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta
        dFY_dP = (sr * tp * cpi * vpitch + cr * tp * cpi * vyaw) * delta

        tfjac = copy.deepcopy(f)
        tfjac[Index.X, Index.ROLL] = dFx_dR
        tfjac[Index.X, Index.PITCH] = dFx_dP
        tfjac[Index.X, Index.YAW] = dFx_dY
        tfjac[Index.Y, Index.ROLL] = dFy_dR
        tfjac[Index.Y, Index.PITCH] = dFy_dP
        tfjac[Index.Y, Index.YAW] = dFy_dY
        tfjac[Index.Z, Index.ROLL] = dFz_dR
        tfjac[Index.Z, Index.PITCH] = dFz_dP
        tfjac[Index.ROLL, Index.ROLL] = dFR_dR
        tfjac[Index.ROLL, Index.PITCH] = dFR_dP
        # tfjac[Index.ROLL, Index.YAW] = dFR_dY
        tfjac[Index.PITCH, Index.ROLL] = dFP_dR
        # tfjac[Index.PITCH, Index.PITCH] = dFP_dP
        # tfjac[Index.PITCH, Index.YAW] = dFP_dY
        tfjac[Index.YAW, Index.ROLL] = dFY_dR
        tfjac[Index.YAW, Index.PITCH] = dFY_dP
        return tfjac

    def print_state(self):
        print("State at time {0}".format(str(self.last_update_time)))
        s = self.state
        c = self.covariance
        print(
            "\tXYZ : ({0}, {1}, {2})".format(str(s[0]), str(s[1]), str(s[2]))
        )
        print(
            "\tRPY : ({0}, {1}, {2})".format(str(s[3]), str(s[4]), str(s[5]))
        )
        print(
            "\tVXYZ: ({0}, {1}, {2})".format(str(s[6]), str(s[7]), str(s[8]))
        )
        print(
            "\tCXYZ: ({0}, {1}, {2})".format(
                str(c[0, 0]), str(c[1, 1]), str(c[2, 2])
            )
        )
        print(
            "\tCRPY: ({0}, {1}, {2})".format(
                str(c[3, 3]), str(c[4, 4]), str(c[5, 5])
            )
        )
        print(
            "\tCVs : ({0}, {1}, {2})".format(
                str(c[6, 6]), str(c[7, 7]), str(c[8, 8])
            )
        )
        print(
            "\tCWs : ({0}, {1}, {2})".format(
                str(c[9, 9]), str(c[10, 10]), str(c[11, 11])
            )
        )

    def print_report(self):
        Console.info("EKF measurements report:")
        for key in self.measurements:
            Console.info(
                "\t",
                key,
                "measurements:",
                self.measurements[key].dropped,
                "/",
                self.measurements[key].total,
                "dropped",
            )


class ExtendedKalmanFilter(object):
    __slots__ = [
        "ekf",
        "initial_estimate_covariance",
        "process_noise_covariance",
        "sensors_std",
        "dr_list",
        "usbl_list",
    ]

    def __init__(
        self,
        initial_estimate_covariance,
        process_noise_covariance,
        sensors_std,
        dr_list,
        usbl_list,
    ):
        """
        Get the first USBL, DVL and Orientation reading for EKF initialization
        """

        self.initial_estimate_covariance = initial_estimate_covariance
        self.process_noise_covariance = process_noise_covariance
        self.sensors_std = sensors_std
        self.dr_list = dr_list
        self.usbl_list = usbl_list

    def run(self, timestamp_list=None):
        state0 = self.build_state(self.dr_list[0])
        dr_idx, usbl_idx = 0, 0
        # Get first measurement (e.g. zero) as a start
        start_time = self.dr_list[0].epoch_timestamp
        current_time = start_time

        self.ekf = EkfImpl()
        self.ekf.set_state(state0)
        self.ekf.set_last_update_time(current_time)
        self.ekf.set_covariance(self.initial_estimate_covariance)
        self.ekf.set_process_noise_covariance(self.process_noise_covariance)

        # Case when no empty list is provided
        if timestamp_list is not None:
            if len(timestamp_list) == 0:
                timestamp_list = None

        timestamp_list_idx = 0
        if timestamp_list is not None:
            # Advance until the time in the list is greather than current_time
            while timestamp_list[timestamp_list_idx] < current_time:
                timestamp_list_idx += 1

        next_prediction_time = None

        # print('-------------------------------')
        # print("Running EKF...")
        # self.ekf.print_state()
        # print('-------------------------------')
        while dr_idx < len(self.dr_list):
            # Show progress
            if timestamp_list is None:
                Console.progress(dr_idx, len(self.dr_list) * 2)
            else:
                Console.progress(timestamp_list_idx, len(timestamp_list) * 2)

            dr_stamp = self.dr_list[dr_idx].epoch_timestamp
            if usbl_idx < len(self.usbl_list) and len(self.usbl_list) > 0:
                usbl_stamp = self.usbl_list[usbl_idx].epoch_timestamp
            else:
                # Fake a posterior USBL measurement to force EKF to read DR
                usbl_stamp = dr_stamp + 1

            last_update_time = self.ekf.get_last_update_time()

            if timestamp_list is not None:
                if len(timestamp_list) > timestamp_list_idx:
                    next_prediction_time = timestamp_list[timestamp_list_idx]
                else:
                    # Fake a posterior time to force EKF to use DR
                    next_prediction_time = dr_stamp + 1

                if (
                    next_prediction_time < dr_stamp
                    and next_prediction_time < usbl_stamp
                ):
                    self.ekf.predict(
                        next_prediction_time,
                        next_prediction_time - last_update_time,
                    )
                    timestamp_list_idx += 1
                    # Iterate again
                    continue

            m = Measurement(self.sensors_std)

            if dr_stamp < usbl_stamp:
                m.from_synced_orientation_body_velocity(self.dr_list[dr_idx])
                dr_idx += 1
            elif usbl_idx < len(self.usbl_list):
                m.from_usbl(self.usbl_list[usbl_idx])
                #self.ekf.print_state()
                #print(m)
                usbl_idx += 1

            last_update_delta = m.time - last_update_time

            if last_update_delta >= 0:
                # print('Predict')
                self.ekf.predict(m.time, last_update_delta)
                # self.ekf.print_state()
            # print('Correct')
            self.ekf.correct(m)
            # self.ekf.print_state()
        self.ekf.smooth(enable=True)
        self.ekf.print_report()

    def get_result(self):
        return self.ekf.get_states()

    def get_smoothed_result(self):
        return self.ekf.get_smoothed_states()

    def build_state(self, init_dr):
        # Create a state from dead reckoning
        x = init_dr.northings
        y = init_dr.eastings
        z = init_dr.depth
        roll = init_dr.roll * math.pi / 180.0
        pitch = init_dr.pitch * math.pi / 180.0
        heading = init_dr.yaw * math.pi / 180.0
        vx = init_dr.x_velocity
        vy = init_dr.y_velocity
        vz = init_dr.z_velocity
        state = np.array(
            [[x, y, z, roll, pitch, heading, vx, vy, vz, 0, 0, 0]]
        )
        return state.T


def save_ekf_to_list(ekf_states, mission, vehicle, dead_reckoning_dvl_list):
    ekf_list = []
    dr_idx = 1
    for s in ekf_states:
        b = s.toSyncedOrientationBodyVelocity()

        # Offset the measurements from the DVL to the robot origin
        [x_offset, y_offset, z_offset] = body_to_inertial(
            b.roll,
            b.pitch,
            b.yaw,
            vehicle.origin.surge - vehicle.dvl.surge,
            vehicle.origin.sway - vehicle.dvl.sway,
            vehicle.origin.heave - vehicle.dvl.heave,
        )
        b.northings += x_offset
        b.eastings += y_offset
        b.depth += z_offset

        # Transform to lat lon using origins
        b.latitude, b.longitude = metres_to_latlon(
            mission.origin.latitude,
            mission.origin.longitude,
            b.eastings,
            b.northings,
        )

        # Interpolate altitude from DVL
        while (
            dr_idx < len(dead_reckoning_dvl_list)
            and dead_reckoning_dvl_list[dr_idx].epoch_timestamp
            < b.epoch_timestamp
        ):
            dr_idx += 1
        b.altitude = interpolate(
            b.epoch_timestamp,
            dead_reckoning_dvl_list[dr_idx - 1].epoch_timestamp,
            dead_reckoning_dvl_list[dr_idx].epoch_timestamp,
            dead_reckoning_dvl_list[dr_idx - 1].altitude,
            dead_reckoning_dvl_list[dr_idx].altitude,
        )
        ekf_list.append(b)
    return ekf_list


def update_camera_list(
    camera_list: List[Camera],
    ekf_list: List[SyncedOrientationBodyVelocity],
    origin_offsets,
    camera1_offsets,
    latlon_reference,
):
    ekf_idx = 0
    c_idx = 0
    while c_idx < len(camera_list) and ekf_idx < len(ekf_list):
        cam_ts = camera_list[c_idx].epoch_timestamp
        ekf_ts = ekf_list[ekf_idx].epoch_timestamp
        if cam_ts < ekf_ts:
            if not camera_list[c_idx].updated:
                Console.error(
                    "There is a camera entry with index",
                    c_idx,
                    "that is not updated to EKF...",
                )
            c_idx += 1
        elif cam_ts > ekf_ts:
            ekf_idx += 1
        elif cam_ts == ekf_ts:
            camera_list[c_idx].fromSyncedBodyVelocity(
                ekf_list[ekf_idx],
                origin_offsets,
                camera1_offsets,
                latlon_reference,
            )
            c_idx += 1
            ekf_idx += 1
    return camera_list

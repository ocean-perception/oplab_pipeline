# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""
from auv_nav.auv_parsers.sensors import BodyVelocity, Usbl
from auv_nav.auv_parsers.sensors import Depth, Orientation

import math
import numpy as np
import copy


class Index():
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
    DIM = 15


class EkfState(object):
    def __init__(self, time, state, covariance):
        # The real-valued time, in seconds, since some epoch
        self.time = time

        # A 15 dimensional state vector
        self.state = state

        # A 15x15 covariance matrix
        self.covariance = covariance

    def set(self, state, covariance):
        self.state = state
        self.covariance = covariance

    def get(self):
        return self.state, self.covariance

    def get_time(self):
        return self.time


class Measurement(object):
    def __init__(self):
        # The measurement and its associated covariance
        self.measurement = np.zeros(15, dtype=float)
        self.covariance = np.zeros((15, 15), dtype=float)

        # This defines which variables within this measurement
        # actually get passed into the filter.
        self.update_vector = np.zeros(15, dtype=int)

        # The real-valued time, in seconds, since some epoch
        self.time = 0.0

        # The Mahalanobis distance threshold in number of sigmas
        self.mahalanobis_threshold = 1e20

    def from_depth(self, value):
        self.time = value.depth_timestamp
        self.measurement[Index.Z] = value.depth
        self.covariance[Index.Z, Index.Z] = value.depth_std
        self.update_vector[Index.Z] = 1

    def from_dvl(self, value):
        self.time = value.epoch_timestamp
        self.measurement[Index.VX] = value.x_velocity
        self.measurement[Index.VY] = value.y_velocity
        self.measurement[Index.VZ] = value.z_velocity
        self.covariance[Index.VX, Index.VX] = value.x_velocity_std
        self.covariance[Index.VY, Index.VY] = value.y_velocity_std
        self.covariance[Index.VZ, Index.VZ] = value.z_velocity_std
        self.update_vector[Index.VX] = 1
        self.update_vector[Index.VY] = 1
        self.update_vector[Index.VZ] = 1

    def from_usbl(self, value):
        self.time = value.epoch_timestamp
        self.measurement[Index.X] = value.northings
        self.measurement[Index.Y] = value.eastings
        self.covariance[Index.X, Index.X] = value.northings_std
        self.covariance[Index.Y, Index.Y] = value.eastings_std
        self.update_vector[Index.X] = 1
        self.update_vector[Index.Y] = 1

    def from_orientation(self, value):
        self.time = value.epoch_timestamp
        self.measurement[Index.ROLL] = value.roll*math.pi/180.
        self.measurement[Index.PITCH] = value.pitch*math.pi/180.
        self.measurement[Index.YAW] = value.yaw*math.pi/180.
        self.covariance[Index.ROLL, Index.ROLL] = value.roll_std*math.pi/180.
        self.covariance[Index.PITCH,
                        Index.PITCH] = value.pitch_std*math.pi/180.
        self.covariance[Index.YAW, Index.YAW] = value.yaw_std*math.pi/180.
        self.update_vector[Index.ROLL] = 1
        self.update_vector[Index.PITCH] = 1
        self.update_vector[Index.YAW] = 1


class EkfImpl(object):
    def __init__(self):
        self.covariance = np.array([])
        self.state = np.array([])
        self.last_measurement_time = 0.0
        self.initialized = False
        self.last_update_time = 0.0
        self.predicted_state = np.array([])
        self.process_noise_covariance = np.array([])
        self.transfer_function = np.array([])
        self.transfer_function_jacobian = np.array([])
        self.states_vector = []
        self.smoothed_states_vector = []

    def get_smoothed_states(self):
        return self.smoothed_states_vector

    def get_last_update_time(self):
        return self.last_update_time

    def get_last_measurement_time(self):
        return self.last_measurement_time

    def set_state(self, state):
        self.state = state.astype(float)

    def set_process_noise_covariance(self, pnc):
        self.process_noise_covariance = np.mat(pnc).astype(float)

    def set_last_update_time(self, time):
        self.last_update_time = time

    def set_last_measurement_time(self, time):
        self.last_measurement_time = time

    def set_covariance(self, cov):
        self.covariance = np.mat(cov).astype(float)

    def wrap_state_angles(self):
        self.state[Index.ROLL] = self.clamp_rotation(self.state[Index.ROLL])
        self.state[Index.PITCH] = self.clamp_rotation(self.state[Index.PITCH])
        self.state[Index.YAW] = self.clamp_rotation(self.state[Index.YAW])

    def clamp_rotation(self, rotation):
        while rotation > math.pi:
            rotation -= 2*math.pi
        while rotation < -math.pi:
            rotation += 2*math.pi
        return rotation

    def check_mahalanobis_distance(self, innovation, innovation_cov, nsigmas):
        sq_mahalanobis = np.dot(innovation.T, innovation_cov@innovation)
        threshold = nsigmas * nsigmas
        if sq_mahalanobis >= threshold:
            return False
        else:
            return True

    def predict(self, timestamp, delta):
        f = self.compute_transfer_function(delta, self.state)
        A = self.compute_transfer_function_jacobian(delta, self.state, f)
        # (1) Project the state forward: x = Ax + Bu (really, x = f(x, u))
        self.state = f @ self.state

        # (2) Project the error forward: P = J * P * J' + Q
        self.covariance = (A @ self.covariance @ A.T)
        self.covariance += delta * self.process_noise_covariance

        # (3) Save the state for posterior smoothing
        s = EkfState(timestamp, self.state, self.covariance)
        self.states_vector.append(s)

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
        state_subset = np.zeros((update_size, 1), dtype=float)
        meas_subset = np.zeros((update_size, 1), dtype=float)
        meas_cov_subset = np.zeros((update_size, update_size), dtype=float)
        state_to_meas_subset = np.zeros((update_size,
                                         Index.DIM), dtype=float)
        kalman_gain_subset = np.zeros((update_size, update_size), dtype=float)
        innovation_subset = np.zeros((update_size, 1), dtype=float)

        for i, upd_i in enumerate(update_indices):
            meas_subset[i] = measurement.measurement[upd_i]
            state_subset[i] = self.state[upd_i]
            for j, upd_j in enumerate(update_indices):
                meas_cov_subset[i, j] = \
                    measurement.covariance[upd_i, upd_j]
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

        # (1) Compute the Kalman gain: K = (PH') / (HPH' + R)
        pht = self.covariance @ state_to_meas_subset.T
        hphr_inv = np.linalg.inv(state_to_meas_subset @ pht + meas_cov_subset)
        kalman_gain_subset = pht @ hphr_inv
        innovation_subset = meas_subset - state_subset

        # Wrap angles of the innovation_subset
        for i, idx in enumerate(update_indices):
            if idx == Index.ROLL or idx == Index.PITCH or idx == Index.YAW:
                while innovation_subset[i] < -math.pi:
                    innovation_subset[i] += 2*math.pi
                while innovation_subset[i] > math.pi:
                    innovation_subset[i] -= 2*math.pi

        # (2) Check mahalanobis distance
        result = self.check_mahalanobis_distance(
            innovation_subset, hphr_inv,
            measurement.mahalanobis_threshold)
        if result is True:
            # (3) Apply the gain
            self.state += kalman_gain_subset @ innovation_subset
            # (4) Update the estimated covariance
            gain_residual = np.eye(Index.DIM, dtype=float)
            gain_residual -= kalman_gain_subset @ state_to_meas_subset
            self.covariance = (gain_residual
                               @ self.covariance
                               @ gain_residual.T)
            self.covariance += (kalman_gain_subset
                                @ meas_cov_subset
                                @ kalman_gain_subset.T)
            # Wrap state angles
            self.wrap_state_angles()

            # (5) Update the state for posterior smoothing
            if len(self.states_vector) > 0:
                self.states_vector[-1].set(
                    self.state, self.covariance)

    def smooth(self):
        if len(self.states_vector) < 2:
            return
        ns = len(self.states_vector)
        self.smoothed_states_vector = copy.deepcopy(self.states_vector)
        for i in range(ns):
            sf = self.smoothed_states_vector[ns - 1 - i]
            s = self.states_vector[ns - 2 - i]
            x_prior, p_prior = s.get()
            x_smoothed, p_smoothed = sf.get()

            delta = sf.get_time() - s.get_time()

            f = self.compute_transfer_function(delta, self.state)
            A = self.compute_transfer_function_jacobian(delta, self.state, f)

            p_prior_pred = (A @ p_prior @ A.T
                            + self.process_noise_covariance * delta)
            J = p_prior * A.T * np.linalg.inv(p_prior_pred)

            x_prior_smoothed = x_prior + J @ (x_smoothed - f @ x_prior)
            p_prior_smoothed = (
                p_prior + J @ (p_smoothed - p_prior_pred) @ J.T)
            self.smoothed_states_vector[ns-2-i].set(
                x_prior_smoothed, p_prior_smoothed)

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

        f = np.zeros((Index.DIM, Index.DIM), dtype=float)
        f[Index.X, Index.VX] = cy * cp * delta
        f[Index.X, Index.VY] = (cy * sp * sr - sy * cr) * delta
        f[Index.X, Index.VZ] = (cy * sp * cr + sy * sr) * delta
        f[Index.X, Index.AX] = 0.5 * f[Index.X, Index.VX] * delta
        f[Index.X, Index.AY] = 0.5 * f[Index.X, Index.VY] * delta
        f[Index.X, Index.AZ] = 0.5 * f[Index.X, Index.VZ] * delta
        f[Index.Y, Index.VX] = sy * cp * delta
        f[Index.Y, Index.VY] = (sy * sp * sr + cy * cr) * delta
        f[Index.Y, Index.VZ] = (sy * sp * cr - cy * sr) * delta
        f[Index.Y, Index.AX] = 0.5 * f[Index.Y, Index.VX] * delta
        f[Index.Y, Index.AY] = 0.5 * f[Index.Y, Index.VY] * delta
        f[Index.Y, Index.AZ] = 0.5 * f[Index.Y, Index.VZ] * delta
        f[Index.Z, Index.VX] = -sp * delta
        f[Index.Z, Index.VY] = cp * sr * delta
        f[Index.Z, Index.VZ] = cp * cr * delta
        f[Index.Z, Index.AX] = 0.5 * f[Index.Z, Index.VX] * delta
        f[Index.Z, Index.AY] = 0.5 * f[Index.Z, Index.VY] * delta
        f[Index.Z, Index.AZ] = 0.5 * f[Index.Z, Index.VZ] * delta
        f[Index.ROLL, Index.VROLL] = f[Index.X, Index.VX]
        f[Index.ROLL, Index.VPITCH] = f[Index.X, Index.VY]
        f[Index.ROLL, Index.VYAW] = f[Index.X, Index.VZ]
        f[Index.PITCH, Index.VROLL] = f[Index.Y, Index.VX]
        f[Index.PITCH, Index.VPITCH] = f[Index.Y, Index.VY]
        f[Index.PITCH, Index.VYAW] = f[Index.Y, Index.VZ]
        f[Index.YAW, Index.VROLL] = f[Index.Z, Index.VX]
        f[Index.YAW, Index.VPITCH] = f[Index.Z, Index.VY]
        f[Index.YAW, Index.VYAW] = f[Index.Z, Index.VZ]
        f[Index.VX, Index.AX] = delta
        f[Index.VY, Index.AY] = delta
        f[Index.VZ, Index.AZ] = delta
        return f

    def compute_transfer_function_jacobian(self, delta, state, f):
        x = state[Index.X]
        y = state[Index.Y]
        z = state[Index.Z]
        roll = state[Index.ROLL]
        pitch = state[Index.PITCH]
        yaw = state[Index.YAW]
        vx = state[Index.VX]
        vy = state[Index.VY]
        vz = state[Index.VZ]
        vroll = state[Index.VROLL]
        vpitch = state[Index.VPITCH]
        vyaw = state[Index.VYAW]
        ax = state[Index.AX]
        ay = state[Index.AY]
        az = state[Index.AZ]

        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        x_coeff = 0.0
        y_coeff = 0.0
        z_coeff = 0.0
        half_t_squared = 0.5 * delta * delta

        y_coeff = cy * sp * cr + sy * sr
        z_coeff = -cy * sp * sr + sy * cr
        dFx_dR = ((y_coeff * vy + z_coeff * vz) * delta
                  + (y_coeff * ay + z_coeff * az) * half_t_squared)
        dFR_dR = 1 + (y_coeff * vpitch + z_coeff * vyaw) * delta

        x_coeff = -cy * sp
        y_coeff = cy * cp * sr
        z_coeff = cy * cp * cr
        dFx_dP = (
            (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta
            + (x_coeff * ax + y_coeff * ay + z_coeff * az) * half_t_squared)
        dFR_dP = (x_coeff * vroll + y_coeff * vpitch + z_coeff * vyaw) * delta

        x_coeff = -sy * cp
        y_coeff = -sy * sp * sr - cy * cr
        z_coeff = -sy * sp * cr + cy * sr
        dFx_dY = (
            (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta
            + (x_coeff * ax + y_coeff * ay + z_coeff * az) * half_t_squared)
        dFR_dY = (x_coeff * vroll + y_coeff * vpitch + z_coeff * vyaw) * delta

        y_coeff = sy * sp * cr - cy * sr
        z_coeff = -sy * sp * sr - cy * cr
        dFy_dR = ((y_coeff * vy + z_coeff * vz) * delta
                  + (y_coeff * ay + z_coeff * az) * half_t_squared)
        dFP_dR = (y_coeff * vpitch + z_coeff * vyaw) * delta

        x_coeff = -sy * sp
        y_coeff = sy * cp * sr
        z_coeff = sy * cp * cr
        dFy_dP = (
            (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta
            + (x_coeff * ax + y_coeff * ay + z_coeff * az) * half_t_squared)
        dFP_dP = 1 + (x_coeff * vroll + y_coeff * vpitch + z_coeff * vyaw) * delta

        x_coeff = cy * cp
        y_coeff = cy * sp * sr - sy * cr
        z_coeff = cy * sp * cr + sy * sr
        dFy_dY = (
            (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta
            + (x_coeff * ax + y_coeff * ay + z_coeff * az) * half_t_squared)
        dFP_dY = (x_coeff * vroll + y_coeff * vpitch + z_coeff * vyaw) * delta

        y_coeff = cp * cr
        z_coeff = -cp * sr
        dFz_dR = ((y_coeff * vy + z_coeff * vz) * delta
                  + (y_coeff * ay + z_coeff * az) * half_t_squared)
        dFY_dR = (y_coeff * vpitch + z_coeff * vyaw) * delta

        x_coeff = -cp
        y_coeff = -sp * sr
        z_coeff = -sp * cr
        dFz_dP = (
            (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta
            + (x_coeff * ax + y_coeff * ay + z_coeff * az) * half_t_squared)
        dFY_dP = (x_coeff * vroll + y_coeff * vpitch + z_coeff * vyaw) * delta

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
        tfjac[Index.ROLL, Index.YAW] = dFR_dY
        tfjac[Index.PITCH, Index.ROLL] = dFP_dR
        tfjac[Index.PITCH, Index.PITCH] = dFP_dP
        tfjac[Index.PITCH, Index.YAW] = dFP_dY
        tfjac[Index.YAW, Index.ROLL] = dFY_dR
        tfjac[Index.YAW, Index.PITCH] = dFY_dP
        return tfjac


class ExtendedKalmanFilter(object):
    def __init__(self,
                 initial_estimate_covariance,
                 process_noise_covariance,
                 velocity_body_list,
                 orientation_list,
                 depth_list,
                 usbl_list):
        # Prepare a list of sensor readings and sort it.
        sensor_list = copy.deepcopy(velocity_body_list)
        sensor_list.extend(orientation_list)
        sensor_list.extend(depth_list)
        sensor_list.extend(usbl_list)
        sorted_list = sorted(sensor_list)

        """
        Get the first USBL, DVL and Orientation reading for EKF initialization
        """
        state0 = self.get_init_state(velocity_body_list,
                                     orientation_list,
                                     depth_list,
                                     usbl_list)

        # Get first measurement
        start_time = sorted_list[0].epoch_timestamp
        end_time = sorted_list[-1].epoch_timestamp
        current_time = start_time
        step_time = 0.05  # 20 Hz

        ekf = EkfImpl()
        ekf.set_state(state0)
        ekf.set_covariance(initial_estimate_covariance)
        ekf.set_process_noise_covariance(process_noise_covariance)

        sensor_idx = 0
        print("Running EKF...")
        while current_time < end_time:
            measurement_queue = []
            while sorted_list[sensor_idx].epoch_timestamp < current_time:
                m = Measurement()
                if type(sorted_list[sensor_idx]) is BodyVelocity:
                    m.from_dvl(sorted_list[sensor_idx])
                    measurement_queue.append(m)
                elif type(sorted_list[sensor_idx]) is Orientation:
                    m.from_orientation(sorted_list[sensor_idx])
                    measurement_queue.append(m)
                elif type(sorted_list[sensor_idx]) is Depth:
                    m.from_depth(sorted_list[sensor_idx])
                    measurement_queue.append(m)
                elif type(sorted_list[sensor_idx]) is Usbl:
                    m.from_usbl(sorted_list[sensor_idx])
                    measurement_queue.append(m)
                sensor_idx += 1

            # f_meas_time = ekf.getLastMeasurementTime()
            f_upda_time = ekf.get_last_update_time()
            last_update_delta = current_time - f_upda_time
            ekf.predict(current_time, last_update_delta)
            if len(measurement_queue) > 0:
                for m in measurement_queue:
                    ekf.correct(m)
                    ekf.set_last_measurement_time(m.time)
            ekf.set_last_update_time(current_time)
            current_time += step_time
        print("EKF finished, smoothing with EKS...")
        ekf.smooth()
        print("EKS finished")
        self.states = ekf.get_smoothed_states()

    def get_result(self):
        return self.states

    def get_init_state(self,
                       velocity_body_list,
                       orientation_list,
                       depth_list,
                       usbl_list):
        oi = 0
        oready = False
        di = 0
        dready = False
        ui = 0
        uready = False
        vi = 0
        ref_stamp = velocity_body_list[vi].epoch_timestamp

        all_ready = False
        while not all_ready:
            o = orientation_list[oi]
            d = depth_list[di]
            u = usbl_list[ui]
            if o.epoch_timestamp < ref_stamp:
                oi += 1
            else:
                oready = True
            if d.epoch_timestamp < ref_stamp:
                di += 1
            else:
                dready = True
            if u.epoch_timestamp < ref_stamp:
                ui += 1
            else:
                uready = True
            all_ready = oready and dready and uready

        x = usbl_list[ui].northings
        y = usbl_list[ui].eastings
        z = depth_list[di].depth
        roll = orientation_list[oi].roll
        pitch = orientation_list[oi].pitch
        heading = orientation_list[oi].yaw
        vx = velocity_body_list[vi].x_velocity
        vy = velocity_body_list[vi].y_velocity
        vz = velocity_body_list[vi].z_velocity
        state = np.array([[x, y, z,
                          roll, pitch, heading,
                          vx, vy, vz,
                          0, 0, 0,
                          0, 0, 0]])
        return state.T
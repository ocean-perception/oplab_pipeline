# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""
from auv_nav.tools.interpolate import interpolate_dvl
from auv_nav.tools.interpolate import interpolate

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
    DIM = 12


class EkfState(object):
    def __init__(self, time, state, covariance):
        # The real-valued time, in seconds, since some epoch
        self.time = time

        # A 12 dimensional state vector
        self.state = state

        # A 12x12 covariance matrix
        self.covariance = covariance

    def set(self, state, covariance):
        self.state = state
        self.covariance = covariance

    def get(self):
        return self.state, self.covariance

    def get_time(self):
        return self.time


class Measurement(object):
    def __init__(self, sensors_std):
        # The measurement and its associated covariance
        self.measurement = np.zeros(12, dtype=float)
        self.covariance = np.zeros((12, 12), dtype=float)
        self.sensors_std = sensors_std

        # This defines which variables within this measurement
        # actually get passed into the filter.
        self.update_vector = np.zeros(12, dtype=int)

        # The real-valued time, in seconds, since some epoch
        self.time = 0.0

        # The Mahalanobis distance threshold in number of sigmas
        self.mahalanobis_threshold = 1e20

    def from_depth(self, value):
        depth_std_factor = self.sensors_std['depth']['factor']
        depth_std_offset = self.sensors_std['depth']['offset']
        self.time = value.epoch_timestamp
        self.measurement[Index.Z] = value.depth
        if value.depth_std > 0:
            self.covariance[Index.Z, Index.Z] = value.depth_std**2
        else:
            self.covariance[Index.Z, Index.Z] = (
                value.depth*depth_std_factor + depth_std_offset)
        self.update_vector[Index.Z] = 1

    def from_dvl(self, value):
        # Vinnay's dvl_noise model:
        # velocity_std = (-0.0125*((velocity)**2)+0.2*(velocity)+0.2125)/100)
        # assuming noise of x_velocity = y_velocity = z_velocity
        velocity_std_factor = self.sensors_std['dvl']['factor']
        velocity_std_offset = self.sensors_std['dvl']['offset']

        self.time = value.epoch_timestamp
        self.measurement[Index.VX] = value.x_velocity
        self.measurement[Index.VY] = value.y_velocity
        self.measurement[Index.VZ] = value.z_velocity

        if value.x_velocity_std > 0:
            self.covariance[Index.VX, Index.VX] = value.x_velocity_std**2
            self.covariance[Index.VX, Index.VX] = value.y_velocity_std**2
            self.covariance[Index.VX, Index.VX] = value.z_velocity_std**2
        else:
            self.covariance[Index.VX, Index.VX] = (
                abs(value.x_velocity)*velocity_std_factor+velocity_std_offset)
            self.covariance[Index.VY, Index.VY] = (
                abs(value.y_velocity)*velocity_std_factor+velocity_std_offset)
            self.covariance[Index.VZ, Index.VZ] = (
                abs(value.z_velocity)*velocity_std_factor+velocity_std_offset)
        self.update_vector[Index.VX] = 1
        self.update_vector[Index.VY] = 1
        self.update_vector[Index.VZ] = 1

    def from_usbl(self, value):
        usbl_noise_std_offset = self.sensors_std['usbl']['offset']
        usbl_noise_std_factor = self.sensors_std['usbl']['factor']
        distance = math.sqrt(value.northings**2
                             + value.eastings**2
                             + value.depth**2)
        error = (usbl_noise_std_offset + usbl_noise_std_factor*distance)

        self.time = value.epoch_timestamp
        self.measurement[Index.X] = value.northings
        self.measurement[Index.Y] = value.eastings

        if value.northings_std > 0:
            self.covariance[Index.X, Index.X] = value.northings_std**2
            self.covariance[Index.Y, Index.Y] = value.eastings_std**2
        else:
            self.covariance[Index.X, Index.X] = error
            self.covariance[Index.Y, Index.Y] = error
        self.update_vector[Index.X] = 1
        self.update_vector[Index.Y] = 1

    def from_orientation(self, value):
        imu_noise_std_offset = self.sensors_std['orientation']['offset']
        imu_noise_std_factor = self.sensors_std['orientation']['factor']

        self.time = value.epoch_timestamp
        self.measurement[Index.ROLL] = value.roll*math.pi/180.
        self.measurement[Index.PITCH] = value.pitch*math.pi/180.
        self.measurement[Index.YAW] = value.yaw*math.pi/180.

        if value.roll_std > 0:
            self.covariance[Index.ROLL, Index.ROLL] = (value.roll_std*math.pi/180.)**2
            self.covariance[Index.PITCH, Index.PITCH] = (value.pitch*math.pi/180.)**2
            self.covariance[Index.YAW, Index.YAW] = (value.yaw*math.pi/180.)**2
        else:
            self.covariance[Index.ROLL, Index.ROLL] = (
                imu_noise_std_offset + value.roll*imu_noise_std_factor)
            self.covariance[Index.PITCH, Index.PITCH] = (
                imu_noise_std_offset + value.pitch*imu_noise_std_factor)
            self.covariance[Index.YAW, Index.YAW] = (
                imu_noise_std_offset + value.yaw*imu_noise_std_factor)
        self.update_vector[Index.ROLL] = 1
        self.update_vector[Index.PITCH] = 1
        self.update_vector[Index.YAW] = 1

    def from_synced_orientation_body_velocity(self, value):
        self.from_orientation(value)
        self.from_dvl(value)
        self.from_depth(value)


class EkfImpl(object):
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
        self.process_noise_covariance = copy.deepcopy(np.mat(pnc).astype(float))

    def set_last_update_time(self, time):
        self.last_update_time = time

    def set_covariance(self, cov):
        self.covariance = copy.deepcopy(np.mat(cov).astype(float))

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

        # print('Prediction {0}'.format(str(self.state.T)))
        self.last_update_time = timestamp

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
            self.last_update_time = measurement.time

            # (5) Update the state for posterior smoothing
            if len(self.states_vector) > 0:
                self.states_vector[-1].set(
                    self.state, self.covariance)
        else:
            print('Mahalanobis distance too large. Correction not applied.')

    def smooth(self, enable=True):
        if len(self.states_vector) < 2:
            return
        ns = len(self.states_vector)
        self.smoothed_states_vector = copy.deepcopy(self.states_vector)
        if enable:
            for i in range(ns):
                sf = self.smoothed_states_vector[ns - 1 - i]
                s = self.states_vector[ns - 2 - i]
                x_prior, p_prior = s.get()
                x_smoothed, p_smoothed = sf.get()

                delta = sf.get_time() - s.get_time()

                f = self.compute_transfer_function(delta, x_prior)
                A = self.compute_transfer_function_jacobian(delta, x_prior, f)

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
        cpi = 1. / cp
        tp = sp * cpi

        f = np.eye(Index.DIM, dtype=float)
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
        vroll = state[Index.VROLL]
        vpitch = state[Index.VPITCH]
        vyaw = state[Index.VYAW]

        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cpi = 1. / cp
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
        dFx_dP = (
            (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta)
        dFR_dP = (cpi * cpi * sr * vpitch + cpi * cpi * cr * vyaw) * delta

        x_coeff = -sy * cp
        y_coeff = -sy * sp * sr - cy * cr
        z_coeff = -sy * sp * cr + cy * sr
        dFx_dY = (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta
        # dFR_dY = (x_coeff * vroll + y_coeff * vpitch + z_coeff * vyaw) * delta

        y_coeff = sy * sp * cr - cy * sr
        z_coeff = -sy * sp * sr - cy * cr
        dFy_dR = ((y_coeff * vy + z_coeff * vz) * delta)
        dFP_dR = (-sr * vpitch - cr * vyaw) * delta

        x_coeff = -sy * sp
        y_coeff = sy * cp * sr
        z_coeff = sy * cp * cr
        dFy_dP = (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta
        # dFP_dP = 1 + (x_coeff * vroll + y_coeff * vpitch + z_coeff * vyaw) * delta

        x_coeff = cy * cp
        y_coeff = cy * sp * sr - sy * cr
        z_coeff = cy * sp * cr + sy * sr
        dFy_dY = (
            (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta)
        # dFP_dY = (x_coeff * vroll + y_coeff * vpitch + z_coeff * vyaw) * delta

        y_coeff = cp * cr
        z_coeff = -cp * sr
        dFz_dR = (y_coeff * vy + z_coeff * vz) * delta
        dFY_dR = (cr * cpi * vpitch - sr * cpi * vyaw) * delta

        x_coeff = -cp
        y_coeff = -sp * sr
        z_coeff = -sp * cr
        dFz_dP = (x_coeff * vx + y_coeff * vy + z_coeff * vz) * delta
        dFY_dP = (sr * tp * cpi * vpitch - cr * tp * cpi * vyaw) * delta

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
        print('State at time {0}'.format(str(self.last_update_time)))
        s = self.state
        print('XYZ : ({0}, {1}, {2})'.format(str(s[0]), str(s[1]), str(s[2])))
        print('RPY : ({0}, {1}, {2})'.format(str(s[3]), str(s[4]), str(s[5])))
        print('VXYZ: ({0}, {1}, {2})'.format(str(s[6]), str(s[7]), str(s[8])))


class ExtendedKalmanFilter(object):
    def __init__(self,
                 initial_estimate_covariance,
                 process_noise_covariance,
                 sensors_std,
                 dr_list,
                 usbl_list):
        """
        Get the first USBL, DVL and Orientation reading for EKF initialization
        """
        state0, dr_idx, usbl_idx = self.get_init_state(dr_list,
                                                       usbl_list)
        # Get first measurement
        start_time = dr_list[dr_idx].epoch_timestamp
        current_time = start_time

        self.ekf = EkfImpl()
        self.ekf.set_state(state0)
        self.ekf.set_last_update_time(current_time)
        self.ekf.set_covariance(initial_estimate_covariance)
        self.ekf.set_process_noise_covariance(process_noise_covariance)

        # print('-------------------------------')
        # print("Running EKF...")
        # self.ekf.print_state()
        # print('-------------------------------')
        while dr_idx < len(dr_list):
            dr_stamp = dr_list[dr_idx].epoch_timestamp
            if usbl_idx < len(usbl_list):
                usbl_stamp = usbl_list[usbl_idx].epoch_timestamp
            else:
                # Fake a posterior USBL measurement to force EKF to read DR
                usbl_stamp = dr_stamp + 1

            m = Measurement(sensors_std)
            f_upda_time = self.ekf.get_last_update_time()

            # m.from_synced_orientation_body_velocity(dr_list[dr_idx])
            # dr_idx += 1

            if dr_stamp < usbl_stamp:
                m.from_synced_orientation_body_velocity(dr_list[dr_idx])
                dr_idx += 1
            elif usbl_idx < len(usbl_list):
                m.from_usbl(usbl_list[usbl_idx])
                usbl_idx += 1

            last_update_delta = m.time - f_upda_time

            if last_update_delta > 0:
                self.ekf.predict(m.time, last_update_delta)
            self.ekf.correct(m)
        # print("EKF finished, smoothing with EKS...")
        self.ekf.smooth(enable=True)
        # print("EKS finished")

    def get_result(self):
        return self.ekf.get_states()

    def get_smoothed_result(self):
        return self.ekf.get_smoothed_states()

    def build_state(self, usbl, dr):
        x = usbl.northings
        y = usbl.eastings
        z = dr.depth
        roll = dr.roll*math.pi/180.
        pitch = dr.pitch*math.pi/180.
        heading = dr.yaw*math.pi/180.
        vx = dr.x_velocity
        vy = dr.y_velocity
        vz = dr.z_velocity
        state = np.array([[x, y, z,
                          roll, pitch, heading,
                          vx, vy, vz,
                          0, 0, 0]])
        return state

    def get_init_state(self,
                       dr_list,
                       usbl_list):
        dr_index = 0
        state = []

        dr_eastings = []
        dr_northings = []
        for i in range(len(usbl_list)):
            while dr_index < len(dr_list) - 2 and usbl_list[i].epoch_timestamp > dr_list[dr_index+1].epoch_timestamp:
                dr_index += 1
            dr_eastings.append(interpolate(usbl_list[i].epoch_timestamp,
                                           dr_list[dr_index].epoch_timestamp,
                                           dr_list[dr_index + 1].epoch_timestamp,
                                           dr_list[dr_index].eastings,
                                           dr_list[dr_index + 1].eastings))
            dr_northings.append(interpolate(usbl_list[i].epoch_timestamp,
                                            dr_list[dr_index].epoch_timestamp,
                                            dr_list[dr_index + 1].epoch_timestamp,
                                            dr_list[dr_index].northings,
                                            dr_list[dr_index + 1].northings))
        usbl_eastings = [i.eastings for i in usbl_list]
        usbl_northings = [i.northings for i in usbl_list]
        eastings_error = [y - x for x, y in zip(dr_eastings, usbl_eastings)]
        northings_error = [y - x for x, y in zip(dr_northings, usbl_northings)]
        eastings_mean = np.mean(eastings_error)
        northings_mean = np.mean(northings_error)

        # print('Offsetting DR navigation by USBL error offset: ({:.2f}, {:.2f})'.format(northings_mean, eastings_mean))

        dr_index = 0
        usbl_index = 0
        if usbl_list[usbl_index].epoch_timestamp > dr_list[dr_index].epoch_timestamp:
            while dr_index < len(dr_list) and usbl_list[usbl_index].epoch_timestamp > dr_list[dr_index].epoch_timestamp:
                dr_index += 1
        elif dr_list[dr_index].epoch_timestamp > usbl_list[usbl_index].epoch_timestamp:
            while usbl_index < len(usbl_list) and dr_list[dr_index].epoch_timestamp > usbl_list[usbl_index].epoch_timestamp:
                usbl_index += 1

        state = self.build_state(usbl_list[usbl_index], dr_list[dr_index])
        state[0, Index.X] = state[0, Index.X] - northings_mean
        state[0, Index.Y] = state[0, Index.Y] - eastings_mean
        return state.T, dr_index, usbl_index

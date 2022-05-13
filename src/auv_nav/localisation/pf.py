# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import copy
import math

import numpy as np
from numpy.random import randn, uniform

from auv_nav.sensors import SyncedOrientationBodyVelocity
from auv_nav.tools.interpolate import interpolate
from oplab import Console

# Particle Filter implementation using classes
# TODO: multiprocessing or multithreading
# TODO: port uncertainty calculation from Jim
# TODO: use 3D gaussian models for USBL


def gaussian_pdf(mu, sigma, x):
    num = -((mu - x) ** 2) / (sigma**2) / 2.0
    den = math.sqrt(2.0 * math.pi * (sigma**2))
    return math.exp(num) / den


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
    ALT = 9
    DIM = 10


class Particle:
    def __init__(self):
        # The real-valued time, in seconds, since some epoch
        self.time = None

        # A 11-dimensional state vector
        self.state = np.zeros((Index.DIM, 1), dtype=float)

        # The particle trajectory
        self.trajectory = []
        self.trajectory_time = []

        # The measured errors during PF
        self.trajectory_error = []

        # Particle weight
        self.weight = None

    @property
    def eastings(self):
        return self.state[Index.Y, 0]

    @property
    def northings(self):
        return self.state[Index.X, 0]

    def __eq__(self, other):
        return self.weight == other.weight

    def __lt__(self, other):
        return self.weight < other.weight


class UsblObservationModel:
    def __init__(self, usbl_noise_sigma_factor):
        # The measurement
        self.x = None
        self.y = None
        self.z = None
        self.std = None
        self.usbl_noise_sigma_factor = usbl_noise_sigma_factor

    def set_observation(self, value):
        self.x = value.northings
        self.y = value.eastings
        self.z = value.depth
        # TODO: Could we use a 3D Gaussian instead of a 1D?
        sigma = math.sqrt(
            value.northings_std**2 + value.eastings_std**2 + value.depth_std**2
        )
        self.std = self.usbl_noise_sigma_factor * sigma

    def measure(self, p):
        """
        This is the main method of ObservationModel. It takes a state reference
        as argument and is supposed to extract the state's variables to compute
        an importance weight for the state.
        Define this method in your sub-class!
        @param state Reference to the state that has to be weightened.
        @return importance weight for the state (positive, non-zero value).
        """
        weight = 1.0
        if self.x is not None:
            dist = math.sqrt(
                (self.x - p.state[Index.X, 0]) ** 2
                + (self.y - p.state[Index.Y, 0]) ** 2
                + (self.z - p.state[Index.Z, 0]) ** 2
            )
            p.trajectory_error.append(dist)
            weight = gaussian_pdf(0, self.std, dist)
        return weight


class DeadReckoningMovementModel:
    """
    @class MovementModel
    @brief Interface for movement models for particle filters.
    The movement model in a particle filter defines how a particle's state
    changes over time.
    """

    def __init__(self, sensors_std, dvl_noise_sigma_factor, imu_noise_sigma_factor):
        self.sensors_std = sensors_std
        self.dvl_noise_sigma_factor = dvl_noise_sigma_factor
        self.imu_noise_sigma_factor = imu_noise_sigma_factor
        self.movement = np.zeros((Index.DIM, 1), dtype=float)

    def set_movement(self, value):
        self.movement[Index.Z, 0] = value.depth
        self.movement[Index.ROLL, 0] = value.roll * math.pi / 180.0
        self.movement[Index.PITCH, 0] = value.pitch * math.pi / 180.0
        self.movement[Index.YAW, 0] = value.yaw * math.pi / 180.0
        self.movement[Index.VX, 0] = value.x_velocity
        self.movement[Index.VY, 0] = value.y_velocity
        self.movement[Index.VZ, 0] = value.z_velocity
        self.movement[Index.ALT, 0] = value.altitude

    def propagate(self, p, dt):
        """
        This is the main method of MovementModel. It takes a state reference as
        argument and is supposed to extract the state's variables and
        manipulate them. dt means delta t and defines the time in seconds that
        has passed since the last filter update.
        @param state Reference to the state that has to be manipulated.
        @param dt time that has passed since the last filter update in seconds.
        """
        depth_std_factor = self.sensors_std["position_z"]["factor"]
        depth_std_offset = self.sensors_std["position_z"]["offset"]
        if "factor_x" in self.sensors_std["speed"]:
            velocity_std_factor_x = self.sensors_std["speed"]["factor_x"]
            velocity_std_offset_x = self.sensors_std["speed"]["offset_x"]
            velocity_std_factor_y = self.sensors_std["speed"]["factor_y"]
            velocity_std_offset_y = self.sensors_std["speed"]["offset_y"]
            velocity_std_factor_z = self.sensors_std["speed"]["factor_z"]
            velocity_std_offset_z = self.sensors_std["speed"]["offset_z"]
        else:
            velocity_std_factor_x = self.sensors_std["speed"]["factor"]
            velocity_std_offset_x = self.sensors_std["speed"]["offset"]
            velocity_std_factor_y = self.sensors_std["speed"]["factor"]
            velocity_std_offset_y = self.sensors_std["speed"]["offset"]
            velocity_std_factor_z = self.sensors_std["speed"]["factor"]
            velocity_std_offset_z = self.sensors_std["speed"]["offset"]
        imu_noise_std_offset = self.sensors_std["orientation"]["offset"]
        imu_noise_std_factor = self.sensors_std["orientation"]["factor"]

        k_dvl = self.dvl_noise_sigma_factor
        k_imu = self.imu_noise_sigma_factor

        def linear_noise(idx, factor, offset, gain=1.0):
            return (
                self.movement[idx, 0]
                + randn() * (self.movement[idx, 0] * factor + offset) * gain
            )

        # Propagate all states except for X and Y
        p.state[Index.Z, 0] = linear_noise(Index.Z, depth_std_factor, depth_std_offset)
        p.state[Index.ROLL, 0] = linear_noise(
            Index.ROLL, imu_noise_std_factor, imu_noise_std_offset, k_imu
        )
        p.state[Index.PITCH, 0] = linear_noise(
            Index.PITCH, imu_noise_std_factor, imu_noise_std_offset, k_imu
        )
        p.state[Index.YAW, 0] = linear_noise(
            Index.YAW, imu_noise_std_factor, imu_noise_std_offset, k_imu
        )
        p.state[Index.VX, 0] = linear_noise(
            Index.VX, velocity_std_factor_x, velocity_std_offset_x, k_dvl
        )
        p.state[Index.VY, 0] = linear_noise(
            Index.VY, velocity_std_factor_y, velocity_std_offset_y, k_dvl
        )
        p.state[Index.VZ, 0] = linear_noise(
            Index.VZ, velocity_std_factor_z, velocity_std_offset_z, k_dvl
        )

        cr = math.cos(p.state[Index.ROLL, 0])
        sr = math.sin(p.state[Index.ROLL, 0])
        cp = math.cos(p.state[Index.PITCH, 0])
        sp = math.sin(p.state[Index.PITCH, 0])
        cy = math.cos(p.state[Index.YAW, 0])
        sy = math.sin(p.state[Index.YAW, 0])

        f = np.eye(Index.DIM, dtype=float)
        f[Index.X, Index.VX] = cy * cp * dt
        f[Index.X, Index.VY] = (cy * sp * sr - sy * cr) * dt
        f[Index.X, Index.VZ] = (cy * sp * cr + sy * sr) * dt
        f[Index.Y, Index.VX] = sy * cp * dt
        f[Index.Y, Index.VY] = (sy * sp * sr + cy * cr) * dt
        f[Index.Y, Index.VZ] = (sy * sp * cr - cy * sr) * dt
        f[Index.Z, Index.VX] = -sp * dt
        f[Index.Z, Index.VY] = cp * sr * dt
        f[Index.Z, Index.VZ] = cp * cr * dt

        # Propagate the p.state forward
        p.state = f @ p.state
        p.time += dt
        p.trajectory.append(p.state)
        p.trajectory_time.append(p.time)


class ParticleFilter:
    def __init__(
        self,
        num_particles,
        movement_model,
        observation_model,
        expected_iterations=0,
    ):
        self.particles = [Particle()] * num_particles
        self.particles_history = []
        self.iteration = 0
        self.iteration_step = int(float(expected_iterations) / 20.0)
        self.mm = movement_model
        self.om = observation_model
        for p in self.particles:
            p.weight = 1.0 / float(num_particles)
        self.particles_history.append(self.particles)

    def __str__(self):
        a = "Particle Filter with " + str(len(self.particles)) + " particles.\n"
        for i, p in enumerate(self.particles):
            a += " Particle " + str(i) + "\n"
            a += (
                "   (x, y, theta) = ("
                + str(p.x)
                + ", "
                + str(p.y)
                + ", "
                + str(p.theta)
                + ")\n"
            )
            a += "    w = " + str(p.weight) + "\n"
        return a

    def set_prior(self, prior):
        for i in range(len(self.particles)):
            self.particles[i] = copy.deepcopy(prior)
            self.particles[i].weight = 1.0 / float(len(self.particles))

    def set_observation(self, value):
        self.om.set_observation(value)

    def set_movement(self, value):
        self.mm.set_movement(value)

    def should_resample(self):
        return self.get_neff() < (len(self.particles) / 2.0)

    def propagate(self, dt):
        for p in self.particles:
            self.mm.propagate(p, dt)
        if self.iteration == self.iteration_step:
            self.particles_history.append(copy.deepcopy(self.particles))
            self.iteration = 0
        self.iteration += 1

    def measure(self):
        for p in self.particles:
            p.weight *= self.om.measure(p)
        self.particles.sort(reverse=True)
        self.normalize()

    def normalize(self):
        s = [p.weight for p in self.particles]
        norm = np.sum(s)
        # Avoid division by zero
        if norm < 1e-20:
            norm += 1e-20
        for p in self.particles:
            p.weight = p.weight / norm

    def resample(self):
        """Importance resample"""
        inverse_num = 1.0 / len(self.particles)
        # random start in CDF
        start = uniform() * inverse_num
        cumulative_weight = 0.0
        # index to draw from
        source_index = 0
        cumulative_weight += self.particles[source_index].weight
        new_particles = [None] * len(self.particles)

        for dest_index, p in enumerate(self.particles):
            # amount of cumulative weight to reach
            prob_sum = start + inverse_num * dest_index
            # sum weights until
            while prob_sum > cumulative_weight:
                source_index += 1
                if source_index >= len(self.particles):
                    source_index = len(self.particles) - 1
                    break
                # target sum reached
                cumulative_weight += self.particles[source_index].weight
            # copy particle (via assignment operator)
            new_particles[dest_index] = copy.deepcopy(self.particles[source_index])
        # Update the particle list
        self.particles = new_particles

    def get_neff(self):
        """Returns the number of effective particles"""
        weights = [p.weight for p in self.particles]
        return 1.0 / np.sum(np.square(weights))


def ParticleToSyncedOrientationBodyVelocity(p):
    sobv_list = []
    for t, x in zip(p.trajectory_time, p.trajectory):
        m = SyncedOrientationBodyVelocity()
        m.epoch_timestamp = t
        m.northings = x[Index.X, 0]
        m.eastings = x[Index.Y, 0]
        m.depth = x[Index.Z, 0]
        m.roll = x[Index.ROLL, 0] * 180.0 / math.pi
        m.pitch = x[Index.PITCH, 0] * 180.0 / math.pi
        m.yaw = x[Index.YAW, 0] * 180.0 / math.pi
        m.x_velocity = x[Index.VX, 0]
        m.y_velocity = x[Index.VY, 0]
        m.z_velocity = x[Index.VZ, 0]
        m.altitude = x[Index.ALT, 0]
        sobv_list.append(m)
    return sobv_list


def get_prior(dr_list, usbl_list):
    dr_index = 0

    # Interpolate DR to USBL updates
    dr_eastings = []
    dr_northings = []
    for i in range(len(usbl_list)):
        usbl_t = usbl_list[i].epoch_timestamp
        dr_t = dr_list[dr_index + 1].epoch_timestamp
        while dr_index < len(dr_list) - 2 and usbl_t > dr_t:
            usbl_t = usbl_list[i].epoch_timestamp
            dr_t = dr_list[dr_index + 1].epoch_timestamp
            dr_index += 1
        dr_eastings.append(
            interpolate(
                usbl_list[i].epoch_timestamp,
                dr_list[dr_index].epoch_timestamp,
                dr_list[dr_index + 1].epoch_timestamp,
                dr_list[dr_index].eastings,
                dr_list[dr_index + 1].eastings,
            )
        )
        dr_northings.append(
            interpolate(
                usbl_list[i].epoch_timestamp,
                dr_list[dr_index].epoch_timestamp,
                dr_list[dr_index + 1].epoch_timestamp,
                dr_list[dr_index].northings,
                dr_list[dr_index + 1].northings,
            )
        )
    usbl_eastings = [i.eastings for i in usbl_list]
    usbl_northings = [i.northings for i in usbl_list]
    eastings_error = [y - x for x, y in zip(dr_eastings, usbl_eastings)]
    northings_error = [y - x for x, y in zip(dr_northings, usbl_northings)]
    eastings_mean = np.mean(eastings_error)
    northings_mean = np.mean(northings_error)

    dr_index = 0
    usbl_index = 0
    usbl_t = usbl_list[usbl_index].epoch_timestamp
    dr_t = dr_list[usbl_index].epoch_timestamp

    while dr_index < len(dr_list) and usbl_t > dr_t:
        usbl_t = usbl_list[usbl_index].epoch_timestamp
        dr_t = dr_list[usbl_index].epoch_timestamp
        dr_index += 1
    while usbl_index < len(usbl_list) and usbl_t < dr_t:
        usbl_t = usbl_list[usbl_index].epoch_timestamp
        dr_t = dr_list[usbl_index].epoch_timestamp
        usbl_index += 1

    # Fix DR to index zero
    dr_index = 0

    # Build state from first known USBL and DR, and use that displacement
    # error at the start of DR.
    x = dr_list[dr_index].northings + northings_mean
    y = dr_list[dr_index].eastings + eastings_mean
    z = dr_list[dr_index].depth
    alt = dr_list[dr_index].altitude
    roll = dr_list[dr_index].roll * math.pi / 180.0
    pitch = dr_list[dr_index].pitch * math.pi / 180.0
    heading = dr_list[dr_index].yaw * math.pi / 180.0
    vx = dr_list[dr_index].x_velocity
    vy = dr_list[dr_index].y_velocity
    vz = dr_list[dr_index].z_velocity
    prior = Particle()
    prior.state = np.array(
        [
            [x - northings_mean],
            [y - eastings_mean],
            [z],
            [roll],
            [pitch],
            [heading],
            [vx],
            [vy],
            [vz],
            [alt],
        ]
    )
    prior.time = dr_list[0].epoch_timestamp
    return prior, dr_index, usbl_index


def run_particle_filter(
    usbl_list,
    dr_list,
    num_particles,
    sensors_std,
    dvl_noise_sigma_factor,
    imu_noise_sigma_factor,
    usbl_noise_sigma_factor,
    measurement_update_flag=True,
):
    """Execute the particle filter over the dataset
    Args:
         usbl_list (list): List of USBL measurements
         dr_list (list): List of DR measurements
         num_particles (int): Number of particles
         sensors_std (list): List of sensors standard deviations
         dvl_noise_sigma_factor (float): DVL noise std multiplication factor
         imu_noise_sigma_factor (float): IMU noise std multiplication factor
         usbl_noise_sigma_factor (float): USBL noise std multiplication factor
         measurement_update_flag (bool, optional): Whether to perform updates
         or not.
    Returns:
        List: List containing at position
            0: Output PF localisation
            1: USBL data points used in updates
            2: List of particles over time
            3: Northings STD
            4: Eastings STD
            5: Yaw STD
    """
    Console.info("Running Particle Filter with:")
    Console.info("\t* Number of particles: {}".format(num_particles))
    Console.info(
        "\t* DVL noise std: f(x)={}x+{} m/s".format(
            sensors_std["speed"]["factor"], sensors_std["speed"]["offset"]
        )
    )
    Console.info(
        "\t* IMU noise std: f(x)={}x+{} deg".format(
            sensors_std["orientation"]["factor"],
            sensors_std["orientation"]["offset"],
        )
    )
    Console.info(
        "\t* Depth noise std: f(x)={}x+{} meters".format(
            sensors_std["position_z"]["factor"],
            sensors_std["position_z"]["offset"],
        )
    )
    Console.info(
        "\t* USBL noise std: f(x)={}x+{} meters".format(
            sensors_std["position_xy"]["factor"],
            sensors_std["position_xy"]["offset"],
        )
    )
    Console.info("Running {} iterations...".format(len(dr_list)))

    prior, dr_idx, usbl_idx = get_prior(dr_list, usbl_list)

    om = UsblObservationModel(usbl_noise_sigma_factor)
    mm = DeadReckoningMovementModel(
        sensors_std, dvl_noise_sigma_factor, imu_noise_sigma_factor
    )

    pf = ParticleFilter(num_particles, mm, om, expected_iterations=len(dr_list))
    pf.set_prior(prior)

    last_t = dr_list[dr_idx].epoch_timestamp

    resampled_usbl_list = []

    # Loop through all DR
    while dr_idx < len(dr_list):
        Console.progress(dr_idx, len(dr_list))
        dr_stamp = dr_list[dr_idx].epoch_timestamp
        if usbl_idx < len(usbl_list):
            usbl_stamp = usbl_list[usbl_idx].epoch_timestamp
        else:
            # Fake a posterior USBL measurement to force PF to read DR
            usbl_stamp = dr_stamp + 1

        if dr_stamp < usbl_stamp:
            # Compute delta t
            dt = dr_list[dr_idx].epoch_timestamp - last_t
            # Set the current movement
            pf.set_movement(dr_list[dr_idx])
            # and propagate the filter
            pf.propagate(dt)
            last_t = dr_list[dr_idx].epoch_timestamp
            dr_idx += 1
        elif usbl_idx < len(usbl_list):
            # Compute delta t
            dt = usbl_list[usbl_idx].epoch_timestamp - last_t
            # Set the measurement
            pf.set_observation(usbl_list[usbl_idx])
            # Propagate
            pf.propagate(dt)
            # And measure. Resample if NEFF > 0.5
            pf.measure()
            if pf.should_resample():
                pf.resample()
                resampled_usbl_list.append(usbl_list[usbl_idx])
            last_t = usbl_list[usbl_idx].epoch_timestamp
            usbl_idx += 1

    # Find best particle
    best_particle = pf.particles[0]

    # Extract trajectory from it
    pf_list = ParticleToSyncedOrientationBodyVelocity(best_particle)

    # Get remaining bits
    particles_list = pf.particles_history

    # TODO: Compute std
    northings_std = []
    eastings_std = []
    yaw_std = []

    print("Resampled {} times.".format(len(resampled_usbl_list)))

    return [
        pf_list,
        resampled_usbl_list,
        particles_list,
        northings_std,
        eastings_std,
        yaw_std,
    ]

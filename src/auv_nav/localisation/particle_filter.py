# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import math
import random

import numpy

from auv_nav.localisation.particle import Particle
from auv_nav.sensors import SyncedOrientationBodyVelocity
from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.interpolate import interpolate_dvl, interpolate_usbl
from oplab import Console


# create an equation for each noise, and def them for sensors in ae2000, or
# ts1, or ts2. read from mission yaml which sensor used, and automatically
# pick the one desired.
class ParticleFilter:
    def __init__(
        self,
        usbl_data,
        dvl_imu_data,
        N,
        sensors_std,
        dvl_noise_sigma_factor,
        imu_noise_sigma_factor,
        usbl_noise_sigma_factor,
        measurement_update_flag=True,
    ):
        return

    def __new__(
        self,
        usbl_data,
        dvl_imu_data,
        N,
        sensors_std,
        dvl_noise_sigma_factor,
        imu_noise_sigma_factor,
        usbl_noise_sigma_factor,
        measurement_update_flag=True,
    ):
        # self.dvl_noise_sigma_factor = dvl_noise_sigma_factor
        # self.imu_noise_sigma_factor = imu_noise_sigma_factor
        # self.usbl_noise_sigma_factor = usbl_noise_sigma_factor

        """
        def eval(r, p):
            sum = 0.0
            for i in range(len(p)):  # calculate mean error
                dx = (
                    p[i].eastings[-1] - r.eastings[-1] + (world_size / 2.0)
                ) % world_size - (world_size / 2.0)
                dy = (
                    p[i].northings[-1] - r.northings[-1] + (world_size / 2.0)
                ) % world_size - (world_size / 2.0)
                err = math.sqrt(dx * dx + dy * dy)
                sum += err
            return sum / float(len(p))
        """

        # ========== Start Noise models ========== #
        def usbl_noise(usbl_datapoint):  # measurement noise
            # distance = usbl_datapoint.distance_to_ship # lateral_distance,bearing = latlon_to_metres(usbl_datapoint.latitude, usbl_datapoint.longitude, usbl_datapoint.latitude_ship, usbl_datapoint.longitude_ship)  # noqa
            # distance = math.sqrt(lateral_distance**2 + usbl_datapoint.depth**2) # noqa
            # error = usbl_noise_sigma_factor*(usbl_noise_std_offset + usbl_noise_std_factor*distance) # 5 is for the differential GPS, and the distance std factor 0.01 is used as 0.006 is too sall and unrealistic # This is moved to parse_gaps and parse_usbl_dump # noqa
            if usbl_datapoint.northings_std != 0:
                error = usbl_datapoint.northings_std * usbl_noise_sigma_factor
            else:
                usbl_noise_std_offset = 5
                usbl_noise_std_factor = 0.01
                distance = math.sqrt(
                    usbl_datapoint.northings**2
                    + usbl_datapoint.eastings**2
                    + usbl_datapoint.depth**2
                )
                error = usbl_noise_sigma_factor * (
                    usbl_noise_std_offset + usbl_noise_std_factor * distance
                )
            return error

        def dvl_noise(dvl_imu_datapoint, mode="estimate"):  # sensor1 noise
            # Vinnay's dvl_noise model: velocity_std = (-0.0125*((velocity)**2)+0.2*(velocity)+0.2125)/100) assuming noise of x_velocity = y_velocity = z_velocity # noqa
            velocity_std_factor = 0.001  # from catalogue rdi whn1200/600. # should read this in from somewhere else, e.g. json # noqa
            velocity_std_offset = 0.002  # 0.02 # 0.2 #from catalogue rdi whn1200/600. # should read this in from somewhere else # noqa
            x_velocity_std = (
                abs(dvl_imu_datapoint.x_velocity) * velocity_std_factor
                + velocity_std_offset
            )  # (-0.0125*((dvl_imu_datapoint.x_velocity)**2)+0.2*(dvl_imu_datapoint.x_velocity)+0.2125)/100 # noqa
            y_velocity_std = (
                abs(dvl_imu_datapoint.y_velocity) * velocity_std_factor
                + velocity_std_offset
            )  # (-0.0125*((dvl_imu_datapoint.y_velocity)**2)+0.2*(dvl_imu_datapoint.y_velocity)+0.2125)/100 # noqa
            z_velocity_std = (
                abs(dvl_imu_datapoint.z_velocity) * velocity_std_factor
                + velocity_std_offset
            )  # (-0.0125*((dvl_imu_datapoint.z_velocity)**2)+0.2*(dvl_imu_datapoint.z_velocity)+0.2125)/100 # noqa
            if mode == "estimate":
                x_velocity_estimate = random.gauss(
                    dvl_imu_datapoint.x_velocity,
                    dvl_noise_sigma_factor * x_velocity_std,
                )
                y_velocity_estimate = random.gauss(
                    dvl_imu_datapoint.y_velocity,
                    dvl_noise_sigma_factor * y_velocity_std,
                )
                z_velocity_estimate = random.gauss(
                    dvl_imu_datapoint.z_velocity,
                    dvl_noise_sigma_factor * z_velocity_std,
                )
                return (
                    x_velocity_estimate,
                    y_velocity_estimate,
                    z_velocity_estimate,
                )
            elif mode == "std":
                return max([x_velocity_std, y_velocity_std, z_velocity_std])

        def imu_noise(
            previous_dvlimu_data_point,
            current_dvlimu_data_point,
            particle_list_data,
        ):  # sensor2 noise
            imu_noise = (
                0.003 * imu_noise_sigma_factor
            )  # each time_step + 0.003. assuming noise of roll = pitch = yaw
            if particle_list_data == 0:  # for initiation
                roll_estimate = random.gauss(current_dvlimu_data_point.roll, imu_noise)
                pitch_estimate = random.gauss(
                    current_dvlimu_data_point.pitch, imu_noise
                )
                yaw_estimate = random.gauss(current_dvlimu_data_point.yaw, imu_noise)
            else:  # for propagation
                roll_estimate = particle_list_data.roll[-1] + random.gauss(
                    current_dvlimu_data_point.roll - previous_dvlimu_data_point.roll,
                    imu_noise,
                )
                pitch_estimate = particle_list_data.pitch[-1] + random.gauss(
                    current_dvlimu_data_point.pitch - previous_dvlimu_data_point.pitch,
                    imu_noise,
                )
                yaw_estimate = particle_list_data.yaw[-1] + random.gauss(
                    current_dvlimu_data_point.yaw - previous_dvlimu_data_point.yaw,
                    imu_noise,
                )
            if yaw_estimate < 0:
                yaw_estimate += 360
            elif yaw_estimate > 360:
                yaw_estimate -= 360
            return roll_estimate, pitch_estimate, yaw_estimate

        # ========== End Noise models ========== #

        def initialize_particles(
            N, usbl_datapoint, dvl_imu_datapoint, init_dvl_imu_datapoint
        ):
            particles = []
            northings_estimate = (
                usbl_datapoint.northings
                - dvl_imu_datapoint.northings
                + init_dvl_imu_datapoint.northings
            )
            eastings_estimate = (
                usbl_datapoint.eastings
                - dvl_imu_datapoint.eastings
                + init_dvl_imu_datapoint.eastings
            )
            for i in range(N):
                temp_particle = Particle()

                roll_estimate, pitch_estimate, yaw_estimate = imu_noise(
                    0, init_dvl_imu_datapoint, 0
                )
                (
                    x_velocity_estimate,
                    y_velocity_estimate,
                    z_velocity_estimate,
                ) = dvl_noise(init_dvl_imu_datapoint)

                usbl_uncertainty = usbl_noise(usbl_datapoint)
                # usbl_uncertainty = 0

                temp_particle.set(
                    random.gauss(eastings_estimate, usbl_uncertainty),
                    random.gauss(northings_estimate, usbl_uncertainty),
                    init_dvl_imu_datapoint.epoch_timestamp,
                    x_velocity_estimate,
                    y_velocity_estimate,
                    z_velocity_estimate,
                    roll_estimate,
                    pitch_estimate,
                    yaw_estimate,
                    init_dvl_imu_datapoint.altitude,
                    init_dvl_imu_datapoint.depth,
                )
                temp_particle.set_weight(1)
                particles.append(temp_particle)

            # Normalize weights
            weights_list = []
            for i in particles:
                weights_list.append(i.weight)
            normalized_weights = normalize_weights(weights_list)
            for index, particle_ in enumerate(particles):
                particle_.weight = normalized_weights[index]

            return particles

        def normalize_weights(weights_list):
            normalized_weights = []
            for i in weights_list:
                normalized_weights.append(i / sum(weights_list))
            return normalized_weights

        def propagate_particles(particles, previous_data_point, current_data_point):
            for i in particles:
                # Propagation error model
                time_difference = (
                    current_data_point.epoch_timestamp
                    - previous_data_point.epoch_timestamp
                )

                roll_estimate, pitch_estimate, yaw_estimate = imu_noise(
                    previous_data_point, current_data_point, i
                )
                (
                    x_velocity_estimate,
                    y_velocity_estimate,
                    z_velocity_estimate,
                ) = dvl_noise(current_data_point)

                [
                    north_velocity_estimate,
                    east_velocity_estimate,
                    down_velocity_estimate,
                ] = body_to_inertial(
                    roll_estimate,
                    pitch_estimate,
                    yaw_estimate,
                    x_velocity_estimate,
                    y_velocity_estimate,
                    z_velocity_estimate,
                )
                [
                    previous_north_velocity_estimate,
                    previous_east_velocity_estimate,
                    previous_down_velocity_estimate,
                ] = body_to_inertial(
                    i.roll[-1],
                    i.pitch[-1],
                    i.yaw[-1],
                    i.x_velocity[-1],
                    i.y_velocity[-1],
                    i.z_velocity[-1],
                )
                # DR motion model
                northing_estimate = (
                    0.5
                    * time_difference
                    * (north_velocity_estimate + previous_north_velocity_estimate)
                    + i.northings[-1]
                )
                easting_estimate = (
                    0.5
                    * time_difference
                    * (east_velocity_estimate + previous_east_velocity_estimate)
                    + i.eastings[-1]
                )
                i.set(
                    easting_estimate,
                    northing_estimate,
                    current_data_point.epoch_timestamp,
                    x_velocity_estimate,
                    y_velocity_estimate,
                    z_velocity_estimate,
                    roll_estimate,
                    pitch_estimate,
                    yaw_estimate,
                    current_data_point.altitude,
                    current_data_point.depth,
                )

        def measurement_update(
            N, usbl_measurement, particles_list, resample_flag=True
        ):  # updates weights of particles and resamples them # USBL uncertainty follow the readings (0.06/100* depth)! assuming noise of northing = easting # noqa

            # Update weights (particle weighting)
            for i in particles_list[-1]:
                weight = i.measurement_prob(
                    usbl_measurement, usbl_noise(usbl_measurement)
                )
                # weights.append(weight)
                # i.weight.append(weight)
                i.weight = i.weight * weight
            # Normalize weights # this should be in particles...
            weights_list = []
            for i in particles_list[-1]:
                weights_list.append(i.weight)
            normalized_weights = normalize_weights(weights_list)
            for index, particle_ in enumerate(particles_list[-1]):
                particle_.weight = normalized_weights[index]

            # calculate Neff
            weights_list = []
            for i in particles_list[-1]:
                weights_list.append(i.weight)
            effectiveParticleSize = 1 / sum([i**2 for i in weights_list])

            if effectiveParticleSize < len(particles_list[-1]) / 2:
                resample_flag = True
            else:
                resample_flag = False

            if resample_flag:
                # resampling wheel
                temp_particles = []
                index = int(random.random() * N)
                beta = 0.0
                mw = max(weights_list)
                for i in range(N):
                    beta += random.random() * 2.0 * mw
                    while beta > weights_list[index]:
                        beta -= weights_list[index]
                        index = (index + 1) % N
                    temp_particle = Particle()
                    temp_particle.parentID = "{}-{}".format(
                        len(particles_list) - 1, index
                    )
                    particles_list[-1][index].childIDList.append(
                        "{}-{}".format(len(particles_list), len(temp_particles))
                    )
                    temp_particle.set(
                        particles_list[-1][index].eastings[-1],
                        particles_list[-1][index].northings[-1],
                        particles_list[-1][index].timestamps[-1],
                        particles_list[-1][index].x_velocity[-1],
                        particles_list[-1][index].y_velocity[-1],
                        particles_list[-1][index].z_velocity[-1],
                        particles_list[-1][index].roll[-1],
                        particles_list[-1][index].pitch[-1],
                        particles_list[-1][index].yaw[-1],
                        particles_list[-1][index].altitude[-1],
                        particles_list[-1][index].depth[-1],
                    )
                    temp_particle.set_weight(1 / N)  # particles_list[-1][index].weight)
                    # temp_particle.set_error(usbl_measurement) # maybe can remove this? # noqa
                    temp_particles.append(temp_particle)
                return (True, temp_particles)
            else:
                return (False, particles_list)

        def extract_trajectory(final_particle):
            northings_trajectory = final_particle.northings
            eastings_trajectory = final_particle.eastings
            timestamp_list = final_particle.timestamps
            roll_list = final_particle.roll
            pitch_list = final_particle.pitch
            yaw_list = final_particle.yaw
            altitude_list = final_particle.altitude
            depth_list = final_particle.depth

            parentID = final_particle.parentID
            while parentID != "":
                particle_list = int(parentID.split("-")[0])
                element_list = int(parentID.split("-")[1])

                northings_trajectory = (
                    particles_list[particle_list][element_list].northings[:-1]
                    + northings_trajectory
                )
                eastings_trajectory = (
                    particles_list[particle_list][element_list].eastings[:-1]
                    + eastings_trajectory
                )
                timestamp_list = (
                    particles_list[particle_list][element_list].timestamps[:-1]
                    + timestamp_list
                )
                roll_list = (
                    particles_list[particle_list][element_list].roll[:-1] + roll_list
                )
                pitch_list = (
                    particles_list[particle_list][element_list].pitch[:-1] + pitch_list
                )
                yaw_list = (
                    particles_list[particle_list][element_list].yaw[:-1] + yaw_list
                )
                altitude_list = (
                    particles_list[particle_list][element_list].altitude[:-1]
                    + altitude_list
                )
                depth_list = (
                    particles_list[particle_list][element_list].depth[:-1] + depth_list
                )

                parentID = particles_list[particle_list][element_list].parentID
            return (
                northings_trajectory,
                eastings_trajectory,
                timestamp_list,
                roll_list,
                pitch_list,
                yaw_list,
                altitude_list,
                depth_list,
            )

        def mean_trajectory(particles):
            x_list_ = []
            y_list_ = []
            for i in particles:
                x_list_.append(i.weight * i.eastings[-1])
                y_list_.append(i.weight * i.northings[-1])
            x = sum(x_list_)
            y = sum(y_list_)
            return x, y

        x_list = []
        y_list = []

        particles_list = []
        usbl_datapoints = []

        # print('Initializing particles around first point of dead reckoning solution offset by averaged usbl readings') # noqa
        # Interpolate dvl_imu_data to usbl_data to initializing particles at first appropriate usbl timestamp. # noqa
        # if dvl_imu_data[dvl_imu_data_index].epoch_timestamp > usbl_data[usbl_data_index].epoch_timestamp: # noqa
        #     while dvl_imu_data[dvl_imu_data_index].epoch_timestamp > usbl_data[usbl_data_index].epoch_timestamp: # noqa
        #         usbl_data_index += 1
        # interpolate usbl_data to dvl_imu_data to initialize particles
        usbl_data_index = 0
        dvl_imu_data_index = 0
        if (
            usbl_data[usbl_data_index].epoch_timestamp
            > dvl_imu_data[dvl_imu_data_index].epoch_timestamp
        ):
            while (
                usbl_data[usbl_data_index].epoch_timestamp
                > dvl_imu_data[dvl_imu_data_index].epoch_timestamp
            ):
                dvl_imu_data_index += 1
        if (
            dvl_imu_data[dvl_imu_data_index].epoch_timestamp
            == usbl_data[usbl_data_index].epoch_timestamp
        ):
            particles = initialize_particles(
                N,
                usbl_data[usbl_data_index],
                dvl_imu_data[dvl_imu_data_index],
                dvl_imu_data[0],
            )  # *For now assume eastings_std = northings_std, usbl_data[usbl_data_index].eastings_std) # noqa
            usbl_data_index += 1
            dvl_imu_data_index += 1
        elif (
            usbl_data[usbl_data_index].epoch_timestamp
            < dvl_imu_data[dvl_imu_data_index].epoch_timestamp
        ):
            while (
                usbl_data[usbl_data_index + 1].epoch_timestamp
                < dvl_imu_data[dvl_imu_data_index].epoch_timestamp
            ):
                if len(usbl_data) - 2 == usbl_data_index:
                    Console.warn(
                        "USBL data does not span to DVL data. Is your data right?"  # noqa
                    )
                    break
                usbl_data_index += 1
            # interpolated_data = interpolate_data(usbl_data[usbl_data_index].epoch_timestamp, dvl_imu_data[dvl_imu_data_index], dvl_imu_data[dvl_imu_data_index+1]) # noqa
            interpolated_data = interpolate_usbl(
                dvl_imu_data[dvl_imu_data_index].epoch_timestamp,
                usbl_data[usbl_data_index],
                usbl_data[usbl_data_index + 1],
            )
            # dvl_imu_data.insert(dvl_imu_data_index+1, interpolated_data)
            usbl_data.insert(usbl_data_index + 1, interpolated_data)
            particles = initialize_particles(
                N,
                usbl_data[usbl_data_index],
                dvl_imu_data[dvl_imu_data_index + 1],
                dvl_imu_data[0],
            )  # *For now assume eastings_std = northings_std, usbl_data[usbl_data_index].eastings_std) # noqa
            usbl_data_index += 1
            dvl_imu_data_index += 1
        else:
            Console.quit("Check dvl_imu_data and usbl_data in particle_filter.py.")
        usbl_datapoints.append(usbl_data[usbl_data_index - 1])
        particles_list.append(particles)
        # Force to start at DR init
        dvl_imu_data_index = 0

        x_, y_ = mean_trajectory(particles)
        x_list.append(x_)
        y_list.append(y_)

        max_uncertainty = 0
        usbl_uncertainty_list = []
        n = 0
        if measurement_update_flag is True:  # perform resampling
            last_usbl_flag = False
            while dvl_imu_data[dvl_imu_data_index] != dvl_imu_data[-1]:
                Console.progress(dvl_imu_data_index, len(dvl_imu_data))
                time_difference = (
                    dvl_imu_data[dvl_imu_data_index + 1].epoch_timestamp
                    - dvl_imu_data[dvl_imu_data_index].epoch_timestamp
                )
                max_uncertainty += (
                    dvl_noise(dvl_imu_data[dvl_imu_data_index], mode="std")
                    * time_difference
                )  # * 2

                if (
                    dvl_imu_data[dvl_imu_data_index + 1].epoch_timestamp
                    < usbl_data[usbl_data_index].epoch_timestamp
                ):
                    propagate_particles(
                        particles_list[-1],
                        dvl_imu_data[dvl_imu_data_index],
                        dvl_imu_data[dvl_imu_data_index + 1],
                    )
                    dvl_imu_data_index += 1
                else:
                    if not last_usbl_flag:
                        # interpolate, insert, propagate, resample measurement_update, add new particles to list, check and assign parent id, check parents that have no children and delete it (skip this step for now) ### # noqa
                        interpolated_data = interpolate_dvl(
                            usbl_data[usbl_data_index].epoch_timestamp,
                            dvl_imu_data[dvl_imu_data_index],
                            dvl_imu_data[dvl_imu_data_index + 1],
                        )
                        dvl_imu_data.insert(dvl_imu_data_index + 1, interpolated_data)
                        propagate_particles(
                            particles_list[-1],
                            dvl_imu_data[dvl_imu_data_index],
                            dvl_imu_data[dvl_imu_data_index + 1],
                        )

                        usbl_uncertainty_list.append(
                            usbl_noise(usbl_data[usbl_data_index])
                        )

                        n += 1
                        resampled, new_particles = measurement_update(
                            N, usbl_data[usbl_data_index], particles_list
                        )

                        if resampled:
                            particles_list.append(new_particles)
                            usbl_datapoints.append(usbl_data[usbl_data_index])
                            # reset usbl_uncertainty_list
                            usbl_uncertainty_list = []
                        else:
                            particles_list = new_particles

                        if usbl_data[usbl_data_index] == usbl_data[-1]:
                            last_usbl_flag = True
                            dvl_imu_data_index += 1
                        else:
                            usbl_data_index += 1
                            dvl_imu_data_index += 1
                    else:
                        propagate_particles(
                            particles_list[-1],
                            dvl_imu_data[dvl_imu_data_index],
                            dvl_imu_data[dvl_imu_data_index + 1],
                        )
                        dvl_imu_data_index += 1

                x_, y_ = mean_trajectory(particles_list[-1])
                x_list.append(x_)
                y_list.append(y_)

            # print (max_uncertainty)

            # select particle trajectory with largest overall weight
            # particles_weight_list = []
            particles_error_list = []
            for i in range(len(particles_list[-1])):
                parentID = particles_list[-1][i].parentID
                particles_error_list.append([])
                if len(particles_list[-1][i].error) != 0:
                    particles_error_list[-1] += particles_list[-1][i].error
                while parentID != "":
                    particle_list = int(parentID.split("-")[0])
                    element_list = int(parentID.split("-")[1])
                    parentID = particles_list[particle_list][element_list].parentID
                    particles_error_list[-1] += particles_list[particle_list][
                        element_list
                    ].error
            for i in range(len(particles_error_list)):
                particles_error_list[i] = sum(particles_error_list[i])
            selected_particle = particles_list[-1][
                particles_error_list.index(min(particles_error_list))
            ]
            (
                northings_trajectory,
                eastings_trajectory,
                timestamp_list,
                roll_list,
                pitch_list,
                yaw_list,
                altitude_list,
                depth_list,
            ) = extract_trajectory(selected_particle)
        else:  # do not perform resampling, only propagate
            while dvl_imu_data[dvl_imu_data_index] != dvl_imu_data[-1]:
                propagate_particles(
                    particles_list[-1],
                    dvl_imu_data[dvl_imu_data_index],
                    dvl_imu_data[dvl_imu_data_index + 1],
                )
                dvl_imu_data_index += 1

            ## select particle trajectory with least average error (maybe assign weights without resampling and compare total or average weight? actually doesn't really matter because path won't be used anyway, main purpose of this is to see the std plot) # noqa
            particles_error_list = []
            for i in range(len(particles_list[-1])):
                parentID = particles_list[-1][i].parentID
                particles_error_list.append([])
                particles_error_list[-1].append(particles_list[-1][i].error)
                while parentID != "":
                    particle_list = int(parentID.split("-")[0])
                    element_list = int(parentID.split("-")[1])
                    parentID = particles_list[particle_list][element_list].parentID
                    particles_error_list[-1].append(
                        particles_list[particle_list][element_list].error
                    )
            for i in range(len(particles_error_list)):
                particles_error_list[i] = sum(particles_error_list[i]) / len(
                    particles_error_list[i]
                )
            selected_particle = particles_list[-1][
                particles_error_list.index(min(particles_error_list))
            ]
            (
                northings_trajectory,
                eastings_trajectory,
                timestamp_list,
                roll_list,
                pitch_list,
                yaw_list,
                altitude_list,
                depth_list,
            ) = extract_trajectory(selected_particle)

        # calculate northings std, eastings std, yaw std of particles
        northings_std = []
        eastings_std = []
        yaw_std = []

        arr_northings = []
        arr_eastings = []
        arr_yaw = []
        for i in particles_list[0]:
            arr_northings.append([])
            arr_eastings.append([])
            arr_yaw.append([])
        for i in range(len(particles_list)):
            for j in range(len(particles_list[i])):
                if i != len(particles_list) - 1:
                    arr_northings[j] += particles_list[i][j].northings[:-1]
                    arr_eastings[j] += particles_list[i][j].eastings[:-1]
                    arr_yaw[j] += particles_list[i][j].yaw[:-1]
                else:
                    arr_northings[j] += particles_list[i][j].northings
                    arr_eastings[j] += particles_list[i][j].eastings
                    arr_yaw[j] += particles_list[i][j].yaw
        arr_northings = numpy.array(arr_northings)
        arr_eastings = numpy.array(arr_eastings)
        arr_yaw = numpy.array(arr_yaw)

        for i in numpy.std(arr_northings, axis=0):
            northings_std.append(i)
        for i in numpy.std(arr_eastings, axis=0):
            eastings_std.append(i)
        # yaw_std step check for different extreme values around 0 and 360. not sure if this method below is robust. # noqa
        arr_std_yaw = numpy.std(arr_yaw, axis=0)
        arr_yaw_change = []
        for i in range(len(arr_std_yaw)):
            if (
                arr_std_yaw[i] > 30
            ):  # if std is more than 30 deg, means there's two extreme values, so minus 360 for anything above 180 deg. # noqa
                arr_yaw_change.append(i)
            # yaw_std.append(i)
        for i in arr_yaw:
            for j in arr_yaw_change:
                if i[j] > 180:
                    i[j] -= 360
        arr_std_yaw = numpy.std(arr_yaw, axis=0)
        for i in arr_std_yaw:
            yaw_std.append(i)
        # numpy.mean(arr, axis=0)

        pf_fusion_dvl_list = []
        for i in range(len(timestamp_list)):
            pf_fusion_dvl = SyncedOrientationBodyVelocity()
            pf_fusion_dvl.epoch_timestamp = timestamp_list[i]
            pf_fusion_dvl.northings = northings_trajectory[i]
            pf_fusion_dvl.eastings = eastings_trajectory[i]
            pf_fusion_dvl.depth = depth_list[i]
            pf_fusion_dvl.roll = roll_list[i]
            pf_fusion_dvl.pitch = pitch_list[i]
            pf_fusion_dvl.yaw = yaw_list[i]
            pf_fusion_dvl.altitude = altitude_list[i]
            pf_fusion_dvl_list.append(pf_fusion_dvl)

        # plt.scatter(x_list, y_list)

        return (
            pf_fusion_dvl_list,
            usbl_datapoints,
            particles_list,
            northings_std,
            eastings_std,
            yaw_std,
        )

        # include this later! after each resampling. maybe put it inside particle filter class # noqa
        # print (eval(myrobot, particles))

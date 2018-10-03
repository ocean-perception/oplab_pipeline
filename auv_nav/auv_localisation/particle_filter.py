# Particle Filter

# Author: Jin Wei Lim
# Date: 27/12/2017

import math
import random
import numpy
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from auv_nav.auv_conversions.interpolate import interpolate
from auv_nav.auv_coordinates.body_to_inertial import body_to_inertial
from auv_nav.auv_parsers.sensors import SyncedOrientationBodyVelocity, Usbl
from auv_nav.auv_localisation.particle import particle
from auv_nav.auv_coordinates.latlon_wgs84 import latlon_to_metres

# create an equation for each noise, and def them for sensors in ae2000, or ts1, or ts2. read from mission yaml which sensor used, and automatically pick the one desired.

class particle_filter:
    def __init__(self, usbl_data, dvl_imu_data, N, measurement_update_flag, dvl_noise_sigma_factor, imu_noise_sigma_factor, usbl_noise_sigma_factor):
        return

    def __new__(self, usbl_data, dvl_imu_data, N, measurement_update_flag, dvl_noise_sigma_factor, imu_noise_sigma_factor, usbl_noise_sigma_factor):
        # self.dvl_noise_sigma_factor = dvl_noise_sigma_factor
        # self.imu_noise_sigma_factor = imu_noise_sigma_factor
        # self.usbl_noise_sigma_factor = usbl_noise_sigma_factor

        def eval(r, p):
        	    sum = 0.0;
        	    for i in range(len(p)): # calculate mean error
        	        dx = (p[i].eastings[-1] - r.eastings[-1] + (world_size/2.0)) % world_size - (world_size/2.0)
        	        dy = (p[i].northings[-1] - r.northings[-1] + (world_size/2.0)) % world_size - (world_size/2.0)
        	        err = math.sqrt(dx * dx + dy * dy)
        	        sum += err
        	    return sum / float(len(p))

        # ========== Start Noise models ========== #
        def usbl_noise(usbl_datapoint): # measurement noise
            # distance = usbl_datapoint.distance_to_ship # lateral_distance,bearing = latlon_to_metres(usbl_datapoint.latitude, usbl_datapoint.longitude, usbl_datapoint.latitude_ship, usbl_datapoint.longitude_ship) 
            # distance = math.sqrt(lateral_distance**2 + usbl_datapoint.depth**2)
            # error = usbl_noise_sigma_factor*(usbl_noise_std_offset + usbl_noise_std_factor*distance) # 5 is for the differential GPS, and the distance std factor 0.01 is used as 0.006 is too sall and unrealistic # This is moved to parse_gaps and parse_usbl_dump
            if usbl_datapoint.northings_std != 0:
                error = usbl_datapoint.northings_std * usbl_noise_sigma_factor
            else:
                usbl_noise_std_offset = 5
                usbl_noise_std_factor = 0.01
                distance = math.sqrt(usbl_datapoint.northings**2 + usbl_datapoint.eastings**2 + usbl_datapoint.depth**2)
                error = usbl_noise_sigma_factor*(usbl_noise_std_offset + usbl_noise_std_factor*distance)
            return error

        def dvl_noise(dvl_imu_datapoint, mode = 'estimate'): # sensor1 noise
            # Vinnay's dvl_noise model: velocity_std = (-0.0125*((velocity)**2)+0.2*(velocity)+0.2125)/100) assuming noise of x_velocity = y_velocity = z_velocity
            velocity_std_factor=0.001 #from catalogue rdi whn1200/600. # should read this in from somewhere else, e.g. json
            velocity_std_offset=0.002 # 0.02 # 0.2 #from catalogue rdi whn1200/600. # should read this in from somewhere else
            x_velocity_std= abs(dvl_imu_datapoint.x_velocity)*velocity_std_factor+velocity_std_offset # (-0.0125*((dvl_imu_datapoint.x_velocity)**2)+0.2*(dvl_imu_datapoint.x_velocity)+0.2125)/100
            y_velocity_std= abs(dvl_imu_datapoint.y_velocity)*velocity_std_factor+velocity_std_offset # (-0.0125*((dvl_imu_datapoint.y_velocity)**2)+0.2*(dvl_imu_datapoint.y_velocity)+0.2125)/100
            z_velocity_std= abs(dvl_imu_datapoint.z_velocity)*velocity_std_factor+velocity_std_offset # (-0.0125*((dvl_imu_datapoint.z_velocity)**2)+0.2*(dvl_imu_datapoint.z_velocity)+0.2125)/100
            if mode == 'estimate':
                x_velocity_estimate = random.gauss(dvl_imu_datapoint.x_velocity, dvl_noise_sigma_factor*x_velocity_std)
                y_velocity_estimate = random.gauss(dvl_imu_datapoint.y_velocity, dvl_noise_sigma_factor*y_velocity_std)
                z_velocity_estimate = random.gauss(dvl_imu_datapoint.z_velocity, dvl_noise_sigma_factor*z_velocity_std)
                return x_velocity_estimate, y_velocity_estimate, z_velocity_estimate
            elif mode == 'std':
                return max([x_velocity_std, y_velocity_std, z_velocity_std])

        def imu_noise(previous_dvlimu_data_point, current_dvlimu_data_point, particle_list_data): #sensor2 noise
            imu_noise = 0.003 * imu_noise_sigma_factor # each time_step + 0.003. assuming noise of roll = pitch = yaw
            if particle_list_data == 0: # for initiation
                roll_estimate = random.gauss(current_dvlimu_data_point.roll, imu_noise)
                pitch_estimate = random.gauss(current_dvlimu_data_point.pitch, imu_noise)
                yaw_estimate = random.gauss(current_dvlimu_data_point.yaw, imu_noise)
            else: # for propagation
                roll_estimate = particle_list_data.roll[-1] + random.gauss(current_dvlimu_data_point.roll-previous_dvlimu_data_point.roll, imu_noise)
                pitch_estimate = particle_list_data.pitch[-1] + random.gauss(current_dvlimu_data_point.pitch - previous_dvlimu_data_point.pitch, imu_noise)
                yaw_estimate = particle_list_data.yaw[-1] + random.gauss(current_dvlimu_data_point.yaw - previous_dvlimu_data_point.yaw, imu_noise)
            if yaw_estimate<0:
                yaw_estimate+=360
            elif yaw_estimate>360:
                yaw_estimate-=360
            return roll_estimate, pitch_estimate, yaw_estimate
        # ========== End Noise models ========== #

        def initialize_particles(N, usbl_datapoint, dvl_imu_datapoint):
            particles = []
            for i in range(N):
                temp_particle = particle()

                roll_estimate, pitch_estimate, yaw_estimate = imu_noise(0, dvl_imu_datapoint, 0)
                x_velocity_estimate, y_velocity_estimate, z_velocity_estimate = dvl_noise(dvl_imu_datapoint)

                usbl_uncertainty = usbl_noise(usbl_datapoint)
                # usbl_uncertainty = 0

                temp_particle.set(random.gauss(dvl_imu_datapoint.eastings, usbl_uncertainty), random.gauss(dvl_imu_datapoint.northings,usbl_uncertainty), dvl_imu_datapoint.timestamp, x_velocity_estimate, y_velocity_estimate, z_velocity_estimate, roll_estimate, pitch_estimate, yaw_estimate, dvl_imu_datapoint.altitude, dvl_imu_datapoint.depth)
                particles.append(temp_particle)
            return (particles)

        def propagate_particles(particles, previous_data_point, current_data_point):
            for i in particles:
                # Propagation error model
                time_difference = current_data_point.timestamp - previous_data_point.timestamp

                roll_estimate, pitch_estimate, yaw_estimate = imu_noise(previous_data_point, current_data_point, i)
                x_velocity_estimate, y_velocity_estimate, z_velocity_estimate = dvl_noise(current_data_point)

                [north_velocity_estimate,east_velocity_estimate,down_velocity_estimate] = body_to_inertial(roll_estimate, pitch_estimate, yaw_estimate, x_velocity_estimate, y_velocity_estimate, z_velocity_estimate)
                [previous_north_velocity_estimate, previous_east_velocity_estimate, previous_down_velocity_estimate] = body_to_inertial(i.roll[-1], i.pitch[-1], i.yaw[-1], i.x_velocity[-1], i.y_velocity[-1], i.z_velocity[-1])
                # DR motion model
                northing_estimate = 0.5*time_difference*(north_velocity_estimate+previous_north_velocity_estimate) + i.northings[-1]
                easting_estimate = 0.5*time_difference*(east_velocity_estimate+previous_east_velocity_estimate) + i.eastings[-1]
                i.set(easting_estimate, northing_estimate, current_data_point.timestamp, x_velocity_estimate, y_velocity_estimate, z_velocity_estimate, roll_estimate, pitch_estimate, yaw_estimate, current_data_point.altitude, current_data_point.depth)

        def measurement_update(N, usbl_measurement, particles_list, resample_flag=True): # updates weights of particles and resamples them # USBL uncertainty follow the readings (0.06/100* depth)! assuming noise of northing = easting
            # particle weighting 
            # measurement_prob(the sensor measurement! i.e. USBL reading). Over here an example is used ... e.g.[31.622776601683793,53.85164807134504, 31.622776601683793, 53.85164807134504] from Z = myrobot.sense()
            # weights = [] 
            for i in particles_list[-1]:
                weight = i.measurement_prob(usbl_measurement, usbl_noise(usbl_measurement))
                # weights.append(weight)
                i.weight.append(weight)

            # normalised_weights=[]
            # for i in weights:
            #     normalised_weights.append(i/sum(weights))
            # for i in range(len(particles_list[-1])):
            #     particles_list[-1][i].weight.append(normalised_weights[i]) # i.weight = weight

            if resample_flag == True:
                # calculate averaged weight
                averaged_weight = []
                for i in particles_list[-1]:
                    averaged_weight.append(sum(i.weight)/len(i.weight))

                # resampling wheel # should i use weights or normalised weights here?
                temp_particles = []
                index = int(random.random()*N)
                beta=0.0

                # mw=max(weights)
                # for i in range(N):
                #     beta += random.random()*2.0*mw
                #     while beta > weights[index]:
                #         beta-= weights[index]
                #         index = (index + 1)%N
                #     temp_particle = particle()
                #     temp_particle.parentID = '{}-{}'.format(len(particles_list)-1,index)
                #     particles_list[-1][index].childIDList.append('{}-{}'.format(len(particles_list),len(temp_particles)))
                #     temp_particle.set(particles_list[-1][index].eastings[-1], particles_list[-1][index].northings[-1], particles_list[-1][index].timestamps[-1], particles_list[-1][index].x_velocity[-1], particles_list[-1][index].y_velocity[-1], particles_list[-1][index].z_velocity[-1], particles_list[-1][index].roll[-1], particles_list[-1][index].pitch[-1], particles_list[-1][index].yaw[-1], particles_list[-1][index].altitude[-1], particles_list[-1][index].depth[-1])
                #     # temp_particle.set_error(usbl_measurement) # maybe can remove this?
                #     temp_particles.append(temp_particle)

                mw=max(averaged_weight)
                for i in range(N):
                    beta += random.random()*2.0*mw
                    while beta > averaged_weight[index]:
                        beta-= averaged_weight[index]
                        index = (index + 1)%N
                    temp_particle = particle()
                    temp_particle.parentID = '{}-{}'.format(len(particles_list)-1,index)
                    particles_list[-1][index].childIDList.append('{}-{}'.format(len(particles_list),len(temp_particles)))
                    temp_particle.set(particles_list[-1][index].eastings[-1], particles_list[-1][index].northings[-1], particles_list[-1][index].timestamps[-1], particles_list[-1][index].x_velocity[-1], particles_list[-1][index].y_velocity[-1], particles_list[-1][index].z_velocity[-1], particles_list[-1][index].roll[-1], particles_list[-1][index].pitch[-1], particles_list[-1][index].yaw[-1], particles_list[-1][index].altitude[-1], particles_list[-1][index].depth[-1])
                    # temp_particle.set_error(usbl_measurement) # maybe can remove this?
                    temp_particles.append(temp_particle)
                return temp_particles
            else:
                return particles_list

        def interpolate_dvl_data(query_timestamp, data_1, data_2):
            temp_data = SyncedOrientationBodyVelocity()
            temp_data.timestamp = query_timestamp
            temp_data.x_velocity = interpolate(query_timestamp, data_1.timestamp, data_2.timestamp, data_1.x_velocity, data_2.x_velocity)
            temp_data.y_velocity = interpolate(query_timestamp, data_1.timestamp, data_2.timestamp, data_1.y_velocity, data_2.y_velocity)
            temp_data.z_velocity = interpolate(query_timestamp, data_1.timestamp, data_2.timestamp, data_1.z_velocity, data_2.z_velocity)
            temp_data.roll = interpolate(query_timestamp, data_1.timestamp, data_2.timestamp, data_1.roll, data_2.roll)
            temp_data.pitch = interpolate(query_timestamp, data_1.timestamp, data_2.timestamp, data_1.pitch, data_2.pitch)
            if abs(data_2.yaw-data_1.yaw)>180:                        
                if data_2.yaw>data_1.yaw:
                    temp_data.yaw=interpolate(query_timestamp,data_1.timestamp,data_2.timestamp,data_1.yaw,data_2.yaw-360)
                    
                else:
                    temp_data.yaw=interpolate(query_timestamp,data_1.timestamp,data_2.timestamp,data_1.yaw-360,data_2.yaw)
                   
                if temp_data.yaw<0:
                    temp_data.yaw+=360
                    
                elif temp_data.yaw>360:
                    temp_data.yaw-=360  

            else:
                temp_data.yaw=interpolate(query_timestamp,data_1.timestamp,data_2.timestamp,data_1.yaw,data_2.yaw)
            return (temp_data)

        def interpolate_usbl_data(query_timestamp, data_1, data_2):
            temp_data = Usbl()
            temp_data.timestamp = query_timestamp
            temp_data.northings = interpolate(query_timestamp, data_1.timestamp, data_2.timestamp, data_1.northings, data_2.northings)
            temp_data.eastings = interpolate(query_timestamp, data_1.timestamp, data_2.timestamp, data_1.eastings, data_2.eastings)
            temp_data.northings_std = interpolate(query_timestamp, data_1.timestamp, data_2.timestamp, data_1.northings_std, data_2.northings_std)
            temp_data.eastings_std = interpolate(query_timestamp, data_1.timestamp, data_2.timestamp, data_1.eastings_std, data_2.eastings_std)
            temp_data.depth = interpolate(query_timestamp, data_1.timestamp, data_2.timestamp, data_1.depth, data_2.depth)
            return (temp_data)

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
            while parentID != '':
                particle_list = int(parentID.split('-')[0])
                element_list = int(parentID.split('-')[1])

                northings_trajectory = particles_list[particle_list][element_list].northings[:-1] + northings_trajectory
                eastings_trajectory = particles_list[particle_list][element_list].eastings[:-1] + eastings_trajectory
                timestamp_list = particles_list[particle_list][element_list].timestamps[:-1] + timestamp_list
                roll_list = particles_list[particle_list][element_list].roll[:-1] + roll_list
                pitch_list = particles_list[particle_list][element_list].pitch[:-1] + pitch_list
                yaw_list = particles_list[particle_list][element_list].yaw[:-1] + yaw_list
                altitude_list = particles_list[particle_list][element_list].altitude[:-1] + altitude_list
                depth_list = particles_list[particle_list][element_list].depth[:-1] + depth_list

                parentID = particles_list[particle_list][element_list].parentID
            return northings_trajectory, eastings_trajectory, timestamp_list, roll_list, pitch_list, yaw_list, altitude_list, depth_list

        particles_list = []
        usbl_datapoints = []

        print ('Initializing particles around first point of dead reckoning solution offset by averaged usbl readings')
        # Interpolate dvl_imu_data to usbl_data to initializing particles at first appropriate usbl timestamp.
        # if dvl_imu_data[dvl_imu_data_index].timestamp > usbl_data[usbl_data_index].timestamp:
        #     while dvl_imu_data[dvl_imu_data_index].timestamp > usbl_data[usbl_data_index].timestamp:
        #         usbl_data_index += 1
        # interpolate usbl_data to dvl_imu_data to initialize particles 
        usbl_data_index = 0
        dvl_imu_data_index = 0
        if usbl_data[usbl_data_index].timestamp > dvl_imu_data[dvl_imu_data_index].timestamp:
            while usbl_data[usbl_data_index].timestamp > dvl_imu_data[dvl_imu_data_index].timestamp:
                dvl_imu_data_index += 1
        if dvl_imu_data[dvl_imu_data_index].timestamp == usbl_data[usbl_data_index].timestamp:
            particles = initialize_particles(N, usbl_data[usbl_data_index], dvl_imu_data[dvl_imu_data_index]) #*For now assume eastings_std = northings_std, usbl_data[usbl_data_index].eastings_std)
            usbl_data_index += 1
            dvl_imu_data_index += 1
        elif usbl_data[usbl_data_index].timestamp < dvl_imu_data[dvl_imu_data_index].timestamp:
            while usbl_data[usbl_data_index+1].timestamp < dvl_imu_data[dvl_imu_data_index].timestamp:
                usbl_data_index += 1
            # interpolated_data = interpolate_data(usbl_data[usbl_data_index].timestamp, dvl_imu_data[dvl_imu_data_index], dvl_imu_data[dvl_imu_data_index+1])
            interpolated_data = interpolate_usbl_data(dvl_imu_data[dvl_imu_data_index].timestamp, usbl_data[usbl_data_index], usbl_data[usbl_data_index+1])
            # dvl_imu_data.insert(dvl_imu_data_index+1, interpolated_data)
            usbl_data.insert(usbl_data_index+1, interpolated_data)
            particles = initialize_particles(N, usbl_data[usbl_data_index], dvl_imu_data[dvl_imu_data_index+1]) #*For now assume eastings_std = northings_std, usbl_data[usbl_data_index].eastings_std)
            usbl_data_index += 1
            dvl_imu_data_index += 1
        else:
            sys.exit("Fatal error. Check dvl_imu_data and usbl_data in particle_filter.py.")
        usbl_datapoints.append(usbl_data[usbl_data_index-1])
        particles_list.append(particles)

        max_uncertainty = 0
        usbl_uncertainty_list = []
        n = 0
        if measurement_update_flag == True: # perform resampling
            last_usbl_flag = False
            while dvl_imu_data[dvl_imu_data_index] != dvl_imu_data[-1]:
                time_difference = dvl_imu_data[dvl_imu_data_index+1].timestamp - dvl_imu_data[dvl_imu_data_index].timestamp
                max_uncertainty += dvl_noise(dvl_imu_data[dvl_imu_data_index], mode = 'std') * time_difference #* 2

                if dvl_imu_data[dvl_imu_data_index+1].timestamp < usbl_data[usbl_data_index].timestamp:
                    propagate_particles(particles_list[-1], dvl_imu_data[dvl_imu_data_index], dvl_imu_data[dvl_imu_data_index+1])
                    dvl_imu_data_index += 1
                else:
                    if last_usbl_flag == False:
                        #interpolate, insert, propagate, resample measurement_update, add new particles to list, check and assign parent id, check parents that have no children and delete it (skip this step for now) ###
                        interpolated_data = interpolate_dvl_data(usbl_data[usbl_data_index].timestamp, dvl_imu_data[dvl_imu_data_index], dvl_imu_data[dvl_imu_data_index+1])
                        dvl_imu_data.insert(dvl_imu_data_index+1, interpolated_data)
                        propagate_particles(particles_list[-1], dvl_imu_data[dvl_imu_data_index], dvl_imu_data[dvl_imu_data_index+1])

                        usbl_uncertainty_list.append(usbl_noise(usbl_data[usbl_data_index]))
                        usbl_avr_uncertainty = sum(usbl_uncertainty_list)/len(usbl_uncertainty_list)

                        if max_uncertainty >= usbl_avr_uncertainty:
                            print ('RESAMPLED! {}'.format(n), max_uncertainty, usbl_noise(usbl_data[usbl_data_index]))
                            n += 1
                            max_uncertainty = 0
                            new_particles = measurement_update(N, usbl_data[usbl_data_index], particles_list)
                            particles_list.append(new_particles)
                            usbl_datapoints.append(usbl_data[usbl_data_index])
                            #reset usbl_uncertainty_list
                            usbl_uncertainty_list = []
                        else:
                            new_particles = measurement_update(N, usbl_data[usbl_data_index], particles_list, resample_flag = False)
                            particles_list = new_particles

                        if usbl_data[usbl_data_index] == usbl_data[-1]:
                            last_usbl_flag = True
                            dvl_imu_data_index += 1
                        else:
                            usbl_data_index += 1
                            dvl_imu_data_index += 1
                    else:
                        propagate_particles(particles_list[-1], dvl_imu_data[dvl_imu_data_index], dvl_imu_data[dvl_imu_data_index+1])
                        dvl_imu_data_index += 1
                    # if usbl_data[usbl_data_index] == usbl_data[-1]:
                    #     break
                    # usbl_data_index += 1
                    # dvl_imu_data_index += 1

            print (max_uncertainty)
            
            ### select particle trajectory with largest overall weight
            particles_weight_list = []
            for i in range(len(particles_list[-1])):
                parentID = particles_list[-1][i].parentID
                particles_weight_list.append([])
                if particles_list[-1][i].averaged_weight == 0:
                    particles_list[-1][i].averaged_weight = sum(particles_list[-1][i].weight)/len(particles_list[-1][i].weight) # this should be smth like particles.calculate_averaged_weight ... self.avrweioght=....
                particles_weight_list[-1].append(particles_list[-1][i].averaged_weight)
                # print (particles_list[-1][i].weight)
                print (particles_list[-1][i].averaged_weight)
                while parentID != '':
                    particle_list = int(parentID.split('-')[0])
                    element_list = int(parentID.split('-')[1])
                    parentID = particles_list[particle_list][element_list].parentID
                    particles_weight_list[-1].append(particles_list[particle_list][element_list].averaged_weight)
            for i in (range(len(particles_weight_list))):
                particles_weight_list[i] = sum(particles_weight_list[i])
                # particles_weight_list[i] = sum(particles_weight_list[i])/len(particles_weight_list[i]) # Normalize again? shouldn't matter...
            selected_particle = particles_list[-1][particles_weight_list.index(max(particles_weight_list))]
            northings_trajectory, eastings_trajectory, timestamp_list, roll_list, pitch_list, yaw_list, altitude_list, depth_list =  extract_trajectory(selected_particle)
        else: # do not perform resampling, only propagate
            while dvl_imu_data[dvl_imu_data_index] != dvl_imu_data[-1]:
                propagate_particles(particles_list[-1], dvl_imu_data[dvl_imu_data_index], dvl_imu_data[dvl_imu_data_index+1])
                dvl_imu_data_index += 1

            ## select particle trajectory with least average error (maybe assign weights without resampling and compare total or average weight? actually doesn't really matter because path won't be used anyway, main purpose of this is to see the std plot)
            particles_error_list = []
            for i in range(len(particles_list[-1])):
                parentID = particles_list[-1][i].parentID
                particles_error_list.append([])
                particles_error_list[-1].append(particles_list[-1][i].error)
                while parentID != '':
                    particle_list = int(parentID.split('-')[0])
                    element_list = int(parentID.split('-')[1])
                    parentID = particles_list[particle_list][element_list].parentID
                    particles_error_list[-1].append(particles_list[particle_list][element_list].error)
            for i in (range(len(particles_error_list))):
                particles_error_list[i] = sum(particles_error_list[i])/len(particles_error_list[i])
            selected_particle = particles_list[-1][particles_error_list.index(min(particles_error_list))]
            northings_trajectory, eastings_trajectory, timestamp_list, roll_list, pitch_list, yaw_list, altitude_list, depth_list =  extract_trajectory(selected_particle)
        
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
                if i != len(particles_list)-1:
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
        # yaw_std step check for different extreme values around 0 and 360. not sure if this method below is robust.
        arr_std_yaw = numpy.std(arr_yaw, axis=0)
        arr_yaw_change = []
        for i in range(len(arr_std_yaw)):
            if arr_std_yaw[i] > 30: # if std is more than 30 deg, means there's two extreme values, so minus 360 for anything above 180 deg.
                arr_yaw_change.append(i)
            # yaw_std.append(i)
        for i in arr_yaw:
            for j in arr_yaw_change:
                if i[j] > 180:
                    i[j] -= 360
        arr_std_yaw = numpy.std(arr_yaw,axis=0)
        for i in arr_std_yaw:
            yaw_std.append(i)
        # numpy.mean(arr, axis=0)
        
        pf_fusion_dvl_list = []
        for i in range(len(timestamp_list)):
            pf_fusion_dvl = SyncedOrientationBodyVelocity()
            pf_fusion_dvl.timestamp = timestamp_list[i]
            pf_fusion_dvl.northings = northings_trajectory[i]
            pf_fusion_dvl.eastings = eastings_trajectory[i]
            pf_fusion_dvl.depth = depth_list[i]
            pf_fusion_dvl.roll = roll_list[i]
            pf_fusion_dvl.pitch = pitch_list[i]
            pf_fusion_dvl.yaw = yaw_list[i]
            pf_fusion_dvl.altitude = altitude_list[i]
            pf_fusion_dvl_list.append(pf_fusion_dvl)

        return pf_fusion_dvl_list, usbl_datapoints, particles_list, northings_std, eastings_std, yaw_std

        # include this later! after each resampling. maybe put it inside particle filter class
        # print (eval(myrobot, particles))
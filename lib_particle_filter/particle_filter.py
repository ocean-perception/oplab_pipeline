# Particle Filter

# Author: Jin Wei Lim
# Date: 27/12/2017

import math
import random
import numpy
import matplotlib.pyplot as plt

import sys

sys.path.append("..")
from lib_calculus.interpolate import interpolate
from lib_coordinates.body_to_inertial import body_to_inertial
from lib_extract import sensor_classes as sens_cls
from lib_particle_filter.particle import particle
from lib_coordinates.latlon_wgs84 import latlon_to_metres

# create an equation for each noise, and def them for sensors in ae2000, or ts1, or ts2. read from mission yaml which sensor used, and automatically pick the one desired.

class particle_filter:
    def __init__(self, usbl_data, dvl_imu_data, N, measurement_update_flag, dvl_noise_factor, imu_noise_factor, usbl_noise_factor):
        return

    def __new__(self, usbl_data, dvl_imu_data, N, measurement_update_flag, dvl_noise_factor, imu_noise_factor, usbl_noise_factor):

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
            # distance = usbl_datapoint.distance # lateral_distance,bearing = latlon_to_metres(usbl_datapoint.latitude, usbl_datapoint.longitude, usbl_datapoint.latitude_ship, usbl_datapoint.longitude_ship) # distance = math.sqrt(lateral_distance**2 + usbl_datapoint.depth**2)
            # error = 5 + 0.01*distance # 5 is for the differential GPS, and the distance std factor 0.01 is used as 0.006 is too sall and unrealistic # This is moved to parse_gaps and parse_usbl_dump
            error = usbl_datapoint.northings_std * usbl_noise_factor
            return error

        def dvl_noise(dvl_imu_datapoint): # sensor1 noise
            # dvl_noise = (-0.0125*((velocity)**2)+0.2*(velocity)+0.2125)/100) assuming noise of x_velocity = y_velocity = z_velocity
            x_velocity_estimate = random.gauss(dvl_imu_datapoint.x_velocity, dvl_noise_factor*(-0.0125*((dvl_imu_datapoint.x_velocity)**2)+0.2*(dvl_imu_datapoint.x_velocity)+0.2125)/100)
            y_velocity_estimate = random.gauss(dvl_imu_datapoint.y_velocity, dvl_noise_factor*(-0.0125*((dvl_imu_datapoint.y_velocity)**2)+0.2*(dvl_imu_datapoint.y_velocity)+0.2125)/100)
            z_velocity_estimate = random.gauss(dvl_imu_datapoint.z_velocity, dvl_noise_factor*(-0.0125*((dvl_imu_datapoint.z_velocity)**2)+0.2*(dvl_imu_datapoint.z_velocity)+0.2125)/100)
            return x_velocity_estimate, y_velocity_estimate, z_velocity_estimate

        def imu_noise(previous_dvlimu_data_point, current_dvlimu_data_point, particle_list_data): #sensor2 noise
            imu_noise = 0.003 * imu_noise_factor # each time_step + 0.003. assuming noise of roll = pitch = yaw
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

                temp_particle.set(random.gauss(usbl_datapoint.eastings, usbl_noise(usbl_datapoint)), random.gauss(usbl_datapoint.northings,usbl_noise(usbl_datapoint)), usbl_datapoint.timestamp, x_velocity_estimate, y_velocity_estimate, z_velocity_estimate, roll_estimate, pitch_estimate, yaw_estimate, dvl_imu_datapoint.altitude, dvl_imu_datapoint.depth)
                temp_particle.set_error(usbl_datapoint)
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

        def measurement_update(N, usbl_measurement, particles_list): # updates weights of particles and resamples them # USBL uncertainty follow the readings (0.06/100* depth)! assuming noise of northing = easting
            # particle weighting 
            # measurement_prob(the sensor measurement! i.e. USBL reading). Over here an example is used ... e.g.[31.622776601683793,53.85164807134504, 31.622776601683793, 53.85164807134504] from Z = myrobot.sense()
            weights = [] 
            for i in particles_list[-1]:
                weight = i.measurement_prob(usbl_measurement, usbl_noise(usbl_measurement))
                weights.append(weight)
                i.weights = weight
            # particle resampling
            temp_particles = []
            index = int(random.random()*N)
            beta=0.0
            mw=max(weights)
            for i in range(N):
                beta += random.random()*2.0*mw
                while beta > weights[index]:
                    beta-= weights[index]
                    index = (index + 1)%N
                temp_particle = particle()
                temp_particle.parentID = '{}-{}'.format(len(particles_list)-1,index)
                particles_list[-1][index].childIDList.append('{}-{}'.format(len(particles_list),len(temp_particles)))
                temp_particle.set(particles_list[-1][index].eastings[-1], particles_list[-1][index].northings[-1], particles_list[-1][index].timestamps[-1], particles_list[-1][index].x_velocity[-1], particles_list[-1][index].y_velocity[-1], particles_list[-1][index].z_velocity[-1], particles_list[-1][index].roll[-1], particles_list[-1][index].pitch[-1], particles_list[-1][index].yaw[-1], particles_list[-1][index].altitude[-1], particles_list[-1][index].depth[-1])
                temp_particle.set_error(usbl_measurement) # maybe can remove this?
                temp_particles.append(temp_particle)
            return (temp_particles)

        def interpolate_data(query_timestamp, data_1, data_2):
            temp_data = sens_cls.synced_orientation_velocity_body()
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

        usbl_data_index = 0
        dvl_imu_data_index = 0

        particles_list = []
        usbl_datapoints = []

        print ('Initializing particles around first USBL reading')
        # Interpolate dvl_imu_data to usbl_data to initializing particles at first appropriate usbl timestamp.
        if dvl_imu_data[dvl_imu_data_index].timestamp > usbl_data[usbl_data_index].timestamp:
            while dvl_imu_data[dvl_imu_data_index].timestamp > usbl_data[usbl_data_index].timestamp:
                usbl_data_index += 1
        if dvl_imu_data[dvl_imu_data_index].timestamp == usbl_data[usbl_data_index].timestamp:
            particles = initialize_particles(N, usbl_data[usbl_data_index], dvl_imu_data[dvl_imu_data_index]) #*For now assume eastings_std = northings_std, usbl_data[usbl_data_index].eastings_std)
            usbl_data_index += 1
            dvl_imu_data_index += 1
        elif dvl_imu_data[dvl_imu_data_index].timestamp < usbl_data[usbl_data_index].timestamp:
            while dvl_imu_data[dvl_imu_data_index+1].timestamp < usbl_data[usbl_data_index].timestamp:
                dvl_imu_data_index += 1    
            interpolated_data = interpolate_data(usbl_data[usbl_data_index].timestamp, dvl_imu_data[dvl_imu_data_index], dvl_imu_data[dvl_imu_data_index+1])
            dvl_imu_data.insert(dvl_imu_data_index+1, interpolated_data)
            particles = initialize_particles(N, usbl_data[usbl_data_index], dvl_imu_data[dvl_imu_data_index+1]) #*For now assume eastings_std = northings_std, usbl_data[usbl_data_index].eastings_std)
            usbl_data_index += 1
            dvl_imu_data_index += 1
        else:
            sys.exit("Fatal error. Check dvl_imu_data and usbl_data in particle_filter.py.")
        usbl_datapoints.append(usbl_data[usbl_data_index-1])
        particles_list.append(particles)

        if measurement_update_flag == True: # perform resampling
            while dvl_imu_data[dvl_imu_data_index] != dvl_imu_data[-1]:
                if dvl_imu_data[dvl_imu_data_index+1].timestamp < usbl_data[usbl_data_index].timestamp:
                    propagate_particles(particles_list[-1], dvl_imu_data[dvl_imu_data_index], dvl_imu_data[dvl_imu_data_index+1])
                    dvl_imu_data_index += 1
                else:
                    #interpolate, insert, propagate, resample measurement_update, add new particles to list, check and assign parent id, check parents that have no children and delete it (skip this step for now) ###
                    interpolated_data = interpolate_data(usbl_data[usbl_data_index].timestamp, dvl_imu_data[dvl_imu_data_index], dvl_imu_data[dvl_imu_data_index+1])
                    dvl_imu_data.insert(dvl_imu_data_index+1, interpolated_data)
                    propagate_particles(particles_list[-1], dvl_imu_data[dvl_imu_data_index], dvl_imu_data[dvl_imu_data_index+1])
                    new_particles = measurement_update(N, usbl_data[usbl_data_index], particles_list)
                    particles_list.append(new_particles)
                    usbl_datapoints.append(usbl_data[usbl_data_index])
                    if usbl_data[usbl_data_index] == usbl_data[-1]:
                        break

                    usbl_data_index += 1
                    dvl_imu_data_index += 1
            
            ### select particle trajectory with largest overall weight
            particles_weight_list = []
            for i in range(len(particles_list[-1])):
                parentID = particles_list[-1][i].parentID
                particles_weight_list.append([])
                particles_weight_list[-1].append(particles_list[-1][i].weight)
                while parentID != '':
                    particle_list = int(parentID.split('-')[0])
                    element_list = int(parentID.split('-')[1])
                    parentID = particles_list[particle_list][element_list].parentID
                    particles_weight_list[-1].append(particles_list[particle_list][element_list].weight)
            for i in (range(len(particles_weight_list))):
                particles_weight_list[i] = sum(particles_weight_list[i])/len(particles_weight_list[i])
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
            pf_fusion_dvl = sens_cls.synced_orientation_velocity_body()
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
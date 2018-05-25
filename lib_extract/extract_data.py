# extract_data

# Assumes filename_camera of 1, 2, and 3 contains the image number between the last 11 and 4 characters for appropriate csv pose estimate files output. e.g. 'Xviii/Cam51707923/0094853.raw' or 'LM165\001\image0001011.tif'

# Scripts to extract data from nav_standard.json, and combined.auv.raw an save csv files and, if plot is True, save plots

# Author: Blair Thornton
# Date: 14/12/2017


# Import librarys
import sys, os, csv
import yaml, json
import shutil, math
import time, codecs
import operator
#import hashlib, glob

import copy

from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from datetime import datetime

sys.path.append("..")
from lib_calculus.interpolate import interpolate
from lib_localisation.dead_reckoning import dead_reckoning
from lib_localisation.usbl_offset import usbl_offset
from lib_coordinates.body_to_inertial import body_to_inertial

from lib_extract import sensor_classes as sens_cls

class extract_data:
    def __init__(self,filepath,ftype,start_time,finish_time,plot,csv_write,show_plot,plotly):

        interpolate_remove_flag = False

        epoch_start_time = 0
        epoch_finish_time = 0

    # velocity body placeholders (DVL)
        velocity_body_list=[]
        velocity_body_sensor_name=''

    # velocity inertial placeholders
        velocity_inertial_list=[]
        velocity_inertial_sensor_name=''
        
    # orientation placeholders (INS)
        orientation_list=[]
        orientation_sensor_name=''

    # depth placeholders
        depth_list=[]
        depth_sensor_name=''

    # altitude placeholders
        altitude_list=[]
        altitude_sensor_name = ''

    # USBL placeholders
        usbl_list=[]
        usbl_sensor_name = ''

    # camera1 placeholders
        camera1_list=[]
        camera1_sensor_name = '' # original serial_camera1
    # camera2 placeholders
        camera2_list=[]
        camera2_sensor_name = ''

    # camera3 placeholders
        camera3_list=[]
        camera3_sensor_name = ''

    # placeholders for interpolated velocity body measurements based on orientation and transformed coordinates
        dead_reckoning_centre_list=[]
        dead_reckoning_dvl_list=[]
        
    # OPLAB
        if ftype == 'oplab':# or (ftype is not 'acfr'):
            outpath=filepath + 'nav'

            filename='nav_standard.json'        
            print('Loading json file ' + outpath + os.sep + filename)
            with open(outpath + os.sep + filename) as nav_standard:                
                parsed_json_data = json.load(nav_standard)

            print('Loading mission.yaml')    
            mission = filepath +'mission.yaml'
            with open(mission,'r') as stream:
                load_data = yaml.load(stream)
            
            # assigns sensor names from mission.yaml instead of json data packet (instead of looking at json data as TunaSand don't have serial yet)
            for i in range(0,len(load_data)): 
                if 'origin' in load_data:
                    origin_flag=1
                    latitude_reference = load_data['origin']['latitude']
                    longitude_reference = load_data['origin']['longitude']
                    coordinate_reference = load_data['origin']['coordinate_reference_system']
                    date = load_data['origin']['date']
                if 'velocity' in load_data:
                    velocity_body_sensor_name = load_data['velocity']['format']
                    velocity_inertial_sensor_name = load_data['velocity']['format']
                if 'orientation' in load_data:
                    orientation_sensor_name = load_data['orientation']['format']
                if 'depth' in load_data:
                    depth_sensor_name = load_data['depth']['format']
                if 'altitude' in load_data:
                    altitude_sensor_name = load_data['altitude']['format']
                if 'usbl' in load_data:
                    usbl_sensor_name = load_data['usbl']['format']
                if 'image' in load_data:
                    if 'camera1' in load_data['image']:
                        camera1_sensor_name = '_'.join(load_data['image']['camera1'].split('/'))
                    if 'camera2' in load_data['image']:
                        camera2_sensor_name = '_'.join(load_data['image']['camera2'].split('/'))
                    if 'camera3' in load_data['image']:
                        camera3_sensor_name = '_'.join(load_data['image']['camera3'].split('/'))
                

        # getting information of sensor position offset from origin/centre reference point
            print('Loading vehicle.yaml')    
            vehicle = filepath +'vehicle.yaml'
            with open(vehicle,'r') as stream:
                load_data = yaml.load(stream)
        
            for i in range(0,len(load_data)):
                if 'origin' in load_data:
                    origin_x_offset = load_data['origin']['x_offset']
                    origin_y_offset = load_data['origin']['y_offset']
                    origin_z_offset = load_data['origin']['z_offset']
                if 'camera1' in load_data:
                    camera1_x_offset = load_data['camera1']['x_offset']
                    camera1_y_offset = load_data['camera1']['y_offset']
                    camera1_z_offset = load_data['camera1']['z_offset']
                if 'camera2' in load_data:
                    camera2_x_offset = load_data['camera2']['x_offset']
                    camera2_y_offset = load_data['camera2']['y_offset']
                    camera2_z_offset = load_data['camera2']['z_offset']
                if 'camera3' in load_data:
                    camera3_x_offset = load_data['camera1']['x_offset']
                    camera3_y_offset = load_data['camera1']['y_offset']
                    camera3_z_offset = load_data['camera1']['z_offset']
                if 'usbl' in load_data:
                    usbl_x_offset = load_data['usbl']['x_offset']
                    usbl_y_offset = load_data['usbl']['y_offset']
                    usbl_z_offset = load_data['usbl']['z_offset']
                if 'dvl' in load_data:
                    dvl_x_offset = load_data['dvl']['x_offset']
                    dvl_y_offset = load_data['dvl']['y_offset']
                    dvl_z_offset = load_data['dvl']['z_offset']
                if 'depth' in load_data:
                    depth_x_offset = load_data['depth']['x_offset']
                    depth_y_offset = load_data['depth']['y_offset']
                    depth_z_offset = load_data['depth']['z_offset']
                if 'ins' in load_data:
                    ins_x_offset = load_data['ins']['x_offset']
                    ins_y_offset = load_data['ins']['y_offset']
                    ins_z_offset = load_data['ins']['z_offset']
            
            yyyy = int(date[0:4])
            mm =  int(date[5:7])
            dd =  int(date[8:10])

            hours = int(start_time[0:2])
            mins = int(start_time[2:4])
            secs = int(start_time[4:6])
                
            dt_obj = datetime(yyyy,mm,dd,hours,mins,secs)       
            time_tuple = dt_obj.timetuple()
            epoch_start_time = time.mktime(time_tuple) 
                
            hours = int(finish_time[0:2])
            mins = int(finish_time[2:4])
            secs = int(finish_time[4:6])        

            dt_obj = datetime(yyyy,mm,dd,hours,mins,secs)
            time_tuple = dt_obj.timetuple()
            epoch_finish_time = time.mktime(time_tuple) 

        # read in data from json file
            # i here is the number of the data packet
            for i in range(len(parsed_json_data)):

                epoch_timestamp=parsed_json_data[i]['epoch_timestamp']

                if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:                                                          
                    if 'velocity' in parsed_json_data[i]['category']:
                        if 'body' in parsed_json_data[i]['frame']:
                            # confirm time stamps of dvl are aligned with main clock (within a second)
                            # if 'epoch_timestamp_dvl' in parsed_json_data[i]:
                            if abs(parsed_json_data[i]['epoch_timestamp']-parsed_json_data[i]['epoch_timestamp_dvl'])<1:
                                velocity_body = sens_cls.velocity_body()
                                velocity_body.timestamp = parsed_json_data[i]['epoch_timestamp_dvl'] # dvl clock not necessarily synced by phins
                                velocity_body.x_velocity = parsed_json_data[i]['data'][0]['x_velocity']
                                velocity_body.y_velocity = parsed_json_data[i]['data'][1]['y_velocity']
                                velocity_body.z_velocity = parsed_json_data[i]['data'][2]['z_velocity']
                                velocity_body.x_velocity_std = parsed_json_data[i]['data'][0]['x_velocity_std']
                                velocity_body.y_velocity_std = parsed_json_data[i]['data'][1]['y_velocity_std']
                                velocity_body.z_velocity_std = parsed_json_data[i]['data'][2]['z_velocity_std']
                                velocity_body_list.append(velocity_body)

                        if 'inertial' in parsed_json_data[i]['frame']:
                            velocity_inertial = sens_cls.velocity_inertial()
                            velocity_inertial.timestamp = parsed_json_data[i]['epoch_timestamp']
                            velocity_inertial.north_velocity = parsed_json_data[i]['data'][0]['north_velocity']
                            velocity_inertial.east_velocity = parsed_json_data[i]['data'][1]['east_velocity']
                            velocity_inertial.down_velocity = parsed_json_data[i]['data'][2]['down_velocity']
                            velocity_inertial.north_velocity_std = parsed_json_data[i]['data'][0]['north_velocity_std']
                            velocity_inertial.east_velocity_std = parsed_json_data[i]['data'][1]['east_velocity_std']
                            velocity_inertial.down_velocity_std = parsed_json_data[i]['data'][2]['down_velocity_std']
                            velocity_inertial_list.append(velocity_inertial)
                    
                    if 'orientation' in parsed_json_data[i]['category']:
                        orientation = sens_cls.orientation()
                        orientation.timestamp = parsed_json_data[i]['epoch_timestamp']
                        orientation.roll = parsed_json_data[i]['data'][1]['roll']
                        orientation.pitch = parsed_json_data[i]['data'][2]['pitch']
                        orientation.yaw = parsed_json_data[i]['data'][0]['heading']
                        orientation.roll_std = parsed_json_data[i]['data'][1]['roll_std']
                        orientation.pitch_std = parsed_json_data[i]['data'][2]['pitch_std']
                        orientation.yaw_std = parsed_json_data[i]['data'][0]['heading_std']
                        orientation_list.append(orientation)

                    if 'depth' in parsed_json_data[i]['category']:
                        depth = sens_cls.depth()
                        depth.timestamp = parsed_json_data[i]['epoch_timestamp_depth']
                        depth.depth = parsed_json_data[i]['data'][0]['depth']
                        depth.depth_std = parsed_json_data[i]['data'][0]['depth_std']
                        depth_list.append(depth)

                    if 'altitude' in parsed_json_data[i]['category']:
                        altitude = sens_cls.altitude()
                        altitude.timestamp = parsed_json_data[i]['epoch_timestamp']
                        altitude.altitude = parsed_json_data[i]['data'][0]['altitude']
                        altitude_list.append(altitude)

                    if 'usbl' in parsed_json_data[i]['category']:
                        usbl = sens_cls.usbl()
                        usbl.timestamp = parsed_json_data[i]['epoch_timestamp']
                        usbl.latitude = parsed_json_data[i]['data_target'][0]['latitude']
                        usbl.longitude = parsed_json_data[i]['data_target'][1]['longitude']
                        usbl.northings = parsed_json_data[i]['data_target'][2]['northings']
                        usbl.eastings = parsed_json_data[i]['data_target'][3]['eastings']
                        usbl.depth = parsed_json_data[i]['data_target'][4]['depth']
                        usbl.latitude_std = parsed_json_data[i]['data_target'][0]['latitude_std']
                        usbl.longitude_std = parsed_json_data[i]['data_target'][1]['longitude_std']
                        usbl.northings_std = parsed_json_data[i]['data_target'][2]['northings_std']
                        usbl.eastings_std = parsed_json_data[i]['data_target'][3]['eastings_std']
                        usbl_list.append(usbl)

                    if 'image' in parsed_json_data[i]['category']:
                        camera1 = sens_cls.camera()
                        camera1.timestamp = parsed_json_data[i]['camera1'][0]['epoch_timestamp']#LC
                        camera1.filename = parsed_json_data[i]['camera1'][0]['filename']
                        camera1_list.append(camera1)
                        camera2 = sens_cls.camera()
                        camera2.timestamp = parsed_json_data[i]['camera2'][0]['epoch_timestamp']
                        camera2.filename = parsed_json_data[i]['camera2'][0]['filename']
                        camera2_list.append(camera2)

                    if 'laser' in parsed_json_data[i]['category']:
                        camera3 = sens_cls.camera()
                        camera3.timestamp = parsed_json_data[i]['epoch_timestamp']
                        camera3.filename = parsed_json_data[i]['filename']
                        camera3_list.append(camera3)


        # make path for csv and plots
            renavpath = filepath + 'json_renav_' + str(yyyy).zfill(4) + str(mm).zfill(2) + str(dd).zfill(2) + '_' + start_time + '_' + finish_time 
            if os.path.isdir(renavpath) == 0:
                try:
                    os.mkdir(renavpath)
                except Exception as e:
                    print("Warning:",e)

            
            print('Complete parse of:' + outpath + os.sep + filename)
            print('Writing outputs to: ' + renavpath)
        
    # ACFR
        if ftype == 'acfr':# or (ftype is not 'acfr'):
            
            print('Loading mission.cfg')    
            mission = filepath +'mission.cfg'
            with codecs.open(mission,'r',encoding='utf-8', errors='ignore') as filein:
                for line in filein.readlines():             
                    line_split = line.strip().split(' ')
                    if str(line_split[0]) == 'MAG_VAR_LAT':
                        latitude_reference = str(line_split[1])                  
                    if str(line_split[0]) == 'MAG_VAR_LNG':
                        longitude_reference = str(line_split[1])                
                    if str(line_split[0]) == 'MAG_VAR_DATE':
                        date = str(line_split[1])                

            # # setup the time window
            yyyy = int(date[1:5])
            mm =  int(date[6:8])
            dd =  int(date[9:11])

            hours = int(start_time[0:2])
            mins = int(start_time[2:4])
            secs = int(start_time[4:6])
                
            dt_obj = datetime(yyyy,mm,dd,hours,mins,secs)       
            time_tuple = dt_obj.timetuple()
            epoch_start_time = time.mktime(time_tuple) 
                
            hours = int(finish_time[0:2])
            mins = int(finish_time[2:4])
            secs = int(finish_time[4:6])        

            dt_obj = datetime(yyyy,mm,dd,hours,mins,secs)
            time_tuple = dt_obj.timetuple()
            epoch_finish_time = time.mktime(time_tuple) 

            outpath=filepath + 'dRAWLOGS_cv'

            filename='combined.RAW.auv'        
            print('Loading acfr standard RAW.auv file ' + outpath + os.sep + filename)

            with codecs.open(outpath + os.sep + filename,'r',encoding='utf-8', errors='ignore') as filein:
                for line in filein.readlines():
                    line_split = line.split(' ')

                    if str(line_split[0]) == 'RDI:':
                        
                        epoch_timestamp=float(line_split[1])
                        if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:                                      

                            velocity_body = sens_cls.velocity_body()
                            velocity_body.timestamp = float(line_split[1])
                            altitude = sens_cls.altitude()
                            altitude.timestamp = float(line_split[1])

                            for i in range(len(line_split)):
                                value=line_split[i].split(':')
                                if value[0] == 'alt':
                                    altitude.altitude = float(value[1])
                                    altitude.seafloor_depth = 0
                                if value[0] == 'vx':
                                    velocity_body.x_velocity = float(value[1])
                                if value[0] == 'vy':
                                    velocity_body.y_velocity = float(value[1])
                                if value[0] == 'vz':
                                    velocity_body.z_velocity = float(value[1])
                            velocity_body_list.append(velocity_body)
                            altitude_list.append(altitude)

                    if str(line_split[0]) == 'PHINS_COMPASS:':
                        
                        epoch_timestamp=float(line_split[1])
                        if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:                                      

                            velocity_inertial = sens_cls.velocity_inertial()
                            velocity_inertial.timestamp = float(line_split[1])
                            orientation = sens_cls.orientation()
                            orientation.timestamp = float(line_split[1])

                            for i in range(len(line_split)-1):
                                
                                if line_split[i] == 'r:':
                                    orientation.roll = float(line_split[i+1])
                                if line_split[i] == 'p:':
                                    orientation.pitch = float(line_split[i+1])
                                if line_split[i] == 'h:':
                                    orientation.yaw = float(line_split[i+1])

                            velocity_inertial_list.append(velocity_inertial)
                            orientation_list.append(orientation)


                    if str(line_split[0]) == 'PAROSCI:':

                        epoch_timestamp=float(line_split[1])
                        if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:                                      

                            depth = sens_cls.depth()
                            depth.timestamp = float(line_split[1])
                            depth.depth = float(line_split[2])
                            depth_list.append(depth)

                    if str(line_split[0]) == 'SSBL_FIX:':
                        
                        epoch_timestamp=float(line_split[1])
                        
                        if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:                                      

                            usbl = sens_cls.usbl()
                            usbl.timestamp = float(line_split[1])
                        
                            for i in range(len(line_split)-1):
                                
                                if line_split[i] == 'target_x:':
                                    usbl.northings = float(line_split[i+1])
                                if line_split[i] == 'target_y:':
                                    usbl.eastings = float(line_split[i+1])
                                if line_split[i] == 'target_z:':
                                    usbl.depth = float(line_split[i+1])

                            usbl_list.append(usbl)

            # make folder to store csv and plots
            renavpath = filepath + 'acfr_renav_' + str(yyyy).zfill(4) + str(mm).zfill(2) + str(dd).zfill(2) + '_' + start_time + '_' + finish_time 
            if os.path.isdir(renavpath) == 0:
                try:
                    os.mkdir(renavpath)
                except Exception as e:
                    print("Warning:",e)

            print('Complete parse of:' + outpath + os.sep + filename)
            print('Writing outputs to: ' + renavpath)


        
    # interpolate to find the appropriate depth to compute seafloor depth for each altitude measurement
        j=0
        for i in range(len(altitude_list)):        
            while j < len(depth_list)-1 and depth_list[j].timestamp<altitude_list[i].timestamp:
                j=j+1

            if j>=1:                
                altitude_list[i].seafloor_depth=interpolate(altitude_list[i].timestamp,depth_list[j-1].timestamp,depth_list[j].timestamp,depth_list[j-1].depth,depth_list[j].depth)+altitude_list[i].altitude

    # perform coordinate transformations and interpolations of state data to velocity_body time stamps with sensor position offset
        #Assumes the first measurement of velocity_body is the beginning of mission. May not be robust to non-continuous measurements..any (sudden start and stop) will affect it?
        # 
        j=0        
        k=0
        n=0
        start_interpolate_index = 0

        while orientation_list[start_interpolate_index].timestamp<velocity_body_list[0].timestamp:
            start_interpolate_index += 1

        # if start_interpolate_index==0:
        # do something? because time_orientation may be way before time_velocity_body

        if start_interpolate_index==1:
            interpolate_remove_flag = True

        for i in range(start_interpolate_index, len(orientation_list)):#time_velocity_body)):

            # interpolate to find the appropriate dvl time for the orientation measurements
            if orientation_list[i].timestamp>velocity_body_list[-1].timestamp:
                break

            while j<len(velocity_body_list)-1 and orientation_list[i].timestamp>velocity_body_list[j+1].timestamp:
                j += 1

            dead_reckoning_dvl = sens_cls.synced_orientation_velocity_body()
            dead_reckoning_dvl.timestamp = orientation_list[i].timestamp
            dead_reckoning_dvl.roll = orientation_list[i].roll
            dead_reckoning_dvl.pitch = orientation_list[i].pitch
            dead_reckoning_dvl.yaw = orientation_list[i].yaw
            dead_reckoning_dvl.x_velocity = interpolate(orientation_list[i].timestamp,velocity_body_list[j].timestamp,velocity_body_list[j+1].timestamp,velocity_body_list[j].x_velocity,velocity_body_list[j+1].x_velocity)
            dead_reckoning_dvl.y_velocity = interpolate(orientation_list[i].timestamp,velocity_body_list[j].timestamp,velocity_body_list[j+1].timestamp,velocity_body_list[j].y_velocity,velocity_body_list[j+1].y_velocity)
            dead_reckoning_dvl.z_velocity = interpolate(orientation_list[i].timestamp,velocity_body_list[j].timestamp,velocity_body_list[j+1].timestamp,velocity_body_list[j].z_velocity,velocity_body_list[j+1].z_velocity)

            [x_offset,y_offset,z_offset] = body_to_inertial(orientation_list[i].roll, orientation_list[i].pitch, orientation_list[i].yaw, dead_reckoning_dvl.x_velocity, dead_reckoning_dvl.y_velocity, dead_reckoning_dvl.z_velocity)

            dead_reckoning_dvl.north_velocity = x_offset
            dead_reckoning_dvl.east_velocity = y_offset
            dead_reckoning_dvl.down_velocity = z_offset

            while n<len(altitude_list)-1 and orientation_list[i].timestamp>altitude_list[n+1].timestamp:
                n += 1
            dead_reckoning_dvl.altitude = interpolate(orientation_list[i].timestamp,velocity_body_list[n].timestamp,velocity_body_list[n+1].timestamp,altitude_list[n].altitude,altitude_list[n+1].altitude)

            while k < len(depth_list)-1 and depth_list[k].timestamp<orientation_list[i].timestamp:
                k+= 1
            # interpolate to find the appropriate depth for dead_reckoning
            dead_reckoning_dvl.depth = interpolate(orientation_list[i].timestamp,depth_list[k-1].timestamp,depth_list[k].timestamp,depth_list[k-1].depth,depth_list[k].depth)

            dead_reckoning_dvl_list.append(dead_reckoning_dvl)

        # northings eastings dead reckoning solution
        for i in range(len(dead_reckoning_dvl_list)):
            # dead reckoning solution
            if i>=1:
                [dead_reckoning_dvl_list[i].northings_dr, dead_reckoning_dvl_list[i].eastings_dr]=dead_reckoning(dead_reckoning_dvl_list[i].timestamp, dead_reckoning_dvl_list[i-1].timestamp, dead_reckoning_dvl_list[i].north_velocity, dead_reckoning_dvl_list[i-1].north_velocity, dead_reckoning_dvl_list[i].east_velocity, dead_reckoning_dvl_list[i-1].east_velocity, dead_reckoning_dvl_list[i-1].northings_dr, dead_reckoning_dvl_list[i-1].eastings_dr)

        # offset sensor to plot origin/centre of vehicle
        dead_reckoning_centre_list = copy.deepcopy(dead_reckoning_dvl_list) #[:] #.copy()
        for i in range(len(dead_reckoning_centre_list)):
            [x_offset, y_offset, z_offset] = body_to_inertial(dead_reckoning_centre_list[i].roll, dead_reckoning_centre_list[i].pitch, dead_reckoning_centre_list[i].yaw, origin_x_offset - dvl_x_offset, origin_y_offset - dvl_y_offset, origin_z_offset - depth_z_offset)
            dead_reckoning_centre_list[i].northings_dr += x_offset
            dead_reckoning_centre_list[i].eastings_dr += y_offset
            dead_reckoning_centre_list[i].depth += z_offset

        #remove first term if first time_orientation is < velocity_body time
        if interpolate_remove_flag == True:

            # del time_orientation[0]
            del dead_reckoning_centre_list[0]
            del dead_reckoning_dvl_list[0]
            interpolate_remove_flag = False # reset flag
        print('Complete interpolation and coordinate transfomations for velocity_body')
    
    # perform interpolations of state data to velocity_inertial time stamps (without sensor offset and correct imu to dvl flipped interpolation)
        #initialise counters for interpolation
        j=0
        k=0
    
        for i in range(len(velocity_inertial_list)):
                           
            while j< len(orientation_list)-1 and orientation_list[j].timestamp<velocity_inertial_list[i].timestamp:
                j=j+1
            
            if j==1:
                interpolate_remove_flag = True
            else:
                velocity_inertial_list[i].roll_interpolated=interpolate(velocity_inertial_list[i].timestamp,orientation_list[j-1].timestamp,orientation_list[j].timestamp,orientation_list[j-1].roll,orientation_list[j].roll)
                velocity_inertial_list[i].pitch_interpolated=interpolate(velocity_inertial_list[i].timestamp,orientation_list[j-1].timestamp,orientation_list[j].timestamp,orientation_list[j-1].pitch,orientation_list[j].pitch)

                if abs(orientation_list[j].yaw-orientation_list[j-1].yaw)>180:                        
                    if orientation_list[j].yaw>orientation_list[j-1].yaw:
                        velocity_inertial_list[i].yaw_interpolated=interpolate(velocity_inertial_list[i].timestamp,orientation_list[j-1].timestamp,orientation_list[j].timestamp,orientation_list[j-1].yaw,orientation_list[j].yaw-360)
                        
                    else:
                        velocity_inertial_list[i].interpolated=interpolate(velocity_inertial_list[i].timestamp,orientation_list[j-1].timestamp,orientation_list[j].timestamp,orientation_list[j-1].yaw-360,orientation_list[j].yaw)
                       
                    if velocity_inertial_list[i].yaw_interpolated<0:
                        velocity_inertial_list[i].yaw_interpolated+=360
                        
                    elif velocity_inertial_list[i].yaw_interpolated>360:
                        velocity_inertial_list[i].yaw_interpolated-=360  

                else:
                    velocity_inertial_list[i].yaw_interpolated=interpolate(velocity_inertial_list[i].timestamp,orientation_list[j-1].timestamp,orientation_list[j].timestamp,orientation_list[j-1].yaw,orientation_list[j].yaw)
            
            while k< len(depth_list)-1 and depth_list[k].timestamp<velocity_inertial_list[i].timestamp:
                k=k+1

            if k>=1:                
                velocity_inertial_list[i].depth_dr=interpolate(velocity_inertial_list[i].timestamp,depth_list[k-1].timestamp,depth_list[k].timestamp,depth_list[k-1].depth,depth_list[k].depth) # depth_dr directly interpolated from depth sensor
        
        for i in range(len(velocity_inertial_list)):
            if i >= 1:                     
                [velocity_inertial_list[i].northings_dr, velocity_inertial_list[i].eastings_dr]=dead_reckoning(velocity_inertial_list[i].timestamp, velocity_inertial_list[i-1].timestamp, velocity_inertial_list[i].north_velocity, velocity_inertial_list[i-1].north_velocity, velocity_inertial_list[i].east_velocity, velocity_inertial_list[i-1].east_velocity, velocity_inertial_list[i-1].northings_dr, velocity_inertial_list[i-1].eastings_dr)

        if interpolate_remove_flag == True:
            del velocity_inertial_list[0]
            interpolate_remove_flag = False # reset flag
        print('Complete interpolation and coordinate transfomations for velocity_inertial')


    # offset velocity body DR by initial usbl estimate
        # compare time_dead_reckoning and time_usbl
        # find initial position offset
        [northings_usbl_interpolated,eastings_usbl_interpolated] = usbl_offset([i.timestamp for i in dead_reckoning_centre_list],[i.northings_dr for i in dead_reckoning_centre_list],[i.eastings_dr for i in dead_reckoning_centre_list],[i.timestamp for i in usbl_list],[i.northings for i in usbl_list],[i.eastings for i in usbl_list])
        # offset the deadreackoning by this initial estimate
        for i in range(len(dead_reckoning_centre_list)):                 
            dead_reckoning_centre_list[i].northings_dr+=northings_usbl_interpolated
            dead_reckoning_centre_list[i].eastings_dr+=eastings_usbl_interpolated
        for i in range(len(dead_reckoning_dvl_list)):
            dead_reckoning_dvl_list[i].northings_dr+=northings_usbl_interpolated
            dead_reckoning_dvl_list[i].eastings_dr+=eastings_usbl_interpolated

    # offset velocity inertial DR by initial usbl estimate
        # compare time_velocity_inertia and time_usbl
        # find initial position offset
        [northings_usbl_interpolated,eastings_usbl_interpolated] = usbl_offset([i.timestamp for i in velocity_inertial_list],[i.northings_dr for i in velocity_inertial_list],[i.eastings_dr for i in velocity_inertial_list],[i.timestamp for i in usbl_list],[i.northings for i in usbl_list],[i.eastings for i in usbl_list])
        # offset the deadreackoning by this initial estimate
        for i in range(len(velocity_inertial_list)):                
            velocity_inertial_list[i].northings_dr+=northings_usbl_interpolated
            velocity_inertial_list[i].eastings_dr+=eastings_usbl_interpolated        

    # perform interpolations of state data to camera{1/2/3} time stamps
        def camera_setup(camera_list, camera_name, camera_offsets):
            j=0
            n=0
            if camera_list[0].timestamp>dead_reckoning_centre_list[-1].timestamp or camera_list[-1].timestamp<dead_reckoning_centre_list[0].timestamp: #Check if camera activates before dvl and orientation sensors.
                print('{} timestamps does not overlap with dead reckoning data, check timestamp_history.pdf via -v option.'.format(camera_name))
            else:
                camera_overlap_flag = 0
                for i in range(len(camera_list)):
                    if camera_list[i].timestamp<dead_reckoning_centre_list[0].timestamp:
                        camera_overlap_flag = 1
                        pass
                    else:
                        del camera_list[:i]
                        break
                for i in range(len(camera_list)):
                    if j>=len(dead_reckoning_centre_list)-1:
                        del camera_list[i:]
                        camera_overlap_flag = 1
                        break
                    while dead_reckoning_centre_list[j].timestamp < camera_list[i].timestamp:
                        if j+1>len(dead_reckoning_centre_list)-1 or dead_reckoning_centre_list[j+1].timestamp>camera_list[-1].timestamp:
                            break
                        j += 1
                    #if j>=1: ?
                    camera_list[i].roll_interpolated = interpolate(camera_list[i].timestamp,dead_reckoning_centre_list[j-1].timestamp,dead_reckoning_centre_list[j].timestamp,dead_reckoning_centre_list[j-1].roll,dead_reckoning_centre_list[j].roll)
                    camera_list[i].pitch_interpolated = interpolate(camera_list[i].timestamp,dead_reckoning_centre_list[j-1].timestamp,dead_reckoning_centre_list[j].timestamp,dead_reckoning_centre_list[j-1].pitch,dead_reckoning_centre_list[j].pitch)
                    if abs(dead_reckoning_centre_list[j].yaw-dead_reckoning_centre_list[j-1].yaw)>180:
                        if dead_reckoning_centre_list[j].yaw>dead_reckoning_centre_list[j-1].yaw:
                            camera_list[i].yaw_interpolated = interpolate(camera_list[i].timestamp,dead_reckoning_centre_list[j-1].timestamp,dead_reckoning_centre_list[j].timestamp,dead_reckoning_centre_list[j-1].yaw,dead_reckoning_centre_list[j].yaw-360)                       
                        else:
                            camera_list[i].yaw_interpolated = interpolate(camera_list[i].timestamp,dead_reckoning_centre_list[j-1].timestamp,dead_reckoning_centre_list[j].timestamp,dead_reckoning_centre_list[j-1].yaw-360,dead_reckoning_centre_list[j].yaw)
                        if camera_list[i].yaw_interpolated<0:
                            camera_list[i].yaw_interpolated+=360
                        elif camera_list[i].yaw_interpolated>360:
                            camera_list[i].yaw_interpolated-=360  
                    else:
                        camera_list[i].yaw_interpolated = interpolate(camera_list[i].timestamp,dead_reckoning_centre_list[j-1].timestamp,dead_reckoning_centre_list[j].timestamp,dead_reckoning_centre_list[j-1].yaw,dead_reckoning_centre_list[j].yaw)
                    
                    # while n<len(time_altitude)-1 and time_altitude[n+1]<time_camera1[i]:
                    #     n += 1
                    # camera1_altitude.append(interpolate(time_camera1[i],time_altitude[n],time_altitude[n+1],altitude[n],altitude[n+1]))
                    camera_list[i].altitude_interpolated = interpolate(camera_list[i].timestamp,dead_reckoning_centre_list[j-1].timestamp,dead_reckoning_centre_list[j].timestamp,dead_reckoning_centre_list[j-1].altitude,dead_reckoning_centre_list[j].altitude)

                    [x_offset,y_offset,z_offset] = body_to_inertial(camera_list[i].roll_interpolated,camera_list[i].pitch_interpolated,camera_list[i].yaw_interpolated, origin_x_offset - camera_offsets[0], origin_y_offset - camera_offsets[1], origin_z_offset - camera_offsets[2])
                    
                    camera_list[i].northings_dr = interpolate(camera_list[i].timestamp,dead_reckoning_centre_list[j-1].timestamp,dead_reckoning_centre_list[j].timestamp,dead_reckoning_centre_list[j-1].northings_dr,dead_reckoning_centre_list[j].northings_dr)-x_offset
                    camera_list[i].eastings_dr = interpolate(camera_list[i].timestamp,dead_reckoning_centre_list[j-1].timestamp,dead_reckoning_centre_list[j].timestamp,dead_reckoning_centre_list[j-1].eastings_dr,dead_reckoning_centre_list[j].eastings_dr)-y_offset
                    camera_list[i].depth_dr = interpolate(camera_list[i].timestamp,dead_reckoning_centre_list[j-1].timestamp,dead_reckoning_centre_list[j].timestamp,dead_reckoning_centre_list[j-1].depth,dead_reckoning_centre_list[j].depth)-z_offset
                if camera_overlap_flag == 1:
                    print('{} data more than dead reckoning data. Only processed overlapping data and ignored the rest.'.format(camera_name))
                print('Complete interpolation and coordinate transfomations for {}'.format(camera_name))
        if len(camera1_list) > 1:
            camera_setup(camera1_list, camera1_sensor_name, [camera1_x_offset,camera1_y_offset,camera1_z_offset])
        if len(camera2_list) > 1:
            camera_setup(camera2_list, camera2_sensor_name, [camera2_x_offset,camera2_y_offset,camera2_z_offset])
        if len(camera3_list) > 1:
            camera_setup(camera3_list, camera3_sensor_name, [camera3_x_offset,camera3_y_offset,camera3_z_offset])

    # write values out to a csv file
        # create a directory with the time stamp
        if csv_write is True:

            csvpath = renavpath + os.sep + 'csv'

            if os.path.isdir(csvpath) == 0:
                try:
                    os.mkdir(csvpath)
                except Exception as e:
                    print("Warning:",e)

            # Useful for plotly+dash+pandas
            if len(dead_reckoning_centre_list) > 1:
                print("Writing outputs to auv_centre.csv ...")
                with open(csvpath + os.sep + 'auv_centre.csv' ,'w') as fileout:
                    fileout.write('Timestamp, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m]\n')
                for i in range(len(dead_reckoning_centre_list)):
                    with open(csvpath + os.sep + 'auv_centre.csv' ,'a') as fileout:
                        try:
                            fileout.write(str(dead_reckoning_centre_list[i].timestamp)+','+str(dead_reckoning_centre_list[i].northings_dr)+','+str(dead_reckoning_centre_list[i].eastings_dr)+','+str(dead_reckoning_centre_list[i].depth)+','+str(dead_reckoning_centre_list[i].roll)+','+str(dead_reckoning_centre_list[i].pitch)+','+str(dead_reckoning_centre_list[i].yaw)+','+str(dead_reckoning_centre_list[i].altitude)+'\n')
                            fileout.close()
                        except IndexError:
                            break

            if len(usbl_list) > 1:
                print("Writing outputs to auv_usbl.csv ...")
                with open(csvpath + os.sep + 'auv_usbl.csv' ,'w') as fileout:
                    fileout.write('Timestamp, Northing [m], Easting [m], Depth [m]\n')
                for i in range(len(usbl_list)):
                    with open(csvpath + os.sep + 'auv_usbl.csv' ,'a') as fileout:
                        try:
                            fileout.write(str(usbl_list[i].timestamp)+','+str(usbl_list[i].northings)+','+str(usbl_list[i].eastings)+','+str(usbl_list[i].depth)+'\n')
                            fileout.close()
                        except IndexError:
                            break

            if len(velocity_inertial_list) > 1:
                print("Writing outputs to auv_{}.csv ...".format(velocity_inertial_sensor_name))
                with open(csvpath + os.sep + 'auv_{}.csv'.format(velocity_inertial_sensor_name), 'w') as fileout:
                    fileout.write('Timestamp, Northing [m], Easting [m], Depth [m]\n')
                for i in range(len(velocity_inertial_list)):
                    with open(csvpath + os.sep + 'auv_{}.csv'.format(velocity_inertial_sensor_name) ,'a') as fileout:
                        try:
                            fileout.write(str(velocity_inertial_list[i].timestamp)+','+str(velocity_inertial_list[i].northings_dr)+','+str(velocity_inertial_list[i].eastings_dr)+','+str(velocity_inertial_list[i].depth_dr)+'\n')
                            fileout.close()
                        except IndexError:
                            break

            if len(dead_reckoning_dvl_list) > 1:
                print("Writing outputs to auv_dvl.csv ...")
                with open(csvpath + os.sep + 'auv_dvl.csv' ,'w') as fileout:
                    fileout.write('Timestamp, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m]\n')
                for i in range(len(dead_reckoning_dvl_list)):
                    with open(csvpath + os.sep + 'auv_dvl.csv' ,'a') as fileout:
                        try:
                            fileout.write(str(dead_reckoning_dvl_list[i].timestamp)+','+str(dead_reckoning_dvl_list[i].northings_dr)+','+str(dead_reckoning_dvl_list[i].eastings_dr)+','+str(dead_reckoning_dvl_list[i].depth)+str(dead_reckoning_dvl_list[i].roll)+','+str(dead_reckoning_dvl_list[i].pitch)+','+str(dead_reckoning_dvl_list[i].yaw)+','+str(dead_reckoning_dvl_list[i].altitude)+'\n')
                            fileout.close()
                        except IndexError:
                            break
### First column of csv file - image file naming step not very robust, needs improvement
            #*** maybe add timestamp at the last column of image.csv
            def camera_csv(camera_list, camera_name):
                if len(camera_list) > 1:
                    print("Writing outputs to {}.csv ...".format(camera_name))
                    with open(csvpath + os.sep + '{}.csv'.format(camera_name) ,'w') as fileout:
                        fileout.write('Imagenumber, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m]\n')
                    for i in range(len(camera_list)):
                        with open(csvpath + os.sep + '{}.csv'.format(camera_name) ,'a') as fileout:
                            try:
                                imagenumber = camera_list[i].filename[-11:-4]
                                if imagenumber.isdigit():
                                    image_filename=int(imagenumber)
                                else:
                                    image_filename=camera_list[i].filename
                                fileout.write(str(image_filename)+','+str(camera_list[i].northings_dr)+','+str(camera_list[i].eastings_dr)+','+str(camera_list[i].depth_dr)+','+str(camera_list[i].roll_interpolated)+','+str(camera_list[i].pitch_interpolated)+','+str(camera_list[i].yaw_interpolated)+','+str(camera_list[i].altitude_interpolated)+'\n')
                                fileout.close()
                            except IndexError:
                                break
            camera_csv(camera1_list, camera1_sensor_name)
            camera_csv(camera2_list, camera2_sensor_name)
            camera_csv(camera3_list, camera3_sensor_name)

            print('Complete extraction of data: ', csvpath)
        
    # plot data
        if plot is True:
            print('Plotting data ...')
            plotpath = renavpath + os.sep + 'plots'
            
            if os.path.isdir(plotpath) == 0:
                try:
                    os.mkdir(plotpath)
                except Exception as e:
                    print("Warning:",e)

        # orientation
            print('...plotting orientation_vs_time...')
            
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot([i.timestamp for i in orientation_list], [i.yaw for i in orientation_list],'r.',label='Yaw')   
            ax.plot([i.timestamp for i in orientation_list], [i.roll for i in orientation_list],'b.',label='Roll')      
            ax.plot([i.timestamp for i in orientation_list], [i.pitch for i in orientation_list],'g.',label='Pitch')                     
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Degrees')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'orientation_vs_time.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # velocity_body (north,east,down) compared to velocity_inertial
            print('...plotting velocity_vs_time...')
            
            fig=plt.figure(figsize=(10,7))
            ax1 = fig.add_subplot(321)            
            ax1.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.north_velocity for i in dead_reckoning_dvl_list], 'ro',label='DVL')#time_velocity_body,north_velocity_inertia_dvl, 'ro',label='DVL')
            ax1.plot([i.timestamp for i in velocity_inertial_list],[i.north_velocity for i in velocity_inertial_list], 'b.',label=velocity_inertial_sensor_name)
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Velocity, m/s')
            ax1.legend()
            ax1.grid(True)
            ax1.set_title('north velocity')
            ax2 = fig.add_subplot(323)            
            ax2.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.east_velocity for i in dead_reckoning_dvl_list],'ro',label='DVL')
            ax2.plot([i.timestamp for i in velocity_inertial_list],[i.east_velocity for i in velocity_inertial_list],'b.',label=velocity_inertial_sensor_name)
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Velocity, m/s')
            ax2.legend()
            ax2.grid(True)
            ax2.set_title('east velocity')
            ax3 = fig.add_subplot(325)            
            ax3.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.down_velocity for i in dead_reckoning_dvl_list],'ro',label='DVL')#time_velocity_body,down_velocity_inertia_dvl,'r.',label='DVL')
            ax3.plot([i.timestamp for i in velocity_inertial_list],[i.down_velocity for i in velocity_inertial_list],'b.',label=velocity_inertial_sensor_name)
            ax3.set_xlabel('Epoch time, s')
            ax3.set_ylabel('Velocity, m/s')
            ax3.legend()
            ax3.grid(True)
            ax3.set_title('down velocity')
            ax4 = fig.add_subplot(322)
            ax4.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.x_velocity for i in dead_reckoning_dvl_list], 'r.',label='Surge') #time_velocity_body,x_velocity, 'r.',label='Surge')
            ax4.set_xlabel('Epoch time, s')
            ax4.set_ylabel('Velocity, m/s')
            ax4.legend()
            ax4.grid(True)
            ax4.set_title('x velocity')
            ax5 = fig.add_subplot(324)
            ax5.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.y_velocity for i in dead_reckoning_dvl_list], 'g.',label='Sway')#time_velocity_body,y_velocity, 'g.',label='Sway')
            ax5.set_xlabel('Epoch time, s')
            ax5.set_ylabel('Velocity, m/s')
            ax5.legend()
            ax5.grid(True)
            ax5.set_title('y velocity')
            ax6 = fig.add_subplot(326)
            ax6.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.z_velocity for i in dead_reckoning_dvl_list], 'b.',label='Heave')#time_velocity_body,z_velocity, 'b.',label='Heave')
            ax6.set_xlabel('Epoch time, s')
            ax6.set_ylabel('Velocity, m/s')
            ax6.legend()
            ax6.grid(True)
            ax6.set_title('z velocity')
            fig.tight_layout()
            plt.savefig(plotpath + os.sep + 'velocity_vs_time.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # time_dead_reckoning northings eastings depth vs time
            print('...plotting deadreckoning_vs_time...')
            
            fig=plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(221)
            ax1.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.northings_dr for i in dead_reckoning_dvl_list],'r.',label='DVL')#time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
            ax1.plot([i.timestamp for i in velocity_inertial_list],[i.northings_dr for i in velocity_inertial_list],'g.',label=velocity_inertial_sensor_name)
            ax1.plot([i.timestamp for i in usbl_list], [i.northings for i in usbl_list],'b.',label='USBL')
            ax1.plot([i.timestamp for i in dead_reckoning_centre_list],[i.northings_dr for i in dead_reckoning_centre_list],'c.',label='Centre')#time_velocity_body,northings_dead_reckoning,'b.')
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Northings, m')
            ax1.grid(True)
            ax1.legend()#loc='upper right', bbox_to_anchor=(1, -0.2))
            ax1.set_title('Northings')
            ax2 = fig.add_subplot(222)
            ax2.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.eastings_dr for i in dead_reckoning_dvl_list],'r.',label='DVL')#time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
            ax2.plot([i.timestamp for i in velocity_inertial_list],[i.eastings_dr for i in velocity_inertial_list],'g.',label=velocity_inertial_sensor_name)
            ax2.plot([i.timestamp for i in usbl_list], [i.eastings for i in usbl_list],'b.',label='USBL')
            ax2.plot([i.timestamp for i in dead_reckoning_centre_list],[i.eastings_dr for i in dead_reckoning_centre_list],'c.',label='Centre')#time_velocity_body,eastings_dead_reckoning,'b.')
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Eastings, m')
            ax2.grid(True)
            ax2.legend()
            ax2.set_title('Eastings')
            ax3 = fig.add_subplot(223)
            ax3.plot([i.timestamp for i in usbl_list],[i.depth for i in usbl_list],'b.',label='USBL depth') 
            ax3.plot([i.timestamp for i in depth_list],[i.depth for i in depth_list],'g-',label='Depth Sensor') 
            ax3.plot([i.timestamp for i in altitude_list],[i.seafloor_depth for i in altitude_list],'r-',label='Seafloor') 
            ax3.plot([i.timestamp for i in dead_reckoning_centre_list],[i.depth for i in dead_reckoning_centre_list],'c-',label='Centre')#time_velocity_body,eastings_dead_reckoning,'b.')
            ax3.set_xlabel('Epoch time, s')
            ax3.set_ylabel('Depth, m')
            plt.gca().invert_yaxis()
            ax3.grid(True)
            ax3.legend()
            ax3.set_title('Depth')
            ax4 = fig.add_subplot(224)
            ax4.plot([i.timestamp for i in altitude_list],[i.altitude for i in altitude_list],'r.',label='Altitude')              
            ax4.set_xlabel('Epoch time, s')
            ax4.set_ylabel('Altitude, m')
            ax4.set_xlim(min([i.timestamp for i in depth_list]),max([i.timestamp for i in depth_list]))
            ax4.grid(True)
            ax4.legend()
            fig.tight_layout()
            plt.savefig(plotpath + os.sep + 'deadreckoning_vs_time.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # usbl latitude longitude
            print('...plotting usbl_LatLong_vs_NorthEast...')

            fig=plt.figure(figsize=(10,5))
            ax1 = fig.add_subplot(121)
            ax1.plot([i.longitude for i in usbl_list],[i.latitude for i in usbl_list],'b.')                 
            ax1.set_xlabel('Longitude, degrees')
            ax1.set_ylabel('Latitude, degrees')
            ax1.grid(True)
            ax2 = fig.add_subplot(122)
            ax2.plot([i.eastings for i in usbl_list],[i.northings for i in usbl_list],'b.',label='Reference ['+str(latitude_reference)+','+str(longitude_reference)+']')                 
            ax2.set_xlabel('Eastings, m')
            ax2.set_ylabel('Northings, m')
            ax2.grid(True)
            ax2.legend()
            fig.tight_layout()
            plt.savefig(plotpath + os.sep + 'usbl_LatLong_vs_NorthEast.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # uncertainties plot. 
            # Meaningless this way? https://plot.ly/python/line-charts/#filled-lines Something like that?
            # Sum up for the rest too? check if this is the correct way
            # for i in range(len(roll_std)):
            #     if i == 0:
            #         pass
            #     else:
            #         roll_std[i]=roll_std[i] + roll_std[i-1]
            #         pitch_std[i]=pitch_std[i] + pitch_std[i-1]
            #         yaw_std[i]=yaw_std[i] + yaw_std[i-1]
            print('...plotting uncertainties_plot...')

            fig=plt.figure(figsize=(15,7))
            ax1 = fig.add_subplot(231)
            ax1.plot([i.timestamp for i in orientation_list],[i.roll_std for i in orientation_list],'r.',label='roll_std')
            ax1.plot([i.timestamp for i in orientation_list],[i.pitch_std for i in orientation_list],'g.',label='pitch_std')
            ax1.plot([i.timestamp for i in orientation_list],[i.yaw_std for i in orientation_list],'b.',label='yaw_std')
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Angle, degrees')
            ax1.legend()
            ax1.grid(True)
            ax2 = fig.add_subplot(234)
            ax2.plot([i.timestamp for i in depth_list],[i.depth_std for i in depth_list],'b.',label='depth_std')
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Depth, m')
            ax2.legend()
            ax2.grid(True)
            ax3 = fig.add_subplot(232)
            ax3.plot([i.timestamp for i in velocity_body_list],[i.x_velocity_std for i in velocity_body_list],'r.',label='x_velocity_std')
            ax3.plot([i.timestamp for i in velocity_body_list],[i.y_velocity_std for i in velocity_body_list],'g.',label='y_velocity_std')
            ax3.plot([i.timestamp for i in velocity_body_list],[i.z_velocity_std for i in velocity_body_list],'b.',label='z_velocity_std')
            ax3.set_xlabel('Epoch time, s')
            ax3.set_ylabel('Velocity, m/s')
            ax3.legend()
            ax3.grid(True)
            ax4 = fig.add_subplot(235)
            ax4.plot([i.timestamp for i in velocity_inertial_list],[i.north_velocity_std for i in velocity_inertial_list],'r.',label='north_velocity_std_inertia')
            ax4.plot([i.timestamp for i in velocity_inertial_list],[i.east_velocity_std for i in velocity_inertial_list],'g.',label='east_velocity_std_inertia')
            ax4.plot([i.timestamp for i in velocity_inertial_list],[i.down_velocity_std for i in velocity_inertial_list],'b.',label='down_velocity_std_inertia')
            ax4.set_xlabel('Epoch time, s')
            ax4.set_ylabel('Velocity, m/s')
            ax4.legend()
            ax4.grid(True)
            ax5 = fig.add_subplot(233)
            ax5.plot([i.timestamp for i in usbl_list],[i.latitude_std for i in usbl_list],'r.',label='latitude_std_usbl')
            ax5.plot([i.timestamp for i in usbl_list],[i.longitude_std for i in usbl_list],'g.',label='longitude_std_usbl')
            ax5.set_xlabel('Epoch time, s')
            ax5.set_ylabel('LatLong, degrees')
            ax5.legend()
            ax5.grid(True)
            ax6 = fig.add_subplot(236)
            ax6.plot([i.timestamp for i in usbl_list],[i.northings_std for i in usbl_list],'r.',label='northings_std_usbl')
            ax6.plot([i.timestamp for i in usbl_list],[i.eastings_std for i in usbl_list],'g.',label='eastings_std_usbl')
            ax6.set_xlabel('Epoch time, s')
            ax6.set_ylabel('NorthEast, m')
            ax6.legend()
            ax6.grid(True)
            fig.tight_layout()                  
            plt.savefig(plotpath + os.sep + 'uncertainties_plot.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            print('...')

        # DR
            print('...plotting camera1_centre_DVL_{}_DR...'.format(velocity_inertial_sensor_name))
            fig=plt.figure()
            ax = fig.add_subplot(111)
            ax.plot([i.eastings_dr for i in camera1_list],[i.northings_dr for i in camera1_list],'y.',label='Camera1')
            ax.plot([i.eastings_dr for i in dead_reckoning_centre_list],[i.northings_dr for i in dead_reckoning_centre_list],'r.',label='Centre')
            ax.plot([i.eastings_dr for i in dead_reckoning_dvl_list],[i.northings_dr for i in dead_reckoning_dvl_list],'g.',label='DVL')
            ax.plot([i.eastings_dr for i in velocity_inertial_list],[i.northings_dr for i in velocity_inertial_list],'m.',label=velocity_inertial_sensor_name)
            ax.plot([i.eastings for i in usbl_list], [i.northings for i in usbl_list],'c.',label='USBL')
            ax.set_xlabel('Eastings, m')
            ax.set_ylabel('Northings, m')
            ax.legend()#loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)   
            plt.savefig(plotpath + os.sep + 'camera1_centre_DVL_{}_DR.pdf'.format(velocity_inertial_sensor_name), dpi=600, bbox_inches='tight')
            if show_plot==True:
                plt.show()
            plt.close()

            print('Complete plot data: ', plotpath)

            # if show_plot==True:
            #     fig=plt.figure()
            #     ax=fig.add_subplot(111)
            #     axamp=plt.axes([0.2,0.01,0.65,0.03])
            #     sts=Slider(axamp,'Time Stamp',0,len(time_dead_reckoning),valinit=0)#output the timestamp, anad change the gap between jumps
            #     def update(val):
            #         ts=sts.val
            #         ax.clear()
            #         xnew=[]
            #         ynew=[]
            #         for i in range(len(time_dead_reckoning)):
            #             if i < ts:
            #                 xnew.append(eastings_dead_reckoning[i])
            #                 ynew.append(northings_dead_reckoning[i])
            #             else:
            #                 pass
            #         ax.scatter(xnew,ynew)
            #     sts.on_changed(update)
            #     plt.show()
            #     plt.close()

    # plotly data
        if plotly is True:

            print('Plotting plotly data ...')
            plotlypath = renavpath + os.sep + 'plotly_plots'
            
            if os.path.isdir(plotlypath) == 0:
                try:
                    os.mkdir(plotlypath)
                except Exception as e:
                    print("Warning:",e)

            def create_trace(x_list,y_list,trace_name,trace_color):
                trace = go.Scattergl(
                    x=x_list,
                    y=y_list,
                    name=trace_name,
                    mode='lines+markers',
                    marker=dict(color=trace_color),#'rgba(152, 0, 0, .8)'),#,size = 10, line = dict(width = 2,color = 'rgb(0, 0, 0)'),
                    line=dict(color=trace_color)#rgb(205, 12, 24)'))#, width = 4, dash = 'dot')
                    # legendgroup='group11'
                )
                return trace
            
        # orientation
            print('...plotting orientation_vs_time...')

            trace11a = create_trace([i.timestamp for i in orientation_list], [i.yaw for i in orientation_list], 'Yaw', 'red')
            trace11b = create_trace([i.timestamp for i in orientation_list], [i.roll for i in orientation_list], 'Roll', 'blue')
            trace11c = create_trace([i.timestamp for i in orientation_list], [i.pitch for i in orientation_list], 'Pitch', 'green')
            layout = go.Layout(
                title='Orientation vs Time',
                hovermode='closest',
                xaxis=dict(title='Epoch time, s', tickformat='.3f'),
                yaxis=dict(title='Degrees'),
                dragmode='pan',
                )
            config={'scrollZoom': True}
            fig = go.Figure(data=[trace11a, trace11b, trace11c], layout=layout)
            py.plot(fig, config=config, filename=plotlypath + os.sep + 'orientation_vs_time.html', auto_open=False)

        # velocity_body (north,east,down) compared to velocity_inertial
            print('...plotting velocity_vs_time...')

            trace11a = create_trace([i.timestamp for i in dead_reckoning_dvl_list], [i.north_velocity for i in dead_reckoning_dvl_list], 'DVL north velocity', 'red')
            trace11b = create_trace([i.timestamp for i in velocity_inertial_list], [i.north_velocity for i in velocity_inertial_list], '{} north velocity'.format(velocity_inertial_sensor_name), 'blue')
            # plot1=[trace11a, trace11b]
            trace21a = create_trace([i.timestamp for i in dead_reckoning_dvl_list], [i.east_velocity for i in dead_reckoning_dvl_list], 'DVL east velocity', 'red')
            trace21b = create_trace([i.timestamp for i in velocity_inertial_list], [i.east_velocity for i in velocity_inertial_list], '{} east velocity'.format(velocity_inertial_sensor_name), 'blue')
            trace31a = create_trace([i.timestamp for i in dead_reckoning_dvl_list], [i.down_velocity for i in dead_reckoning_dvl_list], 'DVL down velocity', 'red')
            trace31b = create_trace([i.timestamp for i in velocity_inertial_list], [i.down_velocity for i in velocity_inertial_list], '{} down velocity'.format(velocity_inertial_sensor_name), 'blue')
            trace12a = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.x_velocity for i in dead_reckoning_centre_list], 'DVL x velocity', 'red')
            trace22a = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.y_velocity for i in dead_reckoning_centre_list], 'DVL y velocity', 'red')
            trace32a = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.z_velocity for i in dead_reckoning_centre_list], 'DVL z velocity', 'red')
            fig = tools.make_subplots(rows=3, cols=2, subplot_titles=('DVL vs {} - north Velocity'.format(velocity_inertial_sensor_name), 'DVL - x velocity / surge', 'DVL vs {} - east Velocity'.format(velocity_inertial_sensor_name), 'DVL - y velocity / sway', 'DVL vs {} - down Velocity'.format(velocity_inertial_sensor_name), 'DVL - z velocity / heave'))
            fig.append_trace(trace11a, 1, 1)
            fig.append_trace(trace11b, 1, 1)
            fig.append_trace(trace21a, 2, 1)
            fig.append_trace(trace21b, 2, 1)
            fig.append_trace(trace31a, 3, 1)
            fig.append_trace(trace31b, 3, 1)
            fig.append_trace(trace12a, 1, 2)
            fig.append_trace(trace22a, 2, 2)
            fig.append_trace(trace32a, 3, 2)
            fig['layout']['xaxis1'].update(title='Epoch time, s', tickformat='.3f')#xaxis 1 title')
            fig['layout']['xaxis2'].update(title='Epoch time, s', tickformat='.3f')#, range=[10, 50])
            fig['layout']['xaxis3'].update(title='Epoch time, s', tickformat='.3f')#, showgrid=False)
            fig['layout']['xaxis4'].update(title='Epoch time, s', tickformat='.3f')#, type='log')
            fig['layout']['xaxis5'].update(title='Epoch time, s', tickformat='.3f')
            fig['layout']['xaxis6'].update(title='Epoch time, s', tickformat='.3f')
            # def define_layout():
            #   for i in XX:
            #       fig['layout']['xaxis{}'.format(plot_number)].update(title=axis_title) ...
            fig['layout']['yaxis1'].update(title='Velocity, m/s')
            fig['layout']['yaxis2'].update(title='Velocity, m/s')
            fig['layout']['yaxis3'].update(title='Velocity, m/s')
            fig['layout']['yaxis4'].update(title='Velocity, m/s')
            fig['layout']['yaxis5'].update(title='Velocity, m/s')
            fig['layout']['yaxis6'].update(title='Velocity, m/s')
            fig['layout'].update(title='Velocity vs Time Plots (Left column: Inertial frame - north east down | Right column: Body frame - x y z)', dragmode='pan', hovermode='closest')#, hoverlabel={'namelength':'-1'})
            config={'scrollZoom': True}
            py.plot(fig, config=config, filename=plotlypath + os.sep + 'velocity_vs_time.html', auto_open=False)

        # time_dead_reckoning northings eastings depth vs time
            print('...plotting deadreckoning_vs_time...')

            trace11a = create_trace([i.timestamp for i in dead_reckoning_dvl_list], [i.northings_dr for i in dead_reckoning_dvl_list], 'Northing DVL', 'red')
            trace11b = create_trace([i.timestamp for i in velocity_inertial_list], [i.northings_dr for i in velocity_inertial_list], 'Northing {}'.format(velocity_inertial_sensor_name), 'green')
            trace11c = create_trace([i.timestamp for i in usbl_list], [i.northings for i in usbl_list], 'Northing USBL', 'blue')
            trace11d = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.northings_dr for i in dead_reckoning_centre_list], 'Northing Centre', 'orange')
            trace12a = create_trace([i.timestamp for i in dead_reckoning_dvl_list], [i.eastings_dr for i in dead_reckoning_dvl_list], 'Easting DVL', 'red')
            trace12b = create_trace([i.timestamp for i in velocity_inertial_list], [i.eastings_dr for i in velocity_inertial_list], 'Easting {}'.format(velocity_inertial_sensor_name), 'green')
            trace12c = create_trace([i.timestamp for i in usbl_list], [i.eastings for i in usbl_list], 'Easting USBL', 'blue')
            trace12d = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.eastings_dr for i in dead_reckoning_centre_list], 'Easting Centre', 'orange')
            trace21a = create_trace([i.timestamp for i in altitude_list], [i.seafloor_depth for i in altitude_list], 'Depth  Seafloor (Depth Sensor + Altitude)', 'red')
            trace21b = create_trace([i.timestamp for i in depth_list], [i.depth for i in depth_list], 'Depth Sensor', 'purple')
            trace21c = create_trace([i.timestamp for i in usbl_list], [i.depth for i in usbl_list], 'Depth USBL', 'blue')
            trace21d = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.depth for i in dead_reckoning_centre_list], 'Depth Centre', 'orange')
            trace22a = create_trace([i.timestamp for i in altitude_list], [i.altitude for i in altitude_list], 'Altitude', 'red')
            fig = tools.make_subplots(rows=2,cols=2, subplot_titles=('Northings', 'Eastings', 'Depth', 'Altitude'))
            fig.append_trace(trace11a, 1, 1)
            fig.append_trace(trace11b, 1, 1)
            fig.append_trace(trace11c, 1, 1)
            fig.append_trace(trace11d, 1, 1)
            fig.append_trace(trace12a, 1, 2)
            fig.append_trace(trace12b, 1, 2)
            fig.append_trace(trace12c, 1, 2)
            fig.append_trace(trace12d, 1, 2)
            fig.append_trace(trace21a, 2, 1)
            fig.append_trace(trace21b, 2, 1)
            fig.append_trace(trace21c, 2, 1)
            fig.append_trace(trace21d, 2, 1)
            fig.append_trace(trace22a, 2, 2)
            fig['layout']['xaxis1'].update(title='Epoch time, s', tickformat='.3f')
            fig['layout']['xaxis2'].update(title='Epoch time, s', tickformat='.3f')
            fig['layout']['xaxis3'].update(title='Epoch time, s', tickformat='.3f')
            fig['layout']['xaxis4'].update(title='Epoch time, s', tickformat='.3f')
            fig['layout']['yaxis1'].update(title='Northing, m')
            fig['layout']['yaxis2'].update(title='Easting, m')
            fig['layout']['yaxis3'].update(title='Depth, m', autorange='reversed')
            fig['layout']['yaxis4'].update(title='Altitude, m')
            fig['layout'].update(title='Deadreckoning vs Time', dragmode='pan', hovermode='closest')#, hoverlabel={'namelength':'-1'})
            config={'scrollZoom': True}
            py.plot(fig, config=config, filename=plotlypath + os.sep + 'deadreckoning_vs_time.html', auto_open=False)

        # Uncertainty plotly
            trace11a = create_trace([i.timestamp for i in orientation_list], [i.roll_std for i in orientation_list], 'roll std', 'red')
            trace11b = create_trace([i.timestamp for i in orientation_list], [i.pitch_std for i in orientation_list], 'pitch std', 'green')
            trace11c = create_trace([i.timestamp for i in orientation_list], [i.yaw_std for i in orientation_list], 'yaw std', 'blue')
            trace12a = create_trace([i.timestamp for i in velocity_body_list], [i.x_velocity_std for i in velocity_body_list], 'x velocity std', 'red')
            trace12b = create_trace([i.timestamp for i in velocity_body_list], [i.y_velocity_std for i in velocity_body_list], 'y velocity std', 'green')
            trace12c = create_trace([i.timestamp for i in velocity_body_list], [i.z_velocity_std for i in velocity_body_list], 'z velocity std', 'blue')
            trace13a = create_trace([i.timestamp for i in usbl_list], [i.latitude_std for i in usbl_list], 'latitude std usbl', 'red')
            trace13b = create_trace([i.timestamp for i in usbl_list], [i.longitude_std for i in usbl_list], 'longitude std usbl', 'green')
            trace21a = create_trace([i.timestamp for i in depth_list], [i.depth_std for i in depth_list], 'depth std', 'red')
            trace22a = create_trace([i.timestamp for i in velocity_inertial_list], [i.north_velocity_std for i in velocity_inertial_list], 'north velocity std inertial', 'red')
            trace22b = create_trace([i.timestamp for i in velocity_inertial_list], [i.east_velocity_std for i in velocity_inertial_list], 'east velocity std inertial', 'green')
            trace22c = create_trace([i.timestamp for i in velocity_inertial_list], [i.down_velocity_std for i in velocity_inertial_list], 'down velocity std inertial', 'blue')
            trace23a = create_trace([i.timestamp for i in usbl_list], [i.northings_std for i in usbl_list], 'northing std usbl', 'red')
            trace23b = create_trace([i.timestamp for i in usbl_list], [i.eastings_std for i in usbl_list], 'easting std usbl', 'green')
            fig = tools.make_subplots(rows=2, cols=3, subplot_titles=('Orientation uncertainties', 'DVL uncertainties', 'USBL uncertainties', 'Depth uncertainties', '{} uncertainties'.format(velocity_inertial_sensor_name), 'USBL uncertainties'))
            fig.append_trace(trace11a, 1, 1)
            fig.append_trace(trace11b, 1, 1)
            fig.append_trace(trace11c, 1, 1)
            fig.append_trace(trace12a, 1, 2)
            fig.append_trace(trace12b, 1, 2)
            fig.append_trace(trace12c, 1, 2)
            fig.append_trace(trace13a, 1, 3)
            fig.append_trace(trace13b, 1, 3)
            fig.append_trace(trace21a, 2, 1)
            fig.append_trace(trace22a, 2, 2)
            fig.append_trace(trace22b, 2, 2)
            fig.append_trace(trace22c, 2, 2)
            fig.append_trace(trace23a, 2, 3)
            fig.append_trace(trace23b, 2, 3)
            fig['layout']['xaxis1'].update(title='Epoch time, s', tickformat='.3f')#xaxis 1 title')
            fig['layout']['xaxis2'].update(title='Epoch time, s', tickformat='.3f')#, range=[10, 50])
            fig['layout']['xaxis3'].update(title='Epoch time, s', tickformat='.3f')#, showgrid=False)
            fig['layout']['xaxis4'].update(title='Epoch time, s', tickformat='.3f')#, type='log')
            fig['layout']['xaxis5'].update(title='Epoch time, s', tickformat='.3f')
            fig['layout']['xaxis6'].update(title='Epoch time, s', tickformat='.3f')
            fig['layout']['yaxis1'].update(title='Angle, degrees')
            fig['layout']['yaxis2'].update(title='Velocity, m/s')
            fig['layout']['yaxis3'].update(title='LatLong, degrees')
            fig['layout']['yaxis4'].update(title='Depth, m')
            fig['layout']['yaxis5'].update(title='Velocity, m/s')
            fig['layout']['yaxis6'].update(title='NorthEast, m')
            fig['layout'].update(title='Uncertainty Plots', dragmode='pan', hovermode='closest')
            config={'scrollZoom': True}
            py.plot(fig, config=config, filename=plotlypath + os.sep + 'uncertainties_plot.html', auto_open=False)

        # # uncertainties plot. 
        #     #https://plot.ly/python/line-charts/#filled-lines Something like that?

        # # DR plotly slider *include toggle button that switches between lat long and north east
            print('...plotting auv_path...')

            minTimestamp = 99999999999999
            maxTimestamp = -99999999999999
            for i in [[i.timestamp for i in camera1_list], [i.timestamp for i in dead_reckoning_centre_list], [i.timestamp for i in velocity_inertial_list], [i.timestamp for i in usbl_list]]:
                if min(i) < minTimestamp:
                    minTimestamp = min(i)
                if max(i) > maxTimestamp:
                    maxTimestamp = max(i)

            # slider plot
            # time_gap = 240
            time_gap = int((maxTimestamp - minTimestamp)/40)
            epoch_timestamps_slider = list(range(int(minTimestamp), int(maxTimestamp), int(time_gap)))

            figure = {
                'data': [],
                'layout': {},
                'frames': []
            }

            #RANGESLIDER!?
            sliders_dict = {
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 20},
                    'prefix': 'epoch_timestamp:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': []
            }

            # fill in most of layout
            figure['layout']['xaxis'] = {'title': 'Eastings,m'} #'range': [-30, 60], 'title': 'Eastings,m'} 
            figure['layout']['yaxis'] = {'title': 'Northings,m'} #'range': [-20, 90], 'title': 'Northings,m'}
            figure['layout']['hovermode'] = 'closest'
            figure['layout']['dragmode'] = 'pan'
            figure['layout']['updatemenus'] = [
                {
                    'buttons': [
                        {
                            'args': [None, {'frame': {'duration': 500, 'redraw': False},
                                     'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                            'label': 'Play',
                            'method': 'animate'
                        },
                        {
                            'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                            'transition': {'duration': 0}}],
                            'label': 'Pause',
                            'method': 'animate'
                        }
                    ],
                    'direction': 'left',
                    'pad': {'r': 10, 't': 87},
                    'showactive': False,
                    'type': 'buttons',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
                }
            ]

            #make data
            def make_data(name,eastings,northings):
                mode='lines'
                if 'USBL' in name:
                    mode='lines+markers'
                data_dict = {
                    'x': eastings,
                    'y': northings,
                    'mode':'{}'.format(mode),
                    'name': '{}'.format(name)
                }
                figure['data'].append(data_dict)

            make_data('Camera1',[i.eastings_dr for i in camera1_list],[i.northings_dr for i in camera1_list])
            make_data('Centre',[i.eastings_dr for i in dead_reckoning_centre_list],[i.northings_dr for i in dead_reckoning_centre_list])
            make_data('DVL',[i.eastings_dr for i in dead_reckoning_dvl_list],[i.northings_dr for i in dead_reckoning_dvl_list])
            make_data(velocity_inertial_sensor_name,[i.eastings_dr for i in velocity_inertial_list],[i.northings_dr for i in velocity_inertial_list])
            make_data('USBL',[i.eastings for i in usbl_list],[i.northings for i in usbl_list])

            config={'scrollZoom': True}

            py.plot(figure, config=config, filename=plotlypath + os.sep + 'auv_path.html',auto_open=False)

            def make_frame(data,tstamp):
                temp_index=-1#next(x[0] for x in enumerate(data[0]) if x[1] > tstamp)
                for i in range(len(data[1])):
                    if data[1][i] <= tstamp:
                        temp_index=i
                    else:
                        break
                eastings=data[2][:temp_index+1]
                northings=data[3][:temp_index+1]
                data_dict = {
                    'x': eastings,
                    'y': northings,
                    'name': '{}'.format(data[0])
                }
                frame['data'].append(data_dict)

            #make frames
            for i in epoch_timestamps_slider:
                frame = {'data': [], 'name': str(i)}
                
                for j in [['Camera1',[i.timestamp for i in camera1_list],[i.eastings_dr for i in camera1_list],[i.northings_dr for i in camera1_list]],['Centre',[i.timestamp for i in dead_reckoning_centre_list],[i.eastings_dr for i in dead_reckoning_centre_list],[i.northings_dr for i in dead_reckoning_centre_list]],['DVL',[i.timestamp for i in dead_reckoning_dvl_list],[i.eastings_dr for i in dead_reckoning_dvl_list],[i.northings_dr for i in dead_reckoning_dvl_list]],[velocity_inertial_sensor_name,[i.timestamp for i in velocity_inertial_list],[i.eastings_dr for i in velocity_inertial_list],[i.northings_dr for i in velocity_inertial_list]],['USBL',[i.timestamp for i in usbl_list],[i.eastings for i in usbl_list],[i.northings for i in usbl_list]]]:
                    make_frame(j,i)
                figure['frames'].append(frame)
                slider_step = {'args': [
                    [i],
                    {'frame': {'duration': 300, 'redraw': False},
                     'mode': 'immediate',
                   'transition': {'duration': 300}}
                 ],
                 'label': i,
                 'method': 'animate'}
                sliders_dict['steps'].append(slider_step)

            figure['layout']['sliders'] = [sliders_dict]

            py.plot(figure, config=config, filename=plotlypath + os.sep + 'auv_path_slider.html',auto_open=False)

            print('Complete plot data: ', plotlypath)

        print('Completed data extraction: ', renavpath)
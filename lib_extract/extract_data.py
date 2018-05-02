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

class extract_data:
    def __init__(self,filepath,ftype,start_time,finish_time,plot,csv_write,show_plot,plotly):

        interpolate_remove_flag = False

        epoch_start_time = 0
        epoch_finish_time = 0

    # velocity body placeholders (DVL)
        time_velocity_body=[]
        x_velocity=[] #x_velocity_body
        y_velocity=[] #y_velocity_body
        z_velocity=[] #z_velocity_body
        x_velocity_std=[] #x_velocity_body_std
        y_velocity_std=[] #y_velocity_body_std
        z_velocity_std=[] #z_velocity_body_std
        velocity_body_sensor_name = ''

    # velocity inertial placeholders
        time_velocity_inertia=[] #time_velocity_inertial
        north_velocity_inertia=[] #north_velocity_inertial
        east_velocity_inertia=[] #east_velocity_inertial
        down_velocity_inertia=[] #down_velocity_inertial
        north_velocity_std_inertia=[] #north_velocity_inertial_std
        east_velocity_std_inertia=[] #east_velocity_inertial_std
        down_velocity_std_inertia=[] #down_velocity_inertial_std
        roll_inertia_interpolated=[] #roll_inertial_interpolated
        pitch_inertia_interpolated=[] #pitch_inertial_interpolated
        yaw_inertia_interpolated=[] #yaw_intertial_interpolated
        northings_inertia_dead_reckoning=[] #northings_inertial_dead_reckoning
        eastings_inertia_dead_reckoning=[] #eastings_inertial_dead_reckoning
        depth_inertia_dead_reckoning=[] #depth_inertial_dead_reckoning
        velocity_inertial_sensor_name = ''
        
    # orientation placeholders (INS)
        time_orientation=[]
        roll=[]
        pitch=[]
        yaw=[]
        roll_std=[]
        pitch_std=[]
        yaw_std=[]
        orientation_sensor_name = ''

    # placeholders for interpolated velocity body measurements based on orientation and transformed coordinates
        time_dead_reckoning=[] #time_velocity_body_dead_reckoning?

        roll_ins_dead_reckoning=[] #roll_dead_reckoning / roll_velocity_body_dead_reckoning? 
        pitch_ins_dead_reckoning=[] #pitch_dead_reckoning
        yaw_ins_dead_reckoning=[] #yaw_dead_reckoning

        x_velocity_interpolated=[] #x_velocity_body_interpolated
        y_velocity_interpolated=[] #y_velocity_body_interpolated
        z_velocity_interpolated=[] #z_velocity_body_interpolated
        # x_velocity_body_std_interpolated = []
        # y_velocity_body_std_interpolated = []
        # z_velocity_body_std_interpolated = []

        # [north_velocity_body_interpolated,east_velocity_body_interpolated,down_velocity_body_interpolated]=[[],[],[]]?
        north_velocity_inertia_dvl=[] #north_velocity_body_interpolated
        east_velocity_inertia_dvl=[] #east_velocity_body_interpolated
        down_velocity_inertia_dvl=[] #down_velocity_body_interpolated

        altitude_interpolated=[]

        northings_dead_reckoning=[] #northings_velocity_body_dead_reckoning
        eastings_dead_reckoning=[] #
        depth_dead_reckoning=[] #

    # depth placeholders
        time_depth=[]
        depth=[]
        depth_std=[]
        depth_sensor_name = ''

    # altitude placeholders
        time_altitude=[]
        altitude=[]
        seafloor_depth=[] # interpolate depth and add altitude for every altitude measurement
        altitude_sensor_name = ''

    # USBL placeholders
        time_usbl=[]
        latitude_usbl=[]
        longitude_usbl=[]
        latitude_std_usbl=[]
        longitude_std_usbl=[]
        northings_usbl=[]
        eastings_usbl=[]
        depth_usbl=[]
        northings_std_usbl=[]
        eastings_std_usbl=[]
        usbl_sensor_name = ''

    # camera1 placeholders
        time_camera1=[]
        filename_camera1=[]
        camera1_dead_reckoning_northings=[]
        camera1_dead_reckoning_eastings=[]
        camera1_dead_reckoning_depth=[]
        camera1_roll=[]
        camera1_pitch=[]
        camera1_yaw=[]
        camera1_altitude=[]
        camera1_sensor_name = '' # original serial_camera1
    # camera2 placeholders
        time_camera2=[]
        filename_camera2=[]
        camera2_dead_reckoning_northings=[]
        camera2_dead_reckoning_eastings=[]
        camera2_dead_reckoning_depth=[]
        camera2_roll=[]
        camera2_pitch=[]
        camera2_yaw=[]
        camera2_altitude=[]
        camera2_sensor_name = ''

    # camera3 placeholders
        time_camera3=[]
        filename_camera3=[]
        camera3_dead_reckoning_northings=[]
        camera3_dead_reckoning_eastings=[]
        camera3_dead_reckoning_depth=[]
        camera3_roll=[]
        camera3_pitch=[]
        camera3_yaw=[]
        camera3_altitude=[]
        camera3_sensor_name = ''
        
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
            
            # assigns sensor names from mission.yaml instead of json data packet, because TS don't have serial yet.
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
                                time_velocity_body.append(parsed_json_data[i]['epoch_timestamp_dvl']) # dvl clock not necessarily synced by phins                      
                                x_velocity.append(parsed_json_data[i]['data'][0]['x_velocity'])
                                y_velocity.append(parsed_json_data[i]['data'][1]['y_velocity'])
                                z_velocity.append(parsed_json_data[i]['data'][2]['z_velocity'])
                                x_velocity_std.append(parsed_json_data[i]['data'][0]['x_velocity_std'])
                                y_velocity_std.append(parsed_json_data[i]['data'][1]['y_velocity_std'])
                                z_velocity_std.append(parsed_json_data[i]['data'][2]['z_velocity_std'])

                        if 'inertial' in parsed_json_data[i]['frame']:
                            time_velocity_inertia.append(parsed_json_data[i]['epoch_timestamp'])
                            north_velocity_inertia.append(parsed_json_data[i]['data'][0]['north_velocity'])
                            east_velocity_inertia.append(parsed_json_data[i]['data'][1]['east_velocity'])
                            down_velocity_inertia.append(parsed_json_data[i]['data'][2]['down_velocity'])
                            north_velocity_std_inertia.append(parsed_json_data[i]['data'][0]['north_velocity_std'])
                            east_velocity_std_inertia.append(parsed_json_data[i]['data'][1]['east_velocity_std'])
                            down_velocity_std_inertia.append(parsed_json_data[i]['data'][2]['down_velocity_std'])
                            roll_inertia_interpolated.append(0)
                            pitch_inertia_interpolated.append(0)
                            yaw_inertia_interpolated.append(0)
                            northings_inertia_dead_reckoning.append(0)
                            eastings_inertia_dead_reckoning.append(0)
                            depth_inertia_dead_reckoning.append(0)
                    
                    if 'orientation' in parsed_json_data[i]['category']:
                        time_orientation.append(parsed_json_data[i]['epoch_timestamp'])
                        roll.append(parsed_json_data[i]['data'][1]['roll'])
                        pitch.append(parsed_json_data[i]['data'][2]['pitch'])
                        yaw.append(parsed_json_data[i]['data'][0]['heading'])
                        roll_std.append(parsed_json_data[i]['data'][1]['roll_std'])
                        pitch_std.append(parsed_json_data[i]['data'][2]['pitch_std'])
                        yaw_std.append(parsed_json_data[i]['data'][0]['heading_std'])

                    if 'depth' in parsed_json_data[i]['category']:
                        time_depth.append(parsed_json_data[i]['epoch_timestamp_depth'])
                        depth.append(parsed_json_data[i]['data'][0]['depth'])
                        depth_std.append(parsed_json_data[i]['data'][0]['depth_std'])

                    if 'altitude' in parsed_json_data[i]['category']:
                        time_altitude.append(parsed_json_data[i]['epoch_timestamp'])
                        altitude.append(parsed_json_data[i]['data'][0]['altitude'])
                        seafloor_depth.append(0)

                    if 'usbl' in parsed_json_data[i]['category']:                        
                        time_usbl.append(parsed_json_data[i]['epoch_timestamp'])
                        latitude_usbl.append(parsed_json_data[i]['data_target'][0]['latitude'])
                        longitude_usbl.append(parsed_json_data[i]['data_target'][1]['longitude'])
                        northings_usbl.append(parsed_json_data[i]['data_target'][2]['northings'])
                        eastings_usbl.append(parsed_json_data[i]['data_target'][3]['eastings'])
                        depth_usbl.append(parsed_json_data[i]['data_target'][4]['depth'])
                        latitude_std_usbl.append(parsed_json_data[i]['data_target'][0]['latitude_std'])
                        longitude_std_usbl.append(parsed_json_data[i]['data_target'][1]['longitude_std'])
                        northings_std_usbl.append(parsed_json_data[i]['data_target'][2]['northings_std'])
                        eastings_std_usbl.append(parsed_json_data[i]['data_target'][3]['eastings_std'])

                    if 'image' in parsed_json_data[i]['category']:
                        time_camera1.append(parsed_json_data[i]['camera1'][0]['epoch_timestamp'])#LC
                        filename_camera1.append(parsed_json_data[i]['camera1'][0]['filename'])
                        time_camera2.append(parsed_json_data[i]['camera2'][0]['epoch_timestamp'])
                        filename_camera2.append(parsed_json_data[i]['camera2'][0]['filename'])

                    if 'laser' in parsed_json_data[i]['category']:
                        time_camera3.append(parsed_json_data[i]['epoch_timestamp'])#LC
                        filename_camera3.append(parsed_json_data[i]['filename'])


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

                            time_velocity_body.append(float(line_split[1]))
                            time_altitude.append(float(line_split[1]))

                            for i in range(len(line_split)):
                                value=line_split[i].split(':')
                                if value[0] == 'alt':
                                    altitude.append(float(value[1]))
                                    seafloor_depth.append(0)
                                if value[0] == 'vx':
                                    x_velocity.append(float(value[1]))
                                if value[0] == 'vy':
                                    y_velocity.append(float(value[1]))
                                if value[0] == 'vz':
                                    z_velocity.append(float(value[1]))

                    if str(line_split[0]) == 'PHINS_COMPASS:':
                        
                        epoch_timestamp=float(line_split[1])
                        if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:                                      

                            time_velocity_inertia.append(float(line_split[1]))
                            time_orientation.append(float(line_split[1]))

                            for i in range(len(line_split)-1):
                                
                                if line_split[i] == 'r:':
                                    roll.append(float(line_split[i+1]))                                    
                                if line_split[i] == 'p:':
                                    pitch.append(float(line_split[i+1]))
                                if line_split[i] == 'h:':
                                    yaw.append(float(line_split[i+1]))

                            north_velocity_inertia.append(0)
                            east_velocity_inertia.append(0)
                            down_velocity_inertia.append(0)
                            roll_inertia_interpolated.append(0)
                            pitch_inertia_interpolated.append(0)
                            yaw_inertia_interpolated.append(0)
                            northings_inertia_dead_reckoning.append(0)
                            eastings_inertia_dead_reckoning.append(0)
                            depth_inertia_dead_reckoning.append(0)


                    if str(line_split[0]) == 'PAROSCI:':

                        epoch_timestamp=float(line_split[1])
                        if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:                                      

                            time_depth.append(float(line_split[1]))
                            depth.append(float(line_split[2]))                            

                    if str(line_split[0]) == 'SSBL_FIX:':
                        
                        epoch_timestamp=float(line_split[1])
                        
                        if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:                                      

                            time_usbl.append(float(line_split[1]))
                        
                            for i in range(len(line_split)-1):
                                
                                
                                if line_split[i] == 'target_x:':
                                    northings_usbl.append(float(line_split[i+1]))                                    
                                if line_split[i] == 'target_y:':
                                    eastings_usbl.append(float(line_split[i+1]))
                                if line_split[i] == 'target_z:':
                                    depth_usbl.append(float(line_split[i+1]))

                            latitude_usbl.append(0)
                            longitude_usbl.append(0)

            # make folder to store csv and plots
            renavpath = filepath + 'acfr_renav_' + str(yyyy).zfill(4) + str(mm).zfill(2) + str(dd).zfill(2) + '_' + start_time + '_' + finish_time 
            if os.path.isdir(renavpath) == 0:
                try:
                    os.mkdir(renavpath)
                except Exception as e:
                    print("Warning:",e)

            print('Complete parse of:' + outpath + os.sep + filename)
            print('Writing outputs to: ' + renavpath)


    # perform coordinate transformations and interpolations of state data to velocity_body time stamps with sensor position offset
        #Assumes the first measurement of velocity_body is the beginning of mission. May not be robust to non-continuous measurements..any (sudden start and stop) will affect it?
        # 
        j=0        
        k=0
        n=0
        start_interpolate_index = 0

        while time_orientation[start_interpolate_index]<time_velocity_body[0]:
            start_interpolate_index += 1

        # if start_interpolate_index==0:
        # do something? because time_orientation may be way before time_velocity_body

        if start_interpolate_index==1:
            interpolate_remove_flag = True

        for i in range(start_interpolate_index, len(time_orientation)):#time_velocity_body)):

            # interpolate to find the appropriate dvl time for the orientation measurements
            if time_orientation[i]>time_velocity_body[-1]:
                break

            while j<len(time_velocity_body)-1 and time_orientation[i]>time_velocity_body[j+1]:
                j += 1

            time_dead_reckoning.append(time_orientation[i])

            #change these to roll_dvl_dr?....
            roll_ins_dead_reckoning.append(roll[i])
            pitch_ins_dead_reckoning.append(pitch[i])
            yaw_ins_dead_reckoning.append(yaw[i])

            x_velocity_interpolated.append(interpolate(time_orientation[i],time_velocity_body[j],time_velocity_body[j+1],x_velocity[j],x_velocity[j+1]))
            y_velocity_interpolated.append(interpolate(time_orientation[i],time_velocity_body[j],time_velocity_body[j+1],y_velocity[j],y_velocity[j+1]))
            z_velocity_interpolated.append(interpolate(time_orientation[i],time_velocity_body[j],time_velocity_body[j+1],z_velocity[j],z_velocity[j+1]))

            [x_offset,y_offset,z_offset] = body_to_inertial(roll[i], pitch[i], yaw[i], x_velocity_interpolated[i-start_interpolate_index], y_velocity_interpolated[i-start_interpolate_index], z_velocity_interpolated[i-start_interpolate_index])

            north_velocity_inertia_dvl.append(x_offset)
            east_velocity_inertia_dvl.append(y_offset)
            down_velocity_inertia_dvl.append(z_offset)
            
            northings_dead_reckoning.append(0)
            eastings_dead_reckoning.append(0)
            depth_dead_reckoning.append(0)

            while n<len(time_altitude)-1 and time_orientation[i]>time_altitude[n+1]:
                n += 1
            
                # print(n,len(altitude))
            altitude_interpolated.append(interpolate(time_orientation[i],time_velocity_body[n],time_velocity_body[n+1],altitude[n],altitude[n+1]))

            while k < len(time_depth)-1 and time_depth[k]<time_orientation[i]:
                k+= 1
            # interpolate to find the appropriate depth for dead_reckoning
            depth_dead_reckoning[i-start_interpolate_index]=interpolate(time_orientation[i],time_depth[k-1],time_depth[k],depth[k-1],depth[k])


        j=0
        # interpolate to find the appropriate depth to compute seafloor depth for each altitude measurement
        for i in range(len(time_altitude)):        
            while j < len(time_depth)-1 and time_depth[j]<time_altitude[i]:
                j=j+1

            if j>=1:                
                seafloor_depth[i]=interpolate(time_altitude[i],time_depth[j-1],time_depth[j],depth[j-1],depth[j])+altitude[i]

        # northings eastings dead reckoning solution
        for i in range(len(time_dead_reckoning)):
            # dead reckoning solution
            if i>=1:
                [northings_dead_reckoning[i], eastings_dead_reckoning[i]]=dead_reckoning(time_dead_reckoning[i], time_dead_reckoning[i-1], north_velocity_inertia_dvl[i-1], east_velocity_inertia_dvl[i-1], northings_dead_reckoning[i-1], eastings_dead_reckoning[i-1])

        # offset sensor to plot origin/centre of vehicle
        eastings_dead_reckoning_dvl = []
        northings_dead_reckoning_dvl = []
        for i in range(len(time_dead_reckoning)):
            eastings_dead_reckoning_dvl.append(eastings_dead_reckoning[i])
            northings_dead_reckoning_dvl.append(northings_dead_reckoning[i])
            [x_offset, y_offset, z_offset] = body_to_inertial(roll_ins_dead_reckoning[i], pitch_ins_dead_reckoning[i], yaw_ins_dead_reckoning[i], origin_x_offset - dvl_x_offset, origin_y_offset - dvl_y_offset, origin_z_offset - depth_z_offset)
            northings_dead_reckoning[i] = northings_dead_reckoning[i] + x_offset
            eastings_dead_reckoning[i] = eastings_dead_reckoning[i] + y_offset
            depth_dead_reckoning[i] = depth_dead_reckoning[i] + z_offset

        #remove first term if first time_orientation is < velocity_body time
        if interpolate_remove_flag == True:

            # del time_orientation[0]
            del time_dead_reckoning[0]
            del roll_ins_dead_reckoning[0]
            del pitch_ins_dead_reckoning[0]
            del yaw_ins_dead_reckoning[0]            
            del x_velocity_interpolated[0]
            del y_velocity_interpolated[0]
            del z_velocity_interpolated[0]
            del north_velocity_inertia_dvl[0]
            del east_velocity_inertia_dvl[0]
            del down_velocity_inertia_dvl[0]
            del altitude_interpolated[0]
            del northings_dead_reckoning[0]
            del eastings_dead_reckoning[0]
            del northings_dead_reckoning_dvl[0]
            del eastings_dead_reckoning_dvl[0]
            del depth_dead_reckoning[0]
            interpolate_remove_flag = False # reset flag
        print('Complete interpolation and coordinate transfomations for velocity_body')
    
    # perform interpolations of state data to velocity_inertia time stamps (without sensor offset and correct imu to dvl flipped interpolation)
        #initialise counters for interpolation
        j=0
        k=0
    
        for i in range(len(time_velocity_inertia)):  
                           
            while j< len(time_orientation)-1 and time_orientation[j]<time_velocity_inertia[i]:
                j=j+1
            
            if j==1:
                interpolate_remove_flag = True
            else:
                roll_inertia_interpolated[i]=interpolate(time_velocity_inertia[i],time_orientation[j-1],time_orientation[j],roll[j-1],roll[j])
                pitch_inertia_interpolated[i]=interpolate(time_velocity_inertia[i],time_orientation[j-1],time_orientation[j],pitch[j-1],pitch[j])

                if abs(yaw[j]-yaw[j-1])>180:                        
                    if yaw[j]>yaw[j-1]:
                        yaw_inertia_interpolated[i]=interpolate(time_velocity_inertia[i],time_orientation[j-1],time_orientation[j],yaw[j-1],yaw[j]-360)
                        
                    else:
                        yaw_inertia_interpolated[i]=interpolate(time_velocity_inertia[i],time_orientation[j-1],time_orientation[j],yaw[j-1]-360,yaw[j])
                       
                    if yaw_inertia_interpolated[i]<0:
                        yaw_inertia_interpolated[i]=yaw_inertia_interpolated[i]+360
                        
                    elif yaw_inertia_interpolated[i]>360:
                        yaw_inertia_interpolated[i]=yaw_inertia_interpolated[i]-360  

                else:
                    yaw_inertia_interpolated[i]=interpolate(time_velocity_inertia[i],time_orientation[j-1],time_orientation[j],yaw[j-1],yaw[j])
            
            while k< len(time_depth)-1 and time_depth[k]<time_velocity_inertia[i]:
                k=k+1

            if k>=1:                
                depth_inertia_dead_reckoning[i]=interpolate(time_velocity_inertia[i],time_depth[k-1],time_depth[k],depth[k-1],depth[k])
        
        for i in range(len(time_velocity_inertia)):
            if i >= 1:                     
                [northings_inertia_dead_reckoning[i], eastings_inertia_dead_reckoning[i]]=dead_reckoning(time_velocity_inertia[i], time_velocity_inertia[i-1], north_velocity_inertia[i-1], east_velocity_inertia[i-1], northings_inertia_dead_reckoning[i-1], eastings_inertia_dead_reckoning[i-1])

        if interpolate_remove_flag == True:

            del time_velocity_inertia[0]
            del roll_inertia_interpolated[0]
            del pitch_inertia_interpolated[0]
            del yaw_inertia_interpolated[0]
            del north_velocity_inertia[0]
            del east_velocity_inertia[0]
            del down_velocity_inertia[0]
            del north_velocity_std_inertia[0]
            del east_velocity_std_inertia[0]
            del down_velocity_std_inertia[0]
            del northings_inertia_dead_reckoning[0]
            del eastings_inertia_dead_reckoning[0]
            del depth_inertia_dead_reckoning[0]
            interpolate_remove_flag = False # reset flag
        print('Complete interpolation and coordinate transfomations for velocity_inertia')


    # offset velocity body DR by initial usbl estimate
        # compare time_dead_reckoning and time_usbl
        # find initial position offset
        [northings_usbl_interpolated,eastings_usbl_interpolated] = usbl_offset(time_dead_reckoning,northings_dead_reckoning,eastings_dead_reckoning,time_usbl,northings_usbl,eastings_usbl)
        # offset the deadreackoning by this initial estimate
        for i in range(len(northings_dead_reckoning)):                 
            northings_dead_reckoning[i]=northings_dead_reckoning[i]+northings_usbl_interpolated
            eastings_dead_reckoning[i]=eastings_dead_reckoning[i]+eastings_usbl_interpolated

            northings_dead_reckoning_dvl[i]=northings_dead_reckoning_dvl[i]+northings_usbl_interpolated
            eastings_dead_reckoning_dvl[i]=eastings_dead_reckoning_dvl[i]+eastings_usbl_interpolated

    # offset velocity inertial DR by initial usbl estimate
        # compare time_velocity_inertia and time_usbl
        # find initial position offset
        [northings_usbl_interpolated,eastings_usbl_interpolated] = usbl_offset(time_velocity_inertia,northings_inertia_dead_reckoning,eastings_inertia_dead_reckoning,time_usbl,northings_usbl,eastings_usbl)
        # offset the deadreackoning by this initial estimate
        for i in range(len(northings_inertia_dead_reckoning)):                
            northings_inertia_dead_reckoning[i]=northings_inertia_dead_reckoning[i]+northings_usbl_interpolated
            eastings_inertia_dead_reckoning[i]=eastings_inertia_dead_reckoning[i]+eastings_usbl_interpolated        

    # perform interpolations of state data to camera1 time stamps
        if len(time_camera1) > 1:
            j=0
            n=0
            if time_camera1[0]>time_dead_reckoning[-1] or time_camera1[-1]<time_dead_reckoning[0]: #Check if camera activates before dvl and orientation sensors.
                print('Camera1 timestamps does not overlap with dead reckoning data, check timestamp_history.pdf via -v option.')
            else:
                camera1_overlap_flag = 0
                for i in range(len(time_camera1)):
                    if time_camera1[i]<time_dead_reckoning[0]:
                        camera1_overlap_flag = 1
                        pass
                    else:
                        del time_camera1[:i]
                        break
                for i in range(len(time_camera1)):
                    if j>=len(time_dead_reckoning)-1:
                        del time_camera1[i:]
                        camera1_overlap_flag = 1
                        break
                    while time_dead_reckoning[j] < time_camera1[i]:
                        if j+1>len(time_dead_reckoning)-1 or time_dead_reckoning[j+1]>time_camera1[-1]:
                            break
                        j += 1
                    #if j>=1: ?
                    camera1_roll.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],roll_ins_dead_reckoning[j-1],roll_ins_dead_reckoning[j]))
                    camera1_pitch.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],pitch_ins_dead_reckoning[j-1],pitch_ins_dead_reckoning[j]))
                    if abs(yaw_ins_dead_reckoning[j]-yaw_ins_dead_reckoning[j-1])>180:                        
                        if yaw_ins_dead_reckoning[j]>yaw_ins_dead_reckoning[j-1]:
                            camera1_yaw.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],yaw_ins_dead_reckoning[j-1],yaw_ins_dead_reckoning[j]-360))                       
                        else:
                            camera1_yaw.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],yaw_ins_dead_reckoning[j-1]-360,yaw_ins_dead_reckoning[j]))    
                        if camera1_yaw[i]<0:
                            camera1_yaw[i]=camera1_yaw[i]+360
                        elif camera1_yaw[i]>360:
                            camera1_yaw[i]=camera1_yaw[i]-360  
                    else:
                        camera1_yaw.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],yaw_ins_dead_reckoning[j-1],yaw_ins_dead_reckoning[j]))
                    
                    # while n<len(time_altitude)-1 and time_altitude[n+1]<time_camera1[i]:
                    #     n += 1
                    # camera1_altitude.append(interpolate(time_camera1[i],time_altitude[n],time_altitude[n+1],altitude[n],altitude[n+1]))
                    camera1_altitude.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],altitude_interpolated[j-1],altitude_interpolated[j]))

                    [x_offset,y_offset,z_offset] = body_to_inertial(camera1_roll[i],camera1_pitch[i],camera1_yaw[i], origin_x_offset - camera1_x_offset, origin_y_offset - camera1_y_offset, origin_z_offset - camera1_z_offset)
                    
                    camera1_dead_reckoning_northings.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],northings_dead_reckoning[j-1],northings_dead_reckoning[j])-x_offset)
                    camera1_dead_reckoning_eastings.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],eastings_dead_reckoning[j-1],eastings_dead_reckoning[j])-y_offset)
                    camera1_dead_reckoning_depth.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],depth_dead_reckoning[j-1],depth_dead_reckoning[j])-z_offset)
                if camera1_overlap_flag == 1:
                    print('Camera1 data more than dead reckoning data. Only processed overlapping data and ignored the rest.')
                print('Complete interpolation and coordinate transfomations for camera1')

    # perform interpolations of state data to camera2 time stamps
        if len(time_camera2) > 1:
            j=0
            n=0
            if time_camera2[0]>time_dead_reckoning[-1] or time_camera2[-1]<time_dead_reckoning[0]: #Check if camera activates before dvl and orientation sensors.
                print('Camera2 timestamps does not overlap with dead reckoning data, check timestamp_history.pdf via -v option.')
            else:
                camera2_overlap_flag = 0
                for i in range(len(time_camera2)):
                    if time_camera2[i]<time_dead_reckoning[0]:
                        camera2_overlap_flag = 1
                        pass
                    else:
                        del time_camera2[:i]
                        break
                for i in range(len(time_camera2)):
                    if j>=len(time_dead_reckoning)-1:
                        del time_camera2[i:]
                        camera2_overlap_flag = 1
                        break
                    while time_dead_reckoning[j] < time_camera2[i]:
                        if j+1>len(time_dead_reckoning)-1 or time_dead_reckoning[j+1]>time_camera2[-1]:
                            break
                        j += 1
                    #if j>=1: ?
                    camera2_roll.append(interpolate(time_camera2[i],time_dead_reckoning[j-1],time_dead_reckoning[j],roll_ins_dead_reckoning[j-1],roll_ins_dead_reckoning[j]))
                    camera2_pitch.append(interpolate(time_camera2[i],time_dead_reckoning[j-1],time_dead_reckoning[j],pitch_ins_dead_reckoning[j-1],pitch_ins_dead_reckoning[j]))
                    if abs(yaw_ins_dead_reckoning[j]-yaw_ins_dead_reckoning[j-1])>180:                        
                        if yaw_ins_dead_reckoning[j]>yaw_ins_dead_reckoning[j-1]:
                            camera2_yaw.append(interpolate(time_camera2[i],time_dead_reckoning[j-1],time_dead_reckoning[j],yaw_ins_dead_reckoning[j-1],yaw_ins_dead_reckoning[j]-360))                       
                        else:
                            camera2_yaw.append(interpolate(time_camera2[i],time_dead_reckoning[j-1],time_dead_reckoning[j],yaw_ins_dead_reckoning[j-1]-360,yaw_ins_dead_reckoning[j]))    
                        if camera2_yaw[i]<0:
                            camera2_yaw[i]=camera2_yaw[i]+360
                        elif camera2_yaw[i]>360:
                            camera2_yaw[i]=camera2_yaw[i]-360  
                    else:
                        camera2_yaw.append(interpolate(time_camera2[i],time_dead_reckoning[j-1],time_dead_reckoning[j],yaw_ins_dead_reckoning[j-1],yaw_ins_dead_reckoning[j]))
                    
                    # while n<len(time_altitude)-1 and time_altitude[n+1]<time_camera2[i]:
                    #     n += 1
                    # camera2_altitude.append(interpolate(time_camera2[i],time_altitude[n],time_altitude[n+1],altitude[n],altitude[n+1]))
                    camera2_altitude.append(interpolate(time_camera2[i],time_dead_reckoning[j-1],time_dead_reckoning[j],altitude_interpolated[j-1],altitude_interpolated[j]))

                    [x_offset,y_offset,z_offset] = body_to_inertial(camera2_roll[i],camera2_pitch[i],camera2_yaw[i], origin_x_offset - camera2_x_offset, origin_y_offset - camera2_y_offset, origin_z_offset - camera2_z_offset)
                    
                    camera2_dead_reckoning_northings.append(interpolate(time_camera2[i],time_dead_reckoning[j-1],time_dead_reckoning[j],northings_dead_reckoning[j-1],northings_dead_reckoning[j])-x_offset)
                    camera2_dead_reckoning_eastings.append(interpolate(time_camera2[i],time_dead_reckoning[j-1],time_dead_reckoning[j],eastings_dead_reckoning[j-1],eastings_dead_reckoning[j])-y_offset)
                    camera2_dead_reckoning_depth.append(interpolate(time_camera2[i],time_dead_reckoning[j-1],time_dead_reckoning[j],depth_dead_reckoning[j-1],depth_dead_reckoning[j])-z_offset)
                if camera2_overlap_flag == 1:
                    print('Camera2 data more than dead reckoning data. Only processed overlapping data and ignored the rest.')
                print('Complete interpolation and coordinate transfomations for camera2')

    # perform interpolations of state data to camera3 time stamps
        if len(time_camera3) > 1: #simplify these sections (loop through list or raise flag...)
            j=0
            n=0
            if time_camera3[0]>time_dead_reckoning[-1] or time_camera3[-1]<time_dead_reckoning[0]: #Check if camera activates before dvl and orientation sensors.
                print('Camera3 timestamps does not overlap with dead reckoning data, check timestamp_history.pdf via -v option.')
            else:
                camera3_overlap_flag = 0
                for i in range(len(time_camera3)):
                    if time_camera3[i]<time_dead_reckoning[0]:
                        camera3_overlap_flag = 1
                        pass
                    else:
                        del time_camera3[:i]
                        break
                for i in range(len(time_camera3)):
                    if j>=len(time_dead_reckoning)-1:
                        camera3_overlap_flag = 1
                        del time_camera3[i:]
                        break
                    while time_dead_reckoning[j] < time_camera3[i]:
                        if j+1>len(time_dead_reckoning)-1 or time_dead_reckoning[j+1]>time_camera3[-1]:
                            break
                        j += 1
                    camera3_roll.append(interpolate(time_camera3[i],time_dead_reckoning[j-1],time_dead_reckoning[j],roll_ins_dead_reckoning[j-1],roll_ins_dead_reckoning[j]))
                    camera3_pitch.append(interpolate(time_camera3[i],time_dead_reckoning[j-1],time_dead_reckoning[j],pitch_ins_dead_reckoning[j-1],pitch_ins_dead_reckoning[j]))
                    if abs(yaw_ins_dead_reckoning[j]-yaw_ins_dead_reckoning[j-1])>180:                        
                        if yaw_ins_dead_reckoning[j]>yaw_ins_dead_reckoning[j-1]:
                            camera3_yaw.append(interpolate(time_camera3[i],time_dead_reckoning[j-1],time_dead_reckoning[j],yaw_ins_dead_reckoning[j-1],yaw_ins_dead_reckoning[j]-360))                       
                        else:
                            camera3_yaw.append(interpolate(time_camera3[i],time_dead_reckoning[j-1],time_dead_reckoning[j],yaw_ins_dead_reckoning[j-1]-360,yaw_ins_dead_reckoning[j]))    
                        if camera3_yaw[i]<0:
                            camera3_yaw[i]=camera3_yaw[i]+360
                        elif camera3_yaw[i]>360:
                            camera3_yaw[i]=camera3_yaw[i]-360  
                    else:
                        camera3_yaw.append(interpolate(time_camera3[i],time_dead_reckoning[j-1],time_dead_reckoning[j],yaw_ins_dead_reckoning[j-1],yaw_ins_dead_reckoning[j]))
                    
                    # while n<len(time_altitude)-1 and time_altitude[n+1]<time_camera3[i]:
                    #     n += 1
                    # camera3_altitude.append(interpolate(time_camera3[i],time_altitude[n],time_altitude[n+1],altitude[n],altitude[n+1]))

                    camera3_altitude.append(interpolate(time_camera3[i],time_dead_reckoning[j-1],time_dead_reckoning[j],altitude_interpolated[j-1],altitude_interpolated[j]))

                    [x_offset,y_offset,z_offset] = body_to_inertial(camera3_roll[i],camera3_pitch[i],camera3_yaw[i], origin_x_offset - camera3_x_offset, origin_y_offset - camera3_y_offset, origin_z_offset - camera3_z_offset)
                    
                    camera3_dead_reckoning_northings.append(interpolate(time_camera3[i],time_dead_reckoning[j-1],time_dead_reckoning[j],northings_dead_reckoning[j-1],northings_dead_reckoning[j])-x_offset)
                    camera3_dead_reckoning_eastings.append(interpolate(time_camera3[i],time_dead_reckoning[j-1],time_dead_reckoning[j],eastings_dead_reckoning[j-1],eastings_dead_reckoning[j])-y_offset)
                    camera3_dead_reckoning_depth.append(interpolate(time_camera3[i],time_dead_reckoning[j-1],time_dead_reckoning[j],depth_dead_reckoning[j-1],depth_dead_reckoning[j])-z_offset)
                
                if camera3_overlap_flag == 1:
                    print('Camera3 data more than dead reckoning data. Only processed overlapping data and ignored the rest.')

                print('Complete interpolation and coordinate transfomations for camera3')

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
            if len(time_dead_reckoning) > 1:
                print("Writing outputs to auv_centre.csv ...")
                with open(csvpath + os.sep + 'auv_centre.csv' ,'w') as fileout:
                    fileout.write('Timestamp, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m]\n')
                for i in range(len(time_dead_reckoning)):
                    with open(csvpath + os.sep + 'auv_centre.csv' ,'a') as fileout:
                        try:
                            fileout.write(str(time_dead_reckoning[i])+','+str(northings_dead_reckoning[i])+','+str(eastings_dead_reckoning[i])+','+str(depth_dead_reckoning[i])+','+str(roll_ins_dead_reckoning[i])+','+str(pitch_ins_dead_reckoning[i])+','+str(yaw_ins_dead_reckoning[i])+','+str(altitude_interpolated[i])+'\n')
                            fileout.close()
                        except IndexError:
                            break

            if len(time_usbl) > 1:
                print("Writing outputs to usbl.csv ...")
                with open(csvpath + os.sep + 'usbl.csv' ,'w') as fileout:
                    fileout.write('Timestamp, Northing [m], Easting [m], Depth [m]\n')
                for i in range(len(time_usbl)):
                    with open(csvpath + os.sep + 'usbl.csv' ,'a') as fileout:
                        try:
                            fileout.write(str(time_usbl[i])+','+str(northings_usbl[i])+','+str(eastings_usbl[i])+','+str(depth_usbl[i])+'\n')
                            fileout.close()
                        except IndexError:
                            break

            if len(time_velocity_inertia) > 1:
                print("Writing outputs to {}.csv ...".format(velocity_inertial_sensor_name))
                with open(csvpath + os.sep + '{}.csv'.format(velocity_inertial_sensor_name), 'w') as fileout:
                    fileout.write('Timestamp, Northing [m], Easting [m], Depth [m]\n')
                for i in range(len(time_velocity_inertia)):
                    with open(csvpath + os.sep + '{}.csv'.format(velocity_inertial_sensor_name) ,'a') as fileout:
                        try:
                            fileout.write(str(time_velocity_inertia[i])+','+str(northings_inertia_dead_reckoning[i])+','+str(eastings_inertia_dead_reckoning[i])+','+str(depth_inertia_dead_reckoning[i])+'\n')
                            fileout.close()
                        except IndexError:
                            break

            if len(time_dead_reckoning) > 1:
                print("Writing outputs to DVL.csv ...")
                with open(csvpath + os.sep + 'DVL.csv' ,'w') as fileout:
                    fileout.write('Timestamp, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg]\n')
                for i in range(len(time_dead_reckoning)):
                    with open(csvpath + os.sep + 'DVL.csv' ,'a') as fileout:
                        try:
                            fileout.write(str(time_dead_reckoning[i])+','+str(northings_dead_reckoning_dvl[i])+','+str(eastings_dead_reckoning_dvl[i])+','+str([i])+str(roll_ins_dead_reckoning[i])+','+str(pitch_ins_dead_reckoning[i])+','+str(yaw_ins_dead_reckoning[i])+'\n')
                            fileout.close()
                        except IndexError:
                            break
### Image file naming step not very robust, needs improvement
            #*** maybe add timestamp at the last column of image.csv
            if len(time_camera1) > 1:
                print("Writing outputs to {}.csv ...".format(camera1_sensor_name))
                with open(csvpath + os.sep + '{}.csv'.format(camera1_sensor_name) ,'w') as fileout:
                    fileout.write('Imagenumber, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m]\n')
                for i in range(len(time_camera1)):
                    with open(csvpath + os.sep + '{}.csv'.format(camera1_sensor_name) ,'a') as fileout:
                        try:
                            imagenumber = filename_camera1[i][-11:-4]
                            if imagenumber.isdigit():
                                image_filename=int(imagenumber)
                            else:
                                image_filename=filename_camera1[i]
                            fileout.write(str(image_filename)+','+str(camera1_dead_reckoning_northings[i])+','+str(camera1_dead_reckoning_eastings[i])+','+str(camera1_dead_reckoning_depth[i])+','+str(camera1_roll[i])+','+str(camera1_pitch[i])+','+str(camera1_yaw[i])+','+str(camera1_altitude[i])+'\n')
                            fileout.close()
                        except IndexError:
                            break

            if len(time_camera2) > 1:
                print("Writing outputs to {}.csv ...".format(camera2_sensor_name))
                with open(csvpath + os.sep + '{}.csv'.format(camera2_sensor_name) ,'w') as fileout:
                    fileout.write('Imagenumber, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m]\n')
                for i in range(len(time_camera2)):
                    with open(csvpath + os.sep + '{}.csv'.format(camera2_sensor_name) ,'a') as fileout:
                        try:
                            imagenumber = filename_camera2[i][-11:-4]
                            if imagenumber.isdigit():
                                image_filename=int(imagenumber)
                            else:
                                image_filename=filename_camera2[i]
                            fileout.write(str(image_filename)+','+str(camera2_dead_reckoning_northings[i])+','+str(camera2_dead_reckoning_eastings[i])+','+str(camera2_dead_reckoning_depth[i])+','+str(camera2_roll[i])+','+str(camera2_pitch[i])+','+str(camera2_yaw[i])+','+str(camera2_altitude[i])+'\n')
                            fileout.close()
                        except IndexError:
                            break

            if len(time_camera3) > 1:
                print("Writing outputs to {}.csv ...".format(camera1_sensor_name))
                with open(csvpath + os.sep + '{}.csv'.format(camera1_sensor_name) ,'w') as fileout:
                    fileout.write('Imagenumber, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m]\n')
                for i in range(len(time_camera3)):
                    with open(csvpath + os.sep + '{}.csv'.format(camera1_sensor_name) ,'a') as fileout:
                        try:
                            imagenumber = filename_camera3[i][-11:-4]
                            if imagenumber.isdigit():
                                image_filename=int(imagenumber)
                            else:
                                image_filename=filename_camera3[i]
                            fileout.write(str(image_filename)+','+str(camera3_dead_reckoning_northings[i])+','+str(camera3_dead_reckoning_eastings[i])+','+str(camera3_dead_reckoning_depth[i])+','+str(camera3_roll[i])+','+str(camera3_pitch[i])+','+str(camera3_yaw[i])+','+str(camera3_altitude[i])+'\n')
                            fileout.close()
                        except IndexError:
                            break

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
            ax.plot(time_orientation, yaw,'r.',label='Yaw')   
            ax.plot(time_orientation, roll,'b.',label='Roll')      
            ax.plot(time_orientation, pitch,'g.',label='Pitch')                     
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
            ax1.plot(time_dead_reckoning,north_velocity_inertia_dvl, 'ro',label='DVL')#time_velocity_body,north_velocity_inertia_dvl, 'ro',label='DVL')
            ax1.plot(time_velocity_inertia,north_velocity_inertia, 'b.',label=velocity_inertial_sensor_name)
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Velocity, m/s')
            ax1.legend()
            ax1.grid(True)
            ax1.set_title('north velocity')
            ax2 = fig.add_subplot(323)            
            ax2.plot(time_dead_reckoning,east_velocity_inertia_dvl,'ro',label='DVL')
            ax2.plot(time_velocity_inertia,east_velocity_inertia,'b.',label=velocity_inertial_sensor_name)
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Velocity, m/s')
            ax2.legend()
            ax2.grid(True)
            ax2.set_title('east velocity')
            ax3 = fig.add_subplot(325)            
            ax3.plot(time_dead_reckoning,down_velocity_inertia_dvl,'ro',label='DVL')#time_velocity_body,down_velocity_inertia_dvl,'r.',label='DVL')
            ax3.plot(time_velocity_inertia,down_velocity_inertia,'b.',label=velocity_inertial_sensor_name)
            ax3.set_xlabel('Epoch time, s')
            ax3.set_ylabel('Velocity, m/s')
            ax3.legend()
            ax3.grid(True)
            ax3.set_title('down velocity')
            ax4 = fig.add_subplot(322)
            ax4.plot(time_dead_reckoning,x_velocity_interpolated, 'r.',label='Surge') #time_velocity_body,x_velocity, 'r.',label='Surge')
            ax4.set_xlabel('Epoch time, s')
            ax4.set_ylabel('Velocity, m/s')
            ax4.legend()
            ax4.grid(True)
            ax4.set_title('x velocity')
            ax5 = fig.add_subplot(324)
            ax5.plot(time_dead_reckoning,y_velocity_interpolated, 'g.',label='Sway')#time_velocity_body,y_velocity, 'g.',label='Sway')
            ax5.set_xlabel('Epoch time, s')
            ax5.set_ylabel('Velocity, m/s')
            ax5.legend()
            ax5.grid(True)
            ax5.set_title('y velocity')
            ax6 = fig.add_subplot(326)
            ax6.plot(time_dead_reckoning,z_velocity_interpolated, 'b.',label='Heave')#time_velocity_body,z_velocity, 'b.',label='Heave')
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
            ax1.plot(time_dead_reckoning,northings_dead_reckoning_dvl,'r.',label='DVL')#time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
            ax1.plot(time_velocity_inertia,northings_inertia_dead_reckoning,'g.',label=velocity_inertial_sensor_name)
            ax1.plot(time_usbl, northings_usbl,'b.',label='USBL')
            ax1.plot(time_dead_reckoning,northings_dead_reckoning,'c.',label='Centre')#time_velocity_body,northings_dead_reckoning,'b.')
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Northings, m')
            ax1.grid(True)
            ax1.legend()#loc='upper right', bbox_to_anchor=(1, -0.2))
            ax1.set_title('Northings')
            ax2 = fig.add_subplot(222)
            ax2.plot(time_dead_reckoning,eastings_dead_reckoning_dvl,'r.',label='DVL')#time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
            ax2.plot(time_velocity_inertia,eastings_inertia_dead_reckoning,'g.',label=velocity_inertial_sensor_name)
            ax2.plot(time_usbl, eastings_usbl,'b.',label='USBL')
            ax2.plot(time_dead_reckoning,eastings_dead_reckoning,'c.',label='Centre')#time_velocity_body,eastings_dead_reckoning,'b.')
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Eastings, m')
            ax2.grid(True)
            ax2.legend()
            ax2.set_title('Eastings')
            ax3 = fig.add_subplot(223)
            ax3.plot(time_usbl,depth_usbl,'b.',label='USBL depth') 
            ax3.plot(time_depth,depth,'g-',label='Depth Sensor') 
            ax3.plot(time_altitude,seafloor_depth,'r-',label='Seafloor') 
            ax3.plot(time_dead_reckoning,depth_dead_reckoning,'c-',label='Depth DR')#time_velocity_body,eastings_dead_reckoning,'b.')
            ax3.set_xlabel('Epoch time, s')
            ax3.set_ylabel('Depth, m')
            plt.gca().invert_yaxis()
            ax3.grid(True)
            ax3.legend()
            ax3.set_title('Depth')
            ax4 = fig.add_subplot(224)
            ax4.plot(time_altitude,altitude,'r.',label='Altitude')              
            ax4.set_xlabel('Epoch time, s')
            ax4.set_ylabel('Altitude, m')
            ax4.set_xlim(min(time_depth),max(time_depth))
            ax4.grid(True)
            ax4.legend()
            fig.tight_layout()
            plt.savefig(plotpath + os.sep + 'deadreckoning_vs_time.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # usbl latitude longitude
            print('...plotting usbl_LatLong_vs_NorthEast...')

            fig=plt.figure(figsize=(10,5))
            ax1 = fig.add_subplot(121)
            ax1.plot(longitude_usbl,latitude_usbl,'b.')                 
            ax1.set_xlabel('Longitude, degrees')
            ax1.set_ylabel('Latitude, degrees')
            ax1.grid(True)
            ax2 = fig.add_subplot(122)
            ax2.plot(eastings_usbl,northings_usbl,'b.',label='Reference ['+str(latitude_reference)+','+str(longitude_reference)+']')                 
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
            ax1.plot(time_orientation,roll_std,'r.',label='roll_std')
            ax1.plot(time_orientation,pitch_std,'g.',label='pitch_std')
            ax1.plot(time_orientation,yaw_std,'b.',label='yaw_std')
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Angle, degrees')
            ax1.legend()
            ax1.grid(True)
            ax2 = fig.add_subplot(234)
            ax2.plot(time_depth,depth_std,'b.',label='depth_std')
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Depth, m')
            ax2.legend()
            ax2.grid(True)
            ax3 = fig.add_subplot(232)
            ax3.plot(time_velocity_body,x_velocity_std,'r.',label='x_velocity_std')
            ax3.plot(time_velocity_body,y_velocity_std,'g.',label='y_velocity_std')
            ax3.plot(time_velocity_body,z_velocity_std,'b.',label='z_velocity_std')
            ax3.set_xlabel('Epoch time, s')
            ax3.set_ylabel('Velocity, m/s')
            ax3.legend()
            ax3.grid(True)
            ax4 = fig.add_subplot(235)
            ax4.plot(time_velocity_inertia,north_velocity_std_inertia,'r.',label='north_velocity_std_inertia')
            ax4.plot(time_velocity_inertia,east_velocity_std_inertia,'g.',label='east_velocity_std_inertia')
            ax4.plot(time_velocity_inertia,down_velocity_std_inertia,'b.',label='down_velocity_std_inertia')
            ax4.set_xlabel('Epoch time, s')
            ax4.set_ylabel('Velocity, m/s')
            ax4.legend()
            ax4.grid(True)
            ax5 = fig.add_subplot(233)
            ax5.plot(time_usbl,latitude_std_usbl,'r.',label='latitude_std_usbl')
            ax5.plot(time_usbl,longitude_std_usbl,'g.',label='longitude_std_usbl')
            ax5.set_xlabel('Epoch time, s')
            ax5.set_ylabel('LatLong, degrees')
            ax5.legend()
            ax5.grid(True)
            ax6 = fig.add_subplot(236)
            ax6.plot(time_usbl,northings_std_usbl,'r.',label='northings_std_usbl')
            ax6.plot(time_usbl,eastings_std_usbl,'g.',label='eastings_std_usbl')
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
            ax.plot(camera1_dead_reckoning_eastings,camera1_dead_reckoning_northings,'y.',label='Camera1')
            ax.plot(eastings_dead_reckoning,northings_dead_reckoning,'r.',label='Centre')
            ax.plot(eastings_dead_reckoning_dvl,northings_dead_reckoning_dvl,'g.',label='DVL')
            ax.plot(eastings_inertia_dead_reckoning,northings_inertia_dead_reckoning,'m.',label=velocity_inertial_sensor_name)
            ax.plot(eastings_usbl, northings_usbl,'c.',label='USBL')
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

            trace11a = create_trace(time_orientation, yaw, 'Yaw', 'red')
            trace11b = create_trace(time_orientation, roll, 'Roll', 'blue')
            trace11c = create_trace(time_orientation, pitch, 'Pitch', 'green')
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

            trace11a = create_trace(time_dead_reckoning, north_velocity_inertia_dvl, 'DVL north velocity', 'red')
            trace11b = create_trace(time_velocity_inertia, north_velocity_inertia, '{} north velocity'.format(velocity_inertial_sensor_name), 'blue')
            # plot1=[trace11a, trace11b]
            trace21a = create_trace(time_dead_reckoning, east_velocity_inertia_dvl, 'DVL east velocity', 'red')
            trace21b = create_trace(time_velocity_inertia, east_velocity_inertia, '{} east velocity'.format(velocity_inertial_sensor_name), 'blue')
            trace31a = create_trace(time_dead_reckoning, down_velocity_inertia_dvl, 'DVL down velocity', 'red')
            trace31b = create_trace(time_velocity_inertia, down_velocity_inertia, '{} down velocity'.format(velocity_inertial_sensor_name), 'blue')
            trace12a = create_trace(time_dead_reckoning, x_velocity_interpolated, 'DVL x velocity', 'red')
            trace22a = create_trace(time_dead_reckoning, y_velocity_interpolated, 'DVL y velocity', 'red')
            trace32a = create_trace(time_dead_reckoning, z_velocity_interpolated, 'DVL z velocity', 'red')
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

            trace11a = create_trace(time_dead_reckoning, northings_dead_reckoning_dvl, 'Northing DVL', 'red')
            trace11b = create_trace(time_velocity_inertia, northings_inertia_dead_reckoning, 'Northing {}'.format(velocity_inertial_sensor_name), 'green')
            trace11c = create_trace(time_usbl, northings_usbl, 'Northing USBL', 'blue')
            trace11d = create_trace(time_dead_reckoning, northings_dead_reckoning, 'Northing Centre', 'orange')
            trace12a = create_trace(time_dead_reckoning, eastings_dead_reckoning_dvl, 'Easting DVL', 'red')
            trace12b = create_trace(time_velocity_inertia, eastings_inertia_dead_reckoning, 'Easting {}'.format(velocity_inertial_sensor_name), 'green')
            trace12c = create_trace(time_usbl, eastings_usbl, 'Easting USBL', 'blue')
            trace12d = create_trace(time_dead_reckoning, eastings_dead_reckoning, 'Easting Centre', 'orange')
            trace21a = create_trace(time_altitude, seafloor_depth, 'Depth  Seafloor (Depth Sensor + Altitude)', 'red')
            trace21b = create_trace(time_depth, depth, 'Depth Sensor', 'purple')
            trace21c = create_trace(time_usbl, depth_usbl, 'Depth USBL', 'blue')
            trace21d = create_trace(time_dead_reckoning, depth_dead_reckoning, 'Depth Centre', 'orange')
            trace22a = create_trace(time_altitude, altitude, 'Altitude', 'red')
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
            trace11a = create_trace(time_orientation, roll_std, 'roll std', 'red')
            trace11b = create_trace(time_orientation, pitch_std, 'pitch std', 'green')
            trace11c = create_trace(time_orientation, yaw_std, 'yaw std', 'blue')
            trace12a = create_trace(time_velocity_body, x_velocity_std, 'x velocity std', 'red')
            trace12b = create_trace(time_velocity_body, y_velocity_std, 'y velocity std', 'green')
            trace12c = create_trace(time_velocity_body, z_velocity_std, 'z velocity std', 'blue')
            trace13a = create_trace(time_usbl, latitude_std_usbl, 'latitude std usbl', 'red')
            trace13b = create_trace(time_usbl, longitude_std_usbl, 'longitude std usbl', 'green')
            trace21a = create_trace(time_depth, depth_std, 'depth std', 'red')
            trace22a = create_trace(time_velocity_inertia, north_velocity_std_inertia, 'north velocity std inertial', 'red')
            trace22b = create_trace(time_velocity_inertia, east_velocity_std_inertia, 'east velocity std inertial', 'green')
            trace22c = create_trace(time_velocity_inertia, down_velocity_std_inertia, 'down velocity std inertial', 'blue')
            trace23a = create_trace(time_usbl, northings_std_usbl, 'northing std usbl', 'red')
            trace23b = create_trace(time_usbl, eastings_std_usbl, 'easting std usbl', 'green')
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
            for i in [time_camera1, time_dead_reckoning, time_velocity_inertia, time_usbl]:
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

            make_data('Camera1',camera1_dead_reckoning_eastings,camera1_dead_reckoning_northings)
            make_data('Centre',eastings_dead_reckoning,northings_dead_reckoning)
            make_data('DVL',eastings_dead_reckoning_dvl,northings_dead_reckoning_dvl)
            make_data(velocity_inertial_sensor_name,eastings_inertia_dead_reckoning,northings_inertia_dead_reckoning)
            make_data('USBL',eastings_usbl,northings_usbl)

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
                
                for j in [['Camera1',time_camera1,camera1_dead_reckoning_eastings,camera1_dead_reckoning_northings],['Centre',time_dead_reckoning,eastings_dead_reckoning,northings_dead_reckoning],['DVL',time_dead_reckoning,eastings_dead_reckoning_dvl,northings_dead_reckoning_dvl],[velocity_inertial_sensor_name,time_velocity_inertia,eastings_inertia_dead_reckoning,northings_inertia_dead_reckoning],['USBL',time_usbl,eastings_usbl,northings_usbl]]:
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
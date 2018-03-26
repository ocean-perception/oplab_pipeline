# extract_data

# Assumes filename_camera of 1, 2, and 3 contains the image number between the last 11 and 4 characters for appropriate csv pose estimate files output. e.g. 'Xviii/Cam51707923/0094853.raw'

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

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from datetime import datetime

sys.path.append("..")
from lib_calculus.interpolate import interpolate
from lib_localisation.dead_reckoning import dead_reckoning
from lib_localisation.usbl_offset import usbl_offset
from lib_coordinates.body_to_inertial import body_to_inertial

class extract_data:
    def __init__(self,filepath,ftype,start_time,finish_time,plot,csv_write,show_plot):

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

    # velocity inertia placeholders
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
        
    # orientation placeholders (INS)
        time_orientation=[]
        roll=[]
        pitch=[]
        yaw=[]
        roll_std=[]
        pitch_std=[]
        yaw_std=[]
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

    # altitude placeholders
        time_altitude=[]
        altitude=[]
        seafloor_depth=[] # interpolate depth and add altitude for every altitude measurement

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
        
            for i in range(0,len(load_data)): 
                if 'origin' in load_data:
                    origin_flag=1
                    latitude_reference = load_data['origin']['latitude']
                    longitude_reference = load_data['origin']['longitude']
                    coordinate_reference = load_data['origin']['coordinate_reference_system']
                    date = load_data['origin']['date']          

        # compute the offset of sensors
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
                    # origin_flag=1
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
                        serial_camera1=parsed_json_data[i]['camera1'][0]['serial']
                        time_camera2.append(parsed_json_data[i]['camera2'][0]['epoch_timestamp'])
                        filename_camera2.append(parsed_json_data[i]['camera2'][0]['filename'])
                        serial_camera2=parsed_json_data[i]['camera2'][0]['serial']

                    if 'laser' in parsed_json_data[i]['category']:
                        time_camera3.append(parsed_json_data[i]['epoch_timestamp'])#LC
                        filename_camera3.append(parsed_json_data[i]['filename'])
                        serial_camera3=parsed_json_data[i]['serial']


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
            if i>1:
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

            # For plotly+dash+pandas
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

            # For plotly+dash+pandas
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

            # For plotly+dash+pandas
            if len(time_velocity_inertia) > 1:
                print("Writing outputs to PHINS.csv ...")
                with open(csvpath + os.sep + 'PHINS.csv' ,'w') as fileout:
                    fileout.write('Timestamp, Northing [m], Easting [m], Depth [m]\n')
                for i in range(len(time_velocity_inertia)):
                    with open(csvpath + os.sep + 'PHINS.csv' ,'a') as fileout:
                        try:
                            fileout.write(str(time_velocity_inertia[i])+','+str(northings_inertia_dead_reckoning[i])+','+str(eastings_inertia_dead_reckoning[i])+','+str(depth_inertia_dead_reckoning[i])+'\n')
                            fileout.close()
                        except IndexError:
                            break

            # For plotly+dash+pandas
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

            if len(time_camera1) > 1:
                print("Writing outputs to {}.csv ...".format(serial_camera1))
                with open(csvpath + os.sep + '{}.csv'.format(serial_camera1) ,'w') as fileout:
                    fileout.write('Imagenumber, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m]\n')
                for i in range(len(time_camera1)):
                    with open(csvpath + os.sep + '{}.csv'.format(serial_camera1) ,'a') as fileout:
                        try:
                            imagenumber = filename_camera1[i][-11:-4]
                            if imagenumber.isdigit():
                                imagenumber=int(imagenumber)
                            fileout.write(str(imagenumber)+','+str(camera1_dead_reckoning_northings[i])+','+str(camera1_dead_reckoning_eastings[i])+','+str(camera1_dead_reckoning_depth[i])+','+str(camera1_roll[i])+','+str(camera1_pitch[i])+','+str(camera1_yaw[i])+','+str(camera1_altitude[i])+'\n')
                            fileout.close()
                        except IndexError:
                            break

            if len(time_camera2) > 1:
                print("Writing outputs to {}.csv ...".format(serial_camera2))
                with open(csvpath + os.sep + '{}.csv'.format(serial_camera2) ,'w') as fileout:
                    fileout.write('Imagenumber, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m]\n')
                for i in range(len(time_camera2)):
                    with open(csvpath + os.sep + '{}.csv'.format(serial_camera2) ,'a') as fileout:
                        try:
                            imagenumber = filename_camera2[i][-11:-4]
                            if imagenumber.isdigit():
                                imagenumber=int(imagenumber)
                            fileout.write(str(imagenumber)+','+str(camera2_dead_reckoning_northings[i])+','+str(camera2_dead_reckoning_eastings[i])+','+str(camera2_dead_reckoning_depth[i])+','+str(camera2_roll[i])+','+str(camera2_pitch[i])+','+str(camera2_yaw[i])+','+str(camera2_altitude[i])+'\n')
                            fileout.close()
                        except IndexError:
                            break

            if len(time_camera3) > 1:
                print("Writing outputs to {}.csv ...".format(serial_camera3))
                with open(csvpath + os.sep + '{}.csv'.format(serial_camera3) ,'w') as fileout:
                    fileout.write('Imagenumber, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m]\n')
                for i in range(len(time_camera3)):
                    with open(csvpath + os.sep + '{}.csv'.format(serial_camera3) ,'a') as fileout:
                        try:
                            imagenumber = filename_camera3[i][-11:-4]
                            if imagenumber.isdigit():
                                imagenumber = int(imagenumber)
                            fileout.write(str(imagenumber)+','+str(camera3_dead_reckoning_northings[i])+','+str(camera3_dead_reckoning_eastings[i])+','+str(camera3_dead_reckoning_depth[i])+','+str(camera3_roll[i])+','+str(camera3_pitch[i])+','+str(camera3_yaw[i])+','+str(camera3_altitude[i])+'\n')
                            fileout.close()
                        except IndexError:
                            break

            print('Complete extraction of data: ', csvpath)
        
    # plot data ### PLOT THESE ALL IN PLOTLY, Maybe with range slider
        if plot is True:
            print('Plotting data ...')
            plotpath = renavpath + os.sep + 'plots'
            
            if os.path.isdir(plotpath) == 0:
                try:
                    os.mkdir(plotpath)
                except Exception as e:
                    print("Warning:",e)

        # orientation
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_orientation, yaw,'r.',label='Yaw')   
            ax.plot(time_orientation, roll,'b.',label='Roll')      
            ax.plot(time_orientation, pitch,'g.',label='Pitch')                     
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Degrees')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'orientation.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            print('...')

        # velocity_body (north,east,down) compared to velocity_inertial
            fig=plt.figure(figsize=(10,7))
            ax1 = fig.add_subplot(321)            
            ax1.plot(time_dead_reckoning,north_velocity_inertia_dvl, 'ro',label='DVL')#time_velocity_body,north_velocity_inertia_dvl, 'ro',label='DVL')
            ax1.plot(time_velocity_inertia,north_velocity_inertia, 'b.',label='Phins')
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Velocity, m/s')
            ax1.legend()
            ax1.grid(True)
            ax1.set_title('north velocity')
            ax2 = fig.add_subplot(323)            
            ax2.plot(time_dead_reckoning,east_velocity_inertia_dvl,'ro',label='DVL')
            ax2.plot(time_velocity_inertia,east_velocity_inertia,'b.',label='Phins')
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Velocity, m/s')
            ax2.legend()
            ax2.grid(True)
            ax2.set_title('east velocity')
            ax3 = fig.add_subplot(325)            
            ax3.plot(time_dead_reckoning,down_velocity_inertia_dvl,'ro',label='DVL')#time_velocity_body,down_velocity_inertia_dvl,'r.',label='DVL')
            ax3.plot(time_velocity_inertia,down_velocity_inertia,'b.',label='Phins')
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

            print('...')

        # time_dead_reckoning northings eastings depth vs time
            fig=plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(221)
            ax1.plot(time_dead_reckoning,northings_dead_reckoning_dvl,'r.',label='DVL')#time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
            ax1.plot(time_velocity_inertia,northings_inertia_dead_reckoning,'g.',label='PHINS')
            ax1.plot(time_usbl, northings_usbl,'b.',label='USBL')
            ax1.plot(time_dead_reckoning,northings_dead_reckoning,'c.',label='Centre')#time_velocity_body,northings_dead_reckoning,'b.')
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Northings, m')
            ax1.grid(True)
            ax1.legend()#loc='upper right', bbox_to_anchor=(1, -0.2))
            ax1.set_title('Northings')
            ax2 = fig.add_subplot(222)
            ax2.plot(time_dead_reckoning,eastings_dead_reckoning_dvl,'r.',label='DVL')#time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
            ax2.plot(time_velocity_inertia,eastings_inertia_dead_reckoning,'g.',label='PHINS')
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
            plt.savefig(plotpath + os.sep + 'dead_reckoning_vs_time.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            print('...')

        # usbl latitude longitude
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

            print('...')
        # uncertainties plot. 
        #Ultimately include this in plotly error plot! MEANINGLESS THIS WAY. https://plot.ly/python/line-charts/#filled-lines Something like that?
            for i in range(len(roll_std)):
                if i == 0:
                    pass
                else:
                    roll_std[i]=roll_std[i] + roll_std[i-1]
                    pitch_std[i]=pitch_std[i] + pitch_std[i-1]
                    yaw_std[i]=yaw_std[i] + yaw_std[i-1]
            #SUM UP FOR THE REST TOO? IS THIS THE RIGHT WAY?

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
            fig=plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(camera1_dead_reckoning_eastings,camera1_dead_reckoning_northings,'y.',label='Camera1')
            ax.plot(eastings_dead_reckoning,northings_dead_reckoning,'r.',label='Centre')
            ax.plot(eastings_dead_reckoning_dvl,northings_dead_reckoning_dvl,'g.',label='DVL')
            ax.plot(eastings_inertia_dead_reckoning,northings_inertia_dead_reckoning,'m.',label='PHINS')
            ax.plot(eastings_usbl, northings_usbl,'c.',label='USBL')
            ax.set_xlabel('Eastings, m')
            ax.set_ylabel('Northings, m')
            ax.legend()#loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)   
            plt.savefig(plotpath + os.sep + 'camera1_centre_DVL_PHINS_DR.pdf', dpi=600, bbox_inches='tight')
            if show_plot==True:
                plt.show()
            plt.close()

            print('Complete plot data: ', plotpath)

            #maybe plot a arrow showing vehicle heading at each step***!!
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


        print('Completed data extraction: ', renavpath)

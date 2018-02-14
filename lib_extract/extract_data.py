# extract_data

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

from datetime import datetime

sys.path.append("..")
from lib_calculus.interpolate import interpolate
from lib_localisation.dead_reckoning import dead_reckoning
from lib_localisation.usbl_offset import usbl_offset
from lib_coordinates.body_to_inertial import body_to_inertial

class extract_data:
    def __init__(self,filepath,ftype,start_time,finish_time,plot,csv_write):

        interpolate_remove_flag = False
        show_plot = True

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

    # camera placeholders
        time_camera1=[]
        filename_camera1=[]
        time_camera2=[]
        filename_camera2=[]
        time_camera3=[]
        filename_camera3=[]
    # placeholders for DR relative to left camera
        camera1_dead_reckoning_northings=[]
        camera1_dead_reckoning_eastings=[]
        camera1_dead_reckoning_depth=[]
        camera1_roll=[]
        camera1_pitch=[]
        camera1_yaw=[]
        
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
                if 'phins' in load_data:
                    # origin_flag=1
                    phins_x_offset = load_data['phins']['x_offset']
                    phins_y_offset = load_data['phins']['y_offset']
                    phins_z_offset = load_data['phins']['z_offset']
            
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
                            if abs(parsed_json_data[i]['epoch_timestamp']-parsed_json_data[i]['epoch_timestamp_dvl']<1):
                            
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


    # perform coordinate transformations and interpolations of state data to velocity_body time stamps
        #Assumes the first measurement of velocity_body is the beginning of mission. May not be robust to non-continuous measurements..any (sudden start and stop) will affect it?
        # 
        j=0        
        k=0
        n=0
        start_interpolate_index = 0

        while time_orientation[start_interpolate_index]<time_velocity_body[0]:
            start_interpolate_index += 1

        if start_interpolate_index==1:
            interpolate_remove_flag = True

        for i in range(start_interpolate_index, len(time_orientation)):#time_velocity_body)):

            # interpolate to find the appropriate dvl time for the orientation measurements
            if time_orientation[i]>time_velocity_body[-1]:
                break

            while time_orientation[i]>time_velocity_body[j+1] and j<len(time_velocity_body)-1:
                j += 1

            
            time_dead_reckoning.append(time_orientation[i])

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

            ###still need this? maybe plot it compare to time_altitude and depth etc.
            while time_orientation[i]>time_altitude[n+1] and n<len(time_altitude)-1:
                n += 1
            
                # print(n,len(altitude))
            altitude_interpolated.append(interpolate(time_orientation[i],time_velocity_body[n],time_velocity_body[n+1],altitude[n],altitude[n+1]))

            while time_depth[k]<time_orientation[i] and k < len(time_depth)-1:
                k+= 1
            # interpolate to find the appropriate depth for dead_reckoning
            depth_dead_reckoning[i-start_interpolate_index]=interpolate(time_orientation[i],time_depth[k-1],time_depth[k],depth[k-1],depth[k])

        j=0
        # interpolate to find the appropriate depth to compute seafloor depth for each altitude measurement
        for i in range(len(time_altitude)):        
            while time_depth[j]<time_altitude[i] and j < len(time_depth)-1:
                j=j+1

            if j>=1:                
                seafloor_depth[i]=interpolate(time_altitude[i],time_depth[j-1],time_depth[j],depth[j-1],depth[j])+altitude[i]

        # northings eastings dead reckoning solution
        for i in range(len(time_dead_reckoning)):
            # dead reckoning solution
            if i>1:                    
                [northings_dead_reckoning[i], eastings_dead_reckoning[i]]=dead_reckoning(time_dead_reckoning[i], time_dead_reckoning[i-1], north_velocity_inertia_dvl[i-1], east_velocity_inertia_dvl[i-1], northings_dead_reckoning[i-1], eastings_dead_reckoning[i-1])

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
            del northings_dead_reckoning[0]
            del eastings_dead_reckoning[0]
            del depth_dead_reckoning[0]
            interpolate_remove_flag = False # reset flag
    
    # perform interpolations of state data to velocity_inertia time stamps
        #initialise counters for interpolation
        j=0
        k=0
    
        for i in range(len(time_velocity_inertia)):  
                           
            while time_orientation[j]<time_velocity_inertia[i] and j< len(time_orientation)-1:
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
            
            while time_depth[k]<time_velocity_inertia[i] and k< len(time_depth)-1:
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

    # offset velocity body DR by initial usbl estimate
        # compare time_dead_reckoning and time_usbl
        # find initial position offset
        [northings_usbl_interpolated,eastings_usbl_interpolated] = usbl_offset(time_dead_reckoning,northings_dead_reckoning,eastings_dead_reckoning,time_usbl,northings_usbl,eastings_usbl)
        # offset the deadreackoning by this initial estimate
        for i in range(len(northings_dead_reckoning)):                 
            northings_dead_reckoning[i]=northings_dead_reckoning[i]+northings_usbl_interpolated
            eastings_dead_reckoning[i]=eastings_dead_reckoning[i]+eastings_usbl_interpolated

    # offset velocity inertial DR by initial usbl estimate
        # compare time_velocity_inertia and time_usbl
        # find initial position offset
        [northings_usbl_interpolated,eastings_usbl_interpolated] = usbl_offset(time_velocity_inertia,northings_inertia_dead_reckoning,eastings_inertia_dead_reckoning,time_usbl,northings_usbl,eastings_usbl)
        # offset the deadreackoning by this initial estimate
        for i in range(len(northings_inertia_dead_reckoning)):                
            northings_inertia_dead_reckoning[i]=northings_inertia_dead_reckoning[i]+northings_usbl_interpolated
            eastings_inertia_dead_reckoning[i]=eastings_inertia_dead_reckoning[i]+eastings_usbl_interpolated

    # perform interpolations of state data to camera time stamps
        j=0
        for i in range(len(time_camera1)):
            if j>=len(time_dead_reckoning)-1:
                break
            while time_dead_reckoning[j] < time_camera1[i]:
                if j+1>len(time_dead_reckoning)-1 or time_dead_reckoning[j+1]>time_camera1[-1]:
                    break
                j += 1
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
                    
            [x_offset,y_offset,z_offset] = body_to_inertial(camera1_roll[i],camera1_pitch[i],camera1_yaw[i],-phins_x_offset,-phins_y_offset,-phins_z_offset)
            camera1_dead_reckoning_northings.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],northings_dead_reckoning[j-1],northings_dead_reckoning[j])+x_offset)
            camera1_dead_reckoning_eastings.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],eastings_dead_reckoning[j-1],eastings_dead_reckoning[j])+y_offset)
            camera1_dead_reckoning_depth.append(interpolate(time_camera1[i],time_dead_reckoning[j-1],time_dead_reckoning[j],depth_dead_reckoning[j-1],depth_dead_reckoning[j])+z_offset)
    
        print('Complete interpolation and coordinate transfomations for DR visualisation')

    # write values out to a csv file
        # create a directory with the time stamp
        if csv_write is True:

            csvpath = renavpath + os.sep + 'csv'

            if os.path.isdir(csvpath) == 0:
                try:
                    os.mkdir(csvpath)
                except Exception as e:
                    print("Warning:",e)

            with open(csvpath + os.sep + 'camera1.csv' ,'w') as fileout:
                fileout.write('filename, northings(m), eastings(m), depth(m), roll, pitch, yaw\n')
            for i in range(len(time_camera1)):
                with open(csvpath + os.sep + 'camera1.csv' ,'a') as fileout:
                    try:
                        fileout.write(str(filename_camera1[i])+','+str(camera1_dead_reckoning_northings[i])+','+str(camera1_dead_reckoning_eastings[i])+','+str(camera1_dead_reckoning_depth[i])+','+str(camera1_roll[i])+','+str(camera1_pitch[i])+','+str(camera1_yaw[i])+'\n')
                        fileout.close()
                    except IndexError:
                        break

            print('Complete extraction of data: ', csvpath)
        
    # plot data
        if plot is True:
            print('Plotting data')
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

        # velocity_interpolated (x,y,z)
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_dead_reckoning,x_velocity_interpolated, 'r.',label='Surge') #time_velocity_body,x_velocity, 'r.',label='Surge')
            ax.plot(time_dead_reckoning,y_velocity_interpolated, 'g.',label='Sway')#time_velocity_body,y_velocity, 'g.',label='Sway')
            ax.plot(time_dead_reckoning,z_velocity_interpolated, 'b.',label='Heave')#time_velocity_body,z_velocity, 'b.',label='Heave')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Velocity, m/s')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'velocity_body.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # velocity_body (north,east,down) compared to velocity_inertial
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)            
            ax.plot(time_dead_reckoning,north_velocity_inertia_dvl, 'ro',label='DVL')#time_velocity_body,north_velocity_inertia_dvl, 'ro',label='DVL')
            ax.plot(time_velocity_inertia,north_velocity_inertia, 'b.',label='Phins')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Velocity, m/s')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'velocity_inertia_north.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)            
            ax.plot(time_dead_reckoning,east_velocity_inertia_dvl,'ro',label='DVL')
            ax.plot(time_velocity_inertia,east_velocity_inertia,'b.',label='Phins')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Velocity, m/s')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'velocity_inertia_east.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)            
            ax.plot(time_dead_reckoning,down_velocity_inertia_dvl,'r.',label='DVL')#time_velocity_body,down_velocity_inertia_dvl,'r.',label='DVL')
            ax.plot(time_velocity_inertia,down_velocity_inertia,'b.',label='Phins')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Velocity, m/s')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'velocity_inertia_down.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # depth and altitude
            fig=plt.figure(figsize=(4,3))
            ax1 = fig.add_subplot(211)
            ax1.plot(time_depth,depth,'b.',label='Vehicle')                        
            ax1.plot(time_altitude,seafloor_depth,'r.',label='Seafloor')                         
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Depth, m')
            plt.gca().invert_yaxis()
            ax1.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax1.grid(True)          
            ax2 = fig.add_subplot(212)
            ax2.plot(time_altitude,altitude,'r.',label='Altitude')              
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Altitude, m')
            ax2.set_xlim(min(time_depth),max(time_depth))
            ax2.grid(True)          
            plt.savefig(plotpath + os.sep + 'depth_altitude.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # usbl 
            #usbl depth and time
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_usbl,depth_usbl,'b.',label='USBL') 
            ax.plot(time_depth,depth,'ro',label='Vehicle') 
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Depth, m')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'usbl_depth_vehicle.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            # usbl latitude longitude
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(longitude_usbl,latitude_usbl,'b.')                 
            ax.set_xlabel('Longitude, degrees')
            ax.set_ylabel('Latitude, degrees')
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'usbl_latitude_longitude.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            # usbl northings eastings
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(eastings_usbl,northings_usbl,'b.',label='Reference ['+str(latitude_reference)+','+str(longitude_reference)+']')                 
            ax.set_xlabel('Eastings, m')
            ax.set_ylabel('Northings, m')
            ax.grid(True)
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))           
            plt.savefig(plotpath + os.sep + 'usbl_northings_eastings.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            # usbl eastings time
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_usbl, eastings_usbl,'b.')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Eastings, m')
            ax.grid(True)                
            plt.savefig(plotpath + os.sep + 'usbl_eastings_time.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            # usbl northings time
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_usbl, northings_usbl,'b.')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Northings, m')
            ax.grid(True)                
            plt.savefig(plotpath + os.sep + 'usbl_northings_time.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # uncertainties plot
            for i in range(len(roll_std)):
                if i == 0:
                    pass
                else:
                    roll_std[i]=roll_std[i] + roll_std[i-1]
                    pitch_std[i]=pitch_std[i] + pitch_std[i-1]
                    yaw_std[i]=yaw_std[i] + yaw_std[i-1]

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_orientation,roll_std,'r.',label='roll_std')
            ax.plot(time_orientation,pitch_std,'g.',label='pitch_std')
            ax.plot(time_orientation,yaw_std,'b.',label='yaw_std')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Angle, degrees')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'orientation_std.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_depth,depth_std,'b.')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Depth, m')
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'depth_std.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_velocity_body,x_velocity_std,'r.',label='x_velocity_std')
            ax.plot(time_velocity_body,y_velocity_std,'g.',label='y_velocity_std')
            ax.plot(time_velocity_body,z_velocity_std,'b.',label='z_velocity_std')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Velocity, m/s')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'velocity_body_std.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_velocity_inertia,north_velocity_std_inertia,'r.',label='north_velocity_std_inertia')
            ax.plot(time_velocity_inertia,east_velocity_std_inertia,'g.',label='east_velocity_std_inertia')
            ax.plot(time_velocity_inertia,down_velocity_std_inertia,'b.',label='down_velocity_std_inertia')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Velocity, m/s')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'velocity_inertia_std.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_usbl,latitude_std_usbl,'r.',label='latitude_std_usbl')
            ax.plot(time_usbl,longitude_std_usbl,'g.',label='longitude_std_usbl')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('LatLong, degrees')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'usbl_latitude_longitude_std.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_usbl,northings_std_usbl,'r.',label='northings_std_usbl')
            ax.plot(time_usbl,eastings_std_usbl,'g.',label='eastings_std_usbl')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('NorthEast, m')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'usbl_northings_eastings_std.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # dead_reckoning northings eastings
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(eastings_dead_reckoning,northings_dead_reckoning,'b.')
            ax.set_xlabel('Eastings, m')
            ax.set_ylabel('Northings, m')
            ax.grid(True)                
            plt.savefig(plotpath + os.sep + 'dead_reckoning_northings_eastings.pdf', dpi=600, bbox_inches='tight')
            plt.close()                

        # dead_reckoning northings time
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_dead_reckoning,northings_dead_reckoning,'b.')#time_velocity_body,northings_dead_reckoning,'b.')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Northings, m')
            ax.grid(True)                
            plt.savefig(plotpath + os.sep + 'dead_reckoning_northings_time.pdf', dpi=600, bbox_inches='tight')
            plt.close() 

        # dead_reckoning eastings time
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_dead_reckoning,eastings_dead_reckoning,'b.')#time_velocity_body,eastings_dead_reckoning,'b.')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Eastings, m')
            ax.grid(True)                
            plt.savefig(plotpath + os.sep + 'dead_reckoning_eastings_time.pdf', dpi=600, bbox_inches='tight')
            plt.close() 


            print('Complete plot data: ', plotpath)

        # camera1_dead_reckoning
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(camera1_dead_reckoning_eastings,camera1_dead_reckoning_northings,'y.',label='Camera1')
            ax.plot(eastings_dead_reckoning,northings_dead_reckoning,'r.',label='DVL')
            ax.set_xlabel('Eastings, m')
            ax.set_ylabel('Northings, m')
            ax.grid(True)                
            plt.savefig(plotpath + os.sep + 'camera1_dead_reckoning_northings_eastings.pdf', dpi=600, bbox_inches='tight')
            if show_plot==True:
                plt.show()
            plt.close()

        # inertia_dead_reckoning northings eastings
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(eastings_inertia_dead_reckoning,northings_inertia_dead_reckoning,'b.')
            ax.set_xlabel('Eastings, m')
            ax.set_ylabel('Northings, m')
            ax.grid(True)                
            plt.savefig(plotpath + os.sep + 'inertia_dead_reckoning_northings_eastings.pdf', dpi=600, bbox_inches='tight')
            plt.close()                

        # inertia_dead_reckoning northings time
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_velocity_inertia,northings_inertia_dead_reckoning,'b.')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Northings, m')
            ax.grid(True)                
            plt.savefig(plotpath + os.sep + 'inertia_dead_reckoning_northings_time.pdf', dpi=600, bbox_inches='tight')
            plt.close() 

        # inertia_dead_reckoning eastings time
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_velocity_inertia,eastings_inertia_dead_reckoning,'b.')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Eastings, m')
            ax.grid(True)                
            plt.savefig(plotpath + os.sep + 'inertia_dead_reckoning_eastings_time.pdf', dpi=600, bbox_inches='tight')
            plt.close() 

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_dead_reckoning,northings_dead_reckoning,'r.',label='DVL')#time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
            ax.plot(time_velocity_inertia,northings_inertia_dead_reckoning,'g.',label='PHINS')
            ax.plot(time_usbl, northings_usbl,'b.',label='USBL')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Northings, m')
            ax.grid(True)                
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            plt.savefig(plotpath + os.sep + 'body_inertia_usbl_northings_time.pdf', dpi=600, bbox_inches='tight')
            plt.close() 

        # dvl, inertia. usbl eastings time
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_dead_reckoning,eastings_dead_reckoning,'r.',label='DVL')#time_velocity_body,eastings_dead_reckoning,'r.',label='DVL')
            ax.plot(time_velocity_inertia,eastings_inertia_dead_reckoning,'g.',label='PHINS')
            ax.plot(time_usbl, eastings_usbl,'b.',label='USBL')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Eastings, m')
            ax.grid(True)                
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            plt.savefig(plotpath + os.sep + 'body_inertia_usbl_eastings_time.pdf', dpi=600, bbox_inches='tight')
            plt.close() 

        # inertia_dead_reckoning northings eastings
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(camera1_dead_reckoning_eastings,camera1_dead_reckoning_northings,'y.',label='Camera1')
            ax.plot(eastings_dead_reckoning,northings_dead_reckoning,'r.',label='DVL')
            ax.plot(eastings_inertia_dead_reckoning,northings_inertia_dead_reckoning,'g.',label='PHINS')
            ax.plot(eastings_usbl, northings_usbl,'b.',label='USBL')
            ax.set_xlabel('Eastings, m')
            ax.set_ylabel('Northings, m')
            ax.grid(True) 
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))               
            plt.savefig(plotpath + os.sep + 'body_inertia_usbl_northings_eastings.pdf', dpi=600, bbox_inches='tight')
            if show_plot==True:
                plt.show()
            plt.close()

            print('Complete plot data: ', plotpath)         

        print('Completed data extraction: ', renavpath)
        

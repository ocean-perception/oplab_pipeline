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
from pathlib import Path
import plotly.offline as py
import plotly.graph_objs as go

import matplotlib.pyplot as plt

from datetime import datetime

sys.path.append("..")
from lib_calculus.interpolate import interpolate
from lib_localisation.dead_reckoning import dead_reckoning
from lib_localisation.usbl_offset import usbl_offset
from lib_coordinates.body_to_inertial import body_to_inertial
from lib_coordinates.latlon_wgs84 import metres_to_latlon
from lib_extract import sensor_classes as sens_cls
from lib_particle_filter.particle_filter import particle_filter

class extract_data:
    #def __init__(self,filepath,ftype,start_datetime,finish_datetime):
    def __init__(self,filepath,ftype,start_datetime,finish_datetime,plot,csv_write,plotly,output_DR,perform_particle_filter):
    
    # placeholders
        interpolate_remove_flag = False

        # selected start and finish time
        epoch_start_time = 0
        epoch_finish_time = 0

        # velocity body placeholders (DVL)
        velocity_body_list=[]
        velocity_body_sensor_name='vel_body'
        # velocity inertial placeholders
        velocity_inertial_list=[]
        velocity_inertial_sensor_name='vel_inertial'
        # orientation placeholders (INS)
        orientation_list=[]
        orientation_sensor_name='ins'
        # depth placeholders
        depth_list=[]
        depth_sensor_name='depth'
        # altitude placeholders
        altitude_list=[]
        altitude_sensor_name = 'alt'
        # USBL placeholders
        usbl_list=[]
        usbl_sensor_name = 'usbl'

        # camera1 placeholders
        camera1_list=[]
        camera1_pf_list=[]
        camera1_sensor_name = 'cam1' # original serial_camera1
        # camera2 placeholders
        camera2_list=[]
        camera2_pf_list=[]
        camera2_sensor_name = 'cam2'
        # camera3 placeholders
        camera3_list=[]
        camera3_pf_list=[]
        camera3_sensor_name = 'cam3'

        # placeholders for interpolated velocity body measurements based on orientation and transformed coordinates
        dead_reckoning_centre_list=[]
        dead_reckoning_dvl_list=[]

        # placeholders for dvl_imu_data fused with usbl_data using particle filter
        pf_fusion_dvl_list = []
        pf_fusion_centre_list = []
        pf_usbl_datapoints = []
        pf_particles_list = []
        pf_northings_std = []
        pf_eastings_std = []
        pf_yaw_std = []

        # placeholders for chemical data
        chemical_list = []
        chemical_pf_list = []

    # Helper functions
        def datetime_to_epochtime(yyyy, mm, dd, hours, mins, secs):
                dt_obj = datetime(yyyy,mm,dd,hours,mins,secs)       
                time_tuple = dt_obj.timetuple()
                return time.mktime(time_tuple)

    # load localisaion.yaml for particle filter and other setup
        print('Loading localisation.yaml')    
        localisation = filepath + 'localisation.yaml'
        localisation_file = Path(localisation)

        if localisation_file.exists():
            print("Loading existng localisation.yaml at {}".format(localisation))
        else:
            default_localisation = 'default_yaml' + os.sep + 'localisation.yaml'
            print("Cannot find {}, generating default from {}".format(localisation,default_localisation))
            shutil.copy2(default_localisation, filepath) # save localisation yaml to processed directory
        
        with open(localisation,'r') as stream:
            load_localisation = yaml.load(stream)
            if 'usbl_filter' in load_localisation:
                max_auv_speed = load_localisation['usbl_filter']['max_auv_speed']
                sigma_factor = load_localisation['usbl_filter']['sigma_factor']
            if 'particle_filter' in load_localisation:
                dvl_noise_sigma_factor = load_localisation['particle_filter']['dvl_noise_sigma_factor']
                imu_noise_sigma_factor = load_localisation['particle_filter']['imu_noise_sigma_factor']
                usbl_noise_sigma_factor = load_localisation['particle_filter']['usbl_noise_sigma_factor']
                particles_number = load_localisation['particle_filter']['particles_number']
                particles_time_interval = load_localisation['particle_filter']['particles_plot_time_interval']
            if 'csv_output' in load_localisation:
                csv_active = load_localisation['csv_output']['active']

                csv_usbl = load_localisation['csv_output']['usbl']

                csv_dr_auv_centre = load_localisation['csv_output']['dead_reckoning']['auv_centre']
                csv_dr_auv_dvl = load_localisation['csv_output']['dead_reckoning']['auv_dvl']
                csv_dr_camera_1 = load_localisation['csv_output']['dead_reckoning']['camera_1']
                csv_dr_camera_2 = load_localisation['csv_output']['dead_reckoning']['camera_2']
                csv_dr_camera_3 = load_localisation['csv_output']['dead_reckoning']['camera_3']
                csv_dr_chemical = load_localisation['csv_output']['dead_reckoning']['chemical']

                csv_pf_auv_centre = load_localisation['csv_output']['particle_filter']['auv_centre']
                csv_pf_auv_dvl = load_localisation['csv_output']['particle_filter']['auv_dvl']
                csv_pf_camera_1 = load_localisation['csv_output']['particle_filter']['camera_1']
                csv_pf_camera_2 = load_localisation['csv_output']['particle_filter']['camera_2']
                csv_pf_camera_3 = load_localisation['csv_output']['particle_filter']['camera_3']
                csv_pf_chemical = load_localisation['csv_output']['particle_filter']['chemical']

    # get information of sensor position offset from origin/centre reference point from vehicle.yaml
        print('Loading vehicle.yaml')    
        vehicle = filepath +'vehicle.yaml'
        with open(vehicle,'r') as stream:
            vehicle_data = yaml.load(stream)
        if 'origin' in vehicle_data:
            origin_x_offset = vehicle_data['origin']['x_offset']
            origin_y_offset = vehicle_data['origin']['y_offset']
            origin_z_offset = vehicle_data['origin']['z_offset']
        if 'camera1' in vehicle_data:
            camera1_x_offset = vehicle_data['camera1']['x_offset']
            camera1_y_offset = vehicle_data['camera1']['y_offset']
            camera1_z_offset = vehicle_data['camera1']['z_offset']
        if 'camera2' in vehicle_data:
            camera2_x_offset = vehicle_data['camera2']['x_offset']
            camera2_y_offset = vehicle_data['camera2']['y_offset']
            camera2_z_offset = vehicle_data['camera2']['z_offset']
        if 'camera3' in vehicle_data:
            camera3_x_offset = vehicle_data['camera3']['x_offset']
            camera3_y_offset = vehicle_data['camera3']['y_offset']
            camera3_z_offset = vehicle_data['camera3']['z_offset']
        if 'usbl' in vehicle_data:
            usbl_x_offset = vehicle_data['usbl']['x_offset']
            usbl_y_offset = vehicle_data['usbl']['y_offset']
            usbl_z_offset = vehicle_data['usbl']['z_offset']
        if 'dvl' in vehicle_data:
            dvl_x_offset = vehicle_data['dvl']['x_offset']
            dvl_y_offset = vehicle_data['dvl']['y_offset']
            dvl_z_offset = vehicle_data['dvl']['z_offset']
        if 'depth' in vehicle_data:
            depth_x_offset = vehicle_data['depth']['x_offset']
            depth_y_offset = vehicle_data['depth']['y_offset']
            depth_z_offset = vehicle_data['depth']['z_offset']
        if 'ins' in vehicle_data:
            ins_x_offset = vehicle_data['ins']['x_offset']
            ins_y_offset = vehicle_data['ins']['y_offset']
            ins_z_offset = vehicle_data['ins']['z_offset']
        if 'chemical' in vehicle_data:
            chemical_x_offset = vehicle_data['chemical']['x_offset']
            chemical_y_offset = vehicle_data['chemical']['y_offset']
            chemical_z_offset = vehicle_data['chemical']['z_offset']

    # OPLAB mode
        if ftype == 'oplab':# or (ftype is not 'acfr'):
            outpath=filepath + 'nav'

            filename='nav_standard.json'        
            print('Loading json file ' + outpath + os.sep + filename)
            with open(outpath + os.sep + filename) as nav_standard:                
                parsed_json_data = json.load(nav_standard)

            print('Loading mission.yaml')    
            mission = filepath +'mission.yaml'
            with open(mission,'r') as stream:
                mission_data = yaml.load(stream)
            # assigns sensor names from mission.yaml instead of json data packet (instead of looking at json data as TunaSand don't have serial yet)
            if 'origin' in mission_data:
                origin_flag=1
                latitude_reference = mission_data['origin']['latitude']
                longitude_reference = mission_data['origin']['longitude']
                coordinate_reference = mission_data['origin']['coordinate_reference_system']
                date = mission_data['origin']['date']
            if 'velocity' in mission_data:
                velocity_body_sensor_name = mission_data['velocity']['format']
                velocity_inertial_sensor_name = mission_data['velocity']['format']
            if 'orientation' in mission_data:
                orientation_sensor_name = mission_data['orientation']['format']
            if 'depth' in mission_data:
                depth_sensor_name = mission_data['depth']['format']
            if 'altitude' in mission_data:
                altitude_sensor_name = mission_data['altitude']['format']
            if 'usbl' in mission_data:
                usbl_sensor_name = mission_data['usbl']['format']
            if 'image' in mission_data:
                if 'camera1' in mission_data['image']:
                    camera1_sensor_name = '_'.join(mission_data['image']['camera1'].split('/'))
                if 'camera2' in mission_data['image']:
                    camera2_sensor_name = '_'.join(mission_data['image']['camera2'].split('/'))
                if 'camera3' in mission_data['image']:
                    camera3_sensor_name = '_'.join(mission_data['image']['camera3'].split('/'))

        # setup start and finish date time
            if start_datetime == '':
                yyyy = int(date[0:4])
                mm =  int(date[5:7])
                dd =  int(date[8:10])

                hours = 0
                mins = 0
                secs = 0

                start_datetime = date[0:4] + date[5:7] + date[8:10] + '000000'
            else:
                yyyy = int(start_datetime[0:4])
                mm =  int(start_datetime[4:6])
                dd =  int(start_datetime[6:8])

                hours = int(start_datetime[8:10])
                mins = int(start_datetime[10:12])
                secs = int(start_datetime[12:14])

            epoch_start_time = datetime_to_epochtime(yyyy,mm,dd,hours,mins,secs)
            
            if finish_datetime == '':
                yyyy = int(date[0:4])
                mm =  int(date[5:7])
                dd =  int(date[8:10])

                hours = 23
                mins = 59
                secs = 59

                finish_datetime = date[0:4] + date[5:7] + date[8:10] + '235959'
            else:
                yyyy = int(finish_datetime[0:4])
                mm =  int(finish_datetime[4:6])
                dd =  int(finish_datetime[6:8])

                hours = int(finish_datetime[8:10])
                mins = int(finish_datetime[10:12])
                secs = int(finish_datetime[12:14])

            epoch_finish_time = datetime_to_epochtime(yyyy,mm,dd,hours,mins,secs)

        # read in data from json file
            # i here is the number of the data packet
            for i in range(len(parsed_json_data)):
                epoch_timestamp=parsed_json_data[i]['epoch_timestamp']
                if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:                                   
                    if 'velocity' in parsed_json_data[i]['category']:
                        if 'body' in parsed_json_data[i]['frame']:
                            # to check for corrupted data point which have inertial frame data values
                            if 'epoch_timestamp_dvl' in parsed_json_data[i]:
                                # confirm time stamps of dvl are aligned with main clock (within a second)
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
                        usbl.latitude_std = parsed_json_data[i]['data_target'][0]['latitude_std']
                        usbl.longitude = parsed_json_data[i]['data_target'][1]['longitude']
                        usbl.longitude_std = parsed_json_data[i]['data_target'][1]['longitude_std']
                        usbl.northings = parsed_json_data[i]['data_target'][2]['northings']
                        usbl.northings_std = parsed_json_data[i]['data_target'][2]['northings_std']
                        usbl.eastings = parsed_json_data[i]['data_target'][3]['eastings']
                        usbl.eastings_std = parsed_json_data[i]['data_target'][3]['eastings_std']
                        usbl.depth = parsed_json_data[i]['data_target'][4]['depth']
                        usbl.depth_std = parsed_json_data[i]['data_target'][4]['depth_std']
                        usbl.distance_to_ship = parsed_json_data[i]['data_target'][5]['distance_to_ship']
                        # usbl.latitude_ship = parsed_json_data[i]['data_ship'][0]['latitude']
                        # usbl.longitude_ship = parsed_json_data[i]['data_ship'][0]['longitude']
                        # usbl.northings_ship = parsed_json_data[i]['data_ship'][1]['northings']
                        # usbl.eastings_ship = parsed_json_data[i]['data_ship'][1]['eastings']
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

                    if 'chemical' in parsed_json_data[i]['category']:
                        chemical = sens_cls.other()
                        chemical.timestamp = parsed_json_data[i]['epoch_timestamp']
                        chemical.data = parsed_json_data[i]['data']
                        chemical_list.append(chemical)

            camera1_pf_list = copy.deepcopy(camera1_list)
            camera2_pf_list = copy.deepcopy(camera2_list)
            camera3_pf_list = copy.deepcopy(camera3_list)
            chemical_pf_list = copy.deepcopy(chemical_list)

        # make path for processed outputs
            renavpath = filepath + 'json_renav_' + start_datetime[0:8] + '_' + start_datetime[8:14] + '_' + finish_datetime[0:8] + '_' + finish_datetime[8:14]
            if os.path.isdir(renavpath) == 0:
                try:
                    os.mkdir(renavpath)
                except Exception as e:
                    print("Warning:",e)

            print('Complete parse of:' + outpath + os.sep + filename)
            print('Writing outputs to: ' + renavpath)
            # copy to renav folder renavpath + os.sep + 'localisation.yaml'
            # shutil.copy2(localisation, renavpath) # save mission yaml to processed directory

    # ACFR mode
        if ftype == 'acfr':# or (ftype is not 'acfr'):
            
            print('Loading mission.cfg')    
            mission = filepath +'mission.cfg'
            with codecs.open(mission,'r',encoding='utf-8', errors='ignore') as filein:
                for line in filein.readlines():             
                    line_split = line.strip().split(' ')
                    if str(line_split[0]) == 'MAG_VAR_LAT':
                        latitude_reference = float(line_split[1])                  
                    if str(line_split[0]) == 'MAG_VAR_LNG':
                        longitude_reference = float(line_split[1])                
                    if str(line_split[0]) == 'MAG_VAR_DATE':
                        date = str(line_split[1])                

        # setup the time window
            if start_datetime == '':
                yyyy = int(date[1:5])
                mm =  int(date[6:8])
                dd =  int(date[9:11])

                hours = 0
                mins = 0
                secs = 0
            else:
                yyyy = int(start_datetime[0:4])
                mm =  int(start_datetime[4:6])
                dd =  int(start_datetime[6:8])

                hours = int(start_datetime[8:10])
                mins = int(start_datetime[10:12])
                secs = int(start_datetime[12:14])

            epoch_start_time = datetime_to_epochtime(yyyy,mm,dd,hours,mins,secs)
            
            if finish_datetime == '':
                yyyy = int(date[1:5])
                mm =  int(date[6:8])
                dd =  int(date[9:11])

                hours = 23
                mins = 59
                secs = 59
            else:
                yyyy = int(finish_datetime[0:4])
                mm =  int(finish_datetime[4:6])
                dd =  int(finish_datetime[6:8])

                hours = int(finish_datetime[8:10])
                mins = int(finish_datetime[10:12])
                secs = int(finish_datetime[12:14])
                
            epoch_finish_time = datetime_to_epochtime(yyyy,mm,dd,hours,mins,secs)  

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

                    if str(line_split[0]) == 'VIS:':
                        if 'FC' in line_split[3]:
                            camera1 = sens_cls.camera()
                            camera1.timestamp = float(line_split[1])
                            camera1.filename = line_split[3]
                            camera1_list.append(camera1)
                        if 'AC' in line_split[3]:
                            camera2 = sens_cls.camera()
                            camera2.timestamp = float(line_split[1])
                            camera2.filename = line_split[3]
                            camera2_list.append(camera2)
            camera1_pf_list = copy.deepcopy(camera1_list)
            camera2_pf_list = copy.deepcopy(camera2_list)
            # camera3_pf_list = copy.deepcopy(camera3_list)
            # chemical_pf_list = copy.deepcopy(chemical_list)        

            # make folder to store csv and plots
            renavpath = filepath + 'acfr_renav_' + start_datetime[0:8] + '_' + start_datetime[8:14] + '_' + finish_datetime[0:8] + '_' + finish_datetime[8:14]
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

    # filter usbl data outliers
    # interpolate to find the appropriate depth to compute seafloor depth for each altitude measurement
        
        depth_interpolated = []
        depth_std_interpolated = []
        usbl_temp_list = []

        j=0

        for i in range(len(usbl_list)):        
            while j < len(depth_list)-1 and depth_list[j].timestamp<usbl_list[i].timestamp:
                j=j+1

            if j>=1:                
                depth_interpolated.append(interpolate(usbl_list[i].timestamp,depth_list[j-1].timestamp,depth_list[j].timestamp,depth_list[j-1].depth,depth_list[j].depth))
                depth_std_interpolated.append(interpolate(usbl_list[i].timestamp,depth_list[j-1].timestamp,depth_list[j].timestamp,depth_list[j-1].depth_std,depth_list[j].depth_std))

        # not very robust, need to define max_speed carefully. Maybe use kmeans clustering? features = gradient/value compared to nearest few neighbours, k = 2, select group with least std/larger value?        
        
        # first filter by depth as bad USBL data tends to get the depth wrong. This should be a function of the range to the target.
        
        # the usbl and vehicle sensor depths should be within 2 std (95% prob) of each other. If more than that apart reject
        def depth_filter(i):
            j=0
            
            depth_difference=abs((usbl_list[i].depth-usbl_z_offset)-(depth_interpolated[i]-depth_z_offset))
            depth_uncertainty_envelope=abs(usbl_list[i].depth_std)+abs(depth_std_interpolated[i])
     
            if  depth_difference <= sigma_factor*depth_uncertainty_envelope:
                return True
            else:
                return False

        def distance_filter(i,n):          
            lateral_distance = ((usbl_list[i].northings - usbl_list[i+n].northings)**2 + (usbl_list[i].eastings - usbl_list[i+n].eastings)**2)**0.5
            lateral_distance_uncertainty_envelope = (((usbl_list[i].northings_std)**2+(usbl_list[i].eastings_std)**2)**0.5+((usbl_list[i+n].northings_std)**2+(usbl_list[i+n].eastings_std)**2)**0.5)+abs(usbl_list[i].timestamp-usbl_list[i+n].timestamp)*max_auv_speed

            if lateral_distance <= sigma_factor*lateral_distance_uncertainty_envelope:
                return True
            else:
                return False
        
        
        for i in range(len(depth_interpolated)):
            if depth_filter(i) is True:
                usbl_temp_list.append(usbl_list[i])
            i+=1

        print ("Length of original usbl list = {}, Length of depth filtered usbl list = {}".format(len(usbl_list), len(usbl_temp_list)))
        usbl_list = usbl_temp_list
        usbl_temp_list = []

        # the 2 usbl readings should be within maximum possible distance travelled by the vehicle at max speed taking into account the 2 std uncertainty (95%) in the USBL lateral measurements
        

        
        continuity_condition = 2
        #want to check continuity over several readings to get rid of just outliers and not good data around them. Curent approach has a lot of redundncy and can be improved when someone has the bandwidth
        i=continuity_condition
        n=-continuity_condition        
        while i < len(usbl_list)-continuity_condition:
            
            if distance_filter(i,n) is False:                
                n=-continuity_condition
                i+=1
            else:
                n+=1
                if n is 0:
                    n+=1
                if n>continuity_condition:                        
                    usbl_temp_list.append(usbl_list[i])
                    n=-continuity_condition
                    i+=1

            
        print ("Length of depth filtered usbl list = {}, Length of depth and distance filtered usbl list = {}".format(len(usbl_list), len(usbl_temp_list)))
        usbl_list = usbl_temp_list
                       
        #def speed(ii, n):
        #    value = abs((usbl_list[ii].northings - usbl_list[ii-n].northings)**2 + (usbl_list[ii].eastings - usbl_list[ii-n].eastings)**2/(usbl_list[ii].timestamp-usbl_list[ii-n].timestamp))
        #    return value
        # to pick a good starting point. Any thing that says auv is at 5m/s reject
        #i=2
        #while not speed(i, -2)<max_auv_speed and speed(i, -1)<max_auv_speed and speed(i, 1)<max_auv_speed and speed(i, 2)<max_auv_speed:
        #    i+= 1
        #i-=2
        #usbl_temp_list.append(usbl_list[i])
        #while i+2 < len(usbl_list):
        #    if speed(i, -1) < max_auv_speed:
        #        i+=1
        #        usbl_temp_list.append(usbl_list[i])
        #    else:
        #        i+=2
        #print ("Length of original usbl list = {}, Length of filtered usbl list = {}".format(len(usbl_list), len(usbl_temp_list)))
        usbl_list = usbl_temp_list

    # perform coordinate transformations and interpolations of state data to velocity_body time stamps with sensor position offset and perform dead reckoning
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

        # dead reckoning solution
        for i in range(len(dead_reckoning_dvl_list)):
            # dead reckoning solution
            if i>=1:
                [dead_reckoning_dvl_list[i].northings, dead_reckoning_dvl_list[i].eastings]=dead_reckoning(dead_reckoning_dvl_list[i].timestamp, dead_reckoning_dvl_list[i-1].timestamp, dead_reckoning_dvl_list[i].north_velocity, dead_reckoning_dvl_list[i-1].north_velocity, dead_reckoning_dvl_list[i].east_velocity, dead_reckoning_dvl_list[i-1].east_velocity, dead_reckoning_dvl_list[i-1].northings, dead_reckoning_dvl_list[i-1].eastings)

        # offset sensor to plot origin/centre of vehicle
        dead_reckoning_centre_list = copy.deepcopy(dead_reckoning_dvl_list) #[:] #.copy()
        for i in range(len(dead_reckoning_centre_list)):
            [x_offset, y_offset, z_offset] = body_to_inertial(dead_reckoning_centre_list[i].roll, dead_reckoning_centre_list[i].pitch, dead_reckoning_centre_list[i].yaw, origin_x_offset - dvl_x_offset, origin_y_offset - dvl_y_offset, origin_z_offset - dvl_z_offset)
            dead_reckoning_centre_list[i].northings += x_offset
            dead_reckoning_centre_list[i].eastings += y_offset
            # dead_reckoning_centre_list[i].depth += z_offset
        ### correct for altitude and depth offset too!

    # particle filter data fusion of usbl_data and dvl_imu_data
        # if perform_particle_filter == True:
        pf_start_time = time.time()
        [pf_fusion_dvl_list, pf_usbl_datapoints, pf_particles_list, pf_northings_std, pf_eastings_std, pf_yaw_std] = particle_filter(copy.deepcopy(usbl_list), copy.deepcopy(dead_reckoning_dvl_list), particles_number, True, dvl_noise_sigma_factor, imu_noise_sigma_factor, usbl_noise_sigma_factor)
        pf_end_time = time.time()
        pf_elapesed_time = pf_end_time - pf_start_time
        print ("particle filter with {} particles took {} seconds".format(particles_number,pf_elapesed_time)) # maybe save this as text alongside plotly outputs
        for i in range(len(pf_fusion_centre_list)):
            pf_fusion_dvl_list[i].latitude, pf_fusion_dvl_list[i].longitude = metres_to_latlon(latitude_reference, longitude_reference, pf_fusion_dvl_list[i].eastings, pf_fusion_dvl_list[i].northings)
        pf_fusion_centre_list = copy.deepcopy(pf_fusion_dvl_list)
        for i in range(len(pf_fusion_centre_list)):
            [x_offset, y_offset, z_offset] = body_to_inertial(pf_fusion_centre_list[i].roll, pf_fusion_centre_list[i].pitch, pf_fusion_centre_list[i].yaw, origin_x_offset - dvl_x_offset, origin_y_offset - dvl_y_offset, origin_z_offset - dvl_z_offset)
            pf_fusion_centre_list[i].northings += x_offset
            pf_fusion_centre_list[i].eastings += y_offset
                # pf_fusion_centre_list[i].depth += z_offset
        pf_fusion_camera1_list = copy.deepcopy(pf_fusion_dvl_list)
        for i in range(len(pf_fusion_camera1_list)):
            [x_offset, y_offset, z_offset] = body_to_inertial(pf_fusion_camera1_list[i].roll, pf_fusion_camera1_list[i].pitch, pf_fusion_camera1_list[i].yaw, camera1_x_offset - dvl_x_offset, origin_y_offset - camera1_y_offset, origin_z_offset - camera1_z_offset)
            pf_fusion_camera1_list[i].northings += x_offset
            pf_fusion_camera1_list[i].eastings += y_offset
            ### correct for altitude and depth offset too!

        #remove first term if first time_orientation is < velocity_body time
        if interpolate_remove_flag == True:

            # del time_orientation[0]
            del dead_reckoning_centre_list[0]
            del dead_reckoning_dvl_list[0]
            interpolate_remove_flag = False # reset flag
        print('Complete interpolation and coordinate transfomations for velocity_body')
    
    # perform interpolations of state data to velocity_inertial time stamps (without sensor offset and correct imu to dvl flipped interpolation) and perform deadreckoning
        #initialise counters for interpolation
        if len(velocity_inertial_list)>0:

            j=0
            k=0
        
            for i in range(len(velocity_inertial_list)):
                               
                while j< len(orientation_list)-1 and orientation_list[j].timestamp<velocity_inertial_list[i].timestamp:
                    j=j+1
                
                if j==1:
                    interpolate_remove_flag = True
                else:
                    velocity_inertial_list[i].roll=interpolate(velocity_inertial_list[i].timestamp,orientation_list[j-1].timestamp,orientation_list[j].timestamp,orientation_list[j-1].roll,orientation_list[j].roll)
                    velocity_inertial_list[i].pitch=interpolate(velocity_inertial_list[i].timestamp,orientation_list[j-1].timestamp,orientation_list[j].timestamp,orientation_list[j-1].pitch,orientation_list[j].pitch)

                    if abs(orientation_list[j].yaw-orientation_list[j-1].yaw)>180:                        
                        if orientation_list[j].yaw>orientation_list[j-1].yaw:
                            velocity_inertial_list[i].yaw=interpolate(velocity_inertial_list[i].timestamp,orientation_list[j-1].timestamp,orientation_list[j].timestamp,orientation_list[j-1].yaw,orientation_list[j].yaw-360)
                            
                        else:
                            velocity_inertial_list[i].yaw=interpolate(velocity_inertial_list[i].timestamp,orientation_list[j-1].timestamp,orientation_list[j].timestamp,orientation_list[j-1].yaw-360,orientation_list[j].yaw)
                           
                        if velocity_inertial_list[i].yaw<0:
                            velocity_inertial_list[i].yaw+=360
                            
                        elif velocity_inertial_list[i].yaw>360:
                            velocity_inertial_list[i].yaw-=360  

                    else:
                        velocity_inertial_list[i].yaw=interpolate(velocity_inertial_list[i].timestamp,orientation_list[j-1].timestamp,orientation_list[j].timestamp,orientation_list[j-1].yaw,orientation_list[j].yaw)
                
                while k< len(depth_list)-1 and depth_list[k].timestamp<velocity_inertial_list[i].timestamp:
                    k=k+1

                if k>=1:                
                    velocity_inertial_list[i].depth=interpolate(velocity_inertial_list[i].timestamp,depth_list[k-1].timestamp,depth_list[k].timestamp,depth_list[k-1].depth,depth_list[k].depth) # depth directly interpolated from depth sensor
            
            for i in range(len(velocity_inertial_list)):
                if i >= 1:                     
                    [velocity_inertial_list[i].northings, velocity_inertial_list[i].eastings]=dead_reckoning(velocity_inertial_list[i].timestamp, velocity_inertial_list[i-1].timestamp, velocity_inertial_list[i].north_velocity, velocity_inertial_list[i-1].north_velocity, velocity_inertial_list[i].east_velocity, velocity_inertial_list[i-1].east_velocity, velocity_inertial_list[i-1].northings, velocity_inertial_list[i-1].eastings)

            if interpolate_remove_flag == True:
                del velocity_inertial_list[0]
                interpolate_remove_flag = False # reset flag
            print('Complete interpolation and coordinate transfomations for velocity_inertial')

    # offset velocity DR by average usbl estimate
        # offset velocity body DR by average usbl estimate
        [northings_usbl_interpolated,eastings_usbl_interpolated] = usbl_offset([i.timestamp for i in dead_reckoning_centre_list],[i.northings for i in dead_reckoning_centre_list],[i.eastings for i in dead_reckoning_centre_list],[i.timestamp for i in usbl_list],[i.northings for i in usbl_list],[i.eastings for i in usbl_list])
        for i in range(len(dead_reckoning_centre_list)):                 
            dead_reckoning_centre_list[i].northings+=northings_usbl_interpolated
            dead_reckoning_centre_list[i].eastings+=eastings_usbl_interpolated
            dead_reckoning_centre_list[i].latitude, dead_reckoning_centre_list[i].longitude = metres_to_latlon(latitude_reference, longitude_reference, dead_reckoning_centre_list[i].eastings, dead_reckoning_centre_list[i].northings)
        for i in range(len(dead_reckoning_dvl_list)):
            dead_reckoning_dvl_list[i].northings+=northings_usbl_interpolated
            dead_reckoning_dvl_list[i].eastings+=eastings_usbl_interpolated
            dead_reckoning_dvl_list[i].latitude, dead_reckoning_dvl_list[i].longitude = metres_to_latlon(latitude_reference, longitude_reference, dead_reckoning_dvl_list[i].eastings, dead_reckoning_dvl_list[i].northings)

        # offset velocity inertial DR by average usbl estimate
        if len(velocity_inertial_list) > 0:
            [northings_usbl_interpolated,eastings_usbl_interpolated] = usbl_offset([i.timestamp for i in velocity_inertial_list],[i.northings for i in velocity_inertial_list],[i.eastings for i in velocity_inertial_list],[i.timestamp for i in usbl_list],[i.northings for i in usbl_list],[i.eastings for i in usbl_list])
            for i in range(len(velocity_inertial_list)):                
                velocity_inertial_list[i].northings+=northings_usbl_interpolated
                velocity_inertial_list[i].eastings+=eastings_usbl_interpolated
                velocity_inertial_list[i].latitude, velocity_inertial_list[i].longitude = metres_to_latlon(latitude_reference, longitude_reference, velocity_inertial_list[i].eastings, velocity_inertial_list[i].northings)

    # perform interpolations of state data to camera{1/2/3} time stamps for both DR and PF
        def camera_setup(camera_list, camera_name, camera_offsets, _centre_list):
            j=0
            n=0
            if camera_list[0].timestamp>_centre_list[-1].timestamp or camera_list[-1].timestamp<_centre_list[0].timestamp: #Check if camera activates before dvl and orientation sensors.
                print('{} timestamps does not overlap with dead reckoning data, check timestamp_history.pdf via -v option.'.format(camera_name))
            else:
                camera_overlap_flag = 0
                for i in range(len(camera_list)):
                    if camera_list[i].timestamp<_centre_list[0].timestamp:
                        camera_overlap_flag = 1
                        pass
                    else:
                        del camera_list[:i]
                        break
                for i in range(len(camera_list)):
                    if j>=len(_centre_list)-1:
                        del camera_list[i:]
                        camera_overlap_flag = 1
                        break
                    while _centre_list[j].timestamp < camera_list[i].timestamp:
                        if j+1>len(_centre_list)-1 or _centre_list[j+1].timestamp>camera_list[-1].timestamp:
                            break
                        j += 1
                    #if j>=1: ?
                    camera_list[i].roll = interpolate(camera_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].roll,_centre_list[j].roll)
                    camera_list[i].pitch = interpolate(camera_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].pitch,_centre_list[j].pitch)
                    if abs(_centre_list[j].yaw-_centre_list[j-1].yaw)>180:
                        if _centre_list[j].yaw>_centre_list[j-1].yaw:
                            camera_list[i].yaw = interpolate(camera_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].yaw,_centre_list[j].yaw-360)                       
                        else:
                            camera_list[i].yaw = interpolate(camera_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].yaw-360,_centre_list[j].yaw)
                        if camera_list[i].yaw<0:
                            camera_list[i].yaw+=360
                        elif camera_list[i].yaw>360:
                            camera_list[i].yaw-=360  
                    else:
                        camera_list[i].yaw = interpolate(camera_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].yaw,_centre_list[j].yaw)
                    
                    # while n<len(time_altitude)-1 and time_altitude[n+1]<time_camera1[i]:
                    #     n += 1
                    # camera1_altitude.append(interpolate(time_camera1[i],time_altitude[n],time_altitude[n+1],altitude[n],altitude[n+1]))
                    camera_list[i].altitude = interpolate(camera_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].altitude,_centre_list[j].altitude)

                    [x_offset,y_offset,z_offset] = body_to_inertial(camera_list[i].roll,camera_list[i].pitch,camera_list[i].yaw, origin_x_offset - camera_offsets[0], origin_y_offset - camera_offsets[1], origin_z_offset - camera_offsets[2])
                    
                    camera_list[i].northings = interpolate(camera_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].northings,_centre_list[j].northings)-x_offset
                    camera_list[i].eastings = interpolate(camera_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].eastings,_centre_list[j].eastings)-y_offset
                    camera_list[i].depth = interpolate(camera_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].depth,_centre_list[j].depth)#-z_offset
                    camera_list[i].latitude, camera_list[i].longitude = metres_to_latlon(latitude_reference, longitude_reference, camera_list[i].eastings, camera_list[i].northings)
                if camera_overlap_flag == 1:
                    print('{} data more than dead reckoning data. Only processed overlapping data and ignored the rest.'.format(camera_name))
                print('Complete interpolation and coordinate transfomations for {}'.format(camera_name))
        if len(camera1_list) > 1:
            camera_setup(camera1_list, camera1_sensor_name, [camera1_x_offset,camera1_y_offset,camera1_z_offset], dead_reckoning_centre_list)
            if len(pf_fusion_centre_list)>1:
                camera_setup(camera1_pf_list, camera1_sensor_name, [camera1_x_offset,camera1_y_offset,camera1_z_offset], pf_fusion_centre_list)
        if len(camera2_list) > 1:
            camera_setup(camera2_list, camera2_sensor_name, [camera2_x_offset,camera2_y_offset,camera2_z_offset], dead_reckoning_centre_list)
            if len(pf_fusion_centre_list)>1:
                camera_setup(camera2_pf_list, camera2_sensor_name, [camera1_x_offset,camera1_y_offset,camera1_z_offset], pf_fusion_centre_list)
        if len(camera3_list) > 1:
            camera_setup(camera3_list, camera3_sensor_name, [camera3_x_offset,camera3_y_offset,camera3_z_offset], dead_reckoning_centre_list)
            if len(pf_fusion_centre_list)>1:
                camera_setup(camera3_pf_list, camera3_sensor_name, [camera1_x_offset,camera1_y_offset,camera1_z_offset], pf_fusion_centre_list)

    # perform interpolations of state data to chemical time stamps for both DR and PF
        def other_data_setup(data_list, position_offsets, data_name, _centre_list):
            j=0
            n=0
            if data_list[0].timestamp>_centre_list[-1].timestamp or data_list[-1].timestamp<_centre_list[0].timestamp: #Check if data appears before dvl and orientation sensors.
                print('{} timestamps does not overlap with dead reckoning data, check timestamp_history.pdf via -v option.'.format(data_name))
            else:
                camera_overlap_flag = 0
                for i in range(len(data_list)):
                    if data_list[i].timestamp<_centre_list[0].timestamp:
                        camera_overlap_flag = 1
                        pass
                    else:
                        del data_list[:i]
                        break
                for i in range(len(data_list)):
                    if j>=len(_centre_list)-1:
                        del data_list[i:]
                        camera_overlap_flag = 1
                        break
                    while _centre_list[j].timestamp < data_list[i].timestamp:
                        if j+1>len(_centre_list)-1 or _centre_list[j+1].timestamp>data_list[-1].timestamp:
                            break
                        j += 1
                    #if j>=1: ?
                    data_list[i].roll = interpolate(data_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].roll,_centre_list[j].roll)
                    data_list[i].pitch = interpolate(data_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].pitch,_centre_list[j].pitch)
                    if abs(_centre_list[j].yaw-_centre_list[j-1].yaw)>180:
                        if _centre_list[j].yaw>_centre_list[j-1].yaw:
                            data_list[i].yaw = interpolate(data_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].yaw,_centre_list[j].yaw-360)                       
                        else:
                            data_list[i].yaw = interpolate(data_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].yaw-360,_centre_list[j].yaw)
                        if data_list[i].yaw<0:
                            data_list[i].yaw+=360
                        elif data_list[i].yaw>360:
                            data_list[i].yaw-=360  
                    else:
                        data_list[i].yaw = interpolate(data_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].yaw,_centre_list[j].yaw)
                    
                    # while n<len(time_altitude)-1 and time_altitude[n+1]<time_camera1[i]:
                    #     n += 1
                    # camera1_altitude.append(interpolate(time_camera1[i],time_altitude[n],time_altitude[n+1],altitude[n],altitude[n+1]))
                    data_list[i].altitude = interpolate(data_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].altitude,_centre_list[j].altitude)

                    [x_offset,y_offset,z_offset] = body_to_inertial(data_list[i].roll,data_list[i].pitch,data_list[i].yaw, origin_x_offset - position_offsets[0], origin_y_offset - position_offsets[1], origin_z_offset - position_offsets[2])
                    
                    data_list[i].northings = interpolate(data_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].northings,_centre_list[j].northings)-x_offset
                    data_list[i].eastings = interpolate(data_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].eastings,_centre_list[j].eastings)-y_offset
                    data_list[i].depth = interpolate(data_list[i].timestamp,_centre_list[j-1].timestamp,_centre_list[j].timestamp,_centre_list[j-1].depth,_centre_list[j].depth)#-z_offset
                    data_list[i].latitude, data_list[i].longitude = metres_to_latlon(latitude_reference, longitude_reference, data_list[i].eastings, data_list[i].northings)
                if camera_overlap_flag == 1:
                    print('{} data more than dead reckoning data. Only processed overlapping data and ignored the rest.'.format(data_name))
                print('Complete interpolation and coordinate transfomations for {}'.format(data_name))
        if len(chemical_list) > 1:
            other_data_setup(chemical_list, [chemical_x_offset, chemical_y_offset, chemical_z_offset] ,'chemical', dead_reckoning_centre_list)
            if len(pf_fusion_centre_list) > 1:
                other_data_setup(chemical_list, [chemical_x_offset, chemical_y_offset, chemical_z_offset] ,'chemical', pf_fusion_centre_list)

    # plot data in pdf
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
            if len(velocity_inertial_list) > 0:
                ax1.plot([i.timestamp for i in velocity_inertial_list],[i.north_velocity for i in velocity_inertial_list], 'b.',label=velocity_inertial_sensor_name)
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Velocity, m/s')
            ax1.legend()
            ax1.grid(True)
            ax1.set_title('north velocity')
            ax2 = fig.add_subplot(323)            
            ax2.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.east_velocity for i in dead_reckoning_dvl_list],'ro',label='DVL')
            if len(velocity_inertial_list) > 0:
                ax2.plot([i.timestamp for i in velocity_inertial_list],[i.east_velocity for i in velocity_inertial_list],'b.',label=velocity_inertial_sensor_name)
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Velocity, m/s')
            ax2.legend()
            ax2.grid(True)
            ax2.set_title('east velocity')
            ax3 = fig.add_subplot(325)            
            ax3.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.down_velocity for i in dead_reckoning_dvl_list],'ro',label='DVL')#time_velocity_body,down_velocity_inertia_dvl,'r.',label='DVL')
            if len(velocity_inertial_list) > 0:
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
            ax1.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.northings for i in dead_reckoning_dvl_list],'r.',label='DVL')#time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
            if len(velocity_inertial_list) > 0:
                ax1.plot([i.timestamp for i in velocity_inertial_list],[i.northings for i in velocity_inertial_list],'g.',label=velocity_inertial_sensor_name)
            ax1.plot([i.timestamp for i in usbl_list], [i.northings for i in usbl_list],'b.',label='USBL')
            ax1.plot([i.timestamp for i in dead_reckoning_centre_list],[i.northings for i in dead_reckoning_centre_list],'c.',label='Centre')#time_velocity_body,northings_dead_reckoning,'b.')
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Northings, m')
            ax1.grid(True)
            ax1.legend()#loc='upper right', bbox_to_anchor=(1, -0.2))
            ax1.set_title('Northings')
            ax2 = fig.add_subplot(222)
            ax2.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.eastings for i in dead_reckoning_dvl_list],'r.',label='DVL')#time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
            if len(velocity_inertial_list) > 0:
                ax2.plot([i.timestamp for i in velocity_inertial_list],[i.eastings for i in velocity_inertial_list],'g.',label=velocity_inertial_sensor_name)
            ax2.plot([i.timestamp for i in usbl_list], [i.eastings for i in usbl_list],'b.',label='USBL')
            ax2.plot([i.timestamp for i in dead_reckoning_centre_list],[i.eastings for i in dead_reckoning_centre_list],'c.',label='Centre')#time_velocity_body,eastings_dead_reckoning,'b.')
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
            # https://plot.ly/python/line-charts/#filled-lines Something like that?
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
            if len(velocity_inertial_list) > 0:
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

        # DR
            print('...plotting camera1_centre_DVL_{}_DR...'.format(velocity_inertial_sensor_name))
            fig=plt.figure()
            ax = fig.add_subplot(111)
            ax.plot([i.eastings for i in camera1_list],[i.northings for i in camera1_list],'y.',label='Camera1')
            ax.plot([i.eastings for i in dead_reckoning_centre_list],[i.northings for i in dead_reckoning_centre_list],'r.',label='Centre')
            ax.plot([i.eastings for i in dead_reckoning_dvl_list],[i.northings for i in dead_reckoning_dvl_list],'g.',label='DVL')
            if len(velocity_inertial_list) > 0:
                ax.plot([i.eastings for i in velocity_inertial_list],[i.northings for i in velocity_inertial_list],'m.',label=velocity_inertial_sensor_name)
            ax.plot([i.eastings for i in usbl_list], [i.northings for i in usbl_list],'c.',label='USBL')
            ax.plot([i.eastings for i in pf_fusion_dvl_list], [i.northings for i in pf_fusion_dvl_list], 'g', label='PF fusion_DVL')
            ax.set_xlabel('Eastings, m')
            ax.set_ylabel('Northings, m')
            ax.legend()#loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)   
            plt.savefig(plotpath + os.sep + 'camera1_centre_DVL_{}_DR.pdf'.format(velocity_inertial_sensor_name), dpi=600, bbox_inches='tight')
            plt.close()

            print('Complete plot data: ', plotpath)

    # plotly data in html
        #if plotly is True:

        print('Plotting plotly data ...')
        plotlypath = renavpath + os.sep + 'interactive_plots'
        
        if os.path.isdir(plotlypath) == 0:
            try:
                os.mkdir(plotlypath)
            except Exception as e:
                print("Warning:",e)

        def create_trace(x_list,y_list,trace_name,trace_color,visibility=True):
            trace = go.Scattergl(
                x=x_list,
                y=y_list,
                visible=visibility, # True | False | legendonly
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
        if len(velocity_inertial_list) > 0:
            trace11b = create_trace([i.timestamp for i in velocity_inertial_list], [i.north_velocity for i in velocity_inertial_list], '{} north velocity'.format(velocity_inertial_sensor_name), 'blue')
            trace21b = create_trace([i.timestamp for i in velocity_inertial_list], [i.east_velocity for i in velocity_inertial_list], '{} east velocity'.format(velocity_inertial_sensor_name), 'blue')
            trace31b = create_trace([i.timestamp for i in velocity_inertial_list], [i.down_velocity for i in velocity_inertial_list], '{} down velocity'.format(velocity_inertial_sensor_name), 'blue')
        # plot1=[trace11a, trace11b]
        trace21a = create_trace([i.timestamp for i in dead_reckoning_dvl_list], [i.east_velocity for i in dead_reckoning_dvl_list], 'DVL east velocity', 'red')
        trace31a = create_trace([i.timestamp for i in dead_reckoning_dvl_list], [i.down_velocity for i in dead_reckoning_dvl_list], 'DVL down velocity', 'red')
        trace12a = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.x_velocity for i in dead_reckoning_centre_list], 'DVL x velocity', 'red')
        trace22a = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.y_velocity for i in dead_reckoning_centre_list], 'DVL y velocity', 'red')
        trace32a = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.z_velocity for i in dead_reckoning_centre_list], 'DVL z velocity', 'red')
        fig = tools.make_subplots(rows=3, cols=2, subplot_titles=('DVL vs {} - north Velocity'.format(velocity_inertial_sensor_name), 'DVL - x velocity / surge', 'DVL vs {} - east Velocity'.format(velocity_inertial_sensor_name), 'DVL - y velocity / sway', 'DVL vs {} - down Velocity'.format(velocity_inertial_sensor_name), 'DVL - z velocity / heave'))
        fig.append_trace(trace11a, 1, 1)
        if len(velocity_inertial_list) > 0:
            fig.append_trace(trace11b, 1, 1)
            fig.append_trace(trace21b, 2, 1)
            fig.append_trace(trace31b, 3, 1)
        fig.append_trace(trace21a, 2, 1)
        fig.append_trace(trace31a, 3, 1)
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

        trace11a = create_trace([i.timestamp for i in dead_reckoning_dvl_list], [i.northings for i in dead_reckoning_dvl_list], 'Northing DVL', 'red')
        if len(velocity_inertial_list)>0:
            trace11b = create_trace([i.timestamp for i in velocity_inertial_list], [i.northings for i in velocity_inertial_list], 'Northing {}'.format(velocity_inertial_sensor_name), 'green')
            trace12b = create_trace([i.timestamp for i in velocity_inertial_list], [i.eastings for i in velocity_inertial_list], 'Easting {}'.format(velocity_inertial_sensor_name), 'green')
        trace11c = create_trace([i.timestamp for i in usbl_list], [i.northings for i in usbl_list], 'Northing USBL', 'blue')
        trace11d = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.northings for i in dead_reckoning_centre_list], 'Northing Centre', 'orange')
        trace12a = create_trace([i.timestamp for i in dead_reckoning_dvl_list], [i.eastings for i in dead_reckoning_dvl_list], 'Easting DVL', 'red')
        trace12c = create_trace([i.timestamp for i in usbl_list], [i.eastings for i in usbl_list], 'Easting USBL', 'blue')
        trace12d = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.eastings for i in dead_reckoning_centre_list], 'Easting Centre', 'orange')
        trace21a = create_trace([i.timestamp for i in altitude_list], [i.seafloor_depth for i in altitude_list], 'Depth  Seafloor (Depth Sensor + Altitude)', 'red')
        trace21b = create_trace([i.timestamp for i in depth_list], [i.depth for i in depth_list], 'Depth Sensor', 'purple')
        trace21c = create_trace([i.timestamp for i in usbl_list], [i.depth for i in usbl_list], 'Depth USBL', 'blue')
        trace21d = create_trace([i.timestamp for i in dead_reckoning_centre_list], [i.depth for i in dead_reckoning_centre_list], 'Depth Centre', 'orange')
        trace22a = create_trace([i.timestamp for i in altitude_list], [i.altitude for i in altitude_list], 'Altitude', 'red')
        fig = tools.make_subplots(rows=2,cols=2, subplot_titles=('Northings', 'Eastings', 'Depth', 'Altitude'))
        fig.append_trace(trace11a, 1, 1)
        if len(velocity_inertial_list)>0:
            fig.append_trace(trace11b, 1, 1)
            fig.append_trace(trace12b, 1, 2)
        fig.append_trace(trace11c, 1, 1)
        fig.append_trace(trace11d, 1, 1)
        fig.append_trace(trace12a, 1, 2)
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

    # pf uncertainty plotly # maybe make a slider plot for this, or a dot projection slider
        trace11a = create_trace([i.timestamp for i in pf_fusion_dvl_list], pf_northings_std, 'northings_std (m)', 'red')
        trace11b = create_trace([i.timestamp for i in pf_fusion_dvl_list], pf_eastings_std, 'eastings_std (m)', 'blue')
        trace11c = create_trace([i.timestamp for i in pf_fusion_dvl_list], pf_yaw_std, 'yaw_std (deg)', 'green')
        layout = go.Layout(
            title='Particle Filter Uncertainty Plot',
            hovermode='closest',
            xaxis=dict(title='Epoch time, s', tickformat='.3f'),
            yaxis=dict(title='Degrees or Metres'),
            dragmode='pan',
            )
        config={'scrollZoom': True}
        fig = go.Figure(data=[trace11a, trace11b, trace11c], layout=layout)
        py.plot(fig, config=config, filename=plotlypath + os.sep + 'pf_uncertainty.html', auto_open=False)

    # Uncertainty plotly --- https://plot.ly/python/line-charts/#filled-lines Something like that?
        trace11a = create_trace([i.timestamp for i in orientation_list], [i.roll_std for i in orientation_list], 'roll std', 'red')
        trace11b = create_trace([i.timestamp for i in orientation_list], [i.pitch_std for i in orientation_list], 'pitch std', 'green')
        trace11c = create_trace([i.timestamp for i in orientation_list], [i.yaw_std for i in orientation_list], 'yaw std', 'blue')
        trace12a = create_trace([i.timestamp for i in velocity_body_list], [i.x_velocity_std for i in velocity_body_list], 'x velocity std', 'red')
        trace12b = create_trace([i.timestamp for i in velocity_body_list], [i.y_velocity_std for i in velocity_body_list], 'y velocity std', 'green')
        trace12c = create_trace([i.timestamp for i in velocity_body_list], [i.z_velocity_std for i in velocity_body_list], 'z velocity std', 'blue')
        trace13a = create_trace([i.timestamp for i in usbl_list], [i.latitude_std for i in usbl_list], 'latitude std usbl', 'red')
        trace13b = create_trace([i.timestamp for i in usbl_list], [i.longitude_std for i in usbl_list], 'longitude std usbl', 'green')
        trace21a = create_trace([i.timestamp for i in depth_list], [i.depth_std for i in depth_list], 'depth std', 'red')
        if len(velocity_inertial_list)>0:
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
        if len(velocity_inertial_list)>0:
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

    # # DR plotly slider *include toggle button that switches between lat long and north east
        print('...plotting auv_path...')

        # might not be robust in the future
        minTimestamp = 99999999999999
        maxTimestamp = -99999999999999

        plotly_list = []
        if len(camera1_list) > 1:
            plotly_list.append(['dr_camera1', camera1_list, 'legendonly'])
        if len(dead_reckoning_centre_list) > 1:
            plotly_list.append(['dr_centre', dead_reckoning_centre_list, 'legendonly'])
        if len(dead_reckoning_dvl_list) > 1:
           plotly_list.append(['dr_dvl',dead_reckoning_dvl_list, True])
        # if len(velocity_inertial_list) > 1:
        #    plotly_list.append([velocity_inertial_sensor_name, velocity_inertial_list])
        if len(usbl_list) > 1:
            plotly_list.append(['usbl', usbl_list, True])

        for i in plotly_list:
            timestamp_list = [j.timestamp for j in i[1]]
            if min(timestamp_list) < minTimestamp:
                minTimestamp = min(timestamp_list)
            if max(timestamp_list) > maxTimestamp:
                maxTimestamp = max(timestamp_list)

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
        def make_data(name,eastings,northings,mode='lines', visibility=True):
            # mode='lines'
            if 'usbl' in name:
                mode='lines+markers'
            # data_dict = go.Scattergl( # this doesn't work
            #     x = eastings,
            #     y = northings,
            #     visible = visibility, # True | False | legendonly
            #     mode = '{}'.format(mode),
            #     name = '{}'.format(name),
            #     # marker=dict(color=trace_color),#'rgba(152, 0, 0, .8)'),#,size = 10, line = dict(width = 2,color = 'rgb(0, 0, 0)'),
            #     # line=dict(color=trace_color)#rgb(205, 12, 24)'))#, width = 4, dash = 'dot')
            # )
            data_dict = {
                'x': eastings,
                'y': northings,
                'mode':'{}'.format(mode),
                'name': '{}'.format(name),
                'visible': visibility
            }
            figure['data'].append(data_dict)

        for i in plotly_list:
            make_data(i[0], [j.eastings for j in i[1]], [j.northings for j in i[1]], visibility=i[2])
        if len(pf_fusion_centre_list) > 1:

            make_data('pf_camera1', [i.eastings for i in pf_fusion_camera1_list], [i.northings for i in pf_fusion_camera1_list], visibility='legendonly')

            make_data('pf_centre', [i.eastings for i in pf_fusion_centre_list], [i.northings for i in pf_fusion_centre_list], visibility='legendonly')
            make_data('pf_dvl', [i.eastings for i in pf_fusion_dvl_list], [i.northings for i in pf_fusion_dvl_list], visibility=True)
            pf_timestamps_interval = []
            pf_eastings_interval = []
            pf_northings_interval = []
            for i in pf_particles_list[0]:
                pf_timestamps_interval.append(pf_particles_list[0][0].timestamps[0])
                pf_eastings_interval.append(i.eastings[0])
                pf_northings_interval.append(i.northings[0])
            timestamp_value_tracker = pf_particles_list[0][0].timestamps[0]
            
            for i in range(len(pf_particles_list)):
                # timestamp_index_tracker = 0
                for j in range(len(pf_particles_list[i][0].timestamps)):# for j in pf_particles_list[i]:
                    if pf_particles_list[i][0].timestamps[j] - timestamp_value_tracker > particles_time_interval: # 
                        for k in pf_particles_list[i]:    # pf_timestamps_interval.append()
                            pf_timestamps_interval.append(k.timestamps[j])
                            pf_eastings_interval.append(k.eastings[j])
                            pf_northings_interval.append(k.northings[j])
                        timestamp_value_tracker = pf_particles_list[i][0].timestamps[j]
            make_data('pf_dvl_distribution', pf_eastings_interval, pf_northings_interval, mode='markers', visibility=True)
     
                 ### ===== for checking and visualization purposes =====
            # make_data('USBL_1', [pf_usbl_datapoints[0].eastings], [pf_usbl_datapoints[0].northings], mode='markers')
            # make_data('PF_Initialization', [i.eastings[0] for i in pf_particles_list[0]], [i.northings[0] for i in pf_particles_list[0]], mode='markers')
            # make_data('PF_First_Propagation', [i.eastings[-1] for i in pf_particles_list[0]], [i.northings[-1] for i in pf_particles_list[0]], mode='markers')
            # make_data('USBL_2', [pf_usbl_datapoints[1].eastings], [pf_usbl_datapoints[1].northings], mode='markers')
            # make_data('PF_First Resampling', [i.eastings[0] for i in pf_particles_list[1]], [i.northings[0] for i in pf_particles_list[1]], mode='markers')
            # make_data('PF_Second_Propagation', [i.eastings[-1] for i in pf_particles_list[1]], [i.northings[-1] for i in pf_particles_list[1]], mode='markers')
            # make_data('USBL_3', [pf_usbl_datapoints[2].eastings], [pf_usbl_datapoints[2].northings], mode='markers')
            # make_data('PF_Second Resampling', [i.eastings[0] for i in pf_particles_list[2]], [i.northings[0] for i in pf_particles_list[2]], mode='markers')
            # make_data('PF_Third_Propagation', [i.eastings[-1] for i in pf_particles_list[2]], [i.northings[-1] for i in pf_particles_list[2]], mode='markers')
            # make_data('USBL_4', [pf_usbl_datapoints[3].eastings], [pf_usbl_datapoints[3].northings], mode='markers')
            # make_data('PF_Third Resampling', [i.eastings[0] for i in pf_particles_list[3]], [i.northings[0] for i in pf_particles_list[3]], mode='markers')
            # temp_list_eastings = []
            # temp_list_northings = []
            # for i in range(3, len(pf_particles_list)):
            #     for j in pf_particles_list[i]:
            #         temp_list_eastings.append(j.eastings[-1])
            #         temp_list_northings.append(j.northings[-1])
            #     temp_list_eastings.append(pf_particles_list[i][0].eastings[0])
            #     temp_list_northings.append(pf_particles_list[i][0].northings[0])
            # make_data('PF_particles', temp_list_eastings, temp_list_northings, mode='markers')
            # make_data('PF_Final-1_Propagation', [i.eastings[-1] for i in pf_particles_list[-2]], [i.northings[-1] for i in pf_particles_list[-2]], mode='markers')
            # make_data('PF_Final Resampling', [i.eastings[0] for i in pf_particles_list[-1]], [i.northings[0] for i in pf_particles_list[-1]], mode='markers')
            # make_data('PF_Final_Propagation', [i.eastings[-1] for i in pf_particles_list[-1]], [i.northings[-1] for i in pf_particles_list[-1]], mode='markers')
            ### ===== for checking and visualization purposes =====
        config={'scrollZoom': True}

        py.plot(figure, config=config, filename=plotlypath + os.sep + 'auv_path.html',auto_open=False)

        def make_frame(data,tstamp, visibility=True, mode='lines'):
            temp_index=-1#next(x[0] for x in enumerate(data[0]) if x[1] > tstamp)
            for i in range(len(data[1])):
                if data[1][i] <= tstamp:
                    temp_index=i
                else:
                    break
            eastings=data[2][:temp_index+1]
            northings=data[3][:temp_index+1]
            # data_dict = go.Scattergl( # this doesn't work
            #     x = eastings,
            #     y = northings,
            #     visible = visibility, # True | False | legendonly
            #     # mode = '{}'.format(mode),
            #     name = '{}'.format(data[0]),
            #     # marker=dict(color=trace_color),#'rgba(152, 0, 0, .8)'),#,size = 10, line = dict(width = 2,color = 'rgb(0, 0, 0)'),
            #     # line=dict(color=trace_color)#rgb(205, 12, 24)'))#, width = 4, dash = 'dot')
            # )
            data_dict = {
                'x': eastings,
                'y': northings,
                'name': '{}'.format(data[0]),
                'mode': '{}'.format(mode)
            }
            frame['data'].append(data_dict)

        #make frames
        for i in epoch_timestamps_slider:
            frame = {'data': [], 'name': str(i)}
            
            for j in plotly_list:
                make_frame([j[0],[k.timestamp for k in j[1]], [k.eastings for k in j[1]], [k.northings for k in j[1]]],i)
            if len(pf_fusion_camera1_list) > 1:
                make_frame(['pf_camera1',[i.timestamp for i in pf_fusion_camera1_list],[i.eastings for i in pf_fusion_camera1_list],[i.northings for i in pf_fusion_camera1_list]],i)
            if len(pf_fusion_centre_list) > 1:
                make_frame(['pf_centre',[i.timestamp for i in pf_fusion_centre_list],[i.eastings for i in pf_fusion_centre_list],[i.northings for i in pf_fusion_centre_list]],i)
            if len(pf_fusion_dvl_list) > 1:
                make_frame(['pf_dvl', [i.timestamp for i in pf_fusion_dvl_list], [i.eastings for i in pf_fusion_dvl_list], [i.northings for i in pf_fusion_dvl_list]], i)
            if len(pf_timestamps_interval) > 1:
                make_frame(['pf_dvl_distribution',pf_timestamps_interval, pf_eastings_interval, pf_northings_interval], i, mode='markers')

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

    # write values out to a csv file
        # create a directory with the time stamp
        
        #if csv_write is True:        

        csvpath = renavpath + os.sep + 'csv'
        drcsvpath = csvpath + os.sep + 'dead_reckoning'
        pfcsvpath = csvpath + os.sep + 'particle_filter'            

        def write_csv(csv_filepath, data_list, csv_filename, csv_flag):
            #check the relvant folders exist and if note create them           
            csv_file = Path(csvpath)
            if csv_file.exists() is False:
                os.mkdir(csvpath)

            csv_file = Path(csv_filepath)
            if csv_file.exists() is False:
                os.mkdir(csv_filepath)        

            if csv_flag == True:
                print("Writing outputs to {}.csv ...".format(csv_filename))
                with open(csv_filepath + os.sep + '{}.csv'.format(csv_filename) ,'w') as fileout:
                    fileout.write('Timestamp, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m], Latitude [deg], Longitude [deg]\n')
                for i in range(len(data_list)):
                    with open(csv_filepath + os.sep + '{}.csv'.format(csv_filename) ,'a') as fileout:
                        try:
                            fileout.write(str(data_list[i].timestamp)+','+str(data_list[i].northings)+','+str(data_list[i].eastings)+','+str(data_list[i].depth)+','+str(data_list[i].roll)+','+str(data_list[i].pitch)+','+str(data_list[i].yaw)+','+str(data_list[i].altitude)+','+str(data_list[i].latitude)+','+str(data_list[i].longitude)+'\n')
                            fileout.close()
                        except IndexError:
                            break

        ### First column of csv file - image file naming step probably not very robust, needs improvement
        def camera_csv(camera_list, camera_name, csv_filepath, csv_flag):
            

            csv_file = Path(csvpath)
            if csv_file.exists() is False:
                os.mkdir(csvpath)

            csv_file = Path(csv_filepath)
            if csv_file.exists() is False:
                os.mkdir(csv_filepath) 

            if csv_flag == True:
                if len(camera_list) > 1:
                    print("Writing outputs to {}.csv ...".format(camera_name))
                    with open(csv_filepath + os.sep + '{}.csv'.format(camera_name) ,'w') as fileout:
                        fileout.write('Imagenumber, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m], Timestamp, Latitude [deg], Longitude [deg]\n')
                    for i in range(len(camera_list)):
                        with open(csv_filepath + os.sep + '{}.csv'.format(camera_name) ,'a') as fileout:
                            try:
                                imagenumber = camera_list[i].filename[-11:-4]
                                if imagenumber.isdigit():
                                    image_filename=int(imagenumber)
                                else:
                                    image_filename=camera_list[i].filename
                                fileout.write(str(image_filename)+','+str(camera_list[i].northings)+','+str(camera_list[i].eastings)+','+str(camera_list[i].depth)+','+str(camera_list[i].roll)+','+str(camera_list[i].pitch)+','+str(camera_list[i].yaw)+','+str(camera_list[i].altitude)+','+str(camera_list[i].timestamp)+','+str(camera_list[i].latitude)+','+str(camera_list[i].longitude)+'\n')
                                fileout.close()
                            except IndexError:
                                break

        # if this works make all follow this format!
        def other_data_csv(data_list, data_name, csv_filepath, csv_flag):
            
            csv_file = Path(csvpath)
            if csv_file.exists() is False:
                os.mkdir(csvpath)

            csv_file = Path(csv_filepath)
            if csv_file.exists() is False:
                os.mkdir(csv_filepath) 

            if csv_flag == True:
                print("Writing outputs to {}.csv ...".format(data_name))
                # csv_header = 
                csv_row_data_list = []
                for i in data_list:
                    csv_row_data = {'epochtimestamp':i.timestamp,'Northing [m]':i.northings, 'Easting [m]': i.eastings, 'Depth [m]': i.depth, 'Roll [deg]': i.roll, 'Pitch [deg]': i.pitch, 'Heading [deg]': i.yaw, 'Altitude [m]':i.altitude, 'Latitude [deg]': i.latitude, 'Longitude [deg]': i.longitude}
                    for j in i.data:
                        csv_row_data.update({'{} [{}]'.format(j['label'], j['units']):j['value']})
                    csv_row_data_list.append(csv_row_data)
                df = pd.DataFrame(csv_row_data_list)
                df.to_csv(csv_filepath + os.sep + '{}.csv'.format(data_name), header=True, index = False) # , na_rep='-') https://www.youtube.com/watch?v=hmYdzvmcTD8

        if csv_active is True:
            if csv_usbl is True:
                if len(usbl_list) > 1:
                    csv_file = Path(csvpath)
                    if csv_file.exists() is False:
                        os.mkdir(csvpath)

                    print("Writing outputs to auv_usbl.csv ...")
                    with open(csvpath + os.sep + 'auv_usbl.csv' ,'w') as fileout:
                        fileout.write('Timestamp, Northing [m], Easting [m], Depth [m], Latitude [deg], Longitude [deg]\n')
                    for i in range(len(usbl_list)):
                        with open(csvpath + os.sep + 'auv_usbl.csv' ,'a') as fileout:
                            try:
                                fileout.write(str(usbl_list[i].timestamp)+','+str(usbl_list[i].northings)+','+str(usbl_list[i].eastings)+','+str(usbl_list[i].depth)+','+str(usbl_list[i].latitude)+','+str(usbl_list[i].longitude)+'\n')
                                fileout.close()
                            except IndexError:
                                break
            if csv_dr_auv_centre is True:
                write_csv(drcsvpath, dead_reckoning_centre_list, 'auv_dr_centre', csv_dr_auv_centre)
            if csv_dr_auv_dvl is True:    
                write_csv(drcsvpath, dead_reckoning_dvl_list, 'auv_dr_dvl', csv_dr_auv_dvl)
            # if len(velocity_inertial_list) > 1:
            #     write_csv(drcsvpath, velocity_inertial_list, 'auv_{}'.format(velocity_inertial_sensor_name)) # can't use this cuz missing Altitude!
            if csv_dr_camera_1 is True:
                camera_csv(camera1_list, 'auv_dr_' + camera1_sensor_name, drcsvpath, csv_dr_camera_1)
            if csv_dr_camera_2 is True:
                camera_csv(camera2_list, 'auv_dr_' + camera2_sensor_name, drcsvpath, csv_dr_camera_2)
            if csv_dr_camera_3 is True:
                camera_csv(camera3_list, 'auv_dr_' + camera3_sensor_name, drcsvpath, csv_dr_camera_3)
                
            if len(chemical_list) > 1:
                other_data_csv(chemical_list, 'auv_dr_chemical', drcsvpath, csv_dr_chemical)

            # if len(pf_eastings)>1:
            #     write_csv(pfcsvpath, ) # can't use this cuz diff format! write new function for all pf related stuff
            if csv_pf_auv_centre is True:
                write_csv(pfcsvpath, pf_fusion_centre_list, 'auv_pf_centre', csv_pf_auv_centre)
            if csv_pf_auv_dvl is True:    
                write_csv(pfcsvpath, pf_fusion_dvl_list, 'auv_pf_dvl', csv_pf_auv_dvl)    
            if csv_pf_camera_1 is True:
                camera_csv(camera1_pf_list, 'auv_pf_' + camera1_sensor_name, pfcsvpath, csv_pf_camera_1)
            if csv_pf_camera_2 is True:
                camera_csv(camera2_pf_list, 'auv_pf_' + camera2_sensor_name, pfcsvpath, csv_pf_camera_2)
            if csv_pf_camera_3 is True:
                camera_csv(camera3_pf_list, 'auv_pf_' + camera3_sensor_name, pfcsvpath, csv_pf_camera_3)
            
            if len(chemical_list) > 1:
                    other_data_csv(chemical_list, 'auv_pf_chemical' , pfcsvpath, csv_pf_chemical)

        print('Complete extraction of data: ', csvpath)

        print('Completed data extraction: ', renavpath)
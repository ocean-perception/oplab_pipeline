# extract_data

# Scripts to extract data from nav_standard.json, and combined.auv.raw an save csv files and, if plot is True, save plots

# Author: Blair Thornton
# Date: 14/12/2017


# Import librarys
import sys, os, csv
import yaml, json
import shutil, math
import calendar, codecs
import operator
#import hashlib, glob

import matplotlib.pyplot as plt

from datetime import datetime

sys.path.append("..")
from lib_calculus.interpolate import interpolate
from lib_localisation.dead_reckoning import dead_reckoning
from lib_coordinates.body_to_inertial import body_to_inertial

class extract_data:
    def __init__(self, filepath,ftype,start_time,finish_time,plot,csv_write):

        interpolate_remove_flag = False

         # load data should at this point be able to specify time stamp range (see asv_nav)
        time_velocity_body=[]
        x_velocity=[]
        y_velocity=[]
        z_velocity=[]

        time_orientation=[]
        roll=[]
        pitch=[]
        yaw=[]

        time_velocity_inertia=[]
        north_velocity_inertia=[]
        east_velocity_inertia=[]
        down_velocity_inertia=[]
        northings_inertia_dead_reckoning=[]
        eastings_inertia_dead_reckoning=[]
        depth_inertia_dead_reckoning=[]


        time_depth=[]
        depth=[]

        time_altitude=[]
        altitude=[]
            
        # placeholders for interpolated measurements
        roll_interpolated=[]
        pitch_interpolated=[]
        yaw_interpolated=[]
        roll_inertia_interpolated=[]
        pitch_inertia_interpolated=[]
        yaw_inertia_interpolated=[]

        # interpolate depth and add altitude for every altitude measurement
        seafloor_depth=[] 

        # placeholders for transformed coordinates
        north_velocity_inertia_dvl=[]
        east_velocity_inertia_dvl=[]
        down_velocity_inertia_dvl=[]
        northings_dead_reckoning=[]
        eastings_dead_reckoning=[]
        depth_dead_reckoning=[]

        # placeholders for USBL
        time_usbl=[]
        latitude_usbl=[]
        longitude_usbl=[]
        northings_usbl=[]
        eastings_usbl=[]
        depth_usbl=[]



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

            # setup the time window
            yyyy = int(date[0:4])
            mm =  int(date[5:7])
            dd =  int(date[8:10])

            hours = int(start_time[0:2])
            mins = int(start_time[2:4])
            secs = int(start_time[4:6])
                
            dt_obj = datetime(yyyy,mm,dd,hours,mins,secs)       
            time_tuple = dt_obj.utctimetuple()
            epoch_start_time = calendar.timegm(time_tuple) 
                
            hours = int(finish_time[0:2])
            mins = int(finish_time[2:4])
            secs = int(finish_time[4:6])        

            dt_obj = datetime(yyyy,mm,dd,hours,mins,secs)
            time_tuple = dt_obj.utctimetuple()
            epoch_finish_time = calendar.timegm(time_tuple) 


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
                                #time_velocity_body.append(parsed_json_data[i]['epoch_timestamp'])                            
                                x_velocity.append(parsed_json_data[i]['data'][0]['x_velocity'])
                                y_velocity.append(parsed_json_data[i]['data'][1]['y_velocity'])
                                z_velocity.append(parsed_json_data[i]['data'][2]['z_velocity'])
                                roll_interpolated.append(0)
                                pitch_interpolated.append(0)
                                yaw_interpolated.append(0)
                                northings_dead_reckoning.append(0)
                                eastings_dead_reckoning.append(0)
                                depth_dead_reckoning.append(0)


                        if 'inertial' in parsed_json_data[i]['frame']:
                            time_velocity_inertia.append(parsed_json_data[i]['epoch_timestamp'])

                            north_velocity_inertia.append(parsed_json_data[i]['data'][0]['north_velocity'])
                            east_velocity_inertia.append(parsed_json_data[i]['data'][1]['east_velocity'])
                            down_velocity_inertia.append(parsed_json_data[i]['data'][2]['down_velocity'])
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

                    if 'depth' in parsed_json_data[i]['category']:
                        time_depth.append(parsed_json_data[i]['epoch_timestamp_depth'])
                        depth.append(parsed_json_data[i]['data'][0]['depth'])

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

            #make path for csv and plots
            renavpath = filepath + 'json_renav_' + str(yyyy).zfill(4) + str(mm).zfill(2) + str(dd).zfill(2) + '_' + start_time + '_' + finish_time 
            if os.path.isdir(renavpath) == 0:
                try:
                    os.mkdir(renavpath)
                except Exception as e:
                    print("Warning:",e)

            
            print('Complete parse of:' + outpath + os.sep + filename)
            print('Writing outputs to: ' + renavpath)
        

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
            time_tuple = dt_obj.utctimetuple()
            epoch_start_time = calendar.timegm(time_tuple) 
                
            hours = int(finish_time[0:2])
            mins = int(finish_time[2:4])
            secs = int(finish_time[4:6])        

            dt_obj = datetime(yyyy,mm,dd,hours,mins,secs)
            time_tuple = dt_obj.utctimetuple()
            epoch_finish_time = calendar.timegm(time_tuple) 




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

                            roll_interpolated.append(0)
                            pitch_interpolated.append(0)
                            yaw_interpolated.append(0)
                            northings_dead_reckoning.append(0)
                            eastings_dead_reckoning.append(0)
                            depth_dead_reckoning.append(0)

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



        # perform coordinate transformations and interpolations                
        j=0
        k=0
        
        for i in range(len(time_velocity_body)):        
                

            # interpolate to find the appropriate orientation timefor the dvl measurements
            while time_orientation[j]<time_velocity_body[i] and j< len(time_orientation)-1:
                j=j+1
            
            if j==1:
                interpolate_remove_flag = True
            else:
                roll_interpolated[i]=interpolate(time_velocity_body[i],time_orientation[j-1],time_orientation[j],roll[j-1],roll[j])
                pitch_interpolated[i]=interpolate(time_velocity_body[i],time_orientation[j-1],time_orientation[j],pitch[j-1],pitch[j])

                if abs(yaw[j]-yaw[j-1])>180:                        
                    if yaw[j]>yaw[j-1]:
                        yaw_interpolated[i]=interpolate(time_velocity_body[i],time_orientation[j-1],time_orientation[j],yaw[j-1],yaw[j]-360)
                        #yaw_interpolated[i]=((yaw[j]-360)-yaw[j-1])/(time_orientation[j]-time_orientation[j-1])*(time_velocity_body[i]-time_orientation[j-1])+yaw[j-1]                            
                        
                    else:
                        yaw_interpolated[i]=interpolate(time_velocity_body[i],time_orientation[j-1],time_orientation[j],yaw[j-1]-360,yaw[j])
                        #yaw_interpolated[i]=(yaw[j]-(yaw[j-1]-360))/(time_orientation[j]-time_orientation[j-1])*(time_velocity_body[i]-time_orientation[j-1])+yaw[j-1]
                       

                    if yaw_interpolated[i]<0:
                        yaw_interpolated[i]=yaw_interpolated[i]+360
                        

                    elif yaw_interpolated[i]>360:
                        yaw_interpolated[i]=yaw_interpolated[i]-360  

                else:
                    yaw_interpolated[i]=interpolate(time_velocity_body[i],time_orientation[j-1],time_orientation[j],yaw[j-1],yaw[j])
            
            [x_offset,y_offset,z_offset] = body_to_inertial(roll_interpolated[i], pitch_interpolated[i], yaw_interpolated[i], x_velocity[i], y_velocity[i], z_velocity[i])

            north_velocity_inertia_dvl.append(x_offset)
            east_velocity_inertia_dvl.append(y_offset)
            down_velocity_inertia_dvl.append(z_offset)

            while time_depth[k]<time_velocity_body[i] and k < len(time_depth)-1:
                k=k+1

                if k>=1:                
                    #seafloor_depth[i]=(depth[j]-depth[j-1])/(time_depth[j]-time_depth[j-1])*(time_altitude[i]-time_depth[j-1])+depth[j-1]+altitude[i] 
                    #print(k,i,len(time_depth),len(time_velocity_body),len(altitude))                     
                    depth_dead_reckoning[i]=interpolate(time_velocity_body[i],time_depth[k-1],time_depth[k],depth[k-1],depth[k])+altitude[i]
                    

        j=0
        # interpolate to find the appropriate depth to compute seafloor depth for each altitude measurement
        for i in range(len(time_altitude)):        
            while time_depth[j]<time_altitude[i] and j < len(time_depth)-1:
                j=j+1

                if j>=1:                
                    #seafloor_depth[i]=(depth[j]-depth[j-1])/(time_depth[j]-time_depth[j-1])*(time_altitude[i]-time_depth[j-1])+depth[j-1]+altitude[i]  
                    seafloor_depth[i]=interpolate(time_altitude[i],time_depth[j-1],time_depth[j],depth[j-1],depth[j])+altitude[i]

            # interpolate to find the appropriate depth for dead_reckoning
            
            
            

            #dead reckoning solution
        for i in range(len(time_velocity_body)):
            # dead reckoning solution
            if i>1:                    
                [northings_dead_reckoning[i], eastings_dead_reckoning[i]]=dead_reckoning(time_velocity_body[i], time_velocity_body[i-1], north_velocity_inertia_dvl[i-1], east_velocity_inertia_dvl[i-1], northings_dead_reckoning[i-1], eastings_dead_reckoning[i-1])


        #remove first term if first time_velocity_body is < orientation time
        if interpolate_remove_flag == True:

            del time_velocity_body[0]
            del roll_interpolated[0]
            del pitch_interpolated[0]
            del yaw_interpolated[0]
            del x_velocity[0]
            del y_velocity[0]
            del z_velocity[0]
            del north_velocity_inertia_dvl[0]
            del east_velocity_inertia_dvl[0]
            del down_velocity_inertia_dvl[0]
            del northings_dead_reckoning[0]
            del eastings_dead_reckoning[0]
            del depth_dead_reckoning[0]
            interpolate_remove_flag = False # reset flag

        #initialise counters for interpolation
        j=0
        k=0

        #interpolate state data to inertia time stamps
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
                        #yaw_interpolated[i]=((yaw[j]-360)-yaw[j-1])/(time_orientation[j]-time_orientation[j-1])*(time_velocity_body[i]-time_orientation[j-1])+yaw[j-1]                            
                        
                    else:
                        yaw_inertia_interpolated[i]=interpolate(time_velocity_inertia[i],time_orientation[j-1],time_orientation[j],yaw[j-1]-360,yaw[j])
                        #yaw_interpolated[i]=(yaw[j]-(yaw[j-1]-360))/(time_orientation[j]-time_orientation[j-1])*(time_velocity_body[i]-time_orientation[j-1])+yaw[j-1]
                       

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
            del northings_inertia_dead_reckoning[0]
            del eastings_inertia_dead_reckoning[0]
            del depth_inertia_dead_reckoning[0]
            interpolate_remove_flag = False # reset flag

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

            with open(csvpath + os.sep + 'body_velocity.csv' ,'w') as fileout:
               fileout.write('epoch_time (s), x_velocity (m/s), y_velocity (m/s), z_velocity (m/s)\n')
            for i in range(len(time_velocity_body)):        
               with open(csvpath + os.sep + 'body_velocity.csv' ,'a') as fileout:
                   fileout.write(str(time_velocity_body[i])+','+str(x_velocity[i])+','+str(y_velocity[i])+','+str(z_velocity[i])+'\n')
                   fileout.close()

            with open(csvpath + os.sep + 'orientation.csv' ,'w') as fileout:
               fileout.write('epoch_time (s), roll (degrees), pitch (degrees), yaw (degrees)\n')
            for i in range(len(time_orientation)):        
               with open(csvpath + os.sep + 'orientation.csv' ,'a') as fileout:
                   fileout.write(str(time_orientation[i])+','+str(roll[i])+','+str(pitch[i])+','+str(yaw[i])+'\n')
                   fileout.close()

            with open(csvpath + os.sep + 'velocity_inertia.csv' ,'w') as fileout:
               fileout.write('epoch_time (s), north_velocity_inertia (m/s), east_velocity_inertia (m/s), down_velocity_inertia (m/s)\n')
            for i in range(len(time_velocity_inertia)):        
               with open(csvpath + os.sep + 'velocity_inertia.csv' ,'a') as fileout:
                   fileout.write(str(time_velocity_inertia[i])+','+str(north_velocity_inertia[i])+','+str(east_velocity_inertia[i])+','+str(down_velocity_inertia[i])+'\n')
                   fileout.close()                    

            with open(csvpath + os.sep + 'body_timing_interpolated_orientation.csv' ,'w') as fileout:
               fileout.write('epoch_time (s), roll (degrees), pitch (degrees), yaw (degrees)\n')
            for i in range(len(time_velocity_body)):        
               with open(csvpath + os.sep + 'body_timing_interpolated_orientation.csv' ,'a') as fileout:
                   fileout.write(str(time_velocity_body[i])+','+str(roll_interpolated[i])+','+str(pitch_interpolated[i])+','+str(yaw_interpolated[i])+'\n')
                   fileout.close()

            with open(csvpath + os.sep + 'inertia_dvl_velocity.csv' ,'w') as fileout:
               fileout.write('epoch_time (s), north_velocity_inertia (m/s), east_velocity_inertia (m/s), down_velocity_inertia (m/s)\n')
            for i in range(len(time_velocity_body)):        
               with open(csvpath + os.sep + 'inertia_dvl_velocity.csv' ,'a') as fileout:
                   fileout.write(str(time_velocity_body[i])+','+str(north_velocity_inertia_dvl[i])+','+str(east_velocity_inertia_dvl[i])+','+str(down_velocity_inertia_dvl[i])+'\n')
                   fileout.close()

            with open(csvpath + os.sep + 'dead_reckoning.csv' ,'w') as fileout:
               fileout.write('epoch_time (s), northings (m), eastings (m), depth (m), roll (degrees), pitch (degrees), yaw (degrees)\n')
            for i in range(len(time_velocity_body)):        
               with open(csvpath + os.sep + 'dead_reckoning.csv' ,'a') as fileout:
                   fileout.write(str(time_velocity_body[i])+','+str(northings_dead_reckoning[i])+','+str(eastings_dead_reckoning[i])+','+str(depth_dead_reckoning[i])+','+str(roll_interpolated[i])+','+str(pitch_interpolated[i])+','+str(yaw_interpolated[i])+'\n')
                   fileout.close()                  

            with open(csvpath + os.sep + 'inertia_dead_reckoning.csv' ,'w') as fileout:
               fileout.write('epoch_time (s), northings (m), eastings (m), depth (m), roll (degrees), pitch (degrees), yaw (degrees)\n')
            for i in range(len(time_velocity_inertia)):        
               with open(csvpath + os.sep + 'inertia_dead_reckoning.csv' ,'a') as fileout:
                   fileout.write(str(time_velocity_inertia[i])+','+str(northings_inertia_dead_reckoning[i])+','+str(eastings_inertia_dead_reckoning[i])+','+str(depth_inertia_dead_reckoning[i])+','+str(roll_inertia_interpolated[i])+','+str(pitch_inertia_interpolated[i])+','+str(yaw_inertia_interpolated[i])+'\n')
                   fileout.close()                   

            with open(csvpath + os.sep + 'depth.csv' ,'w') as fileout:
               fileout.write('epoch_time (s), depth (m)\n')
            for i in range(len(time_depth)):        
               with open(csvpath + os.sep + 'depth.csv' ,'a') as fileout:
                   fileout.write(str(time_depth[i])+','+str(depth[i])+'\n')
                   fileout.close()

            with open(csvpath + os.sep + 'altitude.csv' ,'w') as fileout:
               fileout.write('epoch_time (s), altitude (m), seafloor_depth (m)\n')
            for i in range(len(time_altitude)):        
               with open(csvpath + os.sep + 'altitude.csv' ,'a') as fileout:
                   fileout.write(str(time_altitude[i])+','+str(altitude[i])+','+str(seafloor_depth[i])+'\n')
                   fileout.close()

            with open(csvpath + os.sep + 'usbl.csv' ,'w') as fileout:
               fileout.write('epoch_time (s), latitude_reference (degrees), longitude_reference (degrees), latitude (degrees), longitude (degrees), northings (m), eastings (m), depth (m)\n')
            
                    
            for i in range(len(time_usbl)):        
               with open(csvpath + os.sep + 'usbl.csv' ,'a') as fileout:
                   fileout.write(str(time_usbl[i])+','+str(latitude_reference)+','+str(longitude_reference)+','+str(latitude_usbl[i])+','+str(longitude_usbl[i])+','+str(northings_usbl[i])+','+str(eastings_usbl[i])+','+str(depth_usbl[i])+'\n')
                   fileout.close()

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

            # orientation_interpolated
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_velocity_body, yaw_interpolated, 'ro',label='Interpolated dvl')
            ax.plot(time_orientation, yaw,'b.',label='Original')                            
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Yaw, degrees')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'orientation_yaw.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_velocity_body, roll_interpolated, 'ro',label='Interpolated dvl')
            ax.plot(time_orientation, roll,'b.',label='Original')                                     
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Roll, degrees')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'orientation_roll.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_velocity_body, pitch_interpolated, 'ro',label='Interpolated dvl')
            ax.plot(time_orientation, pitch,'b.',label='Original')                            
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Pitch, degrees')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'orientation_pitch.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            # velocities in body frame
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_velocity_body,x_velocity, 'r.',label='Surge')
            ax.plot(time_velocity_body,y_velocity, 'g.',label='Sway')
            ax.plot(time_velocity_body,z_velocity, 'b.',label='Heave')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Velocity, m/s')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'velocity_body.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            # velocities in inertial frame
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)            
            ax.plot(time_velocity_body,north_velocity_inertia_dvl, 'ro',label='DVL')
            ax.plot(time_velocity_inertia,north_velocity_inertia, 'b.',label='Phins')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Velocity, m/s')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'velocity_inertia_north.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)            
            ax.plot(time_velocity_body,east_velocity_inertia_dvl,'ro',label='DVL')
            ax.plot(time_velocity_inertia,east_velocity_inertia,'b.',label='Phins')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Velocity, m/s')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'velocity_inertia_east.pdf', dpi=600, bbox_inches='tight')
            plt.close()

            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)            
            ax.plot(time_velocity_body,down_velocity_inertia_dvl,'r.',label='DVL')
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

            # usbl depth and time
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
            ax.plot(time_velocity_body,northings_dead_reckoning,'b.')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Northings, m')
            ax.grid(True)                
            plt.savefig(plotpath + os.sep + 'dead_reckoning_northings_time.pdf', dpi=600, bbox_inches='tight')
            plt.close() 

            # dead_reckoning eastings time
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot(time_velocity_body,eastings_dead_reckoning,'b.')
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Eastings, m')
            ax.grid(True)                
            plt.savefig(plotpath + os.sep + 'dead_reckoning_eastings_time.pdf', dpi=600, bbox_inches='tight')
            plt.close() 
            print('Complete plot data: ', plotpath)

            
            
            # compare dvl, inertia. usbl northings time
            # find initial position offset
            i=0
            j=0
            exit_flag=False
            while i < len(northings_usbl)-1 and exit_flag == False:
     #           print(i,len(northings_usbl))
                if time_usbl[i] < time_velocity_body[0]:
                    j=i
                else:
                    exit_flag = True
                i=i+1
                    
            # interpolate unless end member, where extrapolate 
            if j==0:
                j=j+1 
                            
            # compute the location interpolated USBL value            
            northings_usbl_interpolated=interpolate(time_velocity_body[0],time_usbl[j-1],time_usbl[j],northings_usbl[j-1],northings_usbl[j])
            eastings_usbl_interpolated=interpolate(time_velocity_body[0],time_usbl[j-1],time_usbl[j],eastings_usbl[j-1],eastings_usbl[j])
            
            # print(j, time_usbl[j-1],time_velocity_body[0],time_usbl[j])
            # print(j, northings_usbl[j-1],northings_usbl_interpolated,northings_usbl[j])
            # print(j, eastings_usbl[j-1],eastings_usbl_interpolated,eastings_usbl[j])

            # plt.plot([time_usbl[j-1],time_usbl[j]],[northings_usbl[j-1],northings_usbl[j]],'b-')
            # plt.plot([time_usbl[j],time_velocity_body[0]], [northings_usbl[j],northings_usbl_interpolated],'r--')
            # plt.show()


            # plt.plot([time_usbl[j-1],time_usbl[j]],[eastings_usbl[j-1],eastings_usbl[j]],'b-')
            # plt.plot([time_usbl[j],time_velocity_body[0]], [eastings_usbl[j],eastings_usbl_interpolated],'r--')
            # plt.show()

            # offset the deadreackoning by this initial estimate
            for i in range(len(northings_dead_reckoning)):                
                northings_dead_reckoning[i]=northings_dead_reckoning[i]+northings_usbl_interpolated
                eastings_dead_reckoning[i]=eastings_dead_reckoning[i]+eastings_usbl_interpolated

            

            # compare dvl, inertia. usbl northings time
            # find initial position offset
            i=0
            j=0
            exit_flag=False
            while i < len(northings_usbl)-1 and exit_flag == False:
 #               print(i,len(northings_usbl))
                if time_usbl[i] < time_velocity_inertia[0]:
                    j=i
                else:
                    exit_flag = True
                i=i+1

            # interpolate unless end member, where extrapolate 
            if j==0:
                j=j+1 

            
            northings_usbl_interpolated=interpolate(time_velocity_inertia[0],time_usbl[j-1],time_usbl[j],northings_usbl[j-1],northings_usbl[j])
            eastings_usbl_interpolated=interpolate(time_velocity_inertia[0],time_usbl[j-1],time_usbl[j],eastings_usbl[j-1],eastings_usbl[j])
            

            # print(j, time_usbl[j-1],time_velocity_inertia[0],time_usbl[j])
            # plt.plot([time_usbl[j-1],time_usbl[j]],[northings_usbl[j-1],northings_usbl[j]],'b-')
            # plt.plot([time_usbl[j],time_velocity_inertia[0], [northings_usbl[j],northings_usbl_interpolated]],'r--')
            # plt.show()


            # plt.plot([time_usbl[j-1],time_usbl[j]],[eastings_usbl[j-1],eastings_usbl[j]],'b-')
            # plt.plot([time_usbl[j],time_velocity_inertia[0]], [eastings_usbl[j],eastings_usbl_interpolated],'r--')
            # plt.show()
            northings_usbl_interpolated=northings_usbl_interpolated-northings_inertia_dead_reckoning[0]
            eastings_usbl_interpolated=eastings_usbl_interpolated-eastings_inertia_dead_reckoning[0]
            for i in range(len(northings_inertia_dead_reckoning)):  
                #print(northings_inertia_dead_reckoning[i],eastings_inertia_dead_reckoning[i])
                northings_inertia_dead_reckoning[i]=northings_inertia_dead_reckoning[i]+northings_usbl_interpolated
                eastings_inertia_dead_reckoning[i]=eastings_inertia_dead_reckoning[i]+eastings_usbl_interpolated
                #print(northings_inertia_dead_reckoning[i],northings_usbl_interpolated,northings_inertia_dead_reckoning[0])
                #print(eastings_inertia_dead_reckoning[i],eastings_usbl_interpolated,eastings_inertia_dead_reckoning[0])

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
            ax.plot(time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
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
            ax.plot(time_velocity_body,eastings_dead_reckoning,'r.',label='DVL')
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
            ax.plot(eastings_dead_reckoning,northings_dead_reckoning,'r.',label='DVL')
            ax.plot(eastings_inertia_dead_reckoning,northings_inertia_dead_reckoning,'g.',label='PHINS')
            ax.plot(eastings_usbl, northings_usbl,'b.',label='USBL')
            ax.set_xlabel('Eastings, m')
            ax.set_ylabel('Northings, m')
            ax.grid(True) 
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))               
            plt.savefig(plotpath + os.sep + 'body_inertia_usbl_northings_eastings.pdf', dpi=600, bbox_inches='tight')
            plt.close()                


            print('Complete plot data: ', plotpath)         

        print('Completed data extraction: ', renavpath)
        
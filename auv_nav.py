# auv_nav

# Scripts to parse and interleave raw sensor data and output in acfr or oplab formats
 
# Author: Blair Thornton
# Date: 25/08/2017

"""Parsers for navigation data for oplab standard and acfr standard formats

        inputs are 

        auv_nav.py <options>
            -i <path to mission.yaml>
            -o <output type> 'acfr' or 'oplab'
            -e <path to root processed folder where parsed data exists>
            -s <start time in utc time> hhmmss (only for extract)")
            -f <finish time in utc time> hhmmss (only for extract)")                        
            -p <plot option> (only for extract)


        Arguments:
            path to the "mission.yaml" file, output format 'acfr' or 'oplab'

                #YAML 1.0
                origin:
                    latitude: 26.674083
                    longitude: 127.868054               
                    coordinate_reference_system: wgs84  
                    date: 2017/08/17              

                velocity:
                    format: phins
                    thread: dvl
                    filepath: nav/phins/
                    filename: 20170817_phins.txt
                    timezone: utc
                    timeoffset: 0.0
                    headingoffset: -45.0

                orientation:
                    format: phins
                    filepath: nav/phins/
                    filename: 20170817_phins.txt
                    timezone: utc
                    timeoffset: 0.0
                    headingoffset: -45.0

                depth:
                    format: phins
                    filepath: nav/phins/
                    filename: 20170817_phins.txt
                    timezone: utc
                    timeoffset: 0.0

                altitude:
                    format: phins
                    filepath: nav/phins/
                    filename: 20170817_phins.txt
                    timezone: utc
                    timeoffset: 0.0

                usbl:
                    format: gaps
                    filepath: nav/gaps/
                    timezone: utc
                    timeoffset: 0.0

                image:
                    format: acfr_standard
                    filepath: image/r20170817_041459_UG117_sesoko/i20170817_041459/
                    camera1: LC
                    camera2: RC
                    timezone: utc
                    timeoffset: 0.0


        Returns:
            interleaved navigation and imaging data with output options:

                'acfr' - combined.RAW.auv
                    PHINS_COMPASS: 1444452882.644 r: -2.29 p: 17.21 h: 1.75 std_r: 0 std_p: 0 std_h: 0
                    RDI: 1444452882.644 alt:200 r1:0 r2:0 r3:0 r4:0 h:1.75 p:17.21 r:-2.29 vx:0.403 vy:0 vz:0 nx:0 ny:0 nz:0 COG:0 SOG:0 bt_status:32768 h_true:0 p_gimbal:0 sv: 1500
                    PAROSCI: 1444452882.644 298.289
                    VIS: 1444452882.655 [1444452882.655] sx_073311_image0003805_AC.tif exp: 0
                    VIS: 1444452882.655 [1444452882.655] sx_073311_image0003805_FC.tif exp: 0
                    SSBL_FIX: 1444452883 ship_x: 402.988947 ship_y: 140.275056 target_x: 275.337171 target_y: 304.388346 target_z: 299.2 target_hr: 0 target_sr: 364.347071 target_bearing: 127.876747

                'oplab' - nav_standard.json
                    [{"epoch_timestamp": 1501974125.926, "epoch_timestamp_dvl": 1501974125.875, "class": "measurement", "sensor": "phins", "frame": "body", "category": "velocity", "data": [{"x_velocity": -0.075, "x_velocity_std": 0.200075}, {"y_velocity": 0.024, "y_velocity_std": 0.200024}, {"z_velocity": -0.316, "z_velocity_std": 0.20031600000000002}]},
                    {"epoch_timestamp": 1501974002.1, "class": "measurement", "sensor": "phins", "frame": "inertial", "category": "orientation", "data": [{"heading": 243.777, "heading_std": 2.0}, {"roll": 4.595, "roll_std": 0.1}, {"pitch": 0.165, "pitch_std": 0.1}]},
                    {"epoch_timestamp": 1501974125.926, "epoch_timestamp_dvl": 1501974125.875, "class": "measurement", "sensor": "phins", "frame": "body", "category": "altitude", "data": [{"altitude": 31.53, "altitude_std": 0.3153}, {"sound_velocity": 1546.0, "sound_velocity_correction": 0.0}]},
                    {"epoch_timestamp": 1501974002.7, "epoch_timestamp_depth": 1501974002.674, "class": "measurement", "sensor": "phins", "frame": "inertial", "category": "depth", "data": [{"depth": -0.958, "depth_std": -9.58e-05}]},
                    {"epoch_timestamp": 1502840568.204, "class": "measurement", "sensor": "gaps", "frame": "inertial", "category": "usbl", "data_ship": [{"latitude": 26.66935735000014, "longitude": 127.86623359499968}, {"northings": -526.0556603025898, "eastings": -181.08730736724087}, {"heading": 174.0588800058365}], "data_target": [{"latitude": 26.669344833333334, "latitude_std": -1.7801748803947248e-06}, {"longitude": 127.86607166666667, "longitude_std": -1.992112444781924e-06}, {"northings": -527.4487693247576, "northings_std": 0.19816816183128352}, {"eastings": -197.19537408743128, "eastings_std": 0.19816816183128352}, {"depth": 28.8}]}
                    {"epoch_timestamp": 1501983409.56, "class": "measurement", "sensor": "unagi", "frame": "body", "category": "image", "camera1": [{"epoch_timestamp": 1501983409.56, "filename": "PR_20170816_023649_560_LC16.tif"}], "camera2": [{"epoch_timestamp": 1501983409.56, "filename": "PR_20170816_023649_560_RC16.tif"}]}
                    ]

            These are stored in a mirrored file location where the input raw data is stored as follows with the paths to raw data as defined in mission.yaml
            
            e.g. 
                raw     /<YEAR> /<CRUISE>   /<DIVE> /mission.yaml
                                                    /nav/gaps/
                                                    /nav/phins/
                                                    /image/r20170816_023028_UG069_sesoko/i20170816_023028/

            For this example, the outputs would be stored in the follow location, where folders will be automatically generated

            # for oplab
                processed   /<YEAR> /<CRUISE>   /<DIVE> /nav            /nav_standard.json   
            
            # for acfr
                processed   /<YEAR> /<CRUISE>   /<DIVE> /dRAWLOGS_cv    /combined.RAW.auv   
                                                        /mission.cfg

            An example dataset can be downloaded from the following link with the expected folder structure

                https://drive.google.com/drive/folders/0BzYMMCBxpT8BUF9feFpEclBzV0k?usp=sharing

            Download, extract and specify the folder location and run as
                
                python3 auv_nav.py -i ~/raw/2017/cruise/dive/ -o acfr
                python3 auv_nav.py -i ~/raw/2017/cruise/dive/ -o oplab

            The coordinate frames used are those defined in Thor Fossen Guidance, Navigation and Control of Ocean Vehicles
            
            i.e. Body frame:
                    x-direction: +ve aft to fore
                    y-direction: +ve port to starboard
                    z-direction: +ve bottom to top
            i.e. inertial frame:
                    north-direction: +ve north
                    east-direction: +ve east
                    down-direction: +ve depth downwards

            Parameter naming conventions
                    long and descriptive names should be used with all lower case letters. 

    """

# Import librarys
import sys, os, csv
import yaml, json
import shutil, math
import calendar

import matplotlib.pyplot as plt

from datetime import datetime
from lib_calculus.interpolate import interpolate
from lib_localisation.dead_reckoning import dead_reckoning
from lib_coordinates.body_to_inertial import body_to_inertial
from lib_sensors.parse_phins import parse_phins
from lib_sensors.parse_gaps import parse_gaps
from lib_sensors.parse_acfr_images import parse_acfr_images
from lib_sensors.parse_interlacer import parse_interlacer

def parse_data(filepath,ftype):


    # initiate data and processing flags
    
    
    proc_flag = 0
    
    origin_flag=0
    velocity_flag=0
    orientation_flag=0
    depth_flag =0
    attitude_flag =0
    usbl_flag =0
    image_flag =0    


    print('Loading mission.yaml')    
    mission = filepath+'mission.yaml'
    with open(mission,'r') as stream:
        load_data = yaml.load(stream)
    
    
    for i in range(0,len(load_data)): 
        if 'origin' in load_data:
            origin_flag=1
            latitude_reference = load_data['origin']['latitude']
            longitude_reference = load_data['origin']['longitude']
            coordinate_reference = load_data['origin']['coordinate_reference_system']
            date = load_data['origin']['date']

        if 'velocity' in load_data:
            velocity_flag=1                    
            velocity_format = load_data['velocity']['format']
            velocity_filepath = load_data['velocity']['filepath']
            velocity_filename = load_data['velocity']['filename']
            velocity_timezone = load_data['velocity']['timezone']
            velocity_timeoffset = load_data['velocity']['timeoffset']
            velocity_headingoffset = load_data['velocity']['headingoffset']

        if 'orientation' in load_data:
            orientation_flag=1                    
            orientation_format = load_data['orientation']['format']
            orientation_filepath = load_data['orientation']['filepath']
            orientation_filename = load_data['orientation']['filename']
            time_orientationzone = load_data['orientation']['timezone']
            time_orientationoffset = load_data['orientation']['timeoffset']
            orientation_headingoffset = load_data['orientation']['headingoffset']
    
        if 'depth' in load_data:
            depth_flag=1                    
            depth_format = load_data['depth']['format']
            depth_filepath = load_data['depth']['filepath']
            depth_filename = load_data['depth']['filename']
            time_depthzone = load_data['depth']['timezone']
            time_depthoffset = load_data['depth']['timeoffset']
    
        if 'altitude' in load_data:
            altitude_flag=1                    
            altitude_format = load_data['altitude']['format']
            altitude_filepath = load_data['altitude']['filepath']
            altitude_filename = load_data['altitude']['filename']
            time_altitudezone = load_data['altitude']['timezone']
            time_altitudeoffset = load_data['altitude']['timeoffset']

        if 'usbl' in load_data:
            usbl_flag=1                    
            usbl_format = load_data['usbl']['format']
            usbl_filepath = load_data['usbl']['filepath']
            time_usblzone = load_data['usbl']['timezone']
            time_usbloffset = load_data['usbl']['timeoffset']

        if 'image' in load_data:
            image_flag=1                    
            image_format = load_data['image']['format']
            image_filepath = load_data['image']['filepath']
            camera1_label = load_data['image']['camera1']
            camera2_label = load_data['image']['camera2']
            image_timezone = load_data['image']['timezone']
            image_timeoffset = load_data['image']['timeoffset']


    # generate output path
    print('Generating output paths')    
    sub_path = filepath.split(os.sep)        
    sub_out=sub_path
    outpath=sub_out[0]

    for i in range(1,len(sub_path)):
        if sub_path[i]=='raw':
            sub_out[i] = 'processed'
            proc_flag = 1
        else:
            sub_out[i] = sub_path[i]
        
        outpath = outpath + os.sep + sub_out[i]
        # make the new directories after 'processed' if it doesnt already exist
        if proc_flag == 1:        
            if os.path.isdir(outpath) == 0:
                try:
                    os.mkdir(outpath)
                except Exception as e:
                    print("Warning:",e)
                    
    # check for recognised formats and create nav file
    print('Checking output format')

    if ftype == 'oplab':# or (ftype is not 'acfr'):
        shutil.copy2(mission, outpath) # save mission yaml to processed directory
        outpath = outpath + 'nav'
        filename='nav_standard.json'   
        
        proc_flag=2
    
    elif ftype == 'acfr':# or (ftype is not 'acfr'):        
        with open(outpath + os.sep + 'mission.cfg','w') as fileout:
            data = 'MAG_VAR_LAT ' + str(float(latitude_reference)) + '\n' + 'MAG_VAR_LNG ' + str(float(longitude_reference)) + '\n' + 'MAG_VAR_DATE "' + str(date) + '"\n' + 'MAGNETIC_VAR_DEG ' + str(float(0))
            
            fileout.write(data)
            fileout.close()
                       
        outpath = outpath +'dRAWLOGS_cv'
        filename='combined.RAW.auv'
        proc_flag=2    

    else:
        print('Error: -o',ftype,'not recognised')
        syntax_error()    
    
    # check if output specified is recognised and make file path if not exist
    if proc_flag == 2:        
        if os.path.isdir(outpath) == 0:
            try:
                os.mkdir(outpath)
            except Exception as e:
                print("Warning:",e)

    
        # create file (overwrite if exists)
        with open(outpath + os.sep + filename,'w') as fileout:
            print('Loading raw data')
            # read in, parse data and write data
            if velocity_flag == 1:
                if velocity_format == "phins":
                    parse_phins(filepath + velocity_filepath,velocity_filename,'velocity',velocity_timezone,velocity_timeoffset,velocity_headingoffset,ftype,outpath,filename,fileout)
                    velocity_flag = 0
            if orientation_flag == 1:                
                if orientation_format == "phins":
                    parse_phins(filepath + orientation_filepath,orientation_filename,'orientation',time_orientationzone,time_orientationoffset,orientation_headingoffset,ftype,outpath,filename,fileout)
                    orientation_flag = 0
            if depth_flag == 1:                
                if depth_format == "phins":
                    parse_phins(filepath + depth_filepath,depth_filename,'depth',time_depthzone,time_depthoffset,0,ftype,outpath,filename,fileout)
                    depth_flag = 0
            if altitude_flag == 1:                
                if altitude_format == "phins":
                    parse_phins(filepath + altitude_filepath,altitude_filename,'altitude',time_altitudezone,0,time_altitudeoffset,ftype,outpath,filename,fileout)
                    altitude_flag = 0
            if usbl_flag == 1: # to implement
                if usbl_format == "gaps":
                    parse_gaps(filepath + usbl_filepath,'usbl',time_usblzone,time_usbloffset,latitude_reference,longitude_reference,ftype,outpath,filename,fileout)
                    usbl_flag = 0
            if image_flag == 1: # to implement
                if image_format == "acfr_standard" or image_format == "unagi" :
                    parse_acfr_images(filepath + image_filepath,image_format,camera1_label,camera2_label,'images',image_timezone,image_timeoffset,ftype,outpath,filename,fileout)
                    image_flag = 0

        fileout.close()
        
        #interlace the data based on timestamps
        print('Interlacing data')
        parse_interlacer(ftype,outpath,filename)
        print('Output saved to ' + outpath + os.sep + filename)

        print('Complete parse data')


def extract_data(filepath,ftype,start_time,finish_time,plot):

        interpolate_remove_flag = False

         # load data should at this point be able to specify time stamp range (see asv_nav)
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
                        time_depth.append(parsed_json_data[i]['epoch_timestamp'])
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
                        
            

            # perform coordinate transformations and interpolations                
            for i in range(len(time_velocity_body)):        
                
                # interpolate to find the appropriate orientation timefor the dvl measurements
                j=0               
                while time_orientation[j]<time_velocity_body[i]:
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


                # interpolate to find the appropriate depth to compute seafloor depth for each altitude measurement
                j=0
                while time_depth[j]<time_altitude[i]:
                    j=j+1

                if j>=1:                
                    #seafloor_depth[i]=(depth[j]-depth[j-1])/(time_depth[j]-time_depth[j-1])*(time_altitude[i]-time_depth[j-1])+depth[j-1]+altitude[i]  
                    seafloor_depth[i]=interpolate(time_altitude[i],time_depth[j-1],time_depth[j],depth[j-1],depth[j])+altitude[i]

                # interpolate to find the appropriate depth for dead_reckoning
                j=0
                while time_depth[j]<time_velocity_body[i]:
                    j=j+1

                if j>=1:                
                    #seafloor_depth[i]=(depth[j]-depth[j-1])/(time_depth[j]-time_depth[j-1])*(time_altitude[i]-time_depth[j-1])+depth[j-1]+altitude[i]  
                    depth_dead_reckoning[i]=interpolate(time_velocity_body[i],time_depth[j-1],time_depth[j],depth[j-1],depth[j])+altitude[i]

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

            #interpolate state data to inertia time stamps
            for i in range(len(time_velocity_inertia)):  
                j=0               
                while time_orientation[j]<time_velocity_inertia[i]:
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
                         
                j=0
                while time_depth[j]<time_velocity_inertia[i]:
                    j=j+1

                if j>=1:                
                    depth_inertia_dead_reckoning[i]=interpolate(time_velocity_inertia[i],time_depth[j-1],time_depth[j],depth[j-1],depth[j])
            

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

            # write values out to a csv file
            # create a directory with the time stamp
            renavpath = filepath + 'json_renav_' + str(yyyy) + str(mm) + str(dd) + '_' + start_time + '_' + finish_time 
            if os.path.isdir(renavpath) == 0:
                try:
                    os.mkdir(renavpath)
                except Exception as e:
                    print("Warning:",e)

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
                print('Complete plot data: ', plotpath)
                

        print('Complete extract data: ', csvpath)



    

def syntax_error():
# incorrect usage message
    print("     auv_nav.py <options>")
    print("         -i <path to mission.yaml>")
    print("         -o <output type> 'acfr' or 'oplab'")
    print("         -e <path to root processed folder where parsed data exists>")    
    print("         -s <start time in utc time> hhmmss (only for extract)")
    print("         -f <finish time in utc time> hhmmss (only for extract)")
    print("         -p <plot option> (only for extract)")
    return -1

    

if __name__ == '__main__':


    # initialise flags
    flag_i=0
    flag_o=0
    flag_e=0

    start_time='000000'
    finish_time='235959'
    plot = False

    
    # read in filepath and ftype
    if (int((len(sys.argv)))) < 5:
        print('Error: not enough arguments')
        syntax_error()
    else:   
        # read in filepath, start time and finish time from function call
        for i in range(math.ceil(len(sys.argv))):

            option=sys.argv[i]
        
            if option == "-i":
                filepath=sys.argv[i+1]
                flag_i=1
            if option == "-e":
                filepath=sys.argv[i+1]
                flag_e=1
            if option == "-o":
                ftype=sys.argv[i+1]
                flag_o=1                    
            if option == "-s":
                start_time=sys.argv[i+1]
            if option == "-f":
                finish_time=sys.argv[i+1]
            if option == "-p":
                plot = True

        

        if (flag_i ==1) and (flag_o ==1):
            sys.exit(parse_data(filepath,ftype))   
        if (flag_e ==1) and (flag_o ==1):
            sys.exit(extract_data(filepath,ftype,start_time,finish_time,plot))   
        else:
            syntax_error()
            
        
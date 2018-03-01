# auv_nav

# Scripts to parse and interleave raw sensor data and output in acfr or oplab formats
 
# Author: Blair Thornton
# Date: 25/08/2017

"""Parsers for navigation data for oplab standard and acfr standard formats

        inputs are 

        auv_nav.py <options>
            -i <path to mission.yaml>
            -o <output type> 'acfr' or 'oplab'
            -v <path to root processed folder where parsed data exists> to generate brief visualization summaries for json file data
            -e <path to root processed folder where parsed data exists> to extract useful info from json file data
            -start <start time in utc time> hhmmss (only for extract)
            -finish <finish time in utc time> hhmmss (only for extract)                      
            -plot <plot option> (only for extract)
            -csv <csv write option> (only for extract)
            -showplot <showplot option> (only for extract)

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
                    usbl_FIX: 1444452883 ship_x: 402.988947 ship_y: 140.275056 target_x: 275.337171 target_y: 304.388346 target_z: 299.2 target_hr: 0 target_sr: 364.347071 target_bearing: 127.876747

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
                    z-direction: +ve top to bottom
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
import calendar, codecs
import operator
#import hashlib, glob


#import matplotlib.pyplot as plt

from datetime import datetime
from lib_calculus.interpolate import interpolate
from lib_localisation.dead_reckoning import dead_reckoning
from lib_coordinates.body_to_inertial import body_to_inertial
from lib_extract.extract_data import extract_data
from lib_visualise.display_info import display_info
# from lib_visualise.parse_gaps import parse_gaps
from lib_sensors.parse_phins import parse_phins
from lib_sensors.parse_ae2000 import parse_ae2000
from lib_sensors.parse_gaps import parse_gaps
from lib_sensors.parse_usbl_dump import parse_usbl_dump
from lib_sensors.parse_acfr_images import parse_acfr_images
from lib_sensors.parse_seaxerocks_images import parse_seaxerocks_images
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
            if usbl_format == 'usbl_dump':
                usbl_filename = load_data['usbl']['filename']
                usbl_label = load_data['usbl']['label']

        if 'image' in load_data:
            image_flag=1                    
            image_format = load_data['image']['format']
            image_filepath = load_data['image']['filepath']
            camera1_label = load_data['image']['camera1']
            camera2_label = load_data['image']['camera2']
            if image_format == 'seaxerocks_3':
                camera3_label = load_data['image']['camera3']
            image_timezone = load_data['image']['timezone']
            image_timeoffset = load_data['image']['timeoffset']

    print('Loading vehicle.yaml')    
    vehicle = filepath+'vehicle.yaml'
    with open(vehicle,'r') as stream:
        load_data = yaml.load(stream)
    
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
        shutil.copy2(vehicle, outpath) # save mission yaml to processed directory
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
            if image_flag == 1:
                if image_format == "acfr_standard" or image_format == "unagi" :
                    parse_acfr_images(filepath + image_filepath,image_format,camera1_label,camera2_label,'images',image_timezone,image_timeoffset,ftype,outpath,filename,fileout)
                if image_format == "seaxerocks_3":
                    parse_seaxerocks_images(filepath + image_filepath,image_format,date,camera1_label,camera2_label,camera3_label,'images',image_timezone,image_timeoffset,ftype,outpath,filename,fileout)
                image_flag = 0

            if usbl_flag == 1:
                print('... loading usbl')
                if usbl_format == "gaps":                
                    parse_gaps(filepath + usbl_filepath,'usbl',time_usblzone,time_usbloffset,latitude_reference,longitude_reference,ftype,outpath,filename,fileout)                
                if usbl_format == "usbl_dump":                    
                    parse_usbl_dump(filepath + usbl_filepath,usbl_filename,usbl_label,'usbl',time_usblzone,time_usbloffset,latitude_reference,longitude_reference,ftype,outpath,filename,fileout)                
                usbl_flag = 0

            if velocity_flag == 1:
                print('... loading velocity')
                if velocity_format == "phins":                    
                    parse_phins(filepath + velocity_filepath,velocity_filename,'velocity',velocity_timezone,velocity_timeoffset,velocity_headingoffset,ftype,outpath,filename,fileout)
                if velocity_format == "ae2000":                    
                    parse_ae2000(filepath + velocity_filepath,velocity_filename,'velocity',velocity_timezone,velocity_timeoffset,velocity_headingoffset,ftype,outpath,filename,fileout)
                velocity_flag = 0

            if orientation_flag == 1:                
                print('... loading orientation')
                if orientation_format == "phins":                    
                    parse_phins(filepath + orientation_filepath,orientation_filename,'orientation',time_orientationzone,time_orientationoffset,orientation_headingoffset,ftype,outpath,filename,fileout)
                if orientation_format == "ae2000":                    
                    parse_ae2000(filepath + orientation_filepath,orientation_filename,'orientation',time_orientationzone,time_orientationoffset,orientation_headingoffset,ftype,outpath,filename,fileout)
                orientation_flag = 0

            if depth_flag == 1:                                
                print('... loading depth')
                if depth_format == "phins":    
                    parse_phins(filepath + depth_filepath,depth_filename,'depth',time_depthzone,time_depthoffset,0,ftype,outpath,filename,fileout)
                if depth_format == "ae2000":    
                    parse_ae2000(filepath + depth_filepath,depth_filename,'depth',time_depthzone,time_depthoffset,0,ftype,outpath,filename,fileout)
                depth_flag = 0

            if altitude_flag == 1:                
                print('... loading altitude')
                if altitude_format == "phins":                    
                    parse_phins(filepath + altitude_filepath,altitude_filename,'altitude',time_altitudezone,0,time_altitudeoffset,ftype,outpath,filename,fileout)
                if altitude_format == "ae2000":                    
                    parse_ae2000(filepath + altitude_filepath,altitude_filename,'altitude',time_altitudezone,0,time_altitudeoffset,ftype,outpath,filename,fileout)
                altitude_flag = 0
    
        fileout.close()
        
        #interlace the data based on timestamps
        print('Interlacing data')
        parse_interlacer(ftype,outpath,filename)
        print('Output saved to ' + outpath + os.sep + filename)

        print('Complete parse data')



    

def syntax_error():
# incorrect usage message
    print("     auv_nav.py <options>")
    print("         -i <path to mission.yaml>")
    print("         -o <output type> 'acfr' or 'oplab'")
    print("         -v <path to root processed folder where parsed data exists> to generate brief visualization summaries for json file data")
    print("         -e <path to root processed folder where parsed data exists> to extract useful info from json file data")    
    print("         -start <start time in utc time> hhmmss (only for extract)")
    print("         -finish <finish time in utc time> hhmmss (only for extract)")
    print("         -plot <plot option> (only for extract)")
    print("         -csv <csv write option> (only for extract)")
    print("         -showplot <showplot option> (only for extract)")
    
    return -1

    

if __name__ == '__main__':


    # initialise flags
    flag_i=False
    flag_o=False
    flag_v=False
    flag_e=False
    
    flag_f=False

    start_time='000000'
    finish_time='235959'
    plot = False
    csv_write = False
    show_plot = False
    
    # read in filepath and ftype
    if (int((len(sys.argv)))) < 3:
        print('Error: not enough arguments')
        syntax_error()
    else:   
        # read in filepath, start time and finish time from function call
        for i in range(math.ceil(len(sys.argv))):

            option=sys.argv[i]
        
            if option == "-i":
                filepath=sys.argv[i+1]
                flag_i=True
            elif option == "-o":
                ftype=sys.argv[i+1]
                flag_o=True    
            elif option == "-v":
                filepath=sys.argv[i+1]
                flag_v=True
            elif option == "-e":
                filepath=sys.argv[i+1]
                flag_e=True
            elif option == "-start":
                start_time=sys.argv[i+1]
            elif option == "-finish":
                finish_time=sys.argv[i+1]
            elif option == "-plot":
                plot=True
            elif option == "-csv":
                csv_write=True
            elif option == "-showplot":
                show_plot=True
            elif option[0] == '-':
                print('Error: incorrect use')
                sys.exit(syntax_error())

        if (flag_o ==False):
            print('No ouput option selected, default "oplab", -o "acfr" for acfr_standard')
            ftype='oplab'

        if flag_i ==True:
            sub_path = filepath.split(os.sep)        
            for i in range(1,len(sub_path)):
                if sub_path[i]=='raw':
                    flag_f = True
            if flag_f == True:
                parse_data(filepath,ftype)
            else:
                print('Check folder structure contains "raw"')

        elif flag_v == True:
            sub_path = filepath.split(os.sep)        
            for i in range(1,len(sub_path)):
                if sub_path[i]=='processed':
                    flag_f = True
            if flag_f == True:
                display_info(filepath,ftype)
            else:
                print('Check folder structure contains "processed"')

        elif flag_e==True:
            sub_path = filepath.split(os.sep)        
            for i in range(1,len(sub_path)):
                if sub_path[i]=='processed':
                    flag_f = True
            if (flag_f ==True):
                if (csv_write == False) and (plot == False):
                    print('No extract option selected, default plot (-p) enabled but without csv_write -> type (-csv) to enable')
                    plot = True
                extract_data(filepath,ftype,start_time,finish_time,plot,csv_write,show_plot)
            else:
                print('Check folder structure contains "processed"')                                    

        else:
            print('Error: incorrect use')
            syntax_error()
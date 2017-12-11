# auv_nav

# Scripts to parse and interleave raw sensor data and output in acfr or oplab formats
 
# Author: Blair Thornton
# Date: 25/08/2017

"""Parsers for navigation data for oplab standard and acfr standard formats

        inputs are 

        auv_nav.py <options>
            -i <path to mission.yaml>
            -e <path to root processed folder where parsed data exists>
            -s <start time in utc time> hhmmss (only for extract)")
            -f <finish time in utc time> hhmmss (only for extract)")            
            -o <output type> 'acfr' or 'oplab'


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
                    filepath: nav/phins/
                    filename: 20170817_phins.txt
                    timezone: utc
                    timeoffset: 0.0

                orientation:
                    format: phins
                    filepath: nav/phins/
                    filename: 20170817_phins.txt
                    timezone: utc
                    timeoffset: 0.0

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
                    {"epoch_timestamp": 1502840568.204, "class": "measurement", "sensor": "gaps", "frame": "inertial", "category": "usbl", "data_ship": [{"latitude": 26.66935735000014, "longitude": 127.86623359499968}, {"northings": -526.0556603025898, "eastings": -181.08730736724087}, {"heading": 174.0588800058365}], "data_target": [{"latitude": 26.669344833333334, "latitude_std": -1.7801748803947248e-06}, {"longitude": 127.86607166666667, "longitude_std": -1.992112444781924e-06}, {"northings": -527.4487693247576, "northings_std": 0.19816816183128352}, {"eastings": -197.19537408743128, "eastings_std": 0.19816816183128352}, {"depth": 28.8}]},{"epoch_timestamp": 1501983409.56, "class": "measurement", "sensor": "unagi", "frame": "body", "category": "image", "camera1": [{"epoch_timestamp": 1501983409.56, "filename": "PR_20170816_023649_560_LC16.tif"}], "camera2": [{"epoch_timestamp": 1501983409.56, "filename": "PR_20170816_023649_560_RC16.tif"}]}
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
    """

# Import librarys
import sys, os, csv
import yaml, json
import shutil, math
import calendar

from datetime import datetime
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
            orientation_timezone = load_data['orientation']['timezone']
            orientation_timeoffset = load_data['orientation']['timeoffset']
            orientation_headingoffset = load_data['orientation']['headingoffset']
    
        if 'depth' in load_data:
            depth_flag=1                    
            depth_format = load_data['depth']['format']
            depth_filepath = load_data['depth']['filepath']
            depth_filename = load_data['depth']['filename']
            depth_timezone = load_data['depth']['timezone']
            depth_timeoffset = load_data['depth']['timeoffset']
    
        if 'altitude' in load_data:
            altitude_flag=1                    
            altitude_format = load_data['altitude']['format']
            altitude_filepath = load_data['altitude']['filepath']
            altitude_filename = load_data['altitude']['filename']
            altitude_timezone = load_data['altitude']['timezone']
            altitude_timeoffset = load_data['altitude']['timeoffset']

        if 'usbl' in load_data:
            usbl_flag=1                    
            usbl_format = load_data['usbl']['format']
            usbl_filepath = load_data['usbl']['filepath']
            usbl_timezone = load_data['usbl']['timezone']
            usbl_timeoffset = load_data['usbl']['timeoffset']

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
                    parse_phins(filepath + orientation_filepath,orientation_filename,'orientation',orientation_timezone,orientation_timeoffset,orientation_headingoffset,ftype,outpath,filename,fileout)
                    orientation_flag = 0
            if depth_flag == 1:                
                if depth_format == "phins":
                    parse_phins(filepath + depth_filepath,depth_filename,'depth',depth_timezone,depth_timeoffset,0,ftype,outpath,filename,fileout)
                    depth_flag = 0
            if altitude_flag == 1:                
                if altitude_format == "phins":
                    parse_phins(filepath + altitude_filepath,altitude_filename,'altitude',altitude_timezone,0,altitude_timeoffset,ftype,outpath,filename,fileout)
                    altitude_flag = 0
            if usbl_flag == 1: # to implement
                if usbl_format == "gaps":
                    parse_gaps(filepath + usbl_filepath,'usbl',usbl_timezone,usbl_timeoffset,latitude_reference,longitude_reference,ftype,outpath,filename,fileout)
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


def extract_data(filepath,ftype,start_time,finish_time):

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


            velocity_time=[]
            velocity_x=[]
            velocity_y=[]
            velocity_z=[]

            orientation_time=[]
            orientation_roll=[]
            orientation_pitch=[]
            orientation_yaw=[]

            # i here is the number of the data packet
            for i in range(len(parsed_json_data)):

                epoch_timestamp=parsed_json_data[i]['epoch_timestamp']

                if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:                                      

                    if 'velocity' in parsed_json_data[i]['category']:
                        velocity_time.append(parsed_json_data[i]['epoch_timestamp'])

                        velocity_x.append(parsed_json_data[i]['data'][0]['x_velocity'])
                        velocity_y.append(parsed_json_data[i]['data'][1]['y_velocity'])
                        velocity_z.append(parsed_json_data[i]['data'][2]['z_velocity'])
                    
                    if 'orientation' in parsed_json_data[i]['category']:
                        orientation_time.append(parsed_json_data[i]['epoch_timestamp'])
                        orientation_roll.append(parsed_json_data[i]['data'][1]['roll'])
                        orientation_pitch.append(parsed_json_data[i]['data'][2]['pitch'])
                        orientation_yaw.append(parsed_json_data[i]['data'][0]['heading'])
            
            
            # write values out to a csv file
            # create a directory with the time stamp
            renavpath = filepath + 'json_renav_' + str(yyyy) + str(mm) + str(dd) + '_' + start_time + '_' + finish_time
            if os.path.isdir(renavpath) == 0:
                try:
                    os.mkdir(renavpath)
                except Exception as e:
                    print("Warning:",e)

            with open(renavpath + os.sep + 'velocity.csv' ,'w') as fileout:
               fileout.write('epoch_time, velocity_x, velocity_y, velocity_z \n')
            for i in range(len(velocity_time)):        
               with open(renavpath + os.sep + 'velocity.csv' ,'a') as fileout:
                   fileout.write(str(velocity_time[i])+','+str(velocity_x[i])+','+str(velocity_y[i])+','+str(velocity_z[i])+'\n')
                   fileout.close()

            with open(renavpath + os.sep + 'orientation.csv' ,'w') as fileout:
               fileout.write('epoch_time, roll, pitch, yaw \n')
            for i in range(len(orientation_time)):        
               with open(renavpath + os.sep + 'orientation.csv' ,'a') as fileout:
                   fileout.write(str(orientation_time[i])+','+str(orientation_roll[i])+','+str(orientation_pitch[i])+','+str(orientation_yaw[i])+'\n')
                   fileout.close()

                        # # perform coordinate transformation to calculate the 
            velocity_x_offset=[]
            velocity_y_offset=[]
            velocity_z_offset=[]    

            for i in range(len(velocity_time)):        
                #[x_offset,y_offset,z_offset] = body_to_inertial(orientation_roll[i], orientation_pitch[i], orientation_yaw[i]-45, velocity_x[i], velocity_y[i], velocity_z[i])
                [x_offset,y_offset,z_offset] = body_to_inertial(orientation_roll[i], orientation_pitch[i], orientation_yaw[i], velocity_x[i], velocity_y[i], velocity_z[i])
                velocity_x_offset.append(x_offset)
                velocity_y_offset.append(y_offset)
                velocity_z_offset.append(z_offset)

            with open(renavpath + os.sep + 'velocity_offset.csv' ,'w') as fileout:
               fileout.write('epoch_time, velocity_north, velocity_east, velocity_depth \n')
            for i in range(len(velocity_time)):        
               with open(renavpath + os.sep + 'velocity_offset.csv' ,'a') as fileout:
                   fileout.write(str(velocity_time[i])+','+str(velocity_x_offset[i])+','+str(velocity_y_offset[i])+','+str(velocity_z_offset[i])+'\n')
                   fileout.close()

        print('Complete extract data')



    

def syntax_error():
# incorrect usage message
    print("     auv_nav.py <options>")
    print("         -i <path to mission.yaml>")
    print("         -e <path to root processed folder where parsed data exists>")    
    print("         -o <output type> 'acfr' or 'oplab'")
    print("         -s <start time in utc time> hhmmss (only for extract)")
    print("         -f <finish time in utc time> hhmmss (only for extract)")
    return -1

    

if __name__ == '__main__':


    # initialise flags
    flag_i=0
    flag_o=0
    flag_e=0

    start_time='000000'
    finish_time='235959'

    
    # read in filepath and ftype
    if (int((len(sys.argv)))) < 5:
        print('Error: not enough arguments')
        syntax_error()
    else:   
        # read in filepath, start time and finish time from function call
        for i in range(math.ceil(len(sys.argv)/2)):

            option=option=sys.argv[2*i-1]
            value=sys.argv[2*i]                 

        
            if option == "-i":
                filepath=value
                flag_i=1
            if option == "-e":
                filepath=value
                flag_e=1
            if option == "-o":
                ftype=value
                flag_o=1        
            
            if option == "-s":
                start_time=value                
            if option == "-f":
                finish_time=value   

        

        if (flag_i ==1) and (flag_o ==1):
            sys.exit(parse_data(filepath,ftype))   
        if (flag_e ==1) and (flag_o ==1):
            sys.exit(extract_data(filepath,ftype,start_time,finish_time))   
        else:
            syntax_error()
            
        
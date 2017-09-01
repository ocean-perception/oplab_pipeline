# auv_nav

# Scripts to parse and interleave raw sensor data and output in acfr or oplab formats
 
# Author: Blair Thornton
# Date: 25/08/2017

# Import librarys
import sys, os, csv,json
from lib_sensors.parse_phins import parse_phins
from lib_sensors.parse_gaps import parse_gaps
from lib_sensors.parse_acfr_images import parse_acfr_images

def generate_paths(filepath,ftype):

    """Parsers for navigation data for oplab standard and acfr standard formats

        inputs are 

        nav_parser.py <options>
            -i <path to mission.cfg>
            -o <output type> 'acfr' or 'oplab'


        Arguments:
            path to the "mission.cfg" file, output format 'acfr' or 'oplab'

                origin {
                    latitude = "26.674083";
                    longitude = "127.868054";
                    coordinate_reference_system = "wgs84";
                }
                velocity {
                    format = "phins";
                    filepath = "nav/phins/";
                    filename = "20170816_phins.txt";
                    timezone = "utc";
                    timeoffset = "0.0";
                }
                orientation {
                    format = "phins";
                    filepath = "nav/phins/";
                    filename = "20170816_phins.txt";
                    timezone = "utc";
                    timeoffset = "0.0";
                }
                depth {
                    format = "phins";
                    filepath = "nav/phins/";
                    filename = "20170816_phins.txt";
                    timezone = "utc";
                    timeoffset = "0.0";
                }
                altitude {
                    format = "phins";
                    filepath = "nav/phins/";
                    filename = "20170816_phins.txt";
                    timezone = "utc";
                    timeoffset = "0.0";
                }
                usbl {
                    format = "gaps";
                    filepath = "nav/gaps/";
                    filename = "20170816091630-001.dat";
                    timezone = "utc";
                    timeoffset = "0.0";
                }
                images{
                    format = "acfr_standard";
                    filepath = "image/r20170816_023028_UG069_sesoko/i20170816_023028/";
                    timezone = "utc";
                    timeoffset = "0.0";
                }

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
                    [{"epoch_timestamp": 1501974125.926, "epoch_timestamp_dvl": 1501974125.875, "class": "measurement", "sensor": "phins", "frame": "body", "category": "velocity", "data": [{"xx_velocity": -0.075, "xx_velocity_std": 0.200075}, {"yy_velocity": 0.024, "yy_velocity_std": 0.200024}, {"zz_velocity": -0.316, "zz_velocity_std": 0.20031600000000002}]},
                    {"epoch_timestamp": 1501974002.1, "class": "measurement", "sensor": "phins", "frame": "inertial", "category": "orientation", "data": [{"heading": 243.777, "heading_std": 2.0}, {"roll": 4.595, "roll_std": 0.1}, {"pitch": 0.165, "pitch_std": 0.1}]},
                    {"epoch_timestamp": 1501974125.926, "epoch_timestamp_dvl": 1501974125.875, "class": "measurement", "sensor": "phins", "frame": "body", "category": "altitude", "data": [{"altitude": 31.53, "altitude_std": 0.3153}, {"sound_velocity": 1546.0, "sound_velocity_correction": 0.0}]},
                    {"epoch_timestamp": 1501974002.7, "epoch_timestamp_depth": 1501974002.674, "class": "measurement", "sensor": "phins", "frame": "inertial", "category": "depth", "data": [{"depth": -0.958, "depth_std": -9.58e-05}]},
                    {"epoch_timestamp": 1502840568.204, "class": "measurement", "sensor": "gaps", "frame": "inertial", "category": "usbl", "data_ship": [{"latitude": 26.66935735000014, "longitude": 127.86623359499968}, {"northings": -526.0556603025898, "eastings": -181.08730736724087}, {"heading": 174.0588800058365}], "data_target": [{"latitude": 26.669344833333334, "latitude_std": -1.7801748803947248e-06}, {"longitude": 127.86607166666667, "longitude_std": -1.992112444781924e-06}, {"northings": -527.4487693247576, "northings_std": 0.19816816183128352}, {"eastings": -197.19537408743128, "eastings_std": 0.19816816183128352}, {"depth": 28.8}]},{"epoch_timestamp": 1501983409.56, "class": "measurement", "sensor": "unagi", "frame": "body", "category": "image", "camera1": [{"epoch_timestamp": 1501983409.56, "filename": "PR_20170816_023649_560_LC16.tif"}], "camera2": [{"epoch_timestamp": 1501983409.56, "filename": "PR_20170816_023649_560_RC16.tif"}]}
                    ]
    """
    # initiate data and processing flags
    
    
    proc_flag = 0
    
    origin_flag=0
    velocity_flag=0
    orientation_flag=0
    depth_flag =0
    attitude_flag =0
    usbl_flag =0
    images_flag =0

    # read in the mission.cfg
    print('Loading mission.cfg')    
    mission = filepath+'mission.cfg'
    load_data = open(mission,'r').read().split('}')

    # handling cfg to find data    
    for i in range(0,len(load_data)-1): 
        load_data_sub = load_data[i].split('{')
        if  load_data_sub[0].startswith('\n'):
            temp=load_data_sub[0].split('\n')
            header=temp[1]
        else:
            header=load_data_sub[0]

        configuration=load_data_sub[1]

        heading = header.split(' ')
        configuration_sub = (configuration.split(';'))    

        for i in range(0,len(configuration_sub)-1):            
            # isolate just the label
            value=str(configuration_sub[i]).split('=')
            value_sub=str(value[0]).split('\n\t')   
            label=(value_sub[len(value_sub)-1].split(' '))

            # isolate just the value              
            value_sub=((value[1]).split('"'))        
            setting=value_sub[len(value_sub)-2]

            if heading[0] == "origin":
                origin_flag=1
                if label[0] == "latitude":
                    latitude_reference = float(setting)
                elif label[0] == "longitude":
                    longitude_reference = float(setting)
                elif label[0] == "coordinate_reference_system":
                    coordinate_reference = str(setting)
            
            if heading[0] == "velocity":
                velocity_flag=1                    
                if label[0] == "format":                    
                    velocity_format = str(setting)
                if label[0] == "filepath":
                    velocity_filepath = str(setting)
                if label[0] == "filename":
                     velocity_filename = str(setting)
                if label[0] == "timezone":
                    velocity_timezone = str(setting)
                if label[0] == "timeoffset":
                    velocity_timeoffset = float(setting)
                
            if heading[0] == "orientation":
                orientation_flag=1
                if label[0] == "format":
                    orientation_format = str(setting)
                if label[0] == "filepath":
                    orientation_filepath = str(setting)
                if label[0] == "filename":
                    orientation_filename = str(setting)
                if label[0] == "timezone":
                    orientation_timezone = str(setting)
                if label[0] == "timeoffset":
                    orientation_timeoffset = float(setting)

            if heading[0] == "depth":
                depth_flag=1
                if label[0] == "format":
                    depth_format = str(setting)
                if label[0] == "filepath":
                    depth_filepath = str(setting)
                if label[0] == "filename":
                    depth_filename = str(setting)
                if label[0] == "timezone":
                    depth_timezone = str(setting)
                if label[0] == "timeoffset":
                    depth_timeoffset = float(setting)

            if heading[0] == "altitude":
                altitude_flag=1
                if label[0] == "format":
                    altitude_format = str(setting)
                if label[0] == "filepath":
                    altitude_filepath = str(setting)
                if label[0] == "filename":
                    altitude_filename = str(setting)
                if label[0] == "timezone":
                    altitude_timezone = str(setting)
                if label[0] == "timeoffset":
                    altitude_timeoffset = float(setting)


            if heading[0] == "usbl":
                usbl_flag=1                
                if label[0] == "format":
                    usbl_format = str(setting)
                if label[0] == "filepath":
                    usbl_filepath = str(setting)
                if label[0] == "filename":
                    usbl_filename = str(setting)
                if label[0] == "timezone":
                    usbl_timezone = str(setting)
                if label[0] == "timeoffset":
                    usbl_timeoffset = float(setting)

            if heading[0] == "images":
                image_flag=1;
                if label[0] == "format":
                    image_format = str(setting)
                if label[0] == "filepath":
                    image_filepath = str(setting)                
                if label[0] == "camera1":
                    camera1_label = str(setting)                
                if label[0] == "camera2":
                    camera2_label = str(setting)                                    
                if label[0] == "timezone":
                    image_timezone = str(setting)
                if label[0] == "timeoffset":
                    image_timeoffset = float(setting)


    # generate output path
    print('Generating output paths')    
    sub_path = filepath.split('/')        
    sub_out=sub_path
    outpath=sub_out[0]

    for i in range(1,len(sub_path)):
        if sub_path[i]=='raw':
            sub_out[i] = 'processed'
            proc_flag = 1
        else:
            sub_out[i] = sub_path[i]
        
        outpath = outpath +'/' + sub_out[i]
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
        outpath = outpath + 'nav'
        filename='nav_standard.json'        
        proc_flag=2
    
    elif ftype == 'acfr':# or (ftype is not 'acfr'):        
        outpath = outpath +'/dRAWLOGS_cv'
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
        with open(outpath + '/' + filename,'w') as fileout:
            # read in, parse data and write data
            if velocity_flag == 1:                
                if velocity_format == "phins":
                    parse_phins(filepath + velocity_filepath,velocity_filename,'velocity',velocity_timezone,velocity_timeoffset,ftype,outpath,filename,fileout)
            if orientation_flag == 1:                
                if orientation_format == "phins":
                    parse_phins(filepath + orientation_filepath,orientation_filename,'orientation',orientation_timezone,orientation_timeoffset,ftype,outpath,filename,fileout)
            if depth_flag == 1:                
                if depth_format == "phins":
                    parse_phins(filepath + depth_filepath,depth_filename,'depth',depth_timezone,depth_timeoffset,ftype,outpath,filename,fileout)
            if altitude_flag == 1:                
                if altitude_format == "phins":
                    parse_phins(filepath + altitude_filepath,altitude_filename,'altitude',altitude_timezone,altitude_timeoffset,ftype,outpath,filename,fileout)
            if usbl_flag == 1: # to implement
                if usbl_format == "gaps":
                    parse_gaps(filepath + usbl_filepath,usbl_filename,'usbl',usbl_timezone,usbl_timeoffset,latitude_reference,longitude_reference,ftype,outpath,filename,fileout)
            if image_flag == 1: # to implement
                if image_format == "acfr_standard" or image_format == "unagi" :
                    parse_acfr_images(filepath + image_filepath,image_format,camera1_label,camera2_label,'images',image_timezone,image_timeoffset,ftype,outpath,filename,fileout)

        fileout.close()
        
        #interlace the data based on timestamps



    #     print(velocity_format)
    #     velocity_filepath = str(setting)
    #     velocity_filename = str(setting)
    #     velocity_timezone = str(setting)
    #     velocity_timezone = float(setting)    




    # arrange in time







    

    # return outpath
    

def syntax_error():
# incorrect usage message
    print("     nav_parser.py <options>")
    print("         -i <path to mission.cfg>")
    print("         -o <output type> 'acfr' or 'oplab'")
    return -1

    

if __name__ == '__main__':


    # initialise flags
    flag_i=0
    flag_o=0
    
    # read in filepath and ftype
    if (int((len(sys.argv)))) < 5:
        print('Error: not enough arguments')
        syntax_error()
    else:   

        option=sys.argv[1]
        value=sys.argv[2]

        if option == "-i":
            filepath=value
            flag_i=1
        elif option == "-o":
            ftype=value
            flag_o=1

        option=sys.argv[3]
        value=sys.argv[4]

        if option == "-i":
            filepath=value
            flag_i=1
        elif option == "-o":
            ftype=value
            flag_o=1

        if (flag_i ==1) and (flag_o ==1):
            sys.exit(generate_paths(filepath,ftype))            
        else:
            syntax_error()
            
        
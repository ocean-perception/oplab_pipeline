# auv_nav

# Scripts to parse and interleave raw sensor data and output in acfr or oplab formats
 
# Author: Blair Thornton
# Date: 25/08/2017

# Import librarys
import sys, os, csv
from lib_sensors.parse_phins import parse_phins

def generate_paths(filepath,ftype):

    """Read data (navigation and image) according to config.

    Arguments:
        path to the "mission.cfg" file, output format 'acfr' or 'oplab'

    Returns:
        interleaved navigation and imaging data with output options:
            'acfr' - combined.auv.raw
            'oplab' - nav_standard.dat

            'nav_standard.dat'
            
            <Time>,<Class>,<Sensor >,<category>,<mean>,<uncertainty> 
            
            <Time> epoch time
            <Class> is either a measurement or an EIF, EKF
            <Measurment> is one of the following
            <Measurment>
            Category is one of the 3 things the measurement could be

            <vehicle_angular_velocity> roll_velocity pitch_velocity yaw_velocity
            <vehicle_velocity> xx_velocity yy_velocity zz_velocity
            <earth_velocity> x_velocity y_velocity z_velocity

            <earth_orientation> roll pitch yaw
            <earth_position> x_position y_position depth
            
            Mean is the numerical value in SI units and degrees
            Uncertainty is from the catalogue

            Line may look like this 
            1501974061.453 , measurement , phins , body velocity , xx_velocity , 0.351 , 0.033
            1501974061.453 , measurement , phins , body velocity , yy_velocity , -0.114 , 0.038
            1501974061.453 , measurement , phins , body velocity , zz_velocity , 0.214 , 0.021

            
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
                elif label[0] == "longitude":
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
                images_flag=1;
                if label[0] == "format":
                    image_format = str(setting)
                if label[0] == "filepath":
                    image_filepath = str(setting)                
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
        filename='nav_standard.dat'    
        proc_flag=2
    elif ftype == 'acfr':# or (ftype is not 'acfr'):
        filename='combined.auv.raw'
        proc_flag=2    
    
    # check if output specified is recognised and make file path if not exist
    if proc_flag == 2:        
        outpath = outpath + 'nav'
        if os.path.isdir(outpath) == 0:
            try:
                os.mkdir(outpath)
            except Exception as e:
                print("Warning:",e)
    # create file (overwrite if exists)
        with open(outpath + '/' + filename,'w') as fileout:
            csvread = csv.reader(fileout)
    else:
        print('Error: -o',ftype,'not recognised')
        syntax_error()

    # read in and parse data    
    if velocity_flag == 1:                
        if velocity_format == "phins":            
            parse_phins(filepath + velocity_filepath,velocity_filename,'velocity',velocity_timezone,velocity_timeoffset,ftype,fileout)
            


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
            
        
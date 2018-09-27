# Import librarys
import sys, os, time
import math
from lib_parse_data.parse_data import parse_data
from lib_extract.extract_data import extract_data
from lib_visualise.display_info import display_info


def syntax_error():
# incorrect usage message
    print ("    auv_nav.py [arguments]")
    print ("        required arguments (select one of these at a time):")
    print ("            -i [path to root raw folder]")
    print ("            -v [path to root processed folder where parsed data exists]")
    print ("            -e [path to root processed folder where parsed data exists]")
    print ("        optional arguments:")
    print ("            -o [output format 'oplab' or 'acfr']")
    print ("            -start [start date & time in YYYYMMDDhhmmss] (only for extract)")
    print ("            -finish [finish date & time] YYYYMMDDhhmmss] (only for extract)")
    print ("            -plot <output pdf plots option> (only for extract)")
    print ("            -plotly <output interactive plots using plotly option> (only for extract)")
    print ("            -csv <output csv files option> (only for extract)")
    print ("            -DR <output dead reckoning files option> (only for extract)")
    print ("            -PF <perform particle filter option> (only for extract)")
    
    return -1

if __name__ == '__main__':

    start_time = time.time()
    # initialise flags
    flag_i=False # input path set flag
    flag_v=False # visualisation flag
    flag_e=False # extraction of information flag

    flag_o=False # output type set flag

    start_datetime=''
    finish_datetime=''
    
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
            elif option == "-v":
                filepath=sys.argv[i+1]
                flag_v=True
            elif option == "-e":
                filepath=sys.argv[i+1]
                flag_e=True
            elif option == "-o":
                ftype=sys.argv[i+1]
                flag_o=True
            elif option == "-start":
                start_datetime=sys.argv[i+1]
            elif option == "-finish":
                finish_datetime=sys.argv[i+1]
            elif option[0] == '-':
                print('Error: incorrect use')
                sys.exit(syntax_error())

        if flag_o ==False:
            print('No ouput option specified. Using the default output option "oplab"')
            ftype='oplab'

        if flag_i ==True:
            sub_path = filepath.split(os.sep)     
            flag_r=False # path is a subfolder of a folder called "raw"
            for i in range(1,len(sub_path)):
                if sub_path[i]=='raw':
                    flag_r = True
            if flag_r == True:
                parse_data(filepath,ftype)
            else:
                print('Check folder structure contains "raw"')

        elif flag_v == True:
            sub_path = filepath.split(os.sep)  
            flag_p=False # path is a subfolder of a folder called "processed"      
            for i in range(1,len(sub_path)):
                if sub_path[i]=='processed':
                    flag_p = True
            if flag_p == True:
                display_info(filepath + os.sep,ftype)
            else:
                print('Check folder structure contains "processed"')

        elif flag_e==True:
            sub_path = filepath.split(os.sep)  
            flag_p=False # path is a subfolder of a folder called "processed"      
            for i in range(1,len(sub_path)):
                if sub_path[i]=='processed':
                    flag_p = True
            if (flag_p ==True):
                extract_data(filepath + os.sep,ftype,start_datetime,finish_datetime)
            else:
                print('Check folder structure contains "processed"')

        else:
            print('Error: incorrect use')
            syntax_error()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken = {} mins".format(elapsed_time/60))
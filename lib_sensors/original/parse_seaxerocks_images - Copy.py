# parse_seaxerocks_images

# Scripts to parse sexerocks image acquisition data

# Author: Blair Thornton
# Date: 31/08/2017

import os, operator, sys
import codecs, time, json, glob
# from datetime import datetime

# sys.path.append("..")
from lib_converttime.converttime import date_time_to_epoch

#http://www.json.org/

class parse_seaxerocks_images:
    def __init__(self, filepath, sensor_string, date, camera1_label, camera2_label, camera3_label, category, timezone, timeoffset, ftype, outpath, fileoutname, fileout):

        # parser meta data
        class_string = 'measurement'
        frame_string = 'body'
        category_stereo = 'image'
        category_laser = 'laser'

        epoch_timestamp_stereo = []
        epoch_timestamp_laser = []
        epoch_timestamp_camera1 = []
        epoch_timestamp_camera2 = []
        epoch_timestamp_camera3 = []
        stereo_index=[]
        laser_index=[]
        values = []
        data_list = []
        camera1_index = []
        camera2_index = []
        camera3_index = []
        camera1_filename = []
        camera2_filename = []
        camera3_filename = []

        camera1_serial = list(camera1_label)
        camera2_serial = list(camera2_label)
        camera3_serial = list(camera3_label)

        for i in range(1,len(camera1_label)):
            if camera1_label[i]=='/':
                camera1_serial[i] = '_'

        for i in range(1,len(camera2_label)):
            if camera2_label[i]=='/':
                camera2_serial[i] = '_'

        for i in range(1,len(camera3_label)):
            if camera3_label[i]=='/':
                camera3_serial[i] = '_'        

        camera1_serial=''.join(camera1_serial)
        camera2_serial=''.join(camera2_serial)
        camera3_serial=''.join(camera3_serial)


        i=0
        # read in timezone
        if isinstance(timezone, str):           
            if timezone == 'utc' or  timezone == 'UTC':
                timezone_offset = 0
            elif timezone == 'jst' or  timezone == 'JST':
                timezone_offset = 9;
        else:
            try:
                timezone_offset=float(timezone)
            except ValueError:
                print('Error: timezone', timezone, 'in mission.cfg not recognised, please enter value from UTC in hours')
                return

        # convert to seconds from utc
        # timeoffset = -timezone_offset*60*60 + timeoffset 

        print('Parsing', sensor_string, 'images')
        
        # determine file paths
        camera1_path=camera1_label.split('/')

        with codecs.open(filepath+camera1_path[0]+'/'+'FileTime.csv','r',encoding='utf-8', errors='ignore') as filein:
            for line in filein.readlines():
                stereo_index_timestamps=line.strip().split(',') 

                index_string = stereo_index_timestamps[0]
                date_string = stereo_index_timestamps[1]
                time_string = stereo_index_timestamps[2]
                ms_time_string = stereo_index_timestamps[3]

                # read in date
                if date_string != 'date':#ignore header                 
                    stereo_index.append(index_string)                 
                    yyyy = int(date_string[0:4])
                    mm = int(date_string[4:6])
                    dd = int(date_string[6:8])

                    # read in time          
                    hour=int(time_string[0:2])
                    mins=int(time_string[2:4])
                    secs=int(time_string[4:6])
                    msec=int(ms_time_string[0:3])

                    epoch_time = date_time_to_epoch(yyyy,mm,dd,hour,mins,secs,timezone_offset)
                    # dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
                    # time_tuple = dt_obj.timetuple()
                    # epoch_time = time.mktime(time_tuple)
                    
                    epoch_timestamp_stereo.append(float(epoch_time+msec/1000+timeoffset))
                    
        camera1_list = ["{}.raw".format(i) for i in stereo_index]
        camera2_list = ["{}.raw".format(i) for i in stereo_index]
        # camera1_list = sorted(os.listdir(filepath+camera1_label))
        # camera2_list = sorted(os.listdir(filepath+camera2_label))
        
        # camera1_list = [ line for line in camera1_list if '.raw' in line]
        # camera2_list = [ line for line in camera2_list if '.raw' in line]
        
        for i in range(len(camera1_list)):
            camera1_image=camera1_list[i].split('.')
            camera2_image=camera2_list[i].split('.')

            camera1_index.append(camera1_image[0])
            camera2_index.append(camera2_image[0])

        j=0
        for i in range(len(camera1_list)):                      
            if camera1_index[i]==stereo_index[j]:   # find corresponding timestamp even if some images are deletec          
                epoch_timestamp_camera1.append(epoch_timestamp_stereo[j])
                epoch_timestamp_camera2.append(epoch_timestamp_stereo[j])
                if ftype == 'acfr':                 
                    camera1_filename.append('sx3_' + date[2:4] + date[5:7] + date[8:10] + '_image' + str(camera1_index[i]) + '_FC.png')
                    camera2_filename.append('sx3_' + date[2:4] + date[5:7] + date[8:10] + '_image' + str(camera2_index[i]) + '_AC.png')
                j=j+1
            elif stereo_index[j] > camera1_index[i]:
                j=j+1
            else:
                j=j-1
        
        if ftype == 'oplab':
            camera1_filename = [ line for line in camera1_list]
            camera2_filename = [ line for line in camera2_list]

        for i in range(len(camera1_list)):
            if ftype == 'acfr': 
                data = 'VIS: ' + str(float(epoch_timestamp_camera1[i])) + ' [' + str(float(epoch_timestamp_camera1[i])) + '] ' + str(camera1_filename[i]) + ' exp: 0\n'
                fileout.write(data)
                data = 'VIS: ' + str(float(epoch_timestamp_camera2[i])) + ' [' + str(float(epoch_timestamp_camera2[i])) + '] ' + str(camera2_filename[i]) + ' exp: 0\n'
                fileout.write(data)

            if ftype == 'oplab':                    
                data = {'epoch_timestamp': float(epoch_timestamp_camera1[i]), 'class': class_string, 'sensor': sensor_string, 'frame': frame_string, 'category': category_stereo, 'camera1': [{'epoch_timestamp': float(epoch_timestamp_camera1[i]), 'serial' : camera1_serial, 'filename': str(camera1_label+'/'+camera1_filename[i])}], 'camera2':  [{'epoch_timestamp': float(epoch_timestamp_camera2[i]), 'serial' : camera2_serial,  'filename': str(camera2_label+'/'+camera2_filename[i])}]}                
                data_list.append(data)
                

        with codecs.open(filepath+camera3_label+'/'+'FileTime.csv','r',encoding='utf-8', errors='ignore') as filein:
            for line in filein.readlines():
                laser_index_timestamps=line.strip().split(',')  

                index_string = laser_index_timestamps[0]
                date_string = laser_index_timestamps[1]
                time_string = laser_index_timestamps[2]
                ms_time_string = laser_index_timestamps[3]

                # read in date
                if date_string != 'date':#ignore header                 
                    laser_index.append(index_string)
                
                    yyyy = int(date_string[0:4])
                    mm = int(date_string[4:6])
                    dd = int(date_string[6:8])

                    # read in time          
                    hour=int(time_string[0:2])
                    mins=int(time_string[2:4])
                    secs=int(time_string[4:6])
                    msec=int(ms_time_string[0:3])

                    epoch_time = date_time_to_epoch(yyyy,mm,dd,hour,mins,secs,timezone_offset)
                    # dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
                    # time_tuple = dt_obj.timetuple()
                    # epoch_time = time.mktime(time_tuple)
                    
                    epoch_timestamp_laser.append(float(epoch_time+msec/1000+timeoffset))                    
        
        camera3_list = ["{}".format(i) for i in laser_index] # try use pandas for all parsers, should be faster
        # camera3_list = sorted(os.listdir(filepath+camera3_label))
        # camera3_list = [ line for line in camera3_list if '.' not in line] # do not consider any other files, only consider folder - assuming all the folders only contain intended images
        
        for i in range(len(camera3_list)):
            camera3_filename.append('{}/image{}.xxx'.format(camera3_list[i][:3], camera3_list[i])) # let out format (e.g. '.jpg' or '.tif')
            camera3_index.append(camera3_list[i])
        # for i in range(len(camera3_list)): 
        #     images_filenames=sorted(os.listdir(filepath+camera3_label+'/'+camera3_list[i]))
        #     for j in range(len(images_filenames)):
        #         camera3_filename.append(camera3_list[i]+'/'+images_filenames[j])
        #         camera3_image=images_filenames[j].split('.')
        #         camera3_image_index=camera3_image[0].split('image')
        #         camera3_index.append(camera3_image_index[1])

        j=0
        for i in range(len(camera3_filename)):# original comment: find corresponding timestamp even if some images are deletec
            if camera3_index[i]==laser_index[j]:                
                epoch_timestamp_camera3.append(epoch_timestamp_laser[j])
                j=j+1
            elif laser_index[j] > camera3_index[i]: # Jin: incomplete? it means that laser data is missing for this image file, so no epoch_timestamp data, and do what when this happens?
                j=j+1
            else:
                j=j-1 # Jin: incomplete and possibly wrong? it means that this laser data is extra, with no accompanying image file, so it should be j+1 till index match?

            if ftype == 'oplab':                    
                data = {'epoch_timestamp': float(epoch_timestamp_camera3[i]), 'class': class_string, 'sensor': sensor_string, 'frame': frame_string, 'category': category_laser,  'serial' : camera3_serial, 'filename': str(camera3_filename[i])}
                data_list.append(data)



        if ftype == 'oplab':
            fileout.close()
            for filein in glob.glob(outpath + os.sep + fileoutname):
                try:
                    with open(filein, 'rb') as json_file:                       
                        data_in=json.load(json_file)                        
                        for i in range(len(data_in)):
                            data_list.insert(0,data_in[len(data_in)-i-1])                       
                        
                except ValueError:                  
                    print('Initialising JSON file')

            with open(outpath + os.sep + fileoutname,'w') as fileout:
                json.dump(data_list, fileout)   
                del data_list
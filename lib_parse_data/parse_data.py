# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 12:34:42 2018

@author: Adrian
"""

import multiprocessing
import os
import shutil
import json
import yaml
from threading import Thread


from lib_sensors.parse_phins import parse_phins
from lib_sensors.parse_ae2000 import parse_ae2000
from lib_sensors.parse_gaps import parse_gaps
from lib_sensors.parse_usbl_dump import parse_usbl_dump
from lib_sensors.parse_acfr_images import parse_acfr_images
from lib_sensors.parse_seaxerocks_images import parse_seaxerocks_images
from lib_sensors.parse_interlacer import parse_interlacer
# from lib_sensors.parse_chemical import parse_chemical


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self):
        Thread.join(self)
        return self._return



def parse_data(filepath,ftype):

    # initiate data and processing flags
    proc_flag = 0
    
    origin_flag=0
    velocity_flag=0
    orientation_flag=0
    depth_flag =0
    altitude_flag =0
    usbl_flag =0
    image_flag =0
    chemical_flag = 0

    # load mission.yaml config file
    print('Loading mission.yaml')    
    mission = filepath + os.sep + 'mission.yaml'
    with open(mission,'r') as stream:
        load_data = yaml.load(stream)
    # for i in range(0,len(load_data)): 
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
        if usbl_format == 'gaps':
            usbl_id = load_data['usbl']['id']
    if 'image' in load_data:
        image_flag=1                    
        image_format = load_data['image']['format']
        image_filepath = load_data['image']['filepath']
        camera1_label = load_data['image']['camera1']
        camera2_label = load_data['image']['camera2']
        image_timezone = load_data['image']['timezone']
        image_timeoffset = load_data['image']['timeoffset']
        if image_format == 'seaxerocks_3':
            camera3_label = load_data['image']['camera3']
    # if 'chemical' in load_data:
    #     chemical_flag=1
    #     chemical_format = load_data['chemical']['format']
    #     chemical_filepath = load_data['chemical']['filepath']
    #     chemical_filename = load_data['chemical']['filename']
    #     chemical_timezone = load_data['chemical']['timezone']
    #     chemical_timeoffset = load_data['chemical']['timeoffset']
    #     chemical_data = load_data['chemical']['data']
    
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
        # if os.path.isdir(mission):
        shutil.copy2(mission, outpath) # save mission yaml to processed directory
        vehicle = filepath + os.sep + 'vehicle.yaml'
        # if os.path.isdir(vehicle):
        shutil.copy2(vehicle, outpath) # save vehicle yaml to processed directory
        outpath = outpath + os.sep + 'nav'
        filename='nav_standard.json'
        
        proc_flag=2
    
    elif ftype == 'acfr':# or (ftype is not 'acfr'):        
        with open(outpath + os.sep + 'mission.cfg','w') as fileout:
            data = 'MAG_VAR_LAT ' + str(float(latitude_reference)) + '\n' + 'MAG_VAR_LNG ' + str(float(longitude_reference)) + '\n' + 'MAG_VAR_DATE "' + str(date) + '"\n' + 'MAGNETIC_VAR_DEG ' + str(float(0))
            
            fileout.write(data)
            fileout.close()
                       
        outpath = outpath + os.sep +'dRAWLOGS_cv'
        filename='combined.RAW.auv'
        proc_flag=2    

    else:
        print('Error: -o',ftype,'not recognised')
        #syntax_error()    
        return
    
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

            #thread_list = []

            if multiprocessing.cpu_count() < 4:
                cpu_to_use = 1
            else:
                cpu_to_use = 3
            pool = multiprocessing.Pool(cpu_to_use) #multiprocessing.cpu_count() - 3)
            pool_list = []
    
            # read in, parse data and write data
            if image_flag == 1:
                if image_format == "acfr_standard" or image_format == "unagi" :
                    pool_list.append(pool.apply_async(parse_acfr_images, [filepath + os.sep + image_filepath,image_format,camera1_label,camera2_label,'images',image_timezone,image_timeoffset,ftype,outpath,filename]))
                    # pool_list.append(results)
                    # thread = ThreadWithReturnValue(target=parse_acfr_images, args=[filepath + os.sep + image_filepath,image_format,camera1_label,camera2_label,'images',image_timezone,image_timeoffset,ftype,outpath,filename,fileout])
                    # thread_list.append(thread)
                if image_format == "seaxerocks_3":
                    pool_list.append(pool.apply_async(parse_seaxerocks_images, [filepath + os.sep + image_filepath,image_format,date,camera1_label,camera2_label,camera3_label,'images',image_timezone,image_timeoffset,ftype,outpath,filename]))
                    # pool_list.append(results)
                    # thread = ThreadWithReturnValue(target=parse_seaxerocks_images, args=[filepath + os.sep + image_filepath,image_format,date,camera1_label,camera2_label,camera3_label,'images',image_timezone,image_timeoffset,ftype,outpath,filename,fileout])
                    # thread_list.append(thread)
                image_flag = 0

            if usbl_flag == 1:
                print('Loading usbl data...')
                if usbl_format == "gaps":
                    pool_list.append(pool.apply_async(parse_gaps, [filepath + os.sep + usbl_filepath,'usbl',time_usblzone,time_usbloffset,latitude_reference,longitude_reference,ftype,outpath,filename,usbl_id]))
                    # pool_list.append(results)
                    # thread = ThreadWithReturnValue(target=parse_gaps, args=[filepath + os.sep + usbl_filepath,'usbl',time_usblzone,time_usbloffset,latitude_reference,longitude_reference,ftype,outpath,filename,usbl_id,fileout])
                    # thread_list.append(thread)
                if usbl_format == "usbl_dump":
                    pool_list.append(pool.apply_async(parse_usbl_dump, [filepath + os.sep + usbl_filepath,usbl_filename,usbl_label,'usbl',time_usblzone,time_usbloffset,latitude_reference,longitude_reference,ftype,outpath,filename]))
                    # pool_list.append(results)
                    # thead = ThreadWithReturnValue(target=parse_usbl_dump, args=[filepath + os.sep + usbl_filepath,usbl_filename,usbl_label,'usbl',time_usblzone,time_usbloffset,latitude_reference,longitude_reference,ftype,outpath,filename,fileout])
                    # thread_list.append(thread)
                usbl_flag = 0

            if velocity_flag == 1:
                print('Loading velocity data...')
                if velocity_format == "phins":
                    pool_list.append(pool.apply_async(parse_phins, [filepath + os.sep + velocity_filepath,velocity_filename,'velocity',velocity_timezone,velocity_timeoffset,velocity_headingoffset,ftype,outpath,filename]))
                    # pool_list.append(results)
                    # thread = ThreadWithReturnValue(target=parse_phins, args=[filepath + os.sep + velocity_filepath,velocity_filename,'velocity',velocity_timezone,velocity_timeoffset,velocity_headingoffset,ftype,outpath,filename,fileout])
                    # thread_list.append(thread)
                if velocity_format == "ae2000":
                    pool_list.append(pool.apply_async(parse_ae2000, [filepath + os.sep + velocity_filepath,velocity_filename,'velocity',velocity_timezone,velocity_timeoffset,velocity_headingoffset,ftype,outpath,filename]))
                    # pool_list.append(results)
                    # thread = ThreadWithReturnValue(target=parse_ae2000, args=[filepath + os.sep + velocity_filepath,velocity_filename,'velocity',velocity_timezone,velocity_timeoffset,velocity_headingoffset,ftype,outpath,filename,fileout])
                    # thread_list.append(thread)
                velocity_flag = 0

            if orientation_flag == 1:                
                print('Loading orientation data...')
                if orientation_format == "phins":    
                    pool_list.append(pool.apply_async(parse_phins, [filepath + os.sep + orientation_filepath,orientation_filename,'orientation',time_orientationzone,time_orientationoffset,orientation_headingoffset,ftype,outpath,filename]))
                    # pool_list.append(results)
                    # thread = ThreadWithReturnValue(target=parse_phins, args=[filepath + os.sep + orientation_filepath,orientation_filename,'orientation',time_orientationzone,time_orientationoffset,orientation_headingoffset,ftype,outpath,filename,fileout])
                    # thread_list.append(thread)
                if orientation_format == "ae2000":
                    pool_list.append(pool.apply_async(parse_ae2000, [filepath + os.sep + orientation_filepath,orientation_filename,'orientation',time_orientationzone,time_orientationoffset,orientation_headingoffset,ftype,outpath,filename]))
                    # pool_list.append(results)
                    # thread = ThreadWithReturnValue(target=parse_ae2000, args=[filepath + os.sep + orientation_filepath,orientation_filename,'orientation',time_orientationzone,time_orientationoffset,orientation_headingoffset,ftype,outpath,filename,fileout])
                    # thread_list.append(thread)
                orientation_flag = 0

            if depth_flag == 1:                                
                print('Loading depth data...')
                if depth_format == "phins":
                    pool_list.append(pool.apply_async(parse_phins, [filepath + os.sep + depth_filepath,depth_filename,'depth',time_depthzone,time_depthoffset,0,ftype,outpath,filename]))
                    # pool_list.append(results)
                    # thread = ThreadWithReturnValue(target=parse_phins, args=[filepath + os.sep + depth_filepath,depth_filename,'depth',time_depthzone,time_depthoffset,0,ftype,outpath,filename,fileout])
                    # thread_list.append(thread)
                if depth_format == "ae2000":
                    pool_list.append(pool.apply_async(parse_ae2000, [filepath + os.sep + depth_filepath,depth_filename,'depth',time_depthzone,time_depthoffset,0,ftype,outpath,filename]))
                    # pool_list.append(results)
                    # thread = ThreadWithReturnValue(target=parse_ae2000, args=[filepath + os.sep + depth_filepath,depth_filename,'depth',time_depthzone,time_depthoffset,0,ftype,outpath,filename,fileout])
                    # thread_list.append(thread)
                depth_flag = 0

            if altitude_flag == 1:                
                print('Loading altitude data...')
                if altitude_format == "phins":
                    pool_list.append(pool.apply_async(parse_phins, [filepath + os.sep + altitude_filepath,altitude_filename,'altitude',time_altitudezone,0,time_altitudeoffset,ftype,outpath,filename]))
                    # pool_list.append(results)
                    # thread = ThreadWithReturnValue(target=parse_phins, args=[filepath + os.sep + altitude_filepath,altitude_filename,'altitude',time_altitudezone,0,time_altitudeoffset,ftype,outpath,filename,fileout])
                    # thread_list.append(thread)
                if altitude_format == "ae2000":
                    pool_list.append(pool.apply_async(parse_ae2000, [filepath + os.sep + altitude_filepath,altitude_filename,'altitude',time_altitudezone,0,time_altitudeoffset,ftype,outpath,filename]))
                    # pool_list.append(results)
                    # thread = ThreadWithReturnValue(target=parse_ae2000, args=[filepath + os.sep + altitude_filepath,altitude_filename,'altitude',time_altitudezone,0,time_altitudeoffset,ftype,outpath,filename,fileout])
                    # thread_list.append(thread)
                altitude_flag = 0

            # for thread in thread_list:
            #     thread.start()
            # data_list = []
            # for thread in thread_list:
            #     data = thread.join()
            #     data_list.append(data)

            pool.close()
            pool.join()

            data_list = []

            for i in pool_list:
            	results = i.get()
            	data_list.append(results)
                # print (type(results.get()))
                # print (len(results.get()))
                # data_list.append(results.get())

            if ftype == 'acfr':
                data_string = ''
                for i in data_list:
                    data_string += i
                fileout.write(data_string)
                del data_string
            elif ftype == 'oplab':
                # print('Initialising JSON file')
                data_list_temp = []
                for i in data_list:
                    data_list_temp += i
                json.dump(data_list_temp, fileout)
                del data_list_temp
            del data_list

            # if chemical_flag == 1:
            #     print('Loading chemical data...')
            #     if chemical_format == 'date_time_intensity':
            #         parse_chemical(filepath + os.sep + chemical_filepath, chemical_filename, chemical_timezone, chemical_timeoffset, chemical_data, ftype, outpath, filename, fileout)
            #     chemical_flag = 0
    
        fileout.close()
        
        #interlace the data based on timestamps
        print('Interlacing data...')
        parse_interlacer(ftype,outpath,filename)
        print('Output saved to ' + outpath + os.sep + filename)

        print('Complete parse data')
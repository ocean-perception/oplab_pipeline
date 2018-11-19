# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 12:34:42 2018

@author: Adrian
"""

# Known issues: When running the code more than once from the same console,
# line 206 pool = multiprocessing.Pool(cpu_to_use) causes an error.
# A workaround is to close the console after eachr run


import multiprocessing
import os
import shutil
import json
import yaml
import sys

# sys.path.append("..")
from auv_nav.parsers.parse_phins import parse_phins
from auv_nav.parsers.parse_ae2000 import parse_ae2000
from auv_nav.parsers.parse_gaps import parse_gaps
from auv_nav.parsers.parse_usbl_dump import parse_usbl_dump
from auv_nav.parsers.parse_acfr_images import parse_acfr_images
from auv_nav.parsers.parse_seaxerocks_images import parse_seaxerocks_images
from auv_nav.parsers.parse_interlacer import parse_interlacer
# from lib_sensors.parse_chemical import parse_chemical
from auv_nav.plot.plot_parse_data import plot_parse_data


def parse_data(filepath, ftype):
    # initiate data and processing flags
    filepath = filepath.replace('\\', '/')

    # load mission.yaml config file
    print('Loading mission.yaml')

    mission_file = filepath + '/' + 'mission.yaml'
    mission = yaml.load(open(mission_file, 'r'))

    # Check mission.yaml version
    if 'version' in mission:
        if mission['version'] == 1:
            print('Mission version: 1')
        else:
            print('Mission version 1 expected.')
            print('Your mission version is {0}'.format(mission['version']))
            print('auv_nav will now exit')
            sys.exit()
    else:
        print('Mission version 1 expected.')
        print('You are using and old mission.yaml format that is no longer \n\
              compatible. Please refer to the example mission.yaml file and \n\
              modify yours to fit.')
        print('auv_nav will now exit')
        sys.exit()

    # generate output path
    print('Generating output paths')
    sub_path = filepath.split('/')
    sub_out = sub_path
    outpath = sub_out[0]

    is_subfolder_of_processed = False
    for i in range(1, len(sub_path)):
        if sub_path[i] == 'raw':
            sub_out[i] = 'processed'
            is_subfolder_of_processed = True
        else:
            sub_out[i] = sub_path[i]

        outpath = outpath + '/' + sub_out[i]
        # make the new directories after 'processed' if it doesnt already exist
        if is_subfolder_of_processed:
            if os.path.isdir(outpath) == 0:
                try:
                    os.mkdir(outpath)
                except Exception as e:
                    print("Warning:", e)

    if not is_subfolder_of_processed:
        raise ValueError(
            "The input directory you provided is not a subfolder of a folder called 'raw'")

    # check for recognised formats and create nav file
    print('Checking output format')

    # copy mission.yaml and vehicle.yaml to processed folder for process step
    # if os.path.isdir(mission):
    shutil.copy2(mission_file, outpath)  # save mission yaml to processed directory
    vehicle = filepath + '/' + 'vehicle.yaml'
    # if os.path.isdir(vehicle):
    shutil.copy2(vehicle, outpath)  # save vehicle yaml to processed directory

    if ftype == 'oplab':  # or (ftype is not 'acfr'):
        outpath = outpath + '/' + 'nav'
        filename = 'nav_standard.json'

    elif ftype == 'acfr':  # or (ftype is not 'acfr'):
        outpath = outpath + '/' + 'dRAWLOGS_cv'
        filename = 'combined.RAW.auv'
    else:
        print('Error: -o', ftype, 'not recognised')
        # syntax_error()
        return

    # make file path if not exist
    if os.path.isdir(outpath) == 0:
        try:
            os.mkdir(outpath)
        except Exception as e:
            print("Warning:", e)

    # create file (overwrite if exists)
    with open(outpath + '/' + filename, 'w') as fileout:
        print('Loading raw data')

        if multiprocessing.cpu_count() < 4:
            cpu_to_use = 1
        else:
            cpu_to_use = 3

        try:
            pool = multiprocessing.Pool(cpu_to_use)
        except AttributeError as e:
            print("Error: ", e, "\n===============\nThis error is known to \
                   happen when running the code more than once from the same \
                   console in Spyder. Please run the code from a new console \
                   to prevent this error from happening. You may close the \
                   current console.\n==============")
        pool_list = []

        # read in, parse data and write data
        if 'image' in mission:
            if mission['image']['format'] == "acfr_standard" or mission['image']['format'] == "unagi":
                pool_list.append(
                    pool.apply_async(parse_acfr_images,
                                     [mission['image'], 'images',
                                      ftype, outpath, filename]))
            if mission['image']['format'] == "seaxerocks_3":
                pool_list.append(
                    pool.apply_async(parse_seaxerocks_images,
                                     [mission['image'], 'images',
                                      ftype, outpath, filename]))
        if 'usbl' in mission:
            print('Loading usbl data...')
            if mission['usbl']['format'] == "gaps":
                pool_list.append(
                    pool.apply_async(
                        parse_gaps,
                        [mission['usbl'], 'usbl',
                         mission['origin'],
                         ftype, outpath, filename]))
            if mission['usbl']['format'] == "usbl_dump":
                pool_list.append(
                    pool.apply_async(
                        parse_usbl_dump,
                        [mission['usbl'], 'usbl',
                         mission['origin'],
                         ftype, outpath, filename]))

        if 'velocity' in mission:
            print('Loading velocity data...')
            if mission['velocity']['format'] == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission['velocity'], 'velocity',
                         ftype, outpath, filename]))
            if mission['velocity']['format'] == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission['velocity'], 'velocity',
                         ftype, outpath, filename]))

        if 'orientation' in mission:
            print('Loading orientation data...')
            if mission['orientation']['format'] == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission['orientation'], 'orientation',
                         ftype, outpath, filename]))
            if mission['orientation']['format'] == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission['orientation'], 'orientation',
                         ftype, outpath, filename]))

        if 'depth' in mission:
            print('Loading depth data...')
            if mission['depth']['format'] == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission['depth'], 'depth',
                         ftype, outpath, filename]))
            if mission['depth']['format'] == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission['depth'], 'depth',
                         ftype, outpath, filename]))

        if 'altitude' in mission:
            print('Loading altitude data...')
            if mission['altitude']['format'] == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission['altitude'], 'altitude',
                         ftype, outpath, filename]))
            if mission['altitude']['format'] == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission['altitude'], 'altitude',
                         ftype, outpath, filename]))
        pool.close()
        print('Wait for parsing threads to finish...')
        pool.join()
        print('...done. All parsing threads have finished processing.')
        print('...done loading raw data.')

        print('Compile data list...')
        data_list = []
        for i in pool_list:
            results = i.get()
            data_list.append(results)
            # print (type(results.get()))
            # print (len(results.get()))
            # data_list.append(results.get())
        print('...done compiling data list.')

        print('Writing to output file...')
        if ftype == 'acfr':
            data_string = ''
            for i in data_list:
                data_string += i

            date = epoch_to_day(data_string[0]['epoch_timestamp'])
            data = 'MAG_VAR_LAT ' + str(float(latitude_reference)) + '\n' + 'MAG_VAR_LNG ' + str(float(
                longitude_reference)) + '\n' + 'MAG_VAR_DATE "' + str(date) + '"\n' + 'MAGNETIC_VAR_DEG ' + str(float(0))
            fileout.write(data)
            fileout.write(data_string)
            del data_string
        elif ftype == 'oplab':
            # print('Initialising JSON file')
            data_list_temp = []
            for i in data_list:
                data_list_temp += i
            json.dump(data_list_temp, fileout, indent=2)
            del data_list_temp
        del data_list
        print('... done writing to output file.')

        # if chemical_flag == 1:
        #     print('Loading chemical data...')
        #     if chemical_format == 'date_time_intensity':
        #         parse_chemical(filepath + '/' + chemical_filepath, chemical_filename, chemical_timezone, chemical_timeoffset, chemical_data, ftype, outpath, filename, fileout)
        #     chemical_flag = 0

    fileout.close()
    # interlace the data based on timestamps
    print('Interlacing data...')
    parse_interlacer(ftype, outpath, filename)
    print('...done interlacing data. Output saved to ' + outpath + '/' + filename)

    if ftype == 'oplab':
        plot_parse_data(outpath, ftype)

    print('Complete parse data')

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 12:34:42 2018

@author: Adrian
"""

# Known issues: When running the code more than once from the same console,
# line 206 pool = multiprocessing.Pool(cpu_to_use) causes an error.
# A workaround is to close the console after eachr run


import multiprocessing
import json
from pathlib import Path

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
from auv_nav.tools.time_conversions import epoch_to_day
from auv_nav.tools.folder_structure import get_config_folder, get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.parsers.vehicle import Vehicle
from auv_nav.parsers.mission import Mission


def parse_data(filepath, ftype):
    # initiate data and processing flags
    filepath = Path(filepath).resolve()

    filepath = get_raw_folder(filepath)

    # load mission.yaml config file

    mission_file = filepath / 'mission.yaml'
    vehicle_file = filepath / 'vehicle.yaml'
    mission_file = get_raw_folder(mission_file)
    vehicle_file = get_raw_folder(vehicle_file)
    print('Loading mission.yaml at {0}'.format(mission_file))
    mission = Mission(mission_file)
    print('Loading vehicle.yaml at {0}'.format(vehicle_file))
    vehicle = Vehicle(vehicle_file)

    # std factors and offsets defaults
    # Even if you provide std factors and offset or not, if the sensor
    # has its own std measurements, that measurements will be used instead
    std_factor_usbl = 0.01
    std_offset_usbl = 10.
    std_factor_dvl = 0.001
    std_offset_dvl = 0.002
    std_factor_depth = 0
    std_offset_depth = 0.01
    std_factor_orientation = 0.
    std_offset_orientation = 0.003
    std_factor_altitude = 0.
    std_offset_altitude = 0.01

    if mission.usbl.std_factor == 0 and std_factor_usbl != 0:
        # print('USBL standard deviation factor not provided. Using default of {}'.format(std_factor_usbl))
        mission.usbl.std_factor = std_factor_usbl
    if mission.usbl.std_offset == 0 and std_offset_usbl != 0:
        # print('USBL standard deviation offset not provided. Using default of {}'.format(std_offset_usbl))
        mission.usbl.std_offset = std_offset_usbl
    if mission.velocity.std_factor == 0 and std_factor_dvl != 0:
        # print('DVL standard deviation factor not provided. Using default of {}'.format(std_factor_dvl))
        mission.velocity.std_factor = std_factor_dvl
    if mission.velocity.std_offset == 0 and std_offset_dvl != 0:
        # print('DVL standard deviation offset not provided. Using default of {}'.format(std_offset_dvl))
        mission.velocity.std_offset = std_offset_dvl
    if mission.depth.std_factor == 0 and std_factor_depth != 0:
        # print('Depth standard deviation factor not provided. Using default of {}'.format(std_factor_depth))
        mission.depth.std_factor = std_factor_depth
    if mission.depth.std_offset == 0 and std_offset_depth != 0:
        # print('Depth standard deviation offset not provided. Using default of {}'.format(std_offset_depth))
        mission.depth.std_offset = std_offset_depth
    if mission.orientation.std_factor == 0 and std_factor_orientation != 0:
        # print('Orientation standard deviation factor not provided. Using default of {}'.format(std_factor_orientation))
        mission.orientation.std_factor = std_factor_orientation
    if mission.orientation.std_offset == 0 and std_offset_orientation != 0:
        # print('Orientation standard deviation offset not provided. Using default of {}'.format(std_offset_orientation))
        mission.orientation.std_offset = std_offset_orientation
    if mission.altitude.std_factor == 0 and std_factor_altitude != 0:
        # print('Altitude standard deviation factor not provided. Using default of {}'.format(std_factor_altitude))
        mission.altitude.std_factor = std_factor_altitude
    if mission.altitude.std_offset == 0 and std_offset_altitude != 0:
        # print('Altitude standard deviation offset not provided. Using default of {}'.format(std_offset_altitude))
        mission.altitude.std_offset = std_offset_altitude

    # copy mission.yaml and vehicle.yaml to processed folder for process step
    mission_processed = get_processed_folder(mission_file)
    vehicle_processed = get_processed_folder(vehicle_file)
    mission_file.copy(mission_processed)
    vehicle_file.copy(vehicle_processed)

    # check for recognised formats and create nav file
    outpath = get_processed_folder(filepath)
    print('Checking output format')
    if ftype == 'oplab':  # or (ftype is not 'acfr'):
        outpath = outpath / 'nav'
        filename = 'nav_standard.json'

    elif ftype == 'acfr':  # or (ftype is not 'acfr'):
        filename = 'combined.RAW.auv'
        config_filename = outpath / 'mission.cfg'
        outpath = outpath / 'dRAWLOGS_cv'

        with config_filename.open('w') as f:
            data = ('MAG_VAR_LAT ' + str(float(mission.origin.latitude))
                    + '\nMAG_VAR_LNG ' + str(float(mission.origin.longitude))
                    + '\nMAG_VAR_DATE ' + str(mission.origin.date)
                    + '\nMAGNETIC_VAR_DEG ' + str(float(0)))
            f.write(data)
    else:
        print('Error: -o', ftype, 'not recognised')
        # syntax_error()
        return

    # make file path if not exist
    if not outpath.is_dir():
        try:
            outpath.mkdir()
        except Exception as e:
            print("Warning:", e)

    # create file (overwrite if exists)
    nav_file = outpath / filename
    with nav_file.open('w') as fileout:
        print('Loading raw data')

        if multiprocessing.cpu_count() < 4:
            cpu_to_use = 1
        else:
            cpu_to_use = multiprocessing.cpu_count() - 2

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
        if not mission.image.empty():
            if mission.image.format == "acfr_standard" or mission.image.format == "unagi":
                pool_list.append(
                    pool.apply_async(parse_acfr_images,
                                     [mission, vehicle, 'images',
                                      ftype, outpath, filename]))
            if mission.image.format == "seaxerocks_3":
                pool_list.append(
                    pool.apply_async(parse_seaxerocks_images,
                                     [mission, vehicle, 'images',
                                      ftype, outpath, filename]))
        if not mission.usbl.empty():
            print('Loading usbl data...')
            if mission.usbl.format == "gaps":
                pool_list.append(
                    pool.apply_async(
                        parse_gaps,
                        [mission, vehicle, 'usbl',
                         ftype, outpath, filename]))
            if mission.usbl.format == "usbl_dump":
                pool_list.append(
                    pool.apply_async(
                        parse_usbl_dump,
                        [mission, vehicle, 'usbl',
                         ftype, outpath, filename]))

        if not mission.velocity.empty():
            print('Loading velocity data...')
            if mission.velocity.format == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission, vehicle, 'velocity',
                         ftype, outpath, filename]))
            if mission.velocity.format == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission, vehicle, 'velocity',
                         ftype, outpath, filename]))

        if not mission.orientation.empty():
            print('Loading orientation data...')
            if mission.orientation.format == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission, vehicle, 'orientation',
                         ftype, outpath, filename]))
            if mission.orientation.format == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission, vehicle, 'orientation',
                         ftype, outpath, filename]))

        if not mission.depth.empty():
            print('Loading depth data...')
            if mission.depth.format == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission, vehicle, 'depth',
                         ftype, outpath, filename]))
            if mission.depth.format == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission, vehicle, 'depth',
                         ftype, outpath, filename]))

        if not mission.altitude.empty():
            print('Loading altitude data...')
            if mission.altitude.format == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission, vehicle, 'altitude',
                         ftype, outpath, filename]))
            if mission.altitude.format == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission, vehicle, 'altitude',
                         ftype, outpath, filename]))
        pool.close()
        print('Wait for parsing threads to finish...')
        pool.join()
        print('...done. All parsing threads have finished processing.')
        print('...done loading raw data.')

        print('Compile data list...')
        data_list = [[{
            'epoch_timestamp': 0.0,
            'class': 'origin',
            'category': 'origin',
            'data': [{
                'latitude': mission.origin.latitude,
                'longitude': mission.origin.longitude,
                'crs': mission.origin.crs,
                'date': mission.origin.date
            }]
        }]]
        for i in pool_list:
            results = i.get()
            data_list.append(results)
        print('...done compiling data list.')

        print('Writing to output file...')
        if ftype == 'acfr':
            data_string = ''
            for i in data_list:
                    data_string += ''.join(i)
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
    print('...done interlacing data. Output saved to {}'.format(outpath /filename))

    if ftype == 'oplab':
        plot_parse_data(outpath, ftype)

    print('Complete parse data')

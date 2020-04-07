# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, University of Southampton
All rights reserved.
"""
import multiprocessing
import json
from pathlib import Path

# sys.path.append("..")
from auv_nav.parsers.parse_phins import parse_phins
from auv_nav.parsers.parse_ae2000 import parse_ae2000
from auv_nav.parsers.parse_NOC_nmea import parse_NOC_nmea
from auv_nav.parsers.parse_NOC_polpred import parse_NOC_polpred
from auv_nav.parsers.parse_autosub import parse_autosub
from auv_nav.parsers.parse_gaps import parse_gaps
from auv_nav.parsers.parse_usbl_dump import parse_usbl_dump
from auv_nav.parsers.parse_acfr_images import parse_acfr_images
from auv_nav.parsers.parse_biocam_images import parse_biocam_images
from auv_nav.parsers.parse_biocam_images import correct_timestamps
from auv_nav.parsers.parse_seaxerocks_images import parse_seaxerocks_images
from auv_nav.parsers.parse_interlacer import parse_interlacer
# from lib_sensors.parse_chemical import parse_chemical
from auv_nav.plot.plot_parse_data import plot_parse_data
from auv_nav.tools.time_conversions import epoch_to_day
from auv_nav.tools.folder_structure import get_config_folder, get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.console import Console
from auv_nav.vehicle import Vehicle
from auv_nav.mission import Mission

from auv_nav.tools.interpolate import interpolate
from auv_nav.tools.interpolate import interpolate_sensor_list
from auv_nav.tools.time_conversions import string_to_epoch
from auv_nav.tools.time_conversions import epoch_from_json
from auv_nav.tools.time_conversions import epoch_to_datetime

from auv_nav.sensors import Category


def parse(filepath, force_overwite):
    # initiate data and processing flags
    filepath = Path(filepath).resolve()
    filepath = get_raw_folder(filepath)

    if not force_overwite:
        existing_files = check_output_files_exist(get_processed_folder(filepath))
        if existing_files:
            msg =   'It looks like this dataset has already been parsed.\n' \
                  + 'The following file(s) already exist:\n' \
                  + existing_files \
                  + 'If you would like auv_nav to overwrite existing file, rerun it with the flag -F.\n' \
                  + 'Example:   auv_nav parse -F PATH'
            Console.quit(msg)

    ftype = 'oplab'

    # load mission.yaml config file

    mission_file = filepath / 'mission.yaml'
    vehicle_file = filepath / 'vehicle.yaml'
    mission_file = get_raw_folder(mission_file)
    vehicle_file = get_raw_folder(vehicle_file)
    Console.info('Loading mission.yaml at {0}'.format(mission_file))
    mission = Mission(mission_file)

    Console.info('Loading vehicle.yaml at {0}'.format(vehicle_file))
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
    std_factor_tide = 0
    std_offset_tide = 0.01
    std_factor_orientation = 0.
    std_offset_orientation = 0.003
    std_factor_altitude = 0.
    std_offset_altitude = 0.01

    if mission.usbl.std_factor == 0 and std_factor_usbl != 0:
        Console.warn('USBL standard deviation factor not provided. Using default of {}'.format(std_factor_usbl))
        mission.usbl.std_factor = std_factor_usbl
    if mission.usbl.std_offset == 0 and std_offset_usbl != 0:
        Console.warn('USBL standard deviation offset not provided. Using default of {}'.format(std_offset_usbl))
        mission.usbl.std_offset = std_offset_usbl
    if mission.velocity.std_factor == 0 and std_factor_dvl != 0:
        Console.warn('DVL standard deviation factor not provided. Using default of {}'.format(std_factor_dvl))
        mission.velocity.std_factor = std_factor_dvl
    if mission.velocity.std_offset == 0 and std_offset_dvl != 0:
        Console.warn('DVL standard deviation offset not provided. Using default of {}'.format(std_offset_dvl))
        mission.velocity.std_offset = std_offset_dvl
    if mission.depth.std_factor == 0 and std_factor_depth != 0:
        Console.warn('Depth standard deviation factor not provided. Using default of {}'.format(std_factor_depth))
        mission.depth.std_factor = std_factor_depth
    if mission.depth.std_offset == 0 and std_offset_depth != 0:
        Console.warn('Depth standard deviation offset not provided. Using default of {}'.format(std_offset_depth))
        mission.depth.std_offset = std_offset_depth
    if mission.orientation.std_factor == 0 and std_factor_orientation != 0:
        Console.warn('Orientation standard deviation factor not provided. Using default of {}'.format(std_factor_orientation))
        mission.orientation.std_factor = std_factor_orientation
    if mission.orientation.std_offset == 0 and std_offset_orientation != 0:
        Console.warn('Orientation standard deviation offset not provided. Using default of {}'.format(std_offset_orientation))
        mission.orientation.std_offset = std_offset_orientation
    if mission.altitude.std_factor == 0 and std_factor_altitude != 0:
        Console.warn('Altitude standard deviation factor not provided. Using default of {}'.format(std_factor_altitude))
        mission.altitude.std_factor = std_factor_altitude
    if mission.altitude.std_offset == 0 and std_offset_altitude != 0:
        Console.warn('Altitude standard deviation offset not provided. Using default of {}'.format(std_offset_altitude))
        mission.altitude.std_offset = std_offset_altitude

    # copy mission.yaml and vehicle.yaml to processed folder for process step
    mission_processed = get_processed_folder(mission_file)
    vehicle_processed = get_processed_folder(vehicle_file)

    # Write mission with metadata (username, date and hostname)
    mission.write(mission_processed)
    # mission_file.copy(mission_processed)
    vehicle.write(vehicle_processed)

    # check for recognised formats and create nav file
    outpath = get_processed_folder(filepath)
    outpath = outpath / 'nav'
    filename = 'nav_standard.json'

    # make file path if not exist
    if not outpath.is_dir():
        try:
            outpath.mkdir()
        except Exception as e:
            print("Warning:", e)

    # create file (overwrite if exists)
    nav_file = outpath / filename
    with nav_file.open('w') as fileout:
        Console.info('Loading raw data...')

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
            elif mission.image.format == "seaxerocks_3":
                pool_list.append(
                    pool.apply_async(parse_seaxerocks_images,
                                     [mission, vehicle, 'images',
                                      ftype, outpath, filename]))
            elif mission.image.format == "biocam":
                pool_list.append(
                    pool.apply_async(parse_biocam_images,
                                     [mission, vehicle, 'images',
                                      ftype, outpath, filename]))
            else:
                Console.quit('Mission image format {} not supported.'
                             .format(mission.image.format))
        if not mission.usbl.empty():
            if mission.usbl.format == "gaps":
                pool_list.append(
                    pool.apply_async(
                        parse_gaps,
                        [mission, vehicle, 'usbl',
                         ftype, outpath, filename]))
            elif mission.usbl.format == "usbl_dump":
                pool_list.append(
                    pool.apply_async(
                        parse_usbl_dump,
                        [mission, vehicle, 'usbl',
                         ftype, outpath, filename]))
            elif mission.usbl.format == "NOC_nmea":
                pool_list.append(
                    pool.apply_async(
                        parse_NOC_nmea,
                        [mission, vehicle, 'usbl',
                         ftype, outpath, filename]))
            else:
                Console.quit('Mission usbl format {} not supported.'
                             .format(mission.usbl.format))

        if not mission.velocity.empty():
            if mission.velocity.format == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission, vehicle, 'velocity',
                         ftype, outpath, filename]))
            elif mission.velocity.format == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission, vehicle, 'velocity',
                         ftype, outpath, filename]))
            elif mission.velocity.format == "autosub":
                pool_list.append(
                    pool.apply_async(
                        parse_autosub,
                        [mission, vehicle, 'velocity',
                         ftype, outpath, filename]))
            else:
                Console.quit('Mission velocity format {} not supported.'
                             .format(mission.velocity.format))

        if not mission.orientation.empty():
            if mission.orientation.format == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission, vehicle, 'orientation',
                         ftype, outpath, filename]))
            elif mission.orientation.format == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission, vehicle, 'orientation',
                         ftype, outpath, filename]))
            elif mission.orientation.format == "autosub":
                pool_list.append(
                    pool.apply_async(
                        parse_autosub,
                        [mission, vehicle, 'orientation',
                         ftype, outpath, filename]))
            else:
                Console.quit('Mission orientation format {} not supported.'
                             .format(mission.orientation.format))

        if not mission.depth.empty():
            if mission.depth.format == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission, vehicle, 'depth',
                         ftype, outpath, filename]))
            elif mission.depth.format == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission, vehicle, 'depth',
                         ftype, outpath, filename]))
            elif mission.depth.format == "autosub":
                pool_list.append(
                    pool.apply_async(
                        parse_autosub,
                        [mission, vehicle, 'depth',
                         ftype, outpath, filename]))
            else:
                Console.quit('Mission depth format {} not supported.'
                             .format(mission.depth.format))

        if not mission.altitude.empty():
            if mission.altitude.format == "phins":
                pool_list.append(
                    pool.apply_async(
                        parse_phins,
                        [mission, vehicle, 'altitude',
                         ftype, outpath, filename]))
            elif mission.altitude.format == "ae2000":
                pool_list.append(
                    pool.apply_async(
                        parse_ae2000,
                        [mission, vehicle, 'altitude',
                         ftype, outpath, filename]))
            elif mission.altitude.format == "autosub":
                pool_list.append(
                    pool.apply_async(
                        parse_autosub,
                        [mission, vehicle, 'altitude',
                         ftype, outpath, filename]))
            else:
                Console.quit('Mission altitude format {} not supported.'
                             .format(mission.altitude.format))

        if not mission.tide.empty():
            if mission.tide.format == "NOC_polpred":
                tide_list = parse_NOC_polpred(mission, vehicle, 'tide',
                         ftype, outpath, filename)
            else:
                Console.quit('Mission tide format {} not supported.'
                             .format(mission.tide.format))
        else:
            tide_list = None
        
        pool.close()
        pool.join()

        Console.info('...done loading raw data.')
        Console.info('Compile data list...')

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

        # Set advance True/False flag so not comparing for every i in pool_list
        if mission.image.format == "biocam":
            correcting_timestamps = True
        else:
            correcting_timestamps = False
        for i in pool_list:
            results = i.get()
            # If current retrieved data is DEPTH
            # and if TIDE data is available
            if len(results) < 1:
                continue
            if correcting_timestamps:
                if (results[0]['category'] == 'image'):
                    Console.info("Correction of BioCam cpu timestamps...")
                    results = correct_timestamps(results)
            if (results[0]['category'] == Category.DEPTH or results[0]['category'] == Category.USBL):
                if not mission.tide.empty():
                    # proceed to tidal correction
                    Console.info("Tidal correction of depth vector...")
                    # Offset depth to acknowledge for tides
                    j = 0
                    for k in range(len(results)):
                        while j < len(tide_list) and tide_list[j]['epoch_timestamp'] < results[k]['epoch_timestamp']:
                            j = j + 1

                        if j >= 1:
                            _result = interpolate(
                                results[k]['epoch_timestamp'],
                                tide_list[j-1]['epoch_timestamp'],
                                tide_list[j]['epoch_timestamp'],
                                tide_list[j-1]['data'][0]['height'],
                                tide_list[j]['data'][0]['height'])
                            if results[0]['category'] == Category.DEPTH:
                                results[k]['data'][0]['depth'] = results[k]['data'][0]['depth'] - _result
                            elif results[0]['category'] == Category.USBL:
                                results[k]['data_target'][4]['depth'] = results[k]['data_target'][4]['depth'] - _result
            data_list.append(results)

        Console.info('...done compiling data list.')

        Console.info('Writing to output file...')
        data_list_temp = []
        for i in data_list:
            data_list_temp += i

        json.dump(data_list_temp, fileout, indent=2)

        del data_list_temp
        del data_list

    fileout.close()

    # interlace the data based on timestamps
    Console.info('Interlacing data...')
    parse_interlacer(outpath, filename)
    Console.info('...done interlacing data. Output saved to {}'.format(outpath /filename))
    plot_parse_data(outpath, ftype)
    Console.info('Complete parse data')



def check_output_files_exist(processed_dataset_folder):
    """Check if any of the files exist, which `auv_nav parse` writes to disk"""
    mission_file      = processed_dataset_folder / 'mission.yaml'
    vehicle_file      = processed_dataset_folder / 'vehicle.yaml'
    nav_file          = processed_dataset_folder / 'nav' / 'nav_standard.json'
    data_plot_file    = processed_dataset_folder / 'nav' / 'json_data_info.html'
    history_plot_file = processed_dataset_folder / 'nav' / 'timestamp_history.html'

    existing_files = ''
    if mission_file.exists():
        existing_files += str(mission_file) + '\n'
    if vehicle_file.exists():
        existing_files += str(vehicle_file) + '\n'
    if nav_file.exists():
        existing_files += str(nav_file) + '\n'
    if data_plot_file.exists():
        existing_files += str(data_plot_file) + '\n'
    if history_plot_file.exists():
        existing_files += str(history_plot_file) + '\n'

    return existing_files
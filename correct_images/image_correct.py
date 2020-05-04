# this is the main entry point to correct images
# IMPORT --------------------------------
# all imports go here 
import datetime
import sys
import random
import socket
import uuid
import os
import argparse
import cv2
import imageio
import joblib
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters
import yaml
from scipy import optimize
from tqdm import trange
from pathlib import Path

from auv_nav.tools.console import Console
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder
from auv_nav.parsers.mission import *
from auv_nav.camera_system import *

from correct_images.corrector import *
from correct_images.parser import *

from numpy.linalg import inv
import sys
# -----------------------------------------


# Main function
def main(args=None):
    
    #Console.banner()
    #Console.info('Running correct_images version ' + str(Console.get_version()))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    # subparser debayer
    subparser_debayer = subparsers.add_parser(
        'debayer', help='Debayer without correction')
    subparser_debayer.add_argument(
        'path', help="Path to bayer images.")
    subparser_debayer.add_argument(
        'filetype', help="type of image: raw / tif / tiff")
    subparser_debayer.add_argument(
        '-p', '--pattern', default='GRBG',
        help='Bayer pattern (GRBG for Unagi, BGGR for BioCam)')
    subparser_debayer.add_argument(
        '-i', '--image', default=None, help="Single raw image to test.")
    subparser_debayer.add_argument(
        '-o', '--output', default='.', help="Output folder.")
    subparser_debayer.set_defaults(func=call_debayer)


    # subparser correct
    subparser_correct = subparsers.add_parser(
        'correct', help='Correct images for attenuation / distortion / gamma and debayering')
    subparser_correct.add_argument(
        'path', help="Path to raw directory till dive.")
    subparser_correct.add_argument(
        '-F', '--Force', dest='force', action='store_true',
        help="Force overwrite if correction parameters already exist.")
    subparser_correct.set_defaults(
        func=call_correct)


    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        # parser.print_help(sys.stderr)
        print('No arguments')
    else:
        args = parser.parse_args()
        args.func(args)

def call_debayer(args):
   pass
def call_correct(args):
    path = Path(args.path).resolve()
    
    # resolve paths to raw, processed and config folders
    path_raw_folder = get_raw_folder(path)
    path_config_folder = get_config_folder(path)

    # resolve path to mission.yaml and correct_images.yaml
    path_mission = path_raw_folder / 'mission.yaml'
    path_correct_images = path_config_folder / 'correct_images.yaml'

    # find mission and correct_images yaml files
    if path_mission.exists():
        Console.info('path found to mission.yaml file')
    else:
        Console.quit('mission.yaml file not found in designated path')
    if path_correct_images.exists():
        Console.info('path found to correct_images.yaml file')
    else:
        root = Path(__file__).resolve().parents[1]
        default_file_path = root / 'correct_images/default_yaml/correct_images.yaml'
        default_file_path.copy(path_correct_images)
        Console.warn('Default correct_images.yaml copied to config folder')

    # load mission and correct_config parameters
    mission = Mission(path_mission)
    correct_config = CorrectConfig(path_correct_images)

    # load camera.yaml file path
    camera_yaml_path = path_config_folder / 'camera.yaml'
    print(camera_yaml_path)
    if camera_yaml_path.exists():
        Console.info('path found to camera.yaml file')
    else:
        root = Path(__file__).resolve().parents[1]
        acfr_std_camera_file = 'auv_nav/default_yaml/ts1/SSK17-01/camera.yaml'
        sx3_camera_file = 'auv_nav/default_yaml/ae2000/YK17-23C/camera.yaml'
        biocam_camera_file = 'auv_nav/default_yaml/as6/DY109/camera.yaml'

        if mission.image.format == 'acfr_standard':
            default_file_path = root / acfr_std_camera_file
        elif mission.image.format == 'seaxerocks_3':
            default_file_path = root / sx3_camera_file
        elif mission.image.format == 'biocam':
            default_file_path = root / biocam_camera_file
        print(default_file_path)
        
        default_file_path.copy(camera_yaml_path)
        Console.warn('default camera.yaml file copied to config folder')

    # instantiate the camerasystem and setup cameras from mission and config files / auv_nav
    camerasystem = CameraSystem(camera_yaml_path)

    for camera in camerasystem.cameras:
        print(camera.name)
        print('-----------------------------------------------------')

        if len(camera.image_list) == 0:
            Console.quit('No images found for the camera at the path provided...')
        else:
            corrector = Corrector(args.force, camera, correct_config, path)

            corrector.load_generic_config_parameters()
            corrector.load_camera_specific_config_parameters()
            corrector.get_imagelist()
            # corrector.get_image_properties()
            corrector.create_output_directories()
            #corrector.generate_bayer_numpyfiles(bayer_numpy_filelist)
            corrector.generate_distance_matrix()

            corrector.generate_bayer_numpy_filelist(corrector._imagelist)

            corrector.generate_correction_parameters()
            corrector.process_correction()


if __name__ == '__main__':
    main()


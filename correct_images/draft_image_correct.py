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

from auv_nav.console import Console
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder
#from correct_images.read_mission import read_params
import parameters_
import draft_camera_system
import draft_corrector

from numpy.linalg import inv

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
    # instantiate Corrector object
    corrector = Corrector()

    output_dir = Path(args.output)
    if not output_dir.exists():
        Console.info('Creating output dir {}'.format(output_dir))
        output_dir.mkdir(parents=True)
    else:
        Console.info('Using output dir {}'.format(output_dir))
    if not args.image:
        image_dir = Path(args.path)
        Console.info(
        'Debayering folder {} to {}'.format(image_dir, output_dir))
        image_list = list(image_dir.glob('*.' + args.filetype))
        Console.info('Found ' + str(len(image_list)) + ' images.')
        for image_path in image_list:
            rgb_image = corrector.debayer(image_path, args.pattern, args.filetype)
            image_name = str(image_path.stem) + '.png'
            output_image_path = Path(output_dir) / image_name
            cv2.imwrite(str(output_image_path), img_rgb)

    else:
        single_image = Path(args.image)
        rgb_image = corrector.debayer(single_image, args.pattern, args.filetype)
        image_name = str(single_image.stem) + '.png'
        output_image_path = Path(output_dir) / image_name
        cv2.imwrite(str(output_image_path), img_rgb)

def call_correct(args):
    
    path = Path(args.path).resolve()
    # resolve paths to raw, processed and config folders
    path_raw = get_raw_folder(path)
    path_processed = get_processed_folder(path)
    path_config = get_config_folder(path)
    
    # resolve paths to mission.yaml, correct_config.yaml
    path_mission = path_raw / "mission.yaml"
    path_correct_config = path_config / "correct_images.yaml"

    #print(path_raw)
    #print(path_processed)
    #print('-------------------------------------------------------------')

    # parse parameters from mission and correct_config files
    mission_parameters = parameters_.Parameters(path_mission, 'mission')
    correct_parameters = parameters_.Parameters(path_correct_config, 'correct_config')


    # instantiate camera system
    camerasystem = draft_camera_system.CameraSystem(path_raw, mission_parameters, correct_parameters)
    cameras = camerasystem.read_cameras()

    # instantiate corrector
    corrector = draft_corrector.Corrector(path_processed, correct_parameters)
    
    # correct for each camera in the camerasystem
    for camera in cameras:
        imagename_altitude_df = corrector.read_imagename_altitudes(camera)
        #print(imagename_altitude_df['Imagenumber'])
        _, bayer_file_list = corrector.write_bayer_image(camera, imagename_altitude_df)
        print('-------------------------')
        # print(bayer_file_list)

    # execute corrector
    # corrector.Execute()

if __name__ == '__main__':
    main()


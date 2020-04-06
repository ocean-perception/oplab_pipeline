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
#from correct_images.read_mission import read_params
from parameters_ import *
from draft_camera_system import *
from draft_corrector import *

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
   pass
def call_correct(args):
    
    path = Path(args.path).resolve()
    # resolve paths to raw, processed and config folders
    
    # resolve paths to mission.yaml, correct_config.yaml

    # parse parameters from mission and correct_config files


    # instantiate the camerasystem and setup cameras from mission and config files / auv_nav

    # for each camera:
    # 1. instantiate corrector object
    # 2. call corrector.process_correction()


if __name__ == '__main__':
    main()


# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""


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

from oplab import Console
from oplab import get_raw_folder
from oplab import get_processed_folder
from oplab import get_config_folder
from oplab import Mission
from oplab import CameraSystem, MonoCamera, StereoCamera

from correct_images.corrector import Corrector, load_xviii_bayer_from_binary
from correct_images.parser import CorrectConfig

from numpy.linalg import inv
import sys


# Main function
def main(args=None):

    # enable VT100 Escape Sequence for WINDOWS 10 for Console outputs
    # https://stackoverflow.com/questions/16755142/how-to-make-win32-console-recognize-ansi-vt100-escape-sequences
    os.system("")
    Console.banner()
    Console.info("Running correct_images version " + str(Console.get_version()))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # subparser debayer
    subparser_debayer = subparsers.add_parser(
        "debayer", help="Debayer without correction"
    )
    subparser_debayer.add_argument("path", help="Path to bayer images.")
    subparser_debayer.add_argument("filetype", help="type of image: raw / tif / tiff")
    subparser_debayer.add_argument(
        "-p",
        "--pattern",
        default="GRBG",
        help="Bayer pattern (GRBG for Unagi, BGGR for BioCam)",
    )
    subparser_debayer.add_argument(
        "-i", "--image", default=None, help="Single raw image to test."
    )
    subparser_debayer.add_argument("-o", "--output", default=".", help="Output folder.")
    subparser_debayer.add_argument(
        "-o_format", "--output_format", default="png", help="Output image format."
    )
    subparser_debayer.set_defaults(func=call_debayer)

    # subparser correct
    subparser_correct = subparsers.add_parser(
        "correct",
        help="Correct images for attenuation / distortion / gamma and debayering",
    )
    subparser_correct.add_argument("path", help="Path to raw directory till dive.")
    subparser_correct.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force overwrite if correction parameters already exist.",
    )
    subparser_correct.set_defaults(func=call_correct)

    # subparser parse
    subparser_parse = subparsers.add_parser(
        "parse", help="Compute the correction parameters"
    )
    subparser_parse.add_argument("path", help="Path to raw directory till dive.")
    subparser_parse.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force overwrite if correction parameters already exist.",
    )
    subparser_parse.set_defaults(func=call_parse)

    # subparser process
    subparser_process = subparsers.add_parser(
        "process", help="Compute the correction parameters"
    )
    subparser_process.add_argument("path", help="Path to raw directory till dive.")
    subparser_process.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force overwrite if correction parameters already exist.",
    )
    subparser_process.set_defaults(func=call_process)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
    else:
        args = parser.parse_args()
        args.func(args)


def call_debayer(args):
    """ performs debayer of input bayer images without going through correction pipeline

    Parameters
    -----------
    args : parse_args object
        user provided arguments for path of bayer images, filetype, and optional 
        arguments like bayer pattern, output directory, output format
    """

    def debayer_image(
        image_path, filetype, pattern, output_dir, output_format, corrector
    ):
        Console.info("Debayering image {}".format(image_path.name))
        if filetype == "raw":
            xviii_binary_data = np.fromfile(str(image_path), dtype=np.uint8)
            img = load_xviii_bayer_from_binary(xviii_binary_data, 1024, 1280)
        else:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img_rgb = corrector.debayer(img, pattern)
        img_rgb = img_rgb.astype(np.uint8)
        image_name = str(image_path.stem) + output_format
        output_image_path = Path(output_dir) / image_name
        corrector.write_output_image(img_rgb, image_name, output_dir, output_format)

    output_dir = Path(args.output)
    filetype = args.filetype
    pattern = args.pattern
    output_format = args.output_format
    image_list = []
    corrector = Corrector(True)

    if not output_dir.exists():
        Console.info("Creating output dir {}".format(output_dir))
        output_dir.mkdir(parents=True)
    else:
        Console.info("Using output dir {}".format(output_dir))
    if not args.image:
        image_dir = Path(args.path)
        Console.info("Debayering folder {} to {}".format(image_dir, output_dir))
        image_list = list(image_dir.glob("*." + args.filetype))

    else:
        single_image = Path(args.image)
        image_list.append(single_image)
    Console.info("Found " + str(len(image_list)) + " image(s)...")

    joblib.Parallel(n_jobs=-2, verbose=3)(
        [
            joblib.delayed(debayer_image)(
                image_path, filetype, pattern, output_dir, output_format, corrector
            )
            for image_path in image_list
        ]
    )


def call_parse(args):
    """ performs parsing configuration yaml files and generate image correction parameters

    Parameters
    -----------
    args : parse_args object
        user provided arguments for path of source images
    """

    correct_config, camerasystem = setup(args)
    path = Path(args.path).resolve()

    for camera in camerasystem.cameras:
        Console.info("Parsing for camera", camera.name)

        if len(camera.image_list) == 0:
            Console.info("No images found for the camera at the path provided...")
            continue
        else:
            corrector = Corrector(args.force, camera, correct_config, path)
            ret = corrector.setup()
            if ret < 0:
                Console.warn("Camera not included in correct_images.yaml...")
                continue
            else:
                corrector.generate_attenuation_correction_parameters()
        Console.info("Removing memmaps...")
        memmap_files_path = corrector.memmap_folder.glob("*.map")
        for file in memmap_files_path:
            if file.exists():
                file.unlink()
        print("-----------------------------------------------------")

    Console.info(
        "Parse completed for all cameras. Please run process to develop corrected images..."
    )
    # remove memmaps


def call_process(args):
    """ performs processing on source images using correction parameters generated in parse
    and outputs corrected images

    Parameters
    -----------
    args : parse_args object
        user provided arguments for path of source images
    """

    correct_config, camerasystem = setup(args)
    path = Path(args.path).resolve()

    for camera in camerasystem.cameras:
        Console.info("Processing for camera", camera.name)
        print("-----------------------------------------------------")

        if len(camera.image_list) == 0:
            Console.info("No images found for the camera at the path provided...")
            continue
        else:
            corrector = Corrector(args.force, camera, correct_config, path)
            corrector.load_generic_config_parameters()
            ret = corrector.load_camera_specific_config_parameters()
            if ret < 0:
                Console.warn("Camera not included in correct_images.yaml...")
                continue
            else:
                corrector.get_imagelist()

            # check if necessary folders exist in respective folders
            image_path = Path(corrector._imagelist[0]).resolve()
            image_parent_path = image_path.parents[0]
            output_dir_path = get_processed_folder(image_parent_path)
            output_dir_path = output_dir_path / "attenuation_correction"
            folder_name = "params_" + camera.name
            params_path = output_dir_path / folder_name

            if not params_path.exists():
                Console.quit("Parameters do not exist. Please run parse first...")
            else:
                filepath_attenuation_params = (
                    Path(params_path) / "attenuation_parameters.npy"
                )
                filepath_correction_gains = Path(params_path) / "correction_gains.npy"
                filepath_corrected_mean = Path(params_path) / "image_corrected_mean.npy"
                filepath_corrected_std = Path(params_path) / "image_corrected_std.npy"
                filepath_raw_mean = Path(params_path) / "image_raw_mean.npy"
                filepath_raw_std = Path(params_path) / "image_raw_std.npy"

                # read in image numpy files
                folder_name = "bayer_" + camera.name
                dir_ = Path(output_dir_path) / folder_name
                corrector.bayer_numpy_filelist = list(dir_.glob("*.npy"))

                # read in distance matrix numpy files
                folder_name = "distance_" + camera.name
                dir_ = Path(params_path) / folder_name
                corrector.distance_matrix_numpy_filelist = list(dir_.glob("*.npy"))

                # read parameters from disk
                if filepath_attenuation_params.exists():
                    corrector.image_attenuation_parameters = np.load(
                        filepath_attenuation_params
                    )
                if filepath_correction_gains.exists():
                    corrector.correction_gains = np.load(filepath_correction_gains)
                if filepath_corrected_mean.exists():
                    corrector.image_corrected_mean = np.load(filepath_corrected_mean)
                if filepath_corrected_std.exists():
                    corrector.image_corrected_std = np.load(filepath_corrected_std)
                if filepath_raw_mean.exists():
                    corrector.image_raw_mean = np.load(filepath_raw_mean)
                if filepath_raw_std.exists():
                    corrector.image_raw_std = np.load(filepath_raw_std)

                # check if images exist already
                folder_name = "developed_" + camera.name
                output_path = Path(output_dir_path) / folder_name
                filelist = list(output_path.glob("*.*"))
                if len(filelist) > 0:
                    if not args.force:
                        Console.quit("Overwrite images with process -F ...")

                # invoke process function
                corrector.output_images_folder = output_path
                corrector.process_correction()

    Console.info("Process completed for all cameras...")


def call_correct(args):
    """ performs parse and process in one go. can be used for small datasets 

    Parameters
    -----------
    args : parse_args object
        user provided arguments for path of source images
    """

    correct_config, camerasystem = setup(args)
    path = Path(args.path).resolve()

    for camera in camerasystem.cameras:
        Console.info("Processing camera:", camera.name)

        if len(camera.image_list) == 0:
            Console.info("No images found for the camera at the path provided")
            continue
        else:
            corrector = Corrector(args.force, camera, correct_config, path)
            corrector.setup()
            corrector.generate_attenuation_correction_parameters()
            corrector.process_correction()


def setup(args):
    """ generates correct_config and camerasystem objects from input config yaml files
    
    Parameters
    -----------
    args : parse_args object
        user provided arguments for path of source images
    """

    path = Path(args.path).resolve()

    # resolve paths to raw, processed and config folders
    path_raw_folder = get_raw_folder(path)
    path_config_folder = get_config_folder(path)

    # resolve path to mission.yaml
    path_mission = path_raw_folder / "mission.yaml"

    # find mission and correct_images yaml files
    if path_mission.exists():
        Console.info("File mission.yaml found at", path_mission)
    else:
        Console.quit("File mission.yaml file not found at", path_raw_folder)

    # load mission parameters
    mission = Mission(path_mission)

    # find out default yaml paths
    root = Path(__file__).resolve().parents[1]

    acfr_std_camera_file = "auv_nav/default_yaml/ts1/SSK17-01/camera.yaml"
    sx3_camera_file = "auv_nav/default_yaml/ae2000/YK17-23C/camera.yaml"
    biocam_camera_file = "auv_nav/default_yaml/as6/DY109/camera.yaml"

    acfr_std_correct_config_file = (
        "correct_images/default_yaml/acfr/correct_images.yaml"
    )
    sx3_std_correct_config_file = "correct_images/default_yaml/sx3/correct_images.yaml"
    biocam_std_correct_config_file = (
        "correct_images/default_yaml/biocam/correct_images.yaml"
    )

    if mission.image.format == "acfr_standard":
        camera_yaml_path = root / acfr_std_camera_file
        default_file_path_correct_config = root / acfr_std_correct_config_file
    elif mission.image.format == "seaxerocks_3":
        camera_yaml_path = root / sx3_camera_file
        default_file_path_correct_config = root / sx3_std_correct_config_file
    elif mission.image.format == "biocam":
        camera_yaml_path = root / biocam_camera_file
        default_file_path_correct_config = root / biocam_std_correct_config_file
    else:
        Console.warn(
            "Image system not recognised. Looking for camera.yaml in " "raw folder..."
        )
        camera_yaml_path = path_raw_folder / "camera.yaml"
        if not camera_yaml_path.exists():
            Console.quit(
                "camera.yaml file for new Image system not found within"
                " raw folder..."
            )
        else:
            Console.info("camera.yaml found for new imaging system...")

    # instantiate the camerasystem and setup cameras from mission and config files / auv_nav
    camerasystem = CameraSystem(camera_yaml_path, path_raw_folder)
    if camerasystem.camera_system != mission.image.format:
        Console.quit(
            "Image system not found in camera.yaml file. The mission "
            "you are trying to process does not seem to have a camera"
        )

    # check for correct_config yaml path
    path_correct_images = path_config_folder / "correct_images.yaml"
    if path_correct_images.exists():
        Console.info(
            "Configuration file correct_images.yaml file found at", path_correct_images
        )
    else:
        default_file_path_correct_config.copy(path_correct_images)
        Console.warn(
            "Configuration file not found, copying a default one at",
            path_correct_images,
        )

    # load parameters from correct_config.yaml
    correct_config = CorrectConfig(path_correct_images)
    return correct_config, camerasystem


if __name__ == "__main__":
    main()

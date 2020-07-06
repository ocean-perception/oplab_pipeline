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

from correct_images.corrector import Corrector, load_xviii_bayer_from_binary, trim_csv_files, get_imagename_list
from correct_images.parser import CorrectConfig

from numpy.linalg import inv
import sys

from cv2 import imread, imwrite
from PIL import Image

import math

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
        "process", help="Process image correction"
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


    # subparser rescale image
    subparser_rescale = subparsers.add_parser(
        "rescale", help="Rescale processed images"
    )
    subparser_rescale.add_argument("path", help="Path to image directory")
    subparser_rescale.add_argument("distance_path", help="Path to distance file: either an auv_ekf_<cameraname>.csv or a distance list")
    subparser_rescale.add_argument("hor_open_angle", help="horizontal opening angle for the camera")
    subparser_rescale.add_argument("vert_open_angle", help="vertical opening angle for the camera")
    subparser_rescale.add_argument("target_pixel_size", help="output resolution")
    subparser_rescale.add_argument("output_directory", help="path to output directory")
    subparser_rescale.add_argument("-f", "--filelist", default=None, help="Path to list of images to rescale")
    subparser_rescale.add_argument("-mp", "--maintain_pixels", default="N", help="maintain original number of pixels : Y (Yes) / N (No)")
    subparser_rescale.set_defaults(func=call_rescale)


    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)

    elif len(sys.argv) == 2 and not sys.argv[1] == "-h":
        args = parser.parse_args(["correct", sys.argv[1]])
        print(args)

        if hasattr(args, "func"):
            args.func(args)


    else:
        args = parser.parse_args()
        args.func(args)


def call_debayer(args):
    """Perform debayer of input bayer images without going through correction pipeline

    Parameters
    -----------
    args : parse_args object
        User provided arguments for path of bayer images, filetype, and optional 
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
        image_name = str(image_path.stem)
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
    """Perform parsing of configuration yaml files and generate image correction parameters

    Parameters
    -----------
    args : parse_args object
        User provided arguments for path of source images
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
            ret = corrector.setup('parse')
            if ret < 0:
                Console.warn("Camera not included in correct_images.yaml...")
                continue
            else:
                if corrector.correction_method == "colour_correction":
                    corrector.generate_attenuation_correction_parameters()
                elif corrector.correction_method == "manual_balance":
                    Console.info('run process for manual_balance...')
                    continue

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
    """Perform processing on source images using correction parameters generated in parse
    and outputs corrected images

    Parameters
    -----------
    args : parse_args object
        User provided arguments for path of source images
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
            ret = corrector.setup('process')
            if ret < 0:
                Console.warn("Camera not included in correct_images.yaml...")
                continue
            else:
                if corrector.correction_method == "colour_correction":
                    filepath_attenuation_params = Path(corrector.attenuation_parameters_folder) / "attenuation_parameters.npy"
                    filepath_correction_gains = Path(corrector.attenuation_parameters_folder) / "correction_gains.npy"
                    filepath_corrected_mean = Path(corrector.attenuation_parameters_folder) / "image_corrected_mean.npy"
                    filepath_corrected_std = Path(corrector.attenuation_parameters_folder) / "image_corrected_std.npy"
                    filepath_raw_mean = Path(corrector.attenuation_parameters_folder) / "image_raw_mean.npy"
                    filepath_raw_std = Path(corrector.attenuation_parameters_folder) / "image_raw_std.npy"

                    # read parameters from disk
                    if filepath_attenuation_params.exists():
                        corrector.image_attenuation_parameters = np.load(
                            filepath_attenuation_params
                        )
                    else:
                        if corrector.distance_metric == "altitude" or corrector.distance_metric == "depth_map":
                            Console.quit("Code does not find attenuation_parameters.npy...Please run parse before process...")
                    if filepath_correction_gains.exists():
                        corrector.correction_gains = np.load(filepath_correction_gains)
                    else:
                        if corrector.distance_metric == "altitude" or corrector.distance_metric == "depth_map":
                            Console.quit("Code does not find correction_gains.npy...Please run parse before process...")
                    if filepath_corrected_mean.exists():
                        corrector.image_corrected_mean = np.load(filepath_corrected_mean)
                    else:
                        if corrector.distance_metric == "altitude" or corrector.distance_metric == "depth_map":
                            Console.quit("Code does not find image_corrected_mean.npy...Please run parse before process...")
                    if filepath_corrected_std.exists():
                        corrector.image_corrected_std = np.load(filepath_corrected_std)
                    else:
                        if corrector.distance_metric == "altitude" or corrector.distance_metric == "depth_map":
                            Console.quit("Code does not find image_corrected_std.npy...Please run parse before process...")
                    if filepath_raw_mean.exists():
                        corrector.image_raw_mean = np.load(filepath_raw_mean)
                    else:
                        if corrector.distance_metric == "altitude" or corrector.distance_metric == "depth_map" or corrector.distance_metric == "none":
                            Console.quit("Code does not find image_raw_mean.npy...Please run parse before process...")
                    if filepath_raw_std.exists():
                        corrector.image_raw_std = np.load(filepath_raw_std)
                    else:
                        if corrector.distance_metric == "altitude" or corrector.distance_metric == "depth_map" or corrector.distance_metric == "none":
                            Console.quit("Code does not find image_raw_std.npy...Please run parse before process...")
                    Console.info('Correction parameters loaded...')
                    Console.info('Running process for colour correction...')
                else:
                    Console.info('Running process with manual colour balancing...')

                corrector.process_correction()

    Console.info("Process completed for all cameras...")


def call_correct(args):
    """Perform parse and process in one go. Can be used for small datasets 

    Parameters
    -----------
    args : parse_args object
        User provided arguments for path of source images
    """
    call_parse(args)
    call_process(args)



def call_rescale(args):
    image_directory = args.path
    distance_path = args.distance_path
    hor_open_angle = float(args.hor_open_angle)
    vert_open_angle = float(args.vert_open_angle)
    target_pixel_size = float(args.target_pixel_size)
    output_directory = args.output_directory


    filelist = args.filelist
    maintain_pixels = args.maintain_pixels


    if not filelist is None:
        trim_csv_file_path = Path(image_directory) / "trimmed_csv.csv"
        trim_csv_files(Path(filelist), Path(distance_path), trim_csv_file_path)
        dataframe = pd.read_csv(trim_csv_file_path)
        imagepath_list = dataframe["relative_path"]
        imagenames_list = [Path(image).name for image in imagepath_list]
    else:
        dataframe = pd.read_csv(Path(distance_path))
        imagenames_list = [
            filename
            for filename in os.listdir(image_directory)
                if filename[-4:] == ".jpg" or filename[-4:] == ".png" or filename[-4:] == ".tif"
        ]
    Console.info("Distance values loaded...")
    rescale_images(imagenames_list, image_directory, target_pixel_size, dataframe, output_directory, hor_open_angle, vert_open_angle, maintain_pixels)
    Console.info("Rescaling completed ...")


   
    
def rescale_image(image_path, target_pixel_size, altitude, horizontal_opening_angle, vertical_opening_angle, maintain_pixels):
    image, image_width, image_height = get_image_and_info(image_path)
    pixel_height = get_pixel_size(altitude, image_height, vertical_opening_angle)
    pixel_width = get_pixel_size(altitude, image_width, horizontal_opening_angle)
    
    vertical_rescale = pixel_height / target_pixel_size
    horizontal_rescale = pixel_width / target_pixel_size

    if maintain_pixels == "N" or maintain_pixels == "No":
        size = (int(image_width * horizontal_rescale), int(image_height * vertical_rescale))
        image = Image.fromarray(image.astype("uint8"), "RGB")
        image = image.resize(size, Image.BICUBIC)

    elif maintain_pixels == "Y" or maintain_pixels == "Yes":

        if vertical_rescale < 1 or horizontal_rescale < 1:
            
            step_width = int(1/horizontal_rescale)
            step_height = int(1/vertical_rescale)
            for row in range(0, image_height, step_height):
                for col in range(0, image_width, step_width):
                    image[row:(row+step_height-1), col:(col+step_width-1)] = image[row, col]
            image = Image.fromarray(image.astype("uint8"), "RGB")

        else:
            crop_width = int((1/horizontal_rescale) * image_width)
            crop_height = int((1/vertical_rescale) * image_height)

            # find crop box dimensions
            box_left = int((image_width - crop_width) / 2)
            box_upper = int((image_height - crop_height) / 2)
            box_right = image_width - box_left
            box_lower = image_height - box_upper

            # crop the image to the center
            box = (box_left, box_upper, box_right, box_lower)
            image = Image.fromarray(image.astype("uint8"), "RGB")
            cropped_image = image.crop(box)

            # resize the cropped image to the size of original image
            size = (image_width, image_height)
            image = cropped_image.resize(size, Image.BICUBIC)

    return image



def get_image_and_info(image_path):
    image = imageio.imread(image_path)
    image_shape = image.shape
    return image, image_shape[1], image_shape[0]

# uses given opening angle of camera and the altitude parameter to determine the pixel size
# give width & horizontal, or height & vertical
def get_pixel_size(altitude, image_size, opening_angle):
    image_spatial_size = (
        2 * float(altitude) * float(math.tan(math.radians(opening_angle / 2)))
    )
    pixel_size = image_spatial_size / image_size
    return pixel_size


def rescale_images(
    imagenames_list, image_directory, target_pixel_size, dataframe, output_directory, horizontal_opening_angle, vertical_opening_angle,
    maintain_pixels
):
    Console.info("Rescaling images...")

    for idx in trange(len(imagenames_list)):
        image_name = imagenames_list[idx]
        source_image_path = Path(image_directory) / image_name
        output_image_path = Path(output_directory) / image_name
        image_path_list = dataframe["relative_path"]
        trimmed_path_list = [path 
                            for path in image_path_list
                            if Path(path).name == image_name ]
        trimmed_dataframe = dataframe.loc[dataframe["relative_path"].isin(trimmed_path_list) ]
        altitude = trimmed_dataframe["altitude [m]"]
        rescaled_image = rescale_image(
                source_image_path, target_pixel_size, altitude, horizontal_opening_angle, vertical_opening_angle, maintain_pixels
            )
        imageio.imwrite(output_image_path, rescaled_image, format="PNG-FI")



def setup(args):
    """Generate correct_config and camerasystem objects from input config yaml files
    
    Parameters
    -----------
    args : parse_args object
        User provided arguments for path of source images
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

    # resolve path to camera.yaml file
    temp_path = path_raw_folder / 'camera.yaml'

    if not temp_path.exists():
        Console.info('Not found camera.yaml file in /raw folder...Using default camera.yaml file...')
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
            Console.quit(
                "Image system in camera.yaml does not match with mission.yaml... Provide correct camera.yaml in /raw folder..."
            )
    else:
        Console.info('Found camera.yaml file in /raw folder...')
        camera_yaml_path = temp_path


    # instantiate the camerasystem and setup cameras from mission and config files / auv_nav
    camerasystem = CameraSystem(camera_yaml_path, path_raw_folder)
    if camerasystem.camera_system != mission.image.format:
        Console.quit(
            "Image system in camera.yaml does not match with mission.yaml...Provide correct camera.yaml in /raw folder..."
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

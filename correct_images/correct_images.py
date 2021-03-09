# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""


import argparse
import os
import sys
import time
from pathlib import Path

import imageio
from oplab import (
    CameraSystem,
    Console,
    Mission,
    get_config_folder,
    get_raw_folder,
    get_processed_folder,
)

from correct_images import corrections
from correct_images.corrector import Corrector
from correct_images.parser import CorrectConfig


# Main function
def main(args=None):
    # enable VT100 Escape Sequence for WINDOWS 10 for Console outputs
    # https://stackoverflow.com/questions/16755142/how-to-make-win32-console-recognize-ansi-vt100-escape-sequences # noqa
    os.system("")
    Console.banner()
    Console.info(
        "Running correct_images version " + str(Console.get_version())
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # subparser debayer
    subparser_debayer = subparsers.add_parser(
        "debayer", help="Debayer without correction"
    )
    subparser_debayer.add_argument("path", help="Path to bayer images.")
    subparser_debayer.add_argument(
        "filetype", help="type of image: raw / tif / tiff"
    )
    subparser_debayer.add_argument(
        "-p",
        "--pattern",
        default="GRBG",
        help="Bayer pattern (GRBG for Unagi and SeaXerocks3, BGGR for BioCam)",
    )
    subparser_debayer.add_argument(
        "-i", "--image", default=None, help="Single raw image to test."
    )
    subparser_debayer.add_argument(
        "-o", "--output", default=".", help="Output folder."
    )
    subparser_debayer.add_argument(
        "-o_format",
        "--output_format",
        default="png",
        help="Output image format.",
    )
    subparser_debayer.set_defaults(func=call_debayer)

    # subparser correct
    subparser_correct = subparsers.add_parser(
        "correct",
        help="Correct images for attenuation / distortion / gamma and \
            debayering",
    )
    subparser_correct.add_argument(
        "path", help="Path to raw directory till dive."
    )
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
    subparser_parse.add_argument(
        "path", help="Path to raw directory till dive."
    )
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
    subparser_process.add_argument(
        "path", help="Path to raw directory till dive."
    )
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
    subparser_rescale.add_argument("path", help="Path to raw folder")
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


def call_parse(args):
    """Perform parsing of configuration yaml files and generate image
    correction parameters

    Parameters
    -----------
    args : parse_args object
        User provided arguments for path of source images
    """

    path = Path(args.path).resolve()

    Console.set_logging_file(
        get_processed_folder(path)
        / ('log/' + str(time.time()) + "_correct_images_parse.log")
    )

    correct_config, camerasystem = load_configuration_and_camera_system(path)

    for camera in camerasystem.cameras:
        Console.info("Parsing for camera", camera.name)

        if len(camera.image_list) == 0:
            Console.info(
                "No images found for the camera at the path provided..."
            )
            continue
        else:
            corrector = Corrector(args.force, camera, correct_config, path)
            if corrector.camera_found:
                corrector.parse()

    Console.info(
        "Parse completed for all cameras. Please run process to develop \
        corrected images..."
    )


def call_process(args):
    """Perform processing on source images using correction parameters
    generated in parse and outputs corrected images

    Parameters
    -----------
    args : parse_args object
        User provided arguments for path of source images
    """

    path = Path(args.path).resolve()

    Console.set_logging_file(
        get_processed_folder(path)
        / ('log/' + str(time.time()) + "_correct_images_process.log")
    )

    correct_config, camerasystem = load_configuration_and_camera_system(path)

    for camera in camerasystem.cameras:
        Console.info("Processing for camera", camera.name)

        if len(camera.image_list) == 0:
            Console.info(
                "No images found for the camera at the path provided..."
            )
            continue
        else:
            corrector = Corrector(args.force, camera, correct_config, path)
            if corrector.camera_found:
                corrector.process()
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
    path = Path(args.path).resolve()

    Console.set_logging_file(
        get_processed_folder(path)
        / ('log/' + str(time.time()) + "_correct_images_rescale.log")
    )

    correct_config, camerasystem = load_configuration_and_camera_system(path)

    # install freeimage plugins if not installed
    imageio.plugins.freeimage.download()

    # obtain parameters for rescale from correct_config
    rescale_cameras = correct_config.camerarescale.rescale_cameras

    for camera in rescale_cameras:
        corrections.rescale_camera(path, camerasystem, camera)
    Console.info("Rescaling completed for all cameras ...")


def call_debayer(args):
    image_dir = args.path
    filetype = args.filetype
    pattern = args.pattern
    image = args.image
    output_dir = args.output
    output_format = args.output_format
    corrections.debayer_folder(
        output_dir, filetype, pattern, output_format, image, image_dir
    )
    Console.info("Debayering finished")


def load_configuration_and_camera_system(path):
    """Generate correct_config and camera system objects from input config
    yaml files

    Parameters
    -----------
    path : Path
        User provided Path of source images
    """

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
    temp_path = path_raw_folder / "camera.yaml"

    default_file_path_correct_config = None
    camera_yaml_path = None

    if not temp_path.exists():
        Console.info(
            "Not found camera.yaml file in /raw folder...Using default \
            camera.yaml file..."
        )
        # find out default yaml paths
        root = Path(__file__).resolve().parents[1]

        acfr_std_camera_file = "auv_nav/default_yaml/ts1/SSK17-01/camera.yaml"
        sx3_camera_file = "auv_nav/default_yaml/ae2000/YK17-23C/camera.yaml"
        biocam_camera_file = "auv_nav/default_yaml/as6/DY109/camera.yaml"
        hybis_camera_file = "auv_nav/default_yaml/hybis/camera.yaml"

        acfr_std_correct_config_file = (
            "correct_images/default_yaml/acfr/correct_images.yaml"
        )
        sx3_std_correct_config_file = (
            "correct_images/default_yaml/sx3/correct_images.yaml"
        )
        biocam_std_correct_config_file = (
            "correct_images/default_yaml/biocam/correct_images.yaml"
        )
        hybis_std_correct_config_file = (
            "correct_images/default_yaml/hybis/correct_images.yaml"
        )

        if mission.image.format == "acfr_standard":
            camera_yaml_path = root / acfr_std_camera_file
            default_file_path_correct_config = (
                root / acfr_std_correct_config_file
            )
        elif mission.image.format == "seaxerocks_3":
            camera_yaml_path = root / sx3_camera_file
            default_file_path_correct_config = (
                root / sx3_std_correct_config_file
            )
        elif mission.image.format == "biocam":
            camera_yaml_path = root / biocam_camera_file
            default_file_path_correct_config = (
                root / biocam_std_correct_config_file
            )
        elif mission.image.format == "hybis":
            camera_yaml_path = root / hybis_camera_file
            default_file_path_correct_config = (
                root / hybis_std_correct_config_file
            )
        else:
            Console.quit(
                "Image system in camera.yaml does not match with mission.yaml",
                "Provide correct camera.yaml in /raw folder... ",
            )
    else:
        Console.info("Found camera.yaml file in /raw folder...")
        camera_yaml_path = temp_path

    # instantiate the camera system and setup cameras from mission and
    # config files / auv_nav
    camera_system = CameraSystem(camera_yaml_path, path_raw_folder)
    if camera_system.camera_system != mission.image.format:
        Console.quit(
            "Image system in camera.yaml does not match with mission.yaml...",
            "Provide correct camera.yaml in /raw folder...",
        )

    # check for correct_config yaml path
    path_correct_images = path_config_folder / "correct_images.yaml"
    if path_correct_images.exists():
        Console.info(
            "Configuration file correct_images.yaml file found at",
            path_correct_images,
        )
    else:
        default_file_path_correct_config.copy(path_correct_images)
        Console.warn(
            "Configuration file not found, copying a default one at",
            path_correct_images,
        )

    # load parameters from correct_config.yaml
    correct_config = CorrectConfig(path_correct_images)
    return correct_config, camera_system


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""


import argparse
import os
import string
import sys
import time
from pathlib import Path

import imageio

from correct_images import corrections
from correct_images.corrector import Corrector
from correct_images.parser import CorrectConfig
from oplab import (
    CameraSystem,
    Console,
    Mission,
    get_config_folder,
    get_processed_folder,
    get_raw_folder,
)


# Main function
def main(args=None):
    # enable VT100 Escape Sequence for WINDOWS 10 for Console outputs
    # https://stackoverflow.com/questions/16755142/how-to-make-win32-console-recognize-ansi-vt100-escape-sequences # noqa
    os.system("")
    Console.banner()
    Console.info("Running correct_images version " + str(Console.get_version()))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # subparser correct
    subparser_correct = subparsers.add_parser(
        "correct",
        help="Correct images for attenuation / distortion / gamma and debayering",  # noqa
    )
    subparser_correct.add_argument("path", help="Path to raw directory till dive.")
    subparser_correct.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force overwrite if correction parameters already exist.",
    )
    subparser_correct.add_argument(
        "--suffix",
        dest="suffix",
        default="",
        help="Expected suffix for correct_images configuration and output folders.",
    )
    subparser_correct.set_defaults(func=call_correct)

    # subparser parse
    subparser_parse = subparsers.add_parser(
        "parse", help="Compute the correction parameters"
    )

    #   subparser_parse.add_argument("path", help="Path to raw directory till dive.")

    subparser_parse.add_argument(
        "path",
        #        default=".",
        nargs="+",
        help="Folderpath where the (raw) input data is. Needs to be a \
        subfolder of 'raw' and contain the mission.yaml configuration file.",
    )

    subparser_parse.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force overwrite if correction parameters already exist.",
    )
    subparser_parse.add_argument(
        "--suffix",
        dest="suffix",
        default="",
        help="Expected suffix for correct_images configuration and output folders.",
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
    subparser_process.add_argument(
        "--suffix",
        dest="suffix",
        default="",
        help="Expected suffix for correct_images configuration and output folders.",
    )
    subparser_process.set_defaults(func=call_process)

    # subparser rescale image
    subparser_rescale = subparsers.add_parser(
        "rescale", help="Rescale processed images"
    )
    subparser_rescale.add_argument("path", help="Path to raw folder")
    subparser_rescale.add_argument(
        "--suffix",
        dest="suffix",
        default="",
        help="Expected suffix for correct_images configuration and output folders.",
    )
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

        # Check suffix is only text, digits, dash and underscores
        allowed_chars = string.ascii_letters + "-" + "_" + string.digits
        if all([c in allowed_chars for c in args.suffix]):
            args.func(args)
        else:
            Console.error(
                "Suffix must only contain letters, digits, dash and underscores"
            )


def call_parse(args):
    """Perform parsing of configuration yaml files and generate image
    correction parameters

    Parameters
    -----------
    args : parse_args object
        User provided arguments for path of source images
    """

    # Now args.path is a list of paths (str / os.PathLike objects)
    path_list = [Path(path).resolve() for path in args.path]
    if len(path_list) == 1:
        path = path_list[0]
        Console.info("Single path provided, normal single dive mode...")
    else:
        Console.info(
            "Multiple paths provided [{}]. Checking each path...".format(len(path_list))
        )
        for path in path_list:
            # chec if path is valid
            if not path.exists():
                Console.error("Path", path, "does not exist! Exiting...")  # quit
                sys.exit(1)
            else:
                Console.info("\t", path, " [OK]")

    # Populating the configuration and camerasystem lists for each dive path
    # The camera system is pulled first from <config> folder if available, if not from <raw> folder
    correct_config_list, camerasystem_list = zip(
        *[load_configuration_and_camera_system(path, args.suffix) for path in path_list]
    )
    # correct_config <--- from correct_images.yaml  (<config> folder)
    # camerasystem   <--- from camera.yaml          (<config> folder or from <raw> folder)

    # Let's check that both lists have the same length and are not empty (same number of dives)
    if len(correct_config_list) != len(camerasystem_list):
        Console.error("Number of [camerasystem] and [configuration] differ!")
        sys.exit(1)
    if len(correct_config_list) == 0:
        Console.error("No valid camerasystem/configuration found!")
        sys.exit(1)

    # When in multidive mode, check if all camerasystem are the same. For this we test camera_system.camera_system
    if (
        len(camerasystem_list) > 1
    ):  # this test is still valid for single dive mode, so we could remove this [if] sentence
        camera_system = camerasystem_list[0]
        for cs in camerasystem_list:
            # the first entry will be repeated, no problem with that
            # TODO: Extend is_equivalent() method allowing checking cameras in different orders
            # WARNING: We decide not to use equivalent() here, because it is not robust enough. Enforce same camera order
            if not camera_system.camera_system == cs.camera_system:
                Console.error("Camera systems differ!")
                Console.error(
                    "\tFirst camera system (reference) ", camera_system.camera_system
                )
                Console.error("\tWrong camera system (current)   ", cs.camera_system)
                sys.exit(1)
        Console.warn("Camera systems are the same for all dives.")  # so far so good

    # Check if the correct_config_lists elements are the same (equivalent)
    if len(correct_config_list) > 1:
        correct_config = correct_config_list[0]
        for cc in correct_config_list:
            # Check if the relevant fields of the configuration are the same (including per-dive camera setup)
            if not correct_config.is_equivalent(cc):
                Console.quit("Configurations [correct_config] do not match!")
        Console.warn("Configurations are equivalent for all dives.")
    else:
        # only one element in the list, copy it for single dive mode (this could be moved outside)
        correct_config = correct_config_list[0]

    camerasystem = camerasystem_list[0]
    for camera in camerasystem.cameras:
        # check if the camera also exists in the configuration
        if camera.name not in [
            c.camera_name for c in correct_config.configs.camera_configs
        ]:  # ignore if not present
            Console.warn(
                "Camera [",
                camera.name,
                "] defined in <camera.yaml> but not found in configuration. Skipping...",
            )
        else:
            Console.info("Parsing for camera", camera.name)
            # Create a Corrector object for each camera with empty configuration
            # The configuration and the paths will be populated later on a per-dive basis
            corrector = Corrector(args.force, args.suffix, camera, correct_config=None)
            # call new list-compatible implementation of parse()
            corrector.parse(path_list, correct_config_list)

    Console.info(
        "Parse completed for all cameras. Please run process to develop ",
        "corrected images...",
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

    time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    Console.set_logging_file(
        get_processed_folder(path)
        / ("log/" + time_string + "_correct_images_process.log")
    )

    correct_config, camerasystem = load_configuration_and_camera_system(
        path, args.suffix
    )

    for camera in camerasystem.cameras:
        Console.info("Processing for camera", camera.name)

        if len(camera.image_list) == 0:
            Console.info("No images found for the camera at the path provided...")
            continue
        else:
            corrector = Corrector(args.force, args.suffix, camera, correct_config, path)
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

    time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    Console.set_logging_file(
        get_processed_folder(path)
        / ("log/" + time_string + "_correct_images_rescale.log")
    )

    correct_config, camerasystem = load_configuration_and_camera_system(
        path, args.suffix
    )

    # install freeimage plugins if not installed
    imageio.plugins.freeimage.download()

    if correct_config.camerarescale is None:
        Console.error("Camera rescale configuration not found")
        Console.error(
            "Please populate the correct_images.yaml file with a rescale configuration"
        )
        Console.quit("Malformed correct_images.yaml file")

    # obtain parameters for rescale from correct_config
    rescale_cameras = correct_config.camerarescale.rescale_cameras

    for camera in rescale_cameras:
        corrections.rescale_camera(path, camerasystem, camera)
    Console.info("Rescaling completed for all cameras ...")


def load_configuration_and_camera_system(path, suffix=None):
    """Generate correct_config and camera system objects from input config
    yaml files

    Parameters
    -----------
    path : Path
        User provided Path of source images
    """

    if suffix is not None:
        Console.info("Parsing with suffix:", suffix)

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
    camera_yaml_raw_path = path_raw_folder / "camera.yaml"
    camera_yaml_config_path = path_config_folder / "camera.yaml"

    camera_yaml_path = None
    default_file_path_correct_config = None

    if not camera_yaml_raw_path.exists() and not camera_yaml_config_path.exists():
        Console.info(
            "camera.yaml file not found neither in /raw nor in /config folder.",
            "Using default camera.yaml file for image format",
            mission.image.format,
        )
        # find out default yaml paths
        root = Path(__file__).resolve().parents[1]

        acfr_std_camera_file = "auv_nav/default_yaml/ts1/SSK17-01/camera.yaml"
        sx3_camera_file = "auv_nav/default_yaml/ae2000/YK17-23C/camera.yaml"
        biocam_camera_file = "auv_nav/default_yaml/as6/DY109/camera.yaml"
        biocam4000_15c_camera_file = "auv_nav/default_yaml/alr/jc220/camera.yaml"
        hybis_camera_file = "auv_nav/default_yaml/hybis/camera.yaml"
        ntnu_camera_file = "auv_nav/default_yaml/ntnu_stereo/tautra21/camera.yaml"
        rosbag_camera_file = "auv_nav/default_yaml/rosbag/grassmap/camera.yaml"

        acfr_std_correct_config_file = (
            "correct_images/default_yaml/acfr/correct_images.yaml"
        )
        sx3_std_correct_config_file = (
            "correct_images/default_yaml/sx3/correct_images.yaml"
        )
        biocam_std_correct_config_file = (
            "correct_images/default_yaml/biocam/correct_images.yaml"
        )
        biocam4000_15c_std_correct_config_file = (
            "correct_images/default_yaml/biocam4000_15c/correct_images.yaml"
        )
        hybis_std_correct_config_file = (
            "correct_images/default_yaml/hybis/correct_images.yaml"
        )
        ntnu_std_correct_config_file = (
            "correct_images/default_yaml/ntnu_stereo/correct_images.yaml"
        )
        rosbag_std_correct_config_file = (
            "correct_images/default_yaml/rosbag/correct_images.yaml"
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
        elif mission.image.format == "biocam4000_15c":
            camera_yaml_path = root / biocam4000_15c_camera_file
            default_file_path_correct_config = (
                root / biocam4000_15c_std_correct_config_file
            )
        elif mission.image.format == "hybis":
            camera_yaml_path = root / hybis_camera_file
            default_file_path_correct_config = root / hybis_std_correct_config_file
        elif mission.image.format == "ntnu_stereo":
            camera_yaml_path = root / ntnu_camera_file
            default_file_path_correct_config = root / ntnu_std_correct_config_file
        elif mission.image.format == "rosbag":
            camera_yaml_path = root / rosbag_camera_file
            default_file_path_correct_config = root / rosbag_std_correct_config_file
            camera_yaml_path.copy(camera_yaml_config_path)
            Console.info("Copied camera.yaml file to config folder. Please edit it.")
            Console.info("The file is located at", camera_yaml_config_path)
        else:
            Console.quit(
                "There is currently no default camera file for image format",
                mission.image.format,
                ". Please provide the camera.yaml file in the /config folder and",
                "rerun correct_images.",
            )
    elif camera_yaml_raw_path.exists() and not camera_yaml_config_path.exists():
        Console.info("Found camera.yaml file in /raw folder")
        camera_yaml_path = camera_yaml_raw_path
    elif not camera_yaml_raw_path.exists() and camera_yaml_config_path.exists():
        Console.info("Found camera.yaml file in /config folder")
        camera_yaml_path = camera_yaml_config_path
    elif camera_yaml_raw_path.exists() and camera_yaml_config_path.exists():
        Console.info("Found camera.yaml both in /raw and in /config folder")
        Console.info("Using camera.yaml from /config folder")
        camera_yaml_path = camera_yaml_config_path
    else:
        Console.quit(
            "rosbag image type requires a camera.yaml file in /config folder",
            "Please provide camera.yaml file in /config folder",
        )

    # instantiate the camera system and setup cameras from mission and
    # config files / auv_nav
    camera_system = CameraSystem(camera_yaml_path, path_raw_folder)
    if camera_system.camera_system != mission.image.format:
        Console.quit(
            "Image system in camera.yaml does not match with mission.yaml",
            "Provide correct camera.yaml in /raw folder",
        )
    # check for correct_config yaml path
    path_correct_images = None
    if suffix == "" or suffix is None:
        path_correct_images = path_config_folder / "correct_images.yaml"
    else:
        path_correct_images = path_config_folder / (
            "correct_images_" + suffix + ".yaml"
        )
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

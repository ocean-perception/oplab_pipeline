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

from auv_cal.calibration import Calibrator
from oplab import Console, get_processed_folder


def main(args=None):
    """
    Main entry point for project.
    Args:
        args : list of arguments as if they were input in the command line.
               Leave it None to use sys.argv.
    This notation makes it possible to call the module from the command line as
    well as from a different python module. When called from the command line
    args defaults to None, and parse_args() defaults to using sys.argv[1:].
    When called from a python script or module, pass the arguments as list,
    e.g. main(["parse", "-h"]). This will populate the args parameter.
    """

    # enable VT100 Escape Sequence for WINDOWS 10 for Console outputs
    # https://stackoverflow.com/questions/16755142/how-to-make-win32-console-recognize-ansi-vt100-escape-sequences # noqa
    os.system("")
    Console.banner()
    Console.info("Running auv_cal version " + str(Console.get_version()))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser_mono = subparsers.add_parser(
        "mono", help="Monocular camera calibration using OpenCV."
    )
    subparser_mono.add_argument(
        "path", default=".", help="Folder containing the mission.yaml"
    )
    subparser_mono.add_argument(
        "-F",
        dest="force",
        action="store_true",
        help="Force output file overwite",
    )
    subparser_mono.add_argument(
        "-FF",
        dest="force2",
        action="store_true",
        help="Regenerates and overwrittes all files, including intermediate \
             results",
    )
    subparser_mono.add_argument(
        "-s", "--suffix", default="", help="Suffix added to mono calibration file name."
    )
    subparser_mono.set_defaults(func=call_calibrate_mono)

    subparser_stereo = subparsers.add_parser(
        "stereo", help="Stereo camera calibration using OpenCV."
    )
    subparser_stereo.add_argument(
        "path", default=".", help="Folder containing the mission.yaml"
    )
    subparser_stereo.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force output file overwite",
    )
    subparser_stereo.add_argument(
        "-FF",
        dest="force2",
        action="store_true",
        help="Regenerates and overwrittes all files, including intermediate \
              results",
    )
    subparser_stereo.add_argument(
        "-s",
        "--suffix",
        default="",
        help="Suffix added to stereo calibration file name. Note: the mono calibration files that are ingested for \
              this processing step must not contain any suffix.",
    )
    subparser_stereo.set_defaults(func=call_calibrate_stereo)

    subparser_laser = subparsers.add_parser(
        "laser", help="Laser to camera extrinsic calibration."
    )
    subparser_laser.add_argument(
        "path", default=".", help="Folder containing the images."
    )
    subparser_laser.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force output file overwite",
    )
    subparser_laser.add_argument(
        "-FF",
        dest="force2",
        action="store_true",
        help="Regenerates and overwrittes all files, including intermediate \
              results",
    )
    subparser_laser.add_argument(
        "-s",
        "--suffix",
        default="",
        help="Suffix added to laser calibration filename. Note: the stereo calibration file that is ingested for this \
              processing step must not contain any suffix.",
    )
    subparser_laser.add_argument(
        "-u",
        "--num_uncertainty_planes",
        dest="num_uncert_planes",
        default=300,
        type=int,
        help="Nunber of uncertainty planes generated",
    )
    subparser_laser.set_defaults(func=call_calibrate_laser)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)


def call_calibrate_mono(args):
    time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    Console.set_logging_file(
        get_processed_folder(args.path) / ("log/" + time_string + "_auv_cal_mono.log")
    )
    c = Calibrator(args.path, args.force, args.force2, args.suffix)
    c.mono()


def call_calibrate_stereo(args):
    time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    Console.set_logging_file(
        get_processed_folder(args.path) / ("log/" + time_string + "_auv_cal_stereo.log")
    )
    c = Calibrator(args.path, args.force, args.force2, args.suffix)
    c.stereo()


def call_calibrate_laser(args):
    time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    Console.set_logging_file(
        get_processed_folder(args.path) / ("log/" + time_string + "_auv_cal_laser.log")
    )
    c = Calibrator(
        args.path, args.force, args.force2, args.suffix, args.num_uncert_planes
    )
    c.laser()


if __name__ == "__main__":
    main()

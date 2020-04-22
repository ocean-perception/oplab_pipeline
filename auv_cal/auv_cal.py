# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""
from auv_cal.calibration import Calibrator
from oplab import Console

import sys
import argparse
import os


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

    os.system('')  # enable VT100 Escape Sequence for WINDOWS 10 for Console outputs  https://stackoverflow.com/questions/16755142/how-to-make-win32-console-recognize-ansi-vt100-escape-sequences
    Console.banner()
    Console.info('Running auv_cal version ' + str(Console.get_version()))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser_mono = subparsers.add_parser(
        'mono', help="Monocular camera calibration using OpenCV.")
    subparser_mono.add_argument(
        'path', default=".", help="Folder containing the mission.yaml")
    subparser_mono.add_argument(
        '-F', '--Force', dest='force', action='store_true', help="Force file overwite")
    subparser_mono.set_defaults(func=call_calibrate_mono)

    subparser_stereo = subparsers.add_parser(
        'stereo', help="Stereo camera calibration using OpenCV.")
    subparser_stereo.add_argument(
        'path', default=".", help="Folder containing the mission.yaml")
    subparser_stereo.add_argument(
        '-F', '--Force', dest='force', action='store_true', help="Force file overwite")
    subparser_stereo.set_defaults(func=call_calibrate_stereo)

    subparser_laser = subparsers.add_parser(
        'laser', help="Laser to camera extrinsic calibration.")
    subparser_laser.add_argument(
        'path', default=".", help="Folder containing the images.")
    subparser_laser.add_argument(
        '-F', '--Force', dest='force', action='store_true', help="Force file overwite")
    subparser_laser.set_defaults(func=call_calibrate_laser)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)


def call_calibrate_mono(args):
    c = Calibrator(args.path, args.force)
    c.mono()


def call_calibrate_stereo(args):
    c = Calibrator(args.path, args.force)
    c.stereo()


def call_calibrate_laser(args):
    c = Calibrator(args.path, args.force)
    c.laser()


if __name__ == '__main__':
    main()

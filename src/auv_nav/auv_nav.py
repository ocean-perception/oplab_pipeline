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

import argcomplete

from auv_nav.convert import (
    acfr_to_oplab,
    hybis_to_oplab,
    koyo20rov_to_oplab,
    oplab_to_acfr,
)
from auv_nav.parse import parse
from auv_nav.process import process
from oplab import Console, get_config_folder, get_processed_folder


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
    Console.info("Running auv_nav version " + str(Console.get_version()))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="which",
    )

    """
    Subparsers for the 3 targets 'parse', 'convert' and 'process'
    double-dash arguments (optionally completed with a single-dash, single
    letter abbreviation) are optional. Arguments without the double-dash prefix
    are positional arguments and therefore required
    """
    subparser_parse = subparsers.add_parser(
        "parse",
        help="Parse raw data and converting it to an intermediate \
        dataformat for further processing. Type auv_nav parse -h for help on \
        this target.",
    )
    subparser_parse.add_argument(
        "path",
        default=".",
        nargs="+",
        help="Folderpath where the (raw) input data is. Needs to be a \
        subfolder of 'raw' and contain the mission.yaml configuration file.",
    )
    # prefixing the argument with -- means it's optional
    subparser_parse.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force file \
        overwite",
    )
    subparser_parse.add_argument(
        "--merge",
        dest="merge",
        action="store_true",
        help="Merge multiple dives into a single JSON file. Requires more \
        than one dive PATH.",
    )
    subparser_parse.set_defaults(func=call_parse_data)

    subparser_process = subparsers.add_parser(
        "process",
        help="Process and/or convert data. Data needs to be saved in \
        the intermediate data format generated using auv_nav.py parse. Type \
        auv_nav process -h for help on this target.",
    )
    subparser_process.add_argument(
        "path",
        default=".",
        help="Path to folder where the data to process is. The folder \
        has to be generated using auv_nav parse.",
    )
    subparser_process.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force file \
        overwite",
    )
    subparser_process.add_argument(
        "-s",
        "--start",
        dest="start_datetime",
        default="",
        help="Start date & \
        time in YYYYMMDDhhmmss from which data will be processed. If not set, \
        start at beginning of dataset.",
    )
    subparser_process.add_argument(
        "-e",
        "--end",
        dest="end_datetime",
        default="",
        help="End date & time \
        in YYYYMMDDhhmmss up to which data will be processed. If not set \
        process to end of dataset.",
    )
    subparser_process.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose output to terminal",
    )
    subparser_process.add_argument(
        "-r",
        "--relative_pose_uncertainty",
        dest="relative_pose_uncertainty",
        action="store_true",
        help="Enable relative_pose_uncertainty",
    )
    subparser_process.add_argument(
        "--start_image_identifier",
        dest="start_image_identifier",
        default=None,
        help="Identifier (path) from which onwards states are loaded. Required if relative_pose_uncertainty is True.",
    )
    subparser_process.add_argument(
        "--end_image_identifier",
        dest="end_image_identifier",
        default=None,
        help="Identifier (path) up to which states are loaded. Required if relative_pose_uncertainty is True.",
    )
    subparser_process.set_defaults(func=call_process_data)

    subparser_convert = subparsers.add_parser(
        "convert",
        help="Converts data.",
    )
    subparser_convert.set_defaults(func=show_help)

    # CONVERT subparsers
    subsubparsers = subparser_convert.add_subparsers(dest="which")

    # ACFR to OPLAB CSV
    subparser_oplab_to_acfr = subsubparsers.add_parser(
        "oplab_to_acfr",
        help="Converts an already processed dive to ACFR format",
    )
    subparser_oplab_to_acfr.add_argument(
        "-d",
        "--dive-folder",
        dest="dive_folder",
        help="Input dive path.",
    )
    subparser_oplab_to_acfr.add_argument(
        "-o",
        "--output-folder",
        dest="output_folder",
        help="Path where results will be written.",
    )
    subparser_oplab_to_acfr.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force file overwite",
    )
    subparser_oplab_to_acfr.set_defaults(func=oplab_to_acfr)

    # OPLAB to ACFR
    subparser_acfr_to_oplab = subsubparsers.add_parser(
        "acfr_to_oplab",
        help="Converts a VehiclePosEst.data and/or a StereoPosEst.data to \
            OPLAB csv format",
    )

    subparser_acfr_to_oplab.add_argument(
        "--vehicle-pose",
        dest="vehicle_pose",
        help="vehicle_pose_est.data filepath.",
    )
    subparser_acfr_to_oplab.add_argument(
        "--stereo-pose",
        dest="stereo_pose",
        help="stereo_pose_est.data filepath.",
    )
    subparser_acfr_to_oplab.add_argument(
        "-o",
        "--output-folder",
        dest="output_folder",
        help="Path where results will be written.",
    )
    subparser_acfr_to_oplab.add_argument(
        "-d",
        "--dive-folder",
        dest="dive_folder",
        help="Optional path of an existing processed dive to interpolate \
            laser timestamps to ACFR navigation.",
    )
    subparser_acfr_to_oplab.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force file overwite",
    )
    subparser_acfr_to_oplab.set_defaults(func=acfr_to_oplab)

    # KOYO_20_ROV to OPLAB CSV
    subparser_koyo20rov_to_oplab = subsubparsers.add_parser(
        "koyo20rov_to_oplab",
        help=("Converts koyo20rov navigation fileS (plural) to oplab CSV" " format"),
    )
    subparser_koyo20rov_to_oplab.add_argument(
        "-i",
        "--dive-path",
        dest="dive_path",
        help="Input directory path to dive.",
    )
    subparser_koyo20rov_to_oplab.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force overwite of output files",
    )
    subparser_koyo20rov_to_oplab.set_defaults(func=koyo20rov_to_oplab)

    # HYBIS to OPLAB CSV
    subparser_hybis_to_oplab = subsubparsers.add_parser(
        "hybis_to_oplab",
        help="Converts a hybis navigation file to oplab CSV format",
    )
    subparser_hybis_to_oplab.add_argument(
        "-i",
        "--navigation-file",
        dest="navigation_file",
        help="Input navigation file.",
    )
    subparser_hybis_to_oplab.add_argument(
        "-d",
        "--image-path",
        dest="image_path",
        help="Input image path.",
    )
    subparser_hybis_to_oplab.add_argument(
        "-o",
        "--output-folder",
        dest="output_folder",
        help="Path where results will be written.",
    )
    subparser_hybis_to_oplab.add_argument(
        "--reference-lat",
        dest="reference_lat",
        help="Reference latitude for northing/easting.",
    )
    subparser_hybis_to_oplab.add_argument(
        "--reference-lon",
        dest="reference_lon",
        help="Reference longitude for northing/easting.",
    )
    subparser_hybis_to_oplab.add_argument(
        "-F",
        "--Force",
        dest="force",
        action="store_true",
        help="Force file overwite",
    )
    subparser_hybis_to_oplab.set_defaults(func=hybis_to_oplab)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    args.func(args)


def show_help(args):
    Console.info("Run with -h or --help to show the usage and help")


def call_parse_data(args):
    time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    Console.set_logging_file(
        get_processed_folder(args.path[0])
        / ("log/" + time_string + "_auv_nav_parse.log")
    )
    parse(args.path, args.force, args.merge)


def call_process_data(args):
    time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    Console.set_verbosity(args.verbose)
    Console.set_logging_file(
        get_processed_folder(args.path)
        / ("log/" + time_string + "_auv_nav_process.log")
    )
    auv_nav_path = get_config_folder(args.path) / "auv_nav.yaml"
    if auv_nav_path.exists():
        auv_nav_path_log = get_processed_folder(args.path) / (
            "log/" + time_string + "_auv_nav.yaml"
        )
        auv_nav_path.copy(auv_nav_path_log)
    process(
        args.path,
        args.force,
        args.start_datetime,
        args.end_datetime,
        args.relative_pose_uncertainty,
        args.start_image_identifier,
        args.end_image_identifier,
    )


if __name__ == "__main__":
    main()

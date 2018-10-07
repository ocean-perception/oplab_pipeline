# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""


import os
import sys
import argparse

sys.path.append(os.path.abspath('.'))
from auv_nav.auv_parsers.parse_data import parse_data
from auv_nav.auv_parsers.extract_data import extract_data
from auv_nav.auv_parsers.folder_structure import is_subfolder_of


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

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    """
    Subparsers for the 3 targets 'parse', 'visualise' and 'process'
    double-dash arguments (optionally completed with a single-dash, single
    letter abbreviation) are optional. Arguments without the double-dash prefix
    are positional arguments and therefore required
    """
    subparser_parse = subparsers.add_parser(
        'parse', help="Parse raw data and converting it to an intermediate \
        dataformat for further processing. Type auv_nav parse -h for help on \
        this target.")
    subparser_parse.add_argument(
        'path', help="Folderpath where the (raw) input data is. Needs to be a \
        subfolder of 'raw' and contain the mission.yaml configuration file.")
    # prefixing the argument with -- means it's optional
    subparser_parse.add_argument(
        '-f', '--format', dest='format', default="oplab", help="Format in which \
        the data is output. 'oplab' or 'acfr'. Default: 'oplab'.")
    subparser_parse.set_defaults(func=call_parse_data)

#   subparser_visualise = subparsers.add_parser(
#       'visualise', help="Visualise data. Data needs to be saved in the \
#       intermediate data format generated using auv_nav.py parse. Type \
#       auv_nav visualise -h for help on this target.")
#   subparser_visualise.add_argument(
#       'path', help="Path of folder where the data to visualise is. The \
#       folder has to be generated using auv_nav parse.")
#   subparser_visualise.add_argument(
#       '-f', '--format', dest='format', default="oplab", help="Format in \
#       which the data to be visualised is stored. 'oplab' or 'acfr'. \
#       Default: 'oplab'.")
#   subparser_visualise.set_defaults(func=call_visualise_data)

    subparser_process = subparsers.add_parser(
        'process', help="Process and/or convert data. Data needs to be saved in \
        the intermediate data format generated using auv_nav.py parse. Type \
        auv_nav process -h for help on this target.")
    subparser_process.add_argument(
        'path', help="Path to folder where the data to process is. The folder \
        has to be generated using auv_nav parse.")
    subparser_process.add_argument(
        '-f', '--format', dest='format', default="oplab", help="Format in which \
        the data to be processed is stored. 'oplab' or 'acfr'. Default: \
        'oplab'.")
    subparser_process.add_argument(
        '-s', '--start', dest='start_datetime', default='', help="Start date & \
        time in YYYYMMDDhhmmss from which data will be processed. If not set, \
        start at beginning of dataset.")
    subparser_process.add_argument(
        '-e', '--end', dest='end_datetime', default='', help="End date & time \
        in YYYYMMDDhhmmss up to which data will be processed. If not set \
        process to end of dataset.")
    subparser_process.set_defaults(func=call_process_data)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)


def call_parse_data(args):
    parse_data(args.path, args.format)


# def call_visualise_data(args):
#     display_info(args.path + os.sep, args.format)


def call_process_data(args):
    if not is_subfolder_of(args.path, "processed"):
        raise ValueError("The input directory you provided is not a subfolder \
                         of a folder called 'processed'")
    extract_data(args.path, args.format,
                 args.start_datetime, args.end_datetime)


if __name__ == '__main__':
    main()

import argparse
import sys
import os
from shutil import copyfile

from correct_images.calculate_correction_parameters import calculate_correction_parameters
from correct_images.develop_corrected_images import develop_corrected_image
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder


# def batch_correct(method, target_mean, target_std, filelist, mission):
#
# Author: B.Thornton@soton.ac.uk
# Matlab -> Python translation: jw22g14@soton.ac.uk Jenny Walker
#
# single thread script to run batches of colour_convert for images set in
# filepath with the images to be processed specified in a ascii list
# that is human readable and editable to remove poor images. This can be generated
# from terminal
#
# ls * > flist.txt or dir /s * >flist.txt.
#
# The filepath must contain a subfolder raw.
# The output will bein a mirrored path where raw is replaced by processed.
# The recomended filepath convention is
#
# <base subpath>/raw/<year>/<cruise name>/<dive name>/<image folder subpath>/
#
# which will produce it's outputs in
#
# <base subpath>/processed/<year>/<cruise name>/<dive name>/<image folder subpath>/<method>/<settings>
#
# there is no restriction of <base subpath> and <image folder subpath>
# other than that they shoulld not contain an additional /raw/
#

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args=None):
    """
    Main entry point for project.
    Args:
        args : list of arguments as if they were input in the command line.
               Leave it None to use sys.argv.
    When run from comand line >correct_images_.py -h will populate args parameters
    parse/process -h will populate positional arguments

    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser_correct_attenuation = subparsers.add_parser(
        'parse', help='Calculate attenuation correction parameters')
    subparser_correct_attenuation.add_argument(
        'path', help="Path to raw directory till dive.")
    subparser_correct_attenuation.add_argument(
        '-f','--force', type=str2bool, default = False,
        help="Force overwrite if correction parameters already exist.")
    subparser_correct_attenuation.set_defaults(func=call_calculate_attenuation_correction_parameter)

    subparser_correct_attenuation = subparsers.add_parser(
        'process', help='Develop attenuation corrected images')
    subparser_correct_attenuation.add_argument(
        'path', help="Path to processed directory till dive.")
    subparser_correct_attenuation.add_argument(
        '-f', '--force', type=str2bool, default = False,
        help="Force overwrite if processed images already exist.")

    subparser_correct_attenuation.set_defaults(func=call_develop_corrected_image)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)


def call_calculate_attenuation_correction_parameter(args):
    sr = get_raw_folder(args.path)
    sp = get_processed_folder(args.path)
    sc = get_config_folder(args.path)
    pc = os.path.join(sc, "correct_images.yaml")
    if os.path.isfile(pc) is False:
        print('Config File does not exist in target configuration folder.')
        print('Copying default configuration.')
        copyfile('./correct_images/correct_images.yaml', pc)
        print('Default configuration copied to target configuration folder.')
    path_mission = os.path.join(sr, "mission.yaml")
    path_correct = pc
    path_raw = sr
    path_processed = sp
    calculate_correction_parameters(path_raw, path_processed, path_mission, path_correct, args.force)


def call_develop_corrected_image(args):
    sr = get_raw_folder(args.path)
    sp = get_processed_folder(args.path)
    sc = get_config_folder(args.path)
    pc = os.path.join(sc, "correct_images.yaml")
    if os.path.isfile(pc) is False:
        print('Config File does not exist in target configuration folder.')
        print('Copying default configuration.')
        copyfile('./correct_images/correct_images.yaml', pc)
        print('Default configuration copied to target configuration folder.')
    path_mission = os.path.join(sr, "mission.yaml")
    path_correct = pc
    path_raw = sr
    path_processed = sp

    develop_corrected_image(path_processed,path_mission,path_correct,args.force)


if __name__ == '__main__':
    main()

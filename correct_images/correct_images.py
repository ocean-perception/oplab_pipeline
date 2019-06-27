import argparse
import sys
import os
from shutil import copyfile

from correct_images.calculate_correction_parameters import load_xviii_bayer_from_binary
from correct_images.calculate_correction_parameters import calculate_correction_parameters
from correct_images.develop_corrected_images import develop_corrected_image
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import cv2

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

    subparser_debayer = subparsers.add_parser(
        'debayer', help='Debayer without correction')
    subparser_debayer.add_argument(
        'path', help="Path to bayer image.")
    subparser_debayer.add_argument(
        '-a', '--all', action='store_true', help="For all files in given path or specific file. If True next arguments are ignored")
    subparser_debayer.add_argument(
        '-i', '--image', default=None, help="Single raw image to test.")
    subparser_debayer.add_argument(
        '-e', '--extension', default=None, help="extension type of image.")
    subparser_debayer.set_defaults(func=call_debayer)

    subparser_correct_attenuation = subparsers.add_parser(
        'parse', help='Calculate attenuation correction parameters')
    subparser_correct_attenuation.add_argument(
        'path', help="Path to raw directory till dive.")
    subparser_correct_attenuation.add_argument(
        '-f', '--force', action='store_true',
        help="Force overwrite if correction parameters already exist.")
    subparser_correct_attenuation.set_defaults(func=call_calculate_attenuation_correction_parameter)

    subparser_correct_attenuation = subparsers.add_parser(
        'process', help='Develop attenuation corrected images')
    subparser_correct_attenuation.add_argument(
        'path', help="Path to processed directory till dive.")
    subparser_correct_attenuation.add_argument(
        '-f', '--force', action='store_true',
        help="Force overwrite if processed images already exist.")

    subparser_correct_attenuation.set_defaults(func=call_develop_corrected_image)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)


def call_debayer(args):
    display_min = 1000
    display_max = 10000

    def display(image, display_min, display_max): # copied from Bi Rico
        # Here I set copy=True in order to ensure the original image is not
        # modified. If you don't mind modifying the original image, you can
        # set copy=False or skip this step.
        image = np.array(image, copy=True)
        image.clip(display_min, display_max, out=image)
        image -= display_min
        np.floor_divide(image, (display_max - display_min + 1) / 256,
                        out=image, casting='unsafe')
        return image.astype(np.uint8)
    def lut_display(image, display_min, display_max) :
        lut = np.arange(2**16, dtype='uint32')
        lut = display(lut, display_min, display_max)
        return np.take(lut, image)
    if args.all:
        i = 0
        for f in os.listdir(args.path):
            p = args.path + f
            xviii_binary_data = np.fromfile(os.path.expanduser(p), dtype=np.uint8)
            img = load_xviii_bayer_from_binary(xviii_binary_data)
            img_rgb = np.array(demosaicing_CFA_Bayer_bilinear(img, pattern='GRBG'))
            p_ = './' + str(i) + '.png'
            cv2.imwrite(os.path.expanduser(p_), img_rgb)
            i = i + 1
            print('{} done.'.format(str(f)))
    else:
        p = args.path + args.image + args.extension
        xviii_binary_data = np.fromfile(os.path.expanduser(p), dtype=np.uint8)
        img = load_xviii_bayer_from_binary(xviii_binary_data)
        img_rgb = np.array(demosaicing_CFA_Bayer_bilinear(img, pattern='GRBG'))
        img_lut = lut_display(img_rgb, display_min, display_max)
        p_ = './' + args.image + '.png'
        cv2.imwrite(os.path.expanduser(p_), img_lut)
        print('{0}.{1} done.'.format(args.image, args.extension))


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

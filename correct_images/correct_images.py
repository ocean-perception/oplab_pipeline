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
from pathlib import Path

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
        'path', help="Path to bayer images.")
    subparser_debayer.add_argument(
        '-i', '--image', default=None, help="Single raw image to test.")
    subparser_debayer.add_argument(
        '-o', '--output', default=None, help="Output folder.")
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
    def debayer_image(image_path):
        xviii_binary_data = np.fromfile(str(image_path), dtype=np.uint8)
        img = load_xviii_bayer_from_binary(xviii_binary_data)
        img = img / 128
        img_rgb = np.array(demosaicing_CFA_Bayer_bilinear(img, pattern='GRBG'))
        return img_rgb

    ouput_dir = Path(args.output)
    if not args.image:
        image_dir = Path(args.path)
        print('Debayering folder {} to {}'.format(image_dir, ouput_dir))
        image_list = list(image_dir.glob('*.raw'))
        print('Found ' + str(len(image_list)) + ' images.')

        for image_path in image_list:
            print('Debayering image {}'.format(image_path.name))
            img_rgb = debayer_image(image_path)
            image_name = str(image_path.stem) + '.png'
            output_image_path = Path(ouput_dir) / image_name
            cv2.imwrite(str(output_image_path), img_rgb)
    else:
        single_image = Path(args.image)
        print('Debayering single image {} to {}'.format(single_image.name, ouput_dir))
        img_rgb = debayer_image(single_image)
        image_name = str(single_image.stem) + '.png'
        output_image_path = Path(ouput_dir) / image_name
        cv2.imwrite(str(output_image_path), img_rgb)


def call_calculate_attenuation_correction_parameter(args):
    sr = get_raw_folder(args.path)
    sp = get_processed_folder(args.path)
    sc = get_config_folder(args.path)
    pc = sc / "correct_images.yaml"
    if not pc.exists():
        print('Config File does not exist in target configuration folder.')
        print('Copying default configuration.')

        root = Path(__file__).parents[1]
        default_file = root / 'correct_images/default_yaml' / 'correct_images.yaml'
        print("Cannot find {}, generating default from {}".format(
            pc, default_file))
        # save localisation yaml to processed directory
        default_file.copy(pc)

        print('Default configuration copied to target configuration folder.')
    path_mission = sr / "mission.yaml"
    path_correct = pc
    path_raw = sr
    path_processed = sp
    calculate_correction_parameters(path_raw, path_processed, path_mission, path_correct, args.force)


def call_develop_corrected_image(args):
    sr = get_raw_folder(args.path)
    sp = get_processed_folder(args.path)
    sc = get_config_folder(args.path)
    pc = os.path.join(str(sc), "correct_images.yaml")
    if os.path.isfile(pc) is False:
        print('Config File does not exist in target configuration folder.')
        print('Copying default configuration.')
        copyfile('./correct_images/correct_images.yaml', pc)
        print('Default configuration copied to target configuration folder.')
    path_mission = sr / "mission.yaml"
    path_correct = pc
    path_processed = sp

    develop_corrected_image(path_processed, path_mission, path_correct, args.force)


if __name__ == '__main__':
    main()

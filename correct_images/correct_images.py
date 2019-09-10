import argparse
import sys
import os

from correct_images.calculate_correction_parameters import \
    load_xviii_bayer_from_binary
from correct_images.calculate_correction_parameters import \
    calculate_correction_parameters
from correct_images.develop_corrected_images import develop_corrected_image
from auv_nav.tools.console import Console
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import cv2
from pathlib import Path
import joblib


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


def main(args=None):
    """
    Main entry point for project.
    Args:
        args : list of arguments as if they were input in the command line.
               Leave it None to use sys.argv.
    When run from comand line >correct_images_.py -h will populate args parameters
    parse/process -h will populate positional arguments

    """
    os.system(
        '')  # enable VT100 Escape Sequence for WINDOWS 10 for Console outputs  https://stackoverflow.com/questions/16755142/how-to-make-win32-console-recognize-ansi-vt100-escape-sequences
    Console.banner()
    Console.info(
        'Running correct_images version ' + str(Console.get_version()))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser_debayer = subparsers.add_parser(
        'debayer', help='Debayer without correction')
    subparser_debayer.add_argument(
        'path', help="Path to bayer images.")
    subparser_debayer.add_argument(
        'filetype', help="type of image: raw / tif / tiff")
    subparser_debayer.add_argument(
        '-p', '--pattern', default='GRBG',
        help='Bayer pattern (GRBG for Unagi, BGGR for BioCam)')
    subparser_debayer.add_argument(
        '-i', '--image', default=None, help="Single raw image to test.")
    subparser_debayer.add_argument(
        '-o', '--output', default='.', help="Output folder.")
    subparser_debayer.set_defaults(func=call_debayer)

    subparser_correct_attenuation = subparsers.add_parser(
        'parse', help='Calculate attenuation correction parameters')
    subparser_correct_attenuation.add_argument(
        'path', help="Path to raw directory till dive.")
    subparser_correct_attenuation.add_argument(
        '-F', '--Force', dest='force', action='store_true',
        help="Force overwrite if correction parameters already exist.")
    subparser_correct_attenuation.set_defaults(
        func=call_calculate_attenuation_correction_parameter)

    subparser_correct_attenuation = subparsers.add_parser(
        'process', help='Develop attenuation corrected images')
    subparser_correct_attenuation.add_argument(
        'path', help="Path to processed directory till dive.")
    subparser_correct_attenuation.add_argument(
        '-F', '--Force', dest='force', action='store_true',
        help="Force overwrite if processed images already exist.")

    subparser_correct_attenuation.set_defaults(
        func=call_develop_corrected_image)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
    else:
        args = parser.parse_args()
        args.func(args)


def call_debayer(args):
    def debayer_image(image_path, filetype, pattern, output_dir):
        Console.info('Debayering image {}'.format(image_path.name))
        if filetype is 'raw':
            xviii_binary_data = np.fromfile(str(image_path), dtype=np.uint8)
            img = load_xviii_bayer_from_binary(xviii_binary_data)
            img = img / 128
        else:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img_rgb = np.array(demosaicing_CFA_Bayer_bilinear(img, pattern))
        image_name = str(image_path.stem) + '.png'
        output_image_path = Path(output_dir) / image_name
        cv2.imwrite(str(output_image_path), img_rgb)

    output_dir = Path(args.output)
    if not output_dir.exists():
        Console.info('Creating output dir {}'.format(output_dir))
        output_dir.mkdir(parents=True)
    else:
        Console.info('Using output dir {}'.format(output_dir))
    if not args.image:
        image_dir = Path(args.path)
        Console.info(
            'Debayering folder {} to {}'.format(image_dir, output_dir))
        image_list = list(image_dir.glob('*.' + args.filetype))
        Console.info('Found ' + str(len(image_list)) + ' images.')
        joblib.Parallel(n_jobs=-2)([
            joblib.delayed(debayer_image)(
                image_path,
                args.filetype,
                args.pattern,
                output_dir)
            for image_path in image_list])

    else:
        single_image = Path(args.image)
        debayer_image(single_image, args.filetype, args.pattern, output_dir)


def call_calculate_attenuation_correction_parameter(args):
    calculate_correction_parameters(args.path, args.force)


def call_develop_corrected_image(args):
    develop_corrected_image(args.path, args.force)


if __name__ == '__main__':
    main()

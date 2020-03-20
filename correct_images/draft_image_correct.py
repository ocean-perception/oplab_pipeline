# this is the main entry point to correct images
# IMPORT --------------------------------
# all imports go here 
# -----------------------------------------


# Main function
def main(args=None):
    
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

    subparser_correct = subparsers.add_parser(
        'correct', help='Correct images for attenuation / distortion / gamma and debayering')
    subparser_correct_attenuation.add_argument(
        'path', help="Path to raw directory till dive.")
    subparser_correct_attenuation.add_argument(
        '-F', '--Force', dest='force', action='store_true',
        help="Force overwrite if correction parameters already exist.")
    subparser_correct_attenuation.set_defaults(
        func=call_correct)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
    else:
        args = parser.parse_args()
        args.func(args)

def call_debayer(args):
	# instantiate Corrector object
	corrector = Corrector()

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
        for image_path in image_list:
        	rgb_image = corrector.debayer(image_path, args.pattern, args.filetype)
        	image_name = str(image_path.stem) + '.png'
        	output_image_path = Path(output_dir) / image_name
        	cv2.imwrite(str(output_image_path), img_rgb)

    else:
        single_image = Path(args.image)
        rgb_image = corrector.debayer(single_image, args.pattern, args.filetype)
        image_name = str(single_image.stem) + '.png'
        output_image_path = Path(output_dir) / image_name
        cv2.imwrite(str(output_image_path), img_rgb)

def call_correct(args):
	# resolve paths
	path_processed = get_processed(args.path)
	path_config = get_config(args.path)

	# instantiate camera system
	# instantiat corrector

	# execute corrector

# Encoding: utf-8

from auv_nav.console import Console
from auv_cal.camera_models import StereoCamera
from auv_cal.camera_calibrator import CalibrationException
from auv_cal.camera_calibrator import MonoCalibrator
from auv_cal.camera_calibrator import ChessboardInfo
from auv_cal.camera_calibrator import Patterns
from auv_cal.camera_calibrator import StereoCalibrator
from auv_cal.laser_calibrator import LaserCalibrator
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder
from auv_nav.tools.folder_structure import valid_dive
from pathlib import Path
import yaml
import json


def collect_image_files(image_dirs, file_pattern):
    # TODO try all different formats: png, jpeg, tif
    images = []
    if isinstance(image_dirs, list):
        for d in image_dirs:
            if d.is_dir():
                images.extend(list(d.glob(file_pattern)))
            else:
                Console.warn('Directory '"'{}'"' cannot be found'.format(d))
    else:
        if (image_dirs.is_dir()):
            images = list(image_dirs.glob(file_pattern))
        else:
            Console.warn('Directory '"'{}'"' cannot be found'.format(image_dirs))
    images.sort()
    return images


def check_pattern(config):
    if config["pattern"] == 'Circles' or config["pattern"] == 'circles':
        pattern = Patterns.Circles
    elif config["pattern"] == 'ACircles' or config["pattern"] == 'acircles':
        pattern = Patterns.ACircles
    elif config["pattern"] == 'Chessboard' or config["pattern"] == 'chessboard':
        pattern = Patterns.Chessboard
    else:
        Console.quit('The available patterns are: Circles, Chessboard or ACircles. Please check you did not misspell the pattern type.')
    return pattern


def calibrate_mono(name, filepaths, extension, config, output_file, overwrite):
    Console.info('Looking for {} calibration images in {}'.format(extension, str(filepaths)))
    image_list = collect_image_files(filepaths, extension)
    Console.info('Found ' + str(len(image_list)) + ' images.')
    if len(image_list) < 8:
        Console.error('Too few images. Try to get more.')
        return

    mc = MonoCalibrator(boards=[ChessboardInfo(config["rows"],
                                               config["cols"],
                                               config["size"])],
                        pattern=check_pattern(config),
                        invert=config["invert"],
                        name=name)
    try:
        image_list_file = output_file.with_suffix('.json')
        if overwrite or not image_list_file.exists():
            mc.cal(image_list)
            with image_list_file.open('w') as f:
                json.dump(mc.json, f)
        else:
            mc.cal_from_json(image_list_file, image_list)
        mc.report()
        Console.info('Writting calibration to '"'{}'"''.format(output_file))
        with output_file.open('w') as f:
            f.write(mc.yaml())
    except CalibrationException:
        Console.error('The calibration pattern was not found in the images.')


def calibrate_stereo(left_name, left_filepaths, left_extension, left_calib,
                     right_name, right_filepaths, right_extension, right_calib,
                     config, output_file):
    Console.info('Looking for calibration images in {}'.format(str(left_filepaths)))
    left_image_list = collect_image_files(left_filepaths, left_extension)
    Console.info('Found ' + str(len(left_image_list)) + ' left images.')
    Console.info('Looking for calibration images in {}'.format(str(right_filepaths)))
    right_image_list = collect_image_files(right_filepaths, right_extension)
    Console.info('Found ' + str(len(right_image_list)) + ' right images.')
    if len(left_image_list) < 8 or len(right_image_list) < 8:
        Console.error('Too few images. Try to get more.')
        return
    try:
        with left_calib.with_suffix('.json').open('r') as f:
            left_json = json.load(f)
        with right_calib.with_suffix('.json').open('r') as f:
            right_json = json.load(f)
        model = StereoCamera(left=left_calib, right=right_calib)
        sc = StereoCalibrator(stereo_camera_model=model,
                              boards=[ChessboardInfo(config["rows"],
                                                     config["cols"],
                                                     config["size"])],
                              pattern=check_pattern(config),
                              invert=config["invert"])
        # sc.cal(left_image_list, right_image_list)
        sc.cal_from_json(left_json=left_json, right_json=right_json,)
        sc.report()
        Console.info('Writting calibration to '"'{}'"''.format(output_file))
        with output_file.open('w') as f:
            f.write(sc.yaml())
    except CalibrationException:
        Console.error('The calibration pattern was not found in the images.')


def calibrate_laser(left_name, left_filepath, left_extension,
                    right_name, right_filepath, right_extension,
                    stereo_calibration_file,
                    config, output_file, output_file_b, skip_first=0):
    Console.info('Looking for calibration images in {}'.format(left_filepath))
    left_image_list = collect_image_files(left_filepath, left_extension)
    Console.info('Found ' + str(len(left_image_list)) + ' left images.')
    Console.info('Looking for calibration images in {}'.format(right_filepath))
    right_image_list = collect_image_files(right_filepath, right_extension)
    Console.info('Found ' + str(len(right_image_list)) + ' right images.')
    if len(left_image_list) < 8 or len(right_image_list) < 8:
        Console.error('Too few images. Try to get more.')
        return
    try:
        model = StereoCamera(stereo_calibration_file)

        def check_prop(name, dic_to_check, default_value):
            if name in dic_to_check:
                return dic_to_check[name]
            else:
                return default_value

        lc = LaserCalibrator(stereo_camera_model=model,
                             config=config)
        lc.cal(left_image_list[skip_first:], right_image_list[skip_first:])
        Console.info('Writting calibration to '"'{}'"''.format(output_file))
        with output_file.open('w') as f:
            f.write(lc.yaml())
        if not 'two_lasers' in config:
            return
        if config['two_lasers']:
            Console.info('Writting calibration to '"'{}'"''.format(output_file_b))
            with output_file_b.open('w') as f:
                f.write(lc.yaml_b())
    except CalibrationException:
        Console.error('The calibration pattern was not found in the images.')


def build_filepath(base, paths):
    filepaths = []
    if not isinstance(paths, list):
        filepaths = base / str(paths)
    else:
        for p in paths:
            filepaths.append(base / p)
    return filepaths


class Calibrator():
    def __init__(self, filepath, force_overwite=False):
        filepath = Path(filepath).resolve()
        self.filepath = get_raw_folder(filepath)
        
        if not valid_dive(self.filepath):
            Console.error('This code expects you to run it inside a dive folder.')
            Console.quit('The folder specified is not a valid dive.')

        self.filepath = self.filepath.parent

        # Create the calibration folder at the same level as the dives
        self.output_path = get_processed_folder(self.filepath) / 'calibration'
        self.configuration_path = get_config_folder(self.filepath) / 'calibration'
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        if not self.configuration_path.exists():
            self.configuration_path.mkdir(parents=True)

        self.fo = force_overwite

        calibration_config_file = self.configuration_path / 'calibration.yaml'
        if calibration_config_file.exists():
            Console.info("Loading existing calibration.yaml at {}".format(calibration_config_file))
        else:
            root = Path(__file__).parents[1]
            default_file = root / 'auv_cal/default_yaml' / 'default_calibration.yaml'
            Console.warn("Cannot find {}, generating default from {}".format(
                calibration_config_file, default_file))
            # save localisation yaml to processed directory
            default_file.copy(calibration_config_file)

            Console.warn('Edit the file at: \n\t' + str(calibration_config_file))
            Console.warn('Try to use relative paths to the calibration datasets')
            Console.quit('Modify the file calibration.yaml and run this code again.')

        with calibration_config_file.open('r') as stream:
            self.calibration_config = yaml.safe_load(stream)

    def mono(self):
        for c in self.calibration_config['cameras']:
            cam_name = c['name']
            # Find if the calibration file exists
            calibration_file = self.output_path / str('mono_' + cam_name + '.yaml')
            Console.info('Looking for a calibration file at ' + str(calibration_file))
            if calibration_file.exists() and not self.fo:
                Console.warn('The camera ' + c['name'] + ' has already been calibrated. If you want to overwrite the JSON, use the -F flag.')
            else:
                Console.info('The camera is not calibrated, running mono calibration...')
            filepaths = build_filepath(self.filepath, c['camera_calibration']['path'])

            if not 'glob_pattern' in c['camera_calibration']:
                Console.error('Could not find the key glob_pattern for the camera ', c['name'])
                Console.quit('glob_pattern expected in calibration.yaml')

            calibrate_mono(cam_name,
                           filepaths,
                           str(c['camera_calibration']['glob_pattern']),
                           self.calibration_config['camera_calibration'],
                           calibration_file,
                           self.fo)

    def stereo(self):
        if len(self.calibration_config['cameras']) > 1:
            c0 = self.calibration_config['cameras'][0]
            c1 = self.calibration_config['cameras'][1]
            calibration_file = self.output_path / str('stereo_' + c0['name'] + '_' + c1['name'] + '.yaml')
            Console.info('Looking for a calibration file at ' + str(calibration_file))
            if calibration_file.exists() and not self.fo:
                Console.warn('The stereo pair ' + c0['name'] + '_' + c1['name'] + ' has already been calibrated. If you want to overwrite the calibration, use the -F flag.')
            else:
                Console.info('The stereo camera is not calibrated, running stereo calibration...')

                left_filepaths = build_filepath(self.filepath, c0['camera_calibration']['path'])
                right_filepaths = build_filepath(self.filepath, c1['camera_calibration']['path'])
                left_name = c0['name']
                if not 'glob_pattern' in c0['camera_calibration']:
                    Console.error('Could not find the key glob_pattern for the camera ', c0['name'])
                    Console.quit('glob_pattern expected in calibration.yaml')
                left_extension = str(c0['camera_calibration']['glob_pattern'])
                right_name = c1['name']
                if not 'glob_pattern' in c1['camera_calibration']:
                    Console.error('Could not find the key glob_pattern for the camera ', c1['name'])
                    Console.quit('glob_pattern expected in calibration.yaml')
                right_extension = str(c1['camera_calibration']['glob_pattern'])
                left_calibration_file = self.output_path / str('mono_' + left_name + '.yaml')
                right_calibration_file = self.output_path / str('mono_' + right_name + '.yaml')
                if not left_calibration_file.exists() or not right_calibration_file.exists():
                    if not left_calibration_file.exists():
                        Console.warn('Could not find a monocular calibration file ' + str(left_calibration_file) + '...')
                    if not right_calibration_file.exists():
                        Console.warn('Could not find a monocular calibration file ' + str(right_calibration_file) + '...')
                    self.mono()
                if left_calibration_file.exists() and right_calibration_file.exists():
                    Console.info('Loading previous monocular calibrations at \
                                  \n\t * {}\n\t * {}'.format(str(left_calibration_file), str(right_calibration_file)))
                calibrate_stereo(left_name,
                                 left_filepaths,
                                 left_extension,
                                 left_calibration_file,
                                 right_name,
                                 right_filepaths,
                                 right_extension,
                                 right_calibration_file,
                                 self.calibration_config['camera_calibration'],
                                 calibration_file)
        # Check for a second stereo pair
        if len(self.calibration_config['cameras']) > 2:
            c0 = self.calibration_config['cameras'][0]
            c1 = self.calibration_config['cameras'][2]
            calibration_file = self.output_path / str('stereo_' + c0['name'] + '_' + c1['name'] + '.yaml')
            Console.info('Looking for a calibration file at ' + str(calibration_file))
            if calibration_file.exists() and not self.fo:
                Console.warn('The stereo pair ' + c0['name'] + '_' + c1['name'] + ' has already been calibrated. If you want to overwrite the calibration, use the -F flag.')
            else:
                Console.info('The stereo camera is not calibrated, running stereo calibration...')
                left_name = c0['name']
                left_filepaths = build_filepath(self.filepath, c0['camera_calibration']['path'])
                left_extension = str(c0['camera_calibration']['glob_pattern'])
                right_name = c1['name']
                right_filepaths = build_filepath(self.filepath, c1['camera_calibration']['path'])
                right_extension = str(c1['camera_calibration']['glob_pattern'])
                left_calibration_file = self.output_path / str('mono_' + left_name + '.yaml')
                right_calibration_file = self.output_path / str('mono_' + right_name + '.yaml')
                if not left_calibration_file.exists() or not right_calibration_file.exists():
                    if not left_calibration_file.exists():
                        Console.warn('Could not find a monocular calibration file ' + str(left_calibration_file) + '...')
                    if not right_calibration_file.exists():
                        Console.warn('Could not find a monocular calibration file ' + str(right_calibration_file) + '...')
                    self.mono()
                if left_calibration_file.exists() and right_calibration_file.exists():
                    Console.info('Loading previous monocular calibrations at \
                                  \n\t * {}\n\t * {}'.format(str(left_calibration_file), str(right_calibration_file)))
                calibrate_stereo(left_name,
                                 left_filepaths,
                                 left_extension,
                                 left_calibration_file,
                                 right_name,
                                 right_filepaths,
                                 right_extension,
                                 right_calibration_file,
                                 self.calibration_config['camera_calibration'],
                                 calibration_file)

    def laser(self):
        if 'laser_calibration' in self.calibration_config['cameras'][0]:
            c0 = self.calibration_config['cameras'][0]
            c1 = self.calibration_config['cameras'][1]
            if len(self.calibration_config['cameras']) > 2:
                c2 = self.calibration_config['cameras'][2]
            calibration_file = self.output_path / 'laser_calibration_top.yaml'
            calibration_file_b = self.output_path / 'laser_calibration_bottom.yaml'
            Console.info('Looking for a calibration file at ' + str(calibration_file))
            if calibration_file.exists() and not self.fo:
                Console.warn('The laser planes from cameras ' + c0['name'] + ' and ' + c1['name'] + ' have already been calibrated. If you want to overwite the calibration, use the -F flag.')
            else:
                Console.info('The laser planes are not calibrated, running laser calibration...')
                # Check if the stereo pair has already been calibrated
                stereo_calibration_file = self.output_path / str('stereo_' + c0['name'] + '_' + c1['name'] + '.yaml')
                if not stereo_calibration_file.exists():
                    Console.warn('Could not find a stereo calibration file ' + str(stereo_calibration_file) + '...')
                    self.stereo()
                left_name = c0['name']
                left_filepath = self.filepath / str(c0['laser_calibration']['path'])
                left_extension = str(c0['laser_calibration']['glob_pattern'])
                right_name = c1['name']
                right_filepath = self.filepath / str(c1['laser_calibration']['path'])
                right_extension = str(c1['laser_calibration']['glob_pattern'])
                if not 'skip_first' in self.calibration_config:
                    self.calibration_config['skip_first'] = 0
                calibrate_laser(left_name,
                                left_filepath,
                                left_extension,
                                right_name,
                                right_filepath,
                                right_extension,
                                stereo_calibration_file,
                                self.calibration_config['laser_calibration'],
                                calibration_file,
                                calibration_file_b,
                                self.calibration_config['skip_first'])

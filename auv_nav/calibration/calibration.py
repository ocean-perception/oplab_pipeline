# Encoding: utf-8

from auv_nav.tools.console import Console
from auv_nav.parsers.vehicle import Vehicle
from auv_nav.parsers.mission import Mission
from auv_nav.parsers.camera_calibration import StereoCamera
from auv_nav.calibration.calibrator import CalibrationException
from auv_nav.calibration.calibrator import MonoCalibrator
from auv_nav.calibration.calibrator import ChessboardInfo
from auv_nav.calibration.calibrator import Patterns
from auv_nav.calibration.calibrator import StereoCalibrator
from auv_nav.calibration.laser_calibrator import LaserCalibrator
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder
from pathlib import Path
import cv2
import yaml
import glob


def collect_image_files(image_dir, file_pattern):
    # TODO try all different formats: png, jpeg, tif
    images = list(image_dir.glob(file_pattern))
    images.sort()
    return images


def check_pattern(config):
    if config["pattern"] == 'Circles':
        pattern = Patterns.Circles
    elif config["pattern"] == 'ACircles':
        pattern = Patterns.ACircles
    elif config["pattern"] == 'Chessboard':
        pattern = Patterns.Chessboard
    else:
        Console.quit('The available patterns are: Circles, Chessboard or ACircles. Please check you did not misspell the pattern type.')
    return pattern


def calibrate_mono(name, filepath, extension, config, output_file):
    Console.info('Looking for {} calibration images in {}'.format(extension, filepath))
    image_list = collect_image_files(filepath, extension)
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
        mc.cal(image_list)
        mc.report()
        Console.info('Writting calibration to '"'{}'"''.format(output_file))
        with open(output_file, 'w') as f:
            f.write(mc.yaml())
    except CalibrationException:
        Console.error('The calibration pattern was not found in the images.')


def calibrate_stereo(left_name, left_filepath, left_extension, left_calib,
                     right_name, right_filepath, right_extension, right_calib,
                     config, output_file):
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
        model = StereoCamera(left=left_calib, right=right_calib)
        sc = StereoCalibrator(stereo_camera_model=model,
                              boards=[ChessboardInfo(config["rows"],
                                                     config["cols"],
                                                     config["size"])],
                              pattern=check_pattern(config),
                              invert=config["invert"],
                              name=left_name,
                              name2=right_name)
        sc.cal(left_image_list, right_image_list)
        sc.report()
        Console.info('Writting calibration to '"'{}'"''.format(output_file))
        with open(output_file, 'w') as f:
            f.write(sc.yaml())
    except CalibrationException:
        Console.error('The calibration pattern was not found in the images.')


def calibrate_laser(left_name, left_filepath, left_extension,
                    right_name, right_filepath, right_extension,
                    stereo_calibration_file,
                    config,
                    output_file):
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
        lc = LaserCalibrator(stereo_camera_model=model)
        lc.cal(left_image_list, right_image_list)
        Console.info('Writting calibration to '"'{}'"''.format(output_file))
        with open(output_file, 'w') as f:
            f.write(lc.yaml())
    except CalibrationException:
        Console.error('The calibration pattern was not found in the images.')


class Calibrator():
    def __init__(self, filepath, force_overwite=False):
        filepath = Path(filepath).resolve()
        self.filepath = get_raw_folder(filepath)
        self.fo = force_overwite

        # load mission.yaml config file
        mission_file = get_raw_folder(filepath) / 'mission.yaml'
        Console.info('Loading mission.yaml at {0}'.format(mission_file))
        self.mission = Mission(mission_file)

        self.calibration_path = get_processed_folder(filepath) / 'calibration'
        if not self.calibration_path.exists():
            self.calibration_path.mkdir(parents=True)

        calibration_config_file = get_config_folder(filepath)  / 'calibration.yaml'
        if calibration_config_file.exists():
            Console.info("Loading existing calibration.yaml at {}".format(calibration_config_file))
        else:
            root = Path(__file__).parents[1]
            default_file = root / 'default_yaml' / 'default_calibration.yaml'
            Console.warn("Cannot find {}, generating default from {}".format(
                calibration_config_file, default_file))
            # save localisation yaml to processed directory
            default_file.copy(calibration_config_file)

        with calibration_config_file.open('r') as stream:
            self.calibration_config = yaml.safe_load(stream)

    def mono(self):
        for c in self.calibration_config['cameras']:
            cam_name = c['name']
            # Find if the calibration file exists
            calibration_file = self.calibration_path / str('mono_' + cam_name + '.yaml')
            Console.info('Looking for a calibration file at ' + calibration_file)
            if calibration_file.exists() and not self.fo:
                Console.warn('The camera ' + c.name + ' has already been calibrated. If you want to overwrite the calibration, use the -F flag.')
            else:
                Console.info('The camera is not calibrated, running mono calibration...')
                calibrate_mono(cam_name,
                               self.filepath / str(c['camera_calibration']['path']),
                               '*.' + str(c['camera_calibration']['extension']),
                               self.calibration_config['camera_calibration'],
                               calibration_file)

    def stereo(self):
        if len(self.calibration_config['cameras']) > 1:
            c0 = self.calibration_config['cameras'][0]
            c1 = self.calibration_config['cameras'][1]
            calibration_file = self.calibration_path / str('stereo_' + c0['name'] + '_' + c1['name'] + '.yaml')
            Console.info('Looking for a calibration file at ' + calibration_file)
            if calibration_file.exists() and not self.fo:
                Console.warn('The stereo pair ' + c0['name'] + '_' + c1['name'] + ' has already been calibrated. If you want to overwrite the calibration, use the -F flag.')
            else:
                Console.info('The stereo camera is not calibrated, running stereo calibration...')
                left_name = ''
                left_filepath = ''
                left_extension = ''
                right_name = ''
                right_filepath = ''
                right_extension = ''
                if self.mission.image.format == 'seaxerocks_3':
                    left_name = c0['name']
                    left_filepath = self.filepath / str(c0['camera_calibration']['path'])
                    left_extension = '*.' + str(c0['camera_calibration']['extension'])
                    right_name = c1['name']
                    right_filepath = self.filepath / str(c1['camera_calibration']['path'])
                    right_extension = '*.' + str(c1['camera_calibration']['extension'])
                elif self.mission.image.format == 'acfr_standard':
                    left_name = c0['name']
                    left_filepath = self.filepath / str(c0['camera_calibration']['path'])
                    left_extension = '*LC16.' + str(c0['camera_calibration']['extension'])
                    right_name = c1['name']
                    right_filepath = self.filepath / str(c1['camera_calibration']['path'])
                    right_extension = '*RC16.' + str(c1['camera_calibration']['extension'])

                left_calibration_file = self.calibration_path / str('mono_' + left_name + '.yaml')
                right_calibration_file = self.calibration_path / str('mono_' + right_name + '.yaml')
                if not left_calibration_file.exists() or not right_calibration_file.exists():
                    if not left_calibration_file.exists():
                        Console.warn('Could not find a monocular calibration file ' + left_calibration_file + '...')
                    if not right_calibration_file.exists():
                        Console.warn('Could not find a monocular calibration file ' + right_calibration_file + '...')
                    self.mono()
                if left_calibration_file.exists() and right_calibration_file.exists():
                    Console.info('Loading previous monocular calibrations at \
                                  \n\t * {}\n\t * {}'.format(str(left_calibration_file), str(right_calibration_file)))

                calibrate_stereo(left_name,
                                 left_filepath,
                                 left_extension,
                                 left_calibration_file,
                                 right_name,
                                 right_filepath,
                                 right_extension,
                                 right_calibration_file,
                                 self.calibration_config['camera_calibration'],
                                 calibration_file)
        if len(self.calibration_config['cameras']) > 2:
            c0 = self.calibration_config['cameras'][0]
            c2 = self.calibration_config['cameras'][2]
            calibration_file = self.calibration_path / str('stereo_' + c0['name'] + '_' + c2['name'] + '.yaml')
            if calibration_file.exists() and not self.fo:
                Console.warn('The stereo pair ' + c0['name'] + '_' + c2['name'] + ' has already been calibrated. If you want to overwrite the calibration, use the -F flag.')
            else:
                calibrate_stereo(c0['name'],
                                 self.filepath / str(c0['camera_calibration']['path']),
                                 '*.' + str(c0['camera_calibration']['extension']),
                                 c2['name'],
                                 self.filepath / str(c2['camera_calibration']['path']),
                                 '*.' + str(c2['camera_calibration']['extension']),
                                 self.calibration_config['camera_calibration'],
                                 calibration_file)

    def laser(self):
        if 'laser_calibration' in self.calibration_config['cameras'][0]:
            c0 = self.calibration_config['cameras'][0]
            c1 = self.calibration_config['cameras'][1]
            c2 = self.calibration_config['cameras'][2]
            calibration_file = self.calibration_path / 'laser_calibration.yaml'
            Console.info('Looking for a calibration file at ' + calibration_file)
            if calibration_file.exists() and not self.fo:
                Console.warn('The laser planes from cameras ' + c0['name'] + ' and ' + c1['name'] + ' have already been calibrated. If you want to overwite the calibration, use the -F flag.')
            else:
                Console.info('The laser planes are not calibrated, running laser calibration...')
                # Check if the stereo pair has already been calibrated
                stereo_calibration_file = self.calibration_path / str('stereo_' + c0['name'] + '_' + c1['name'] + '.yaml')
                if not stereo_calibration_file.exists():
                    Console.warn('Could not find a stereo calibration file ' + stereo_calibration_file + '...')
                    self.stereo()
                left_name = c0['name']
                left_filepath = self.filepath / str(c0['laser_calibration']['path'])
                left_extension = '*.' + str(c0['laser_calibration']['extension'])
                right_name = c1['name']
                right_filepath = self.filepath / str(c1['laser_calibration']['path'])
                right_extension = '*.' + str(c1['laser_calibration']['extension'])
                calibrate_laser(left_name,
                                left_filepath,
                                left_extension,
                                right_name,
                                right_filepath,
                                right_extension,
                                stereo_calibration_file,
                                self.calibration_config['laser_calibration'],
                                calibration_file)

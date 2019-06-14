# Encoding: utf-8

from auv_nav.tools.console import Console
from auv_nav.parsers.vehicle import Vehicle
from auv_nav.parsers.mission import Mission
from auv_nav.calibration.calibrator import CalibrationException
from auv_nav.calibration.calibrator import MonoCalibrator
from auv_nav.calibration.calibrator import ChessboardInfo
from auv_nav.calibration.calibrator import Patterns
from auv_nav.calibration.calibrator import StereoCalibrator
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


def calibrate_mono(name, filepath, extension, output_file):
    Console.info('Looking for {} calibration images in {}'.format(extension, filepath))
    image_list = collect_image_files(filepath, extension)
    Console.info('Found ' + str(len(image_list)) + ' images.')
    if len(image_list) < 8:
        Console.error('Too few images. Try to get more.')
        return
    mc = MonoCalibrator(boards=[ChessboardInfo(7, 7, 0.1)],
                        pattern=Patterns.Circles,
                        invert=True,
                        name=name)
    try:
        mc.cal(image_list)
        mc.report()
        Console.info('Writting calibration to '"'calibration.yaml'"' ')
        with open(output_file, 'w') as f:
            f.write(mc.yaml())
    except CalibrationException:
        Console.error('The calibration pattern was not found in the images.')


def calibrate_stereo(left_name, left_filepath, right_name, right_filepath, left_extension, right_extension, output_file):
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
        sc = StereoCalibrator(boards=[ChessboardInfo(7, 7, 0.1)],
                              pattern=Patterns.Circles,
                              invert=True,
                              name=left_name,
                              name2=right_name)
        sc.cal(left_image_list, right_image_list)
        sc.report()
        Console.info('Writting calibration to '"'calibration_left.yaml'"' and '"'calibration_right.yaml'"'')
        with open(output_file, 'w') as f:
            f.write(sc.yaml())
    except CalibrationException:
        Console.error('The calibration pattern was not found in the images.')


def calibrate_laser(filepath, extension, force_overwite):
    Console.warn('Not implemented')


class Calibrator():
    def __init__(self, filepath, extension, force_overwite=False):
        self.extension = extension
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
            if calibration_file.exists() and not self.fo:
                Console.warn('The camera ' + c.name + ' has already been calibrated. If you want to overwrite the calibration, use the -F flag.')
            else:
                calibrate_mono(cam_name,
                               self.filepath / str(c['camera_calibration']),
                               '*.' + self.extension,
                               calibration_file)

    def stereo(self):
        if len(self.calibration_config['cameras']) > 1:
            c0 = self.calibration_config['cameras'][0]
            c1 = self.calibration_config['cameras'][1]
            calibration_file = self.calibration_path / str('stereo_' + c0['name'] + '_' + c1['name'] + '.yaml')
            if calibration_file.exists() and not self.fo:
                Console.warn('The stereo pair ' + c0['name'] + '_' + c1['name'] + ' has already been calibrated. If you want to overwrite the calibration, use the -F flag.')
            else:

                if self.mission.image.format == 'seaxerocks_3':
                    calibrate_stereo(c0['name'], self.filepath / str(c0['camera_calibration']),
                                 c1['name'], self.filepath / str(c1['camera_calibration']),
                                 '*.' + self.extension,
                                 '*.' + self.extension,
                                 calibration_file)
                elif self.mission.image.format == 'acfr_standard':
                    calibrate_stereo(c0['name'], self.filepath / str(c0['camera_calibration']),
                                 c1['name'], self.filepath / str(c1['camera_calibration']),
                                 '*LC16.' + self.extension,
                                 '*RC16.' + self.extension,
                                 calibration_file)
        if len(self.calibration_config['cameras']) > 2:
            c0 = self.calibration_config['cameras'][0]
            c2 = self.calibration_config['cameras'][2]
            calibration_file = self.calibration_path / str('stereo_' + c0['name'] + '_' + c2['name'] + '.yaml')
            if calibration_file.exists() and not self.fo:
                Console.warn('The stereo pair ' + c0['name'] + '_' + c2['name'] + ' has already been calibrated. If you want to overwrite the calibration, use the -F flag.')
            else:
                calibrate_stereo(c0['name'], self.filepath / str(c0['camera_calibration']),
                                 c2['name'], self.filepath / str(c2['camera_calibration']),
                                 '*.' + self.extension,
                                 '*.' + self.extension,
                                 calibration_file)

    def laser(self):
        Console.warn('Not implemented')
        """
        if self.mission.cameras[0].laser_calibration:
            c0 = self.mission.cameras[0]
            c1 = self.mission.cameras[1]
            c2 = self.mission.cameras[2]
            calibration_file = calibration_path / 'laser_calibration.yaml'
            if calibration_file.exists() and not force_overwite:
                Console.warn('The camera ' + c.name + ' has already been calibrated. If you want to overwite the calibration, use the -F flag.')
            else:
                calibrate_laser(c0.laser_calibration,
                                c1.laser_calibration,
                                c2.laser_calibration)
        """

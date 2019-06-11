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
from pathlib import Path
import cv2
import glob


class Calibrator():
    def __init__(self, filepath, force_overwite=False):
        filepath = Path(filepath).resolve()
        self.filepath = get_raw_folder(filepath)

        # load mission.yaml config file
        mission_file = filepath / 'mission.yaml'
        vehicle_file = filepath / 'vehicle.yaml'
        mission_file = get_raw_folder(mission_file)
        vehicle_file = get_raw_folder(vehicle_file)
        Console.info('Loading mission.yaml at {0}'.format(mission_file))
        self.mission = Mission(mission_file)

        if self.mission.calibration is None:
            Console.error('Your mission file does not contain calibration information. Please edit the mission.yaml file located at:\n' + mission_file)

        Console.info('Loading vehicle.yaml at {0}'.format(vehicle_file))
        self.vehicle = Vehicle(vehicle_file)

        calibration_path = get_processed_folder(filepath) / 'calibration'
        if not calibration_path.exists():
            calibration_path.mkdir(parents=True)

        Console.info('Locating calibration files...')
        for c in self.mission.cameras:
            # Find if the calibration file exists
            calibration_file = calibration_path / 'mono_' + c.name + '.yaml'
            if calibration_file.exists() and not force_overwite:
                Console.warn('The camera ' + c.name + ' has already been calibrated. If you want to overwite the calibration, use the -F flag.')
            else:
                calibrate_mono(c.camera_calibration, calibration_file)
        if len(self.mission.cameras) > 1:
            c0 = self.mission.cameras[0]
            c1 = self.mission.cameras[1]
            calibration_file = calibration_path / 'stereo_' + c0.name + '_' + c1.name + '.yaml'
            if calibration_file.exists() and not force_overwite:
                Console.warn('The camera ' + c.name + ' has already been calibrated. If you want to overwite the calibration, use the -F flag.')
            else:
                calibrate_stereo(c0.camera_calibration,
                                 c1.camera_calibration)
        if len(self.mission.cameras) > 2:
            c0 = self.mission.cameras[0]
            c2 = self.mission.cameras[2]
            calibration_file = calibration_path / 'stereo_' + c.name + '.yaml'
            if calibration_file.exists() and not force_overwite:
                Console.warn('The camera ' + c.name + ' has already been calibrated. If you want to overwite the calibration, use the -F flag.')
            else:
                calibrate_stereo(c0.camera_calibration,
                                 c2.camera_calibration)
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


def collect_image_files(image_dir, file_pattern):
    # TODO try all different formats: png, jpeg, tif
    images = glob.glob(image_dir + '/' + file_pattern)
    images.sort()
    return images


def calibrate_mono(filepath, extension, force_overwite):
    Console.warn('calibrate_mono')
    image_list = collect_image_files(filepath, '*.' + extension)
    Console.info('Found ' + str(len(image_list)) + ' images.')
    if len(image_list) < 8:
        Console.error('Too few images. Try to get more.')
        return
    mc = MonoCalibrator(boards=[ChessboardInfo(7, 7, 0.1)],
                        pattern=Patterns.Circles,
                        invert=True)
    try:
        mc.cal(image_list)
        mc.report()
        Console.info('Writting calibration to '"'calibration.yaml'"' ')
        with open('calibration.yaml', 'w') as f:
            f.write(mc.yaml())
    except CalibrationException:
        Console.error('The calibration pattern was not found in the images.')


def calibrate_stereo(left_filepath, right_filepath, extension, force_overwite):
    Console.warn('calibrate_stereo')
    left_image_list = collect_image_files(left_filepath, '*.' + extension)
    right_image_list = collect_image_files(right_filepath, '*.' + extension)
    Console.info('Found ' + str(len(left_image_list)) + ' left images.')
    Console.info('Found ' + str(len(right_image_list)) + ' right images.')
    if len(left_image_list) < 8 or len(right_image_list) < 8:
        Console.error('Too few images. Try to get more.')
        return
    try:
        sc = StereoCalibrator(boards=[ChessboardInfo(7, 7, 0.1)],
                              pattern=Patterns.Circles,
                              invert=True)
        sc.cal(left_image_list, right_image_list)
        sc.report()
        lyaml, ryaml, eyaml = sc.yaml()
        Console.info('Writting calibration to '"'calibration_left.yaml'"' and '"'calibration_right.yaml'"'')
        with open('calibration_left.yaml', 'w') as f:
            f.write(lyaml)
        with open('calibration_right.yaml', 'w') as f:
            f.write(ryaml)
        with open('calibration_extrinsics.yaml', 'w') as f:
            f.write(eyaml)
    except CalibrationException:
        Console.error('The calibration pattern was not found in the images.')


def calibrate_laser(filepath, extension, force_overwite):
    Console.warn('Not implemented')

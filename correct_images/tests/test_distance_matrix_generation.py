import unittest
import yaml
import os
from pathlib import Path
from auv_nav.camera_system import *
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder
from auv_nav.parsers.mission import *
from auv_nav.camera_system import *
from correct_images.corrector import *
from correct_images.parser import *
from auv_nav.tools.console import Console
import glob



ACFR_IMG_WIDTH = 2464
ACFR_IMG_HEIGHT = 2056
ACFR_IMG_CHANNELS = 1

SX3_IMG_WIDTH = 1280 
SX3_IMG_HEIGHT = 1024
SX3_IMG_CHANNELS = 1

LM165_IMG_WIDTH = 1392
LM165_IMG_HEIGHT = 1040

BIOCAM_IMG_WIDTH = 2560
BIOCAM_IMG_HEIGHT = 2160
BIOCAM_IMG_CHANNELS = 1


# Requirements:
# -----------------------------------------
# 1. path to mission.yaml for each dataset to be provided in test.yaml
# 2. Make sure the JSON path is provided correctly in the correct_config.yaml within configuration

class testCaseCorrector(unittest.TestCase):
	
	def load_params(self):
		file = './test.yaml'
		file_path = Path(file).resolve()
		with file_path.open('r') as stream:
			params = yaml.safe_load(stream)
		self.path_acfr = params['path_mission_acfr']
		self.path_sx3 = params['path_mission_sx3']
		self.path_biocam = params['path_mission_biocam']

		root = Path(__file__).resolve().parents[1]
		self.path_acfr_full = root / self.path_acfr
		self.path_sx3_full = root / self.path_sx3
		self.path_biocam_full = root / self.path_biocam
		

		# resolve paths to config folders
		path_config_folder_acfr = get_config_folder(self.path_acfr_full)
		path_config_folder_sx3 = get_config_folder(self.path_sx3_full)
		path_config_folder_biocam = get_config_folder(self.path_biocam_full)

		# load correct config parameters for acfr, sx3 and biocam
		path_correct_images = path_config_folder_acfr / 'correct_images.yaml'
		self.correct_config_acfr = CorrectConfig(path_correct_images)
		self.camera_file_acfr = path_config_folder_acfr / 'camera.yaml'

		path_correct_images = path_config_folder_sx3 / 'correct_images.yaml'
		self.correct_config_sx3 = CorrectConfig(path_correct_images)
		self.camera_file_sx3 = path_config_folder_sx3 / 'camera.yaml'

		path_correct_images = path_config_folder_biocam / 'correct_images.yaml'
		self.correct_config_biocam = CorrectConfig(path_correct_images)
		self.camera_file_biocam = path_config_folder_biocam / 'camera.yaml'


	# Feature being tested: Corrector::get_distance_matrix()
	def test_acfr_get_distance_matrix(self):

		self.load_params()

		cs = CameraSystem(self.camera_file_acfr, self.path_acfr_full)

		for camera in cs.cameras:
			if len(camera.image_list) == 0:
				Console.quit('No images found for the camera at the path provided...')
			else:
				corrector = Corrector(True, camera, self.correct_config_acfr, self.path_acfr_full)
				corrector.load_generic_config_parameters()
				corrector.load_camera_specific_config_parameters()
				corrector.get_imagelist()
				corrector.create_output_directories()
				corrector.generate_distance_matrix()
				self.assertEqual(len(corrector._imagelist), 
				len(corrector.distance_matrix_numpy_filelist),
				'Length of distance matrix filelist does not match with imagelist')

				for idx in corrector.altitude_based_filtered_indices:
					distance_matrix = np.load(corrector.distance_matrix_numpy_filelist[idx])
					self.assertEqual(distance_matrix.shape[0], ACFR_IMG_HEIGHT, 'Dimension mismatch: height')
					self.assertEqual(distance_matrix.shape[1], ACFR_IMG_WIDTH, 'Dimension mismatch: width')
					out_of_range = (np.abs(distance_matrix) < corrector.altitude_min).any() or (np.abs(distance_matrix) > corrector.altitude_max).any()
					self.assertEqual(out_of_range, False, 'distance matrix values out of altitude bounds error')

		

	def test_sx3_get_distance_matrix(self):

		self.load_params()

		cs = CameraSystem(self.camera_file_sx3, self.path_sx3_full)

		for camera in cs.cameras:
			if len(camera.image_list) == 0:
				Console.quit('No images found for the camera at the path provided...')
			else: 
				corrector = Corrector(True, camera, self.correct_config_sx3, self.path_sx3_full)
				corrector.load_generic_config_parameters()
				corrector.load_camera_specific_config_parameters()
				corrector.get_imagelist()
				corrector.create_output_directories()
				corrector.generate_distance_matrix()
				self.assertEqual(len(corrector._imagelist), 
				len(corrector.distance_matrix_numpy_filelist),
				'Length of distance matrix filelist does not match with imagelist')

				for idx in corrector.altitude_based_filtered_indices:
					distance_matrix = np.load(corrector.distance_matrix_numpy_filelist[idx])
					if camera.name == 'LM165':
						self.assertEqual(distance_matrix.shape[0], LM165_IMG_HEIGHT, 'Dimension mismatch: height')
						self.assertEqual(distance_matrix.shape[1], LM165_IMG_WIDTH, 'Dimension mismatch: width')
					else:
						self.assertEqual(distance_matrix.shape[0], SX3_IMG_HEIGHT, 'Dimension mismatch: height')
						self.assertEqual(distance_matrix.shape[1], SX3_IMG_WIDTH, 'Dimension mismatch: width')
					out_of_range = (np.abs(distance_matrix) < corrector.altitude_min).any() or (np.abs(distance_matrix) > corrector.altitude_max).any()
					self.assertEqual(out_of_range, False, 'distance matrix values out of altitude bounds error')
	
	def test_biocam_get_distance_matrix(self):

		self.load_params()

		cs = CameraSystem(self.camera_file_biocam, self.path_biocam_full)

		for camera in cs.cameras:
			if len(camera.image_list) == 0:
				Console.quit('No images found for the camera at the path provided...')
			else:
				corrector = Corrector(True, camera, self.correct_config_biocam, self.path_biocam_full)
				corrector.load_generic_config_parameters()
				corrector.load_camera_specific_config_parameters()
				corrector.get_imagelist()
				corrector.create_output_directories()
				corrector.generate_distance_matrix()
				self.assertEqual(len(corrector._imagelist), 
				len(corrector.distance_matrix_numpy_filelist),
				'Length of distance matrix filelist does not match with imagelist')

				for idx in corrector.altitude_based_filtered_indices:
					distance_matrix = np.load(corrector.distance_matrix_numpy_filelist[idx])
					self.assertEqual(distance_matrix.shape[0], BIOCAM_IMG_HEIGHT, 'Dimension mismatch: height')
					self.assertEqual(distance_matrix.shape[1], BIOCAM_IMG_WIDTH, 'Dimension mismatch: width')
					out_of_range = (np.abs(distance_matrix) < corrector.altitude_min).any() or (np.abs(distance_matrix) > corrector.altitude_max).any()
					self.assertEqual(out_of_range, False, 'distance matrix values out of altitude bounds error')
	
    		
if __name__ == '__main__':
    unittest.main()







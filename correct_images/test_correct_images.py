import unittest
import os
from correct_images.parameters_ import *
from correct_images.draft_camera_system import *
from correct_images.draft_corrector import *

ACFR_IMG_WIDTH = 2464
ACFR_IMG_HEIGHT = 2056
ACFR_IMG_CHANNELS = 1

SX3_IMG_WIDTH = 1280 
SX3_IMG_HEIGHT = 1024
SX3_IMG_CHANNELS = 1

BIOCAM_IMG_WIDTH = 2560
BIOCAM_IMG_HEIGHT = 2160
BIOCAM_IMG_CHANNELS = 1


# Requirements:
# write the following to test.yaml file:
# 1. path to mission.yaml file in the test.yaml file
# 2. camera_system: acfr_standard / seaxerocks_3 / biocam


class CorrectorTestCase(unittest.TestCase):
	
	def read_test_yaml(self):
		with open('./test.yaml') as params:
			path_ = params['path']
			imaging_system = params['image_system']
		return path_

	# Feature being tested: Corrector::get_image_properties()
	def test_get_image_properties(self):
		path_, imaging_system = self.read_test_yaml()
		root = Path(__file__).parents[1]
		if imaging_system == 'acfr_standard':
        	camera_file = 'default_yaml/ts1/SSK17-01/camera.yaml'
        elif imaging_system == 'seaxerocks_3':
        	camera_file = 'default_yaml/ae2000/YK17-23C/camera.yaml'
    	elif imaging_system == 'biocam':
    		camera_file = 'default_yaml/as6/DY109/camera.yaml'

		cs = CameraSystem(root / camera_file)
		
		for camera in cs.cameras:
			corrector = Corrector(camera, path_)
			result_img_width, result_image_height, result_image_channels = corrector.get_image_properties()
			if cs.camera_system == 'acfr_standard':
				self.assertEqual(result_img_width, ACFR_IMG_WIDTH, 'incorrect image width')
				self.assertEqual(result_image_height, ACFR_IMG_HEIGHT, 'incorrect image height')
				self.assertEqual(result_image_channels, ACFR_IMG_CHANNELS, 'incorrect image channels')
			elif cs.camera_system == 'seaxerocks_3':
				self.assertEqual(result_img_width, SX3_IMG_WIDTH, 'incorrect image width')
				self.assertEqual(result_image_height, SX3_IMG_HEIGHT, 'incorrect image height')
				self.assertEqual(result_image_channels, SX3_IMG_CHANNELS, 'incorrect image channels')
			elif cs.camera_system == 'biocam':
				self.assertEqual(result_img_width, BIOCAM_IMG_WIDTH, 'incorrect image width')
				self.assertEqual(result_image_height, BIOCAM_IMG_HEIGHT, 'incorrect image height')
				self.assertEqual(result_image_channels, BIOCAM_IMG_CHANNELS, 'incorrect image channels')

	# Feature being tested: Corrector::get_distance_matrix()
	def test_get_distance_matrix(self):
		path_, imaging_system = self.read_test_yaml()
		root = Path(__file__).parents[1]
		if imaging_system == 'acfr_standard':
        	camera_file = 'default_yaml/ts1/SSK17-01/camera.yaml'
        elif imaging_system == 'seaxerocks_3':
        	camera_file = 'default_yaml/ae2000/YK17-23C/camera.yaml'
    	elif imaging_system == 'biocam':
    		camera_file = 'default_yaml/as6/DY109/camera.yaml'

		cs = CameraSystem(root / camera_file)

		for camera in cs.cameras:
			corrector = Corrector(camera, path_)
			result_distance_metric, result_distance_matrix = corrector.get_distance_matrix()
			matrix_dims = result_distance_matrix.shape

			if result_distance_metric == 'altitude' or result_distance_metric == 'depth_map':
				self.assertEqual(len(matrix_dims), 2, 'Distance matrix dimensions are incorrect')
				if cs.camera_system == 'acfr_standard':
					self.assertEqual(matrix_dims[0], ACFR_IMG_WIDTH, 'incorrect dimension for distance matrix')
					self.assertEqual(matrix_dims[1], ACFR_IMG_HEIGHT, 'incorrect dimension for distance matrix')
				elif cs.camera_system == 'seaxerocks_3':
					self.assertEqual(matrix_dims[0], SX3_IMG_WIDTH, 'incorrect dimension for distance matrix')
					self.assertEqual(matrix_dims[1], SX3_IMG_HEIGHT, 'incorrect dimension for distance matrix')
				elif cs.camera_system == 'biocam':
					self.assertEqual(matrix_dims[0], BIOCAM_IMG_WIDTH, 'incorrect dimension for distance matrix')
					self.assertEqual(matrix_dims[1], BIOCAM_IMG_HEIGHT, 'incorrect dimension for distance matrix')
			elif result_distance_metric == 'none':
				self.assertEqual(len(matrix_dims), 0, 'Distance matrix dimensions are incorrect')
	
	# Feature being tested: Corrector::calculate_attenuation_parameters()
	def test_calculate_attenuation_parameters(self):
		path_, imaging_system = self.read_test_yaml()
		root = Path(__file__).parents[1]
		if imaging_system == 'acfr_standard':
        	camera_file = 'default_yaml/ts1/SSK17-01/camera.yaml'
        elif imaging_system == 'seaxerocks_3':
        	camera_file = 'default_yaml/ae2000/YK17-23C/camera.yaml'
    	elif imaging_system == 'biocam':
    		camera_file = 'default_yaml/as6/DY109/camera.yaml'

		cs = CameraSystem(root / camera_file)

		for camera in cs.cameras:
			corrector = Corrector(camera, path_)

			# full path to attenuation parameters output folder for given camera
			output_params_path = corrector.calculate_attenuation_parameters() 

			folders = []
			files = []
			channel_files = []
			# r=root, d=directories, f = files
			for r, d, f in os.walk(output_params_path):
			    for folder in d:
			        folders.append(folder)
	        	for file in f:
        			files.append(file)
        			if 'ch*.png' in file:
        				channel_files.append(file)

        	self.assertIn('bayer_img_mean_atn_crr', folders, 'Attenuation mean parameters do not exist')
        	self.assertIn('bayer_img_std_atn_crr', folders, 'Attenuation std parameters do not exist')
        	self.assertIn('atn_crr_params.npy', files, 'attenuation parameters numpy not generated')
        	self.assertIn('bayer_img_mean_atn_crr.npy', files, 'bayer mean attenuation parameters numpy not generated')
        	self.assertIn('bayer_img_std_atn_crr.npy', files, 'bayer std attenuation parameters numpy not generated')
        	self.assertEqual(len(channel_files), 12, 'inadequate images per channel')
    	










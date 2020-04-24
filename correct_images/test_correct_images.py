import unittest
import os
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

elif cs.camera_system == 'seaxerocks_3':
					
				elif cs.camera_system == 'biocam':
					self.assertEqual(matrix_dims[1], BIOCAM_IMG_WIDTH, 'incorrect dimension for distance matrix')
					self.assertEqual(matrix_dims[2], BIOCAM_IMG_HEIGHT, 'incorrect dimension for distance matrix')

# Requirements:
# write the following to test.yaml file:
# 1. path to mission.yaml file in the test.yaml file for each dataset

class CorrectorTestCase(unittest.TestCase):
	def __init__(self):
		with open('./test.yaml') as params:
			self.path_acfr = params['path_mission_acfr']
			self.path_sx3 = params['path_mission_sx3']
			self.path_biocam = params['path_mission_biocam']
		
		self.root = Path(__file__).parents[1]
		self.camera_file_acfr = 'default_yaml/ts1/SSK17-01/camera.yaml'
		self.camera_file_sx3 = 'default_yaml/ae2000/YK17-23C/camera.yaml'
		self.camera_file_biocam = 'default_yaml/as6/DY109/camera.yaml'

	# Feature being tested: Corrector::get_image_properties()
	def test_acfr_get_image_properties(self):
		
		cs = CameraSystem(self.root / self.camera_file_acfr)
		
		for camera in cs.cameras:
			corrector = Corrector(camera, self.path_acfr)
			result_img_width, result_image_height, result_image_channels = corrector.get_image_properties()
			self.assertEqual(result_img_width, ACFR_IMG_WIDTH, 'incorrect image width')
			self.assertEqual(result_image_height, ACFR_IMG_HEIGHT, 'incorrect image height')
			self.assertEqual(result_image_channels, ACFR_IMG_CHANNELS, 'incorrect image channels')
	
	def test_sx3_get_image_properties(self):

		cs = CameraSystem(self.root / self.camera_file_sx3)
		
		for camera in cs.cameras:
			corrector = Corrector(camera, self.path_sx3)
			result_img_width, result_image_height, result_image_channels = corrector.get_image_properties()
			self.assertEqual(result_img_width, SX3_IMG_WIDTH, 'incorrect image width')
			self.assertEqual(result_image_height, SX3_IMG_HEIGHT, 'incorrect image height')
			self.assertEqual(result_image_channels, SX3_IMG_CHANNELS, 'incorrect image channels')
	
	def test_biocam_get_image_properties(self):

		cs = CameraSystem(self.root / self.camera_file_biocam)
		
		for camera in cs.cameras:
			corrector = Corrector(camera, self.path_biocam)
			result_img_width, result_image_height, result_image_channels = corrector.get_image_properties()
			self.assertEqual(result_img_width, BIOCAM_IMG_WIDTH, 'incorrect image width')
			self.assertEqual(result_image_height, BIOCAM_IMG_HEIGHT, 'incorrect image height')
			self.assertEqual(result_image_channels, BIOCAM_IMG_CHANNELS, 'incorrect image channels')	
	

	# Feature being tested: Corrector::get_distance_matrix()
	def test_acfr_get_distance_matrix(self):

		cs = CameraSystem(self.root / self.camera_file_acfr)

		for camera in cs.cameras:
			corrector = Corrector(camera, self.path_acfr)
			result_distance_metric, result_distance_matrix = corrector.get_distance_matrix()
			matrix_dims = result_distance_matrix.shape

			if result_distance_metric == 'altitude' or result_distance_metric == 'depth_map':
				self.assertEqual(len(matrix_dims), 3, 'Distance matrix dimensions are incorrect')
				if cs.camera_system == 'acfr_standard':
					self.assertEqual(matrix_dims[1], ACFR_IMG_WIDTH, 'incorrect dimension for distance matrix')
					self.assertEqual(matrix_dims[2], ACFR_IMG_HEIGHT, 'incorrect dimension for distance matrix')
				
			elif result_distance_metric == 'none':
				self.assertEqual(len(matrix_dims), 0, 'Distance matrix dimensions are incorrect')

	def test_sx3_get_distance_matrix(self):

		cs = CameraSystem(self.root / self.camera_file_sx3)

		for camera in cs.cameras:
			corrector = Corrector(camera, self.path_sx3)
			result_distance_metric, result_distance_matrix = corrector.get_distance_matrix()
			matrix_dims = result_distance_matrix.shape

			if result_distance_metric == 'altitude' or result_distance_metric == 'depth_map':
				self.assertEqual(len(matrix_dims), 3, 'Distance matrix dimensions are incorrect')
				if cs.camera_system == 'acfr_standard':
					self.assertEqual(matrix_dims[1], SX3_IMG_WIDTH, 'incorrect dimension for distance matrix')
					self.assertEqual(matrix_dims[2], SX3_IMG_HEIGHT, 'incorrect dimension for distance matrix')
				
			elif result_distance_metric == 'none':
				self.assertEqual(len(matrix_dims), 0, 'Distance matrix dimensions are incorrect')

	def test_biocam_get_distance_matrix(self):

		cs = CameraSystem(self.root / self.camera_file_biocam)

		for camera in cs.cameras:
			corrector = Corrector(camera, self.path_biocam)
			result_distance_metric, result_distance_matrix = corrector.get_distance_matrix()
			matrix_dims = result_distance_matrix.shape

			if result_distance_metric == 'altitude' or result_distance_metric == 'depth_map':
				self.assertEqual(len(matrix_dims), 3, 'Distance matrix dimensions are incorrect')
				if cs.camera_system == 'acfr_standard':
					self.assertEqual(matrix_dims[1], BIOCAM_IMG_WIDTH, 'incorrect dimension for distance matrix')
					self.assertEqual(matrix_dims[2], BIOCAM_IMG_HEIGHT, 'incorrect dimension for distance matrix')
				
			elif result_distance_metric == 'none':
				self.assertEqual(len(matrix_dims), 0, 'Distance matrix dimensions are incorrect')

    		
	# Feature being tested: Corrector::calculate_attenuation_parameters()
	def test_acfr_calculate_attenuation_parameters(self):

		cs = CameraSystem(self.root / self.camera_file_acfr)

		for camera in cs.cameras:
			corrector = Corrector(camera, self.path_acfr)

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
    
    def test_sx3_calculate_attenuation_parameters(self):

		cs = CameraSystem(self.root / self.camera_file_sx3)

		for camera in cs.cameras:
			corrector = Corrector(camera, self.path_sx3)

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

	def test_biocam_calculate_attenuation_parameters(self):

		cs = CameraSystem(self.root / self.camera_file_biocam)

		for camera in cs.cameras:
			corrector = Corrector(camera, self.path_biocam)

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









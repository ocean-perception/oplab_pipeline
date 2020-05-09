import unittest
import yaml
import os
import numpy as np
from pathlib import Path
from correct_images.corrector import *
from auv_nav.tools.console import Console
import pandas as pd
import tempfile


class testCaseCorrector(unittest.TestCase):
	def test_calculate_parameters_RGB(self):

		path_root = Path(__file__).resolve().parents[1]
		test_yaml_path = path_root / 'test.yaml'
		with test_yaml_path.open('r') as stream:
			params = yaml.safe_load(stream)

		image_1 = np.array(params['Test_images']['RGB']['image_1'])
		image_2 = np.array(params['Test_images']['RGB']['image_2'])
		image_3 = np.array(params['Test_images']['RGB']['image_3'])

		distance_1 = np.array(params['Distance']['distance_1'])
		distance_2 = np.array(params['Distance']['distance_2'])
		distance_3 = np.array(params['Distance']['distance_3'])


		# create dummy folders needed
		path_temp = tempfile.mkdtemp()
		message = 'Temporary folder created at ' + path_temp
		Console.info(message)
		path_dummy_folder = Path(path_temp).resolve() / 'dummy_folder'
		
		if not path_dummy_folder.exists():
			path_dummy_folder.mkdir(parents=True)

		path_memmap_folder = path_dummy_folder / 'memmap'
		if not path_memmap_folder.exists():
			path_memmap_folder.mkdir(parents=True)

		path_image_numpy = path_dummy_folder / 'image'
		if not path_image_numpy.exists():
			path_image_numpy.mkdir(parents=True)

		path_distance_numpy = path_dummy_folder / 'distance'
		if not path_distance_numpy.exists():
			path_distance_numpy.mkdir(parents=True)

		
		path_attenuation_parameters = path_dummy_folder / 'params'
		if not path_attenuation_parameters.exists():
			path_attenuation_parameters.mkdir()

		# create image and distance numpy files
		image1_npy = path_image_numpy / 'image_1.npy'
		image2_npy = path_image_numpy / 'image_2.npy'
		image3_npy = path_image_numpy / 'image_3.npy'

		distance1_npy = path_distance_numpy / 'distance_1.npy'
		distance2_npy = path_distance_numpy / 'distance_2.npy'
		distance3_npy = path_distance_numpy / 'distance_3.npy'

		np.save(image1_npy, image_1)
		np.save(image2_npy, image_2)
		np.save(image3_npy, image_3)
		np.save(distance1_npy, distance_1)
		np.save(distance2_npy, distance_2)
		np.save(distance3_npy, distance_3)

		image_numpy_list = []
		distance_numpy_list = []

		image_numpy_list.append(image1_npy)
		image_numpy_list.append(image2_npy)
		image_numpy_list.append(image3_npy)

		distance_numpy_list.append(distance1_npy)
		distance_numpy_list.append(distance2_npy)
		distance_numpy_list.append(distance3_npy)

		# create Corrector object
		corrector = Corrector(True)

		# set Corrector attributes needed
		corrector.image_height = image_1.shape[0]
		corrector.image_width = image_1.shape[1]
		corrector.image_channels = image_1.shape[2]

		corrector.correction_method = 'colour_correction'
		corrector.altitude_based_filtered_indices = [0, 1, 2]
		corrector.memmap_folder = path_memmap_folder

		corrector.bayer_numpy_filelist = image_numpy_list
		corrector.distance_matrix_numpy_filelist = distance_numpy_list
		corrector.attenuation_parameters_folder = path_attenuation_parameters


		corrector.altitude_min = 0.5
		corrector.altitude_max = 3.5

		corrector.smoothing = 'mean'

		# invoke the feature to be tested
		corrector.generate_attenuation_correction_parameters()

		image_mean_red = corrector.image_raw_mean[0]
		image_mean_green = corrector.image_raw_mean[1]
		image_mean_blue = corrector.image_raw_mean[2]

		image_std_red = corrector.image_raw_std[0]
		image_std_green = corrector.image_raw_std[1]
		image_std_blue = corrector.image_raw_std[2]

		image_corrected_mean_blue = corrector.image_corrected_mean[2]
		image_corrected_std_blue = corrector.image_corrected_std[2]


		parameters_red_a = corrector.image_attenuation_parameters[0,:,:,0]
		parameters_red_b = corrector.image_attenuation_parameters[0,:,:,1]
		parameters_red_c = corrector.image_attenuation_parameters[0,:,:,2]
		parameters_green_a = corrector.image_attenuation_parameters[1,:,:,0]
		parameters_green_b = corrector.image_attenuation_parameters[1,:,:,1]
		parameters_green_c = corrector.image_attenuation_parameters[1,:,:,2]
		parameters_blue_a = corrector.image_attenuation_parameters[2,:,:,0]
		parameters_blue_b = corrector.image_attenuation_parameters[2,:,:,1]
		parameters_blue_c = corrector.image_attenuation_parameters[2,:,:,2]

		image_mean_red_int = image_mean_red.astype(int)
		image_mean_green_int = image_mean_green.astype(int)
		image_mean_blue_int = image_mean_blue.astype(int)
		image_std_red_int = image_std_red.astype(int)
		image_std_green_int = image_std_green.astype(int)
		image_std_blue_int = image_std_blue.astype(int)
		image_corrected_mean_blue_int = image_corrected_mean_blue.astype(int)
		image_corrected_std_blue_int = image_corrected_std_blue.astype(int)
		parameters_red_a_int = parameters_red_a.astype(int)
		parameters_red_b_int = parameters_red_b.astype(int)
		parameters_red_c_int = parameters_red_c.astype(int)
		parameters_green_a_int = parameters_green_a.astype(int)
		parameters_green_b_int = parameters_green_b.astype(int)
		parameters_green_c_int = parameters_green_c.astype(int)
		parameters_blue_a_int = parameters_blue_a.astype(int)
		parameters_blue_b_int = parameters_blue_b.astype(int)
		parameters_blue_c_int = parameters_blue_c.astype(int)


		self.assertEqual((image_mean_red_int == 140).all(), True, 'Mean values are incorrect and not same for all pixels along red channel')
		self.assertEqual((image_std_red_int == 74).all(), True, 'Std values are incorrect and not same for all pixels along red channel')
		self.assertEqual((image_mean_green_int == 92).all(), True, 'Mean values are incorrect and not same for all pixels along green channel')
		self.assertEqual((image_std_green_int == 21).all(), True, 'Std values are incorrect and not same for all pixels along green channel')
		self.assertEqual((image_mean_blue_int == 60).all(), True, 'Mean values are incorrect and not same for all pixels along blue channel')
		self.assertEqual((image_std_blue_int == 0).all(), True, 'Std values are incorrect and not same for all pixels along blue channel')
		

		self.assertEqual((parameters_red_a_int == parameters_red_a_int[0,0]).all(), True, 'Parameter values are not same for all pixels along red channel')
		self.assertEqual((parameters_red_b_int == parameters_red_b_int[0,0]).all(), True, 'Parameter values are not same for all pixels along red channel')
		self.assertEqual((parameters_red_c_int == parameters_red_c_int[0,0]).all(), True, 'Parameter values are not same for all pixels along red channel')
		self.assertEqual((parameters_green_a_int == parameters_green_a_int[0,0]).all(), True, 'Parameter values are not same for all pixels along green channel')
		self.assertEqual((parameters_green_b_int == parameters_green_b_int[0,0]).all(), True, 'Parameter values are not same for all pixels along green channel')
		self.assertEqual((parameters_green_c_int == parameters_green_c_int[0,0]).all(), True, 'Parameter values are not same for all pixels along green channel')
		self.assertEqual((parameters_blue_a_int == parameters_blue_a_int[0,0]).all(), True, 'Parameter values are not same for all pixels along blue channell')
		self.assertEqual((parameters_blue_b_int == parameters_blue_b_int[0,0]).all(), True, 'Parameter values are not same for all pixels along blue channel')
		self.assertEqual((parameters_blue_c_int == parameters_blue_c_int[0,0]).all(), True, 'Parameter values are not same for all pixels along blue channel')

		
		self.assertEqual(np.array_equal(image_mean_blue_int, image_corrected_mean_blue_int), True, 'Corrected mean not same for raw mean on blue channel')
		self.assertEqual(np.array_equal(image_std_blue_int, image_corrected_std_blue_int), True, 'Corrected mean not same for raw mean on blue channel')

		
	def test_calculate_parameters_Black_White(self):

		path_root = Path(__file__).resolve().parents[1]
		test_yaml_path = path_root / 'test.yaml'
		with test_yaml_path.open('r') as stream:
			params = yaml.safe_load(stream)

		image_1 = np.array(params['Test_images']['Black_White']['image_1'])
		image_2 = np.array(params['Test_images']['Black_White']['image_2'])
		image_3 = np.array(params['Test_images']['Black_White']['image_3'])

		distance_1 = np.array(params['Distance']['distance_1'])
		distance_2 = np.array(params['Distance']['distance_2'])
		distance_3 = np.array(params['Distance']['distance_3'])


		# create dummy folders needed
		path_temp = tempfile.mkdtemp()
		message = 'Temporary folder created at ' + path_temp
		Console.info(message)
		path_dummy_folder = Path(path_temp).resolve() / 'dummy_folder'

		if not path_dummy_folder.exists():
			path_dummy_folder.mkdir(parents=True)

		path_memmap_folder = path_dummy_folder / 'memmap'
		if not path_memmap_folder.exists():
			path_memmap_folder.mkdir(parents=True)

		path_image_numpy = path_dummy_folder / 'image'
		if not path_image_numpy.exists():
			path_image_numpy.mkdir(parents=True)

		path_distance_numpy = path_dummy_folder / 'distance'
		if not path_distance_numpy.exists():
			path_distance_numpy.mkdir(parents=True)

		path_attenuation_parameters = path_dummy_folder / 'params'
		if not path_attenuation_parameters.exists():
			path_attenuation_parameters.mkdir()

		# create image and distance numpy files
		image1_npy = path_image_numpy / 'image_1.npy'
		image2_npy = path_image_numpy / 'image_2.npy'
		image3_npy = path_image_numpy / 'image_3.npy'

		distance1_npy = path_distance_numpy / 'distance_1.npy'
		distance2_npy = path_distance_numpy / 'distance_2.npy'
		distance3_npy = path_distance_numpy / 'distance_3.npy'

		np.save(image1_npy, image_1)
		np.save(image2_npy, image_2)
		np.save(image3_npy, image_3)
		np.save(distance1_npy, distance_1)
		np.save(distance2_npy, distance_2)
		np.save(distance3_npy, distance_3)

		image_numpy_list = []
		distance_numpy_list = []

		image_numpy_list.append(image1_npy)
		image_numpy_list.append(image2_npy)
		image_numpy_list.append(image3_npy)

		distance_numpy_list.append(distance1_npy)
		distance_numpy_list.append(distance2_npy)
		distance_numpy_list.append(distance3_npy)

		# create Corrector object
		corrector = Corrector(True)

		# set Corrector attributes needed
		corrector.image_height = image_1.shape[0]
		corrector.image_width = image_1.shape[1]
		corrector.image_channels = 1

		corrector.correction_method = 'colour_correction'
		corrector.altitude_based_filtered_indices = [0, 1, 2]
		corrector.memmap_folder = path_memmap_folder

		corrector.bayer_numpy_filelist = image_numpy_list
		corrector.distance_matrix_numpy_filelist = distance_numpy_list
		corrector.attenuation_parameters_folder = path_attenuation_parameters

		corrector.altitude_min = 0.5
		corrector.altitude_max = 3.5

		corrector.smoothing = 'mean'

		# invoke the feature to be tested
		corrector.generate_attenuation_correction_parameters()

		image_mean = corrector.image_raw_mean[0]
		image_std = corrector.image_raw_std[0]

		image_corrected_mean = corrector.image_corrected_mean[0]
		image_corrected_std = corrector.image_corrected_std[0]

		parameters_a = corrector.image_attenuation_parameters[0,:,:,0]
		parameters_b = corrector.image_attenuation_parameters[0,:,:,1]
		parameters_c = corrector.image_attenuation_parameters[0,:,:,2]

		image_mean_int = image_mean.astype(int)
		image_std_int = image_std.astype(int)

		image_corrected_mean_int = image_corrected_mean.astype(int)
		image_corrected_std_int = image_corrected_std.astype(int)


		parameters_a_int = parameters_a.astype(int)
		parameters_b_int = parameters_b.astype(int)
		parameters_c_int = parameters_c.astype(int)


		self.assertEqual((image_mean_int == 140).all(), True, 'Mean values are incorrect and not same for all pixels')
		self.assertEqual((image_std_int == 74).all(), True, 'Std values are incorrect and not same for all pixels')


		self.assertEqual((parameters_a_int == parameters_a_int[0,0]).all(), True, 'Parameter values are not same for all pixels')
		self.assertEqual((parameters_b_int == parameters_b_int[0,0]).all(), True, 'Parameter values are not same for all pixels')
		self.assertEqual((parameters_c_int == parameters_c_int[0,0]).all(), True, 'Parameter values are not same for all pixels')
		
		self.assertEqual((image_corrected_mean_int == image_corrected_mean_int[0,0]).all(), True, 'Corrected Mean values are not same for all pixels')
		self.assertEqual((image_corrected_std_int == image_corrected_std_int[0,0]).all(), True, 'Std values are not same for all pixels')

    		
if __name__ == '__main__':
    unittest.main()







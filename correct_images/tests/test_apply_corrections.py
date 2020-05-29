
 # -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import unittest
import yaml
import os
import numpy as np
from pathlib import Path
from correct_images.corrector import *
from oplab import Console
import pandas as pd
import tempfile


class testCaseCorrector(unittest.TestCase):
	def test_calculate_parameters_RGB(self):

		path_root = Path(__file__).resolve().parents[1]
		test_yaml_path = path_root / 'tests' / 'test.yaml'
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
		corrector.brightness = 45
		corrector.contrast = 15
		corrector.altitude_based_filtered_indices = [0, 1, 2]
		corrector.memmap_folder = path_memmap_folder

		corrector.bayer_numpy_filelist = image_numpy_list
		corrector.distance_matrix_numpy_filelist = distance_numpy_list
		corrector.attenuation_parameters_folder = path_attenuation_parameters

		corrector.altitude_min = 0.5
		corrector.altitude_max = 3.5

		corrector.smoothing = 'mean'

		corrector.undistort = False
		corrector._type = 'grayscale'
		corrector.output_format = 'png'
		corrector.output_images_folder = path_dummy_folder


		# invoke the feature to be tested
		corrector.generate_attenuation_correction_parameters()
		corrector.process_correction(True) # set True to denote test phase for process_correction
		# calculate the mean and std of the corrected images
		correctedimage_memmap_path, correctedimage_memmap = load_memmap_from_numpyfilelist(corrector.memmap_folder, 
										corrector.bayer_numpy_filelist)
		
		# find out the mean and std of the images corrected for target
		# brightness and contrast
		# test success: R, G channels should have same mean, std as target brightness, contrast 
		# 				B channel should have mean as target brightness but std should be 0
		targeted_mean = np.empty((corrector.image_channels, corrector.image_height,
				corrector.image_width))
		targeted_std = np.empty((corrector.image_channels, corrector.image_height,
				corrector.image_width))
		for i in range(corrector.image_channels):
			targeted_image_mean, targeted_image_std = mean_std(correctedimage_memmap[:,:,:,i])
			targeted_mean[i] = np.around(targeted_image_mean)
			targeted_std[i] = np.around(targeted_image_std)

		
		image_mean_red = targeted_mean[0]
		image_mean_green = targeted_mean[1]

		image_std_red = targeted_std[0]
		image_std_green = targeted_std[1]


		image_mean_red_int = image_mean_red.astype(int)
		image_mean_green_int = image_mean_green.astype(int)
		image_std_red_int = image_std_red.astype(int)
		image_std_green_int = image_std_green.astype(int)


		self.assertEqual((image_mean_red_int == corrector.brightness).all(), True, 'Mean values are incorrect and not same for all pixels along red channel')
		self.assertEqual((image_std_red_int == corrector.contrast).all(), True, 'Std values are incorrect and not same for all pixels along red channel')
		self.assertEqual((image_mean_green_int == corrector.brightness).all(), True, 'Mean values are incorrect and not same for all pixels along green channel')
		self.assertEqual((image_std_green_int == corrector.contrast).all(), True, 'Std values are incorrect and not same for all pixels along green channel')
		

		
	def test_calculate_parameters_Black_White(self):

		path_root = Path(__file__).resolve().parents[1]
		test_yaml_path = path_root / 'tests' / 'test.yaml'
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
		corrector.brightness = 45
		corrector.contrast = 15

		corrector.altitude_based_filtered_indices = [0, 1, 2]
		corrector.memmap_folder = path_memmap_folder

		corrector.bayer_numpy_filelist = image_numpy_list
		corrector.distance_matrix_numpy_filelist = distance_numpy_list
		corrector.attenuation_parameters_folder = path_attenuation_parameters

		corrector.altitude_min = 0.5
		corrector.altitude_max = 3.5

		corrector.smoothing = 'mean'

		corrector.undistort = False
		corrector._type = 'grayscale'
		corrector.output_format = 'png'
		corrector.output_images_folder = path_dummy_folder

		# invoke the feature to be tested
		corrector.generate_attenuation_correction_parameters()
		corrector.process_correction(True)


		# calculate the mean and std of the corrected images
		correctedimage_memmap_path, correctedimage_memmap = load_memmap_from_numpyfilelist(corrector.memmap_folder, 
										corrector.bayer_numpy_filelist)
		
		# find out the mean and std of the images corrected for target
		# brightness and contrast
		# test success: R, G channels should have same mean, std as target brightness, contrast 
		# 				B channel should have mean as target brightness but std should be 0
		targeted_mean, targeted_std = mean_std(correctedimage_memmap)
		targeted_mean = np.around(targeted_mean)
		targeted_std = np.around(targeted_std)

		print('Black and White images:')

		image_mean = targeted_mean[0]
		image_std = targeted_std[0]

		image_mean_int = image_mean.astype(int)
		image_std_int = image_std.astype(int)


		self.assertEqual((image_mean_int == corrector.brightness).all(), True, 'Mean values are incorrect and not same for all pixels along red channel')
		self.assertEqual((image_std_int == corrector.contrast).all(), True, 'Std values are incorrect and not same for all pixels along red channel')

		
if __name__ == '__main__':
    unittest.main()







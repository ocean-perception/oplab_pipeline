 # -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import unittest
import os
from pathlib import Path
from correct_images.corrector import Corrector
from oplab import Console
import pandas as pd
import numpy as np
import tempfile


ACFR_IMG_WIDTH = 2464
ACFR_IMG_HEIGHT = 2056

SX3_IMG_WIDTH = 1280 
SX3_IMG_HEIGHT = 1024

LM165_IMG_WIDTH = 1392
LM165_IMG_HEIGHT = 1040

BIOCAM_IMG_WIDTH = 2560
BIOCAM_IMG_HEIGHT = 2160


class TestCaseCorrector(unittest.TestCase):
	def createACFRDummyDataset(self):
		# create a dummy imagelist
		path_temp = tempfile.mkdtemp()
		message = 'Temporary folder created at ' + path_temp
		Console.info(message)
		path_dummy_folder = Path(path_temp).resolve() / 'dummy_folder'
		if not path_dummy_folder.exists():
			path_dummy_folder.mkdir()
		imagelist = []
		for i in range(15):
			imagename = 'image_' + str(i) + '.tif'
			imagepath = path_dummy_folder / imagename
			imagelist.append(imagepath)
		imagenumbers = []
		distances = []
		min_altitude = 2
		max_altitude = 6
		for i in range(15):
			imagenumbers.append('image_'+str(i))
			distance = np.random.uniform(low=min_altitude-1, high=max_altitude+2)
			distances.append(distance)
		dataframe = pd.DataFrame({'relative_path':imagenumbers, 'altitude [m]':distances})
		distance_path = 'json_'
		distance_full_path = path_dummy_folder / distance_path
		distance_csv_path = distance_full_path / 'csv/ekf'
		if not distance_csv_path.exists():
			distance_csv_path.mkdir(parents=True)
		path_to_camera1_csv = distance_csv_path / 'auv_ekf_LC.csv'
		path_to_camera2_csv = distance_csv_path / 'auv_ekf_RC.csv'

		# create the csvs
		dataframe.to_csv(path_to_camera1_csv)
		dataframe.to_csv(path_to_camera2_csv)

		# create the folder where distance numpy files will be generated
		distance_folder = path_dummy_folder / 'distance'
		if not distance_folder.exists():
			distance_folder.mkdir(parents=True)
		self.path_processed = path_dummy_folder
		self.distance_folder  = distance_folder 
		self.distance_path  = distance_path
		self.altitude_min = min_altitude
		self.altitude_max = max_altitude
		self.image_height = ACFR_IMG_HEIGHT
		self.image_width = ACFR_IMG_WIDTH
		self._imagelist = imagelist
		self._camera_image_file_list = 'none'
		# set cameras list
		self.cameras = ['RC', 'LC']
		self.compare_shape = [ACFR_IMG_HEIGHT, ACFR_IMG_WIDTH]

	
	def createSX3DummyDataset(self):
		# create a dummy imagelist
		path_temp = tempfile.mkdtemp()
		message = 'Temporary folder created at ' + path_temp
		Console.info(message)
		path_dummy_folder = Path(path_temp).resolve() / 'dummy_folder'
		if not path_dummy_folder.exists():
			path_dummy_folder.mkdir()
		imagelist = []
		for i in range(15):
			imagename = 'image_' + str(i) + '.tif'
			imagepath = path_dummy_folder / imagename
			imagelist.append(imagepath)
		imagenumbers = []
		distances = []
		min_altitude = 2
		max_altitude = 6
		for i in range(15):
			imagenumbers.append('image_'+str(i))
			distance = np.random.uniform(low=min_altitude-1, high=max_altitude+2)
			distances.append(distance)
		dataframe = pd.DataFrame({'relative_path':imagenumbers, 'altitude [m]':distances})
		distance_path = 'json_'
		distance_full_path = path_dummy_folder / distance_path
		distance_csv_path = distance_full_path / 'csv/ekf'
		if not distance_csv_path.exists():
			distance_csv_path.mkdir(parents=True)
		path_to_camera1_csv = distance_csv_path / 'auv_ekf_Cam51707923.csv'
		path_to_camera2_csv = distance_csv_path / 'auv_ekf_Cam51707925.csv'
		path_to_camera3_csv = distance_csv_path / 'auv_ekf_LM165.csv'

		# create the csvs
		dataframe.to_csv(path_to_camera1_csv)
		dataframe.to_csv(path_to_camera2_csv)
		dataframe.to_csv(path_to_camera3_csv)


		# create the folder where distance numpy files will be generated
		distance_folder = path_dummy_folder / 'distance'
		if not distance_folder.exists():
			distance_folder.mkdir(parents=True)

		
		# set attributes
		self.distance_metric = 'altitude'
		self.path_processed = path_dummy_folder
		self.distance_path = distance_path
		self.distance_folder = distance_folder
		self._camera_image_file_list = 'none'
		self._imagelist = imagelist
		self.image_height = SX3_IMG_HEIGHT
		self.image_width = SX3_IMG_WIDTH

		self.altitude_min = min_altitude
		self.altitude_max = max_altitude
		
		# set cameras list
		self.cameras = ['Cam51707923', 'Cam51707925', 'LM165']
		self.compare_shape = [SX3_IMG_HEIGHT, SX3_IMG_WIDTH]



	def createBiocamDummyDataset(self):
		# create a dummy imagelist
		path_temp = tempfile.mkdtemp()
		message = 'Temporary folder created at ' + path_temp
		Console.info(message)
		path_dummy_folder = Path(path_temp).resolve() / 'dummy_folder'
		if not path_dummy_folder.exists():
			path_dummy_folder.mkdir()
		imagelist = []
		for i in range(15):
			imagename = 'image_' + str(i) + '.tif'
			imagepath = path_dummy_folder / imagename
			imagelist.append(imagepath)
		imagenumbers = []
		distances = []
		min_altitude = 2
		max_altitude = 6
		for i in range(15):
			imagenumbers.append('image_'+str(i))
			distance = np.random.uniform(low=min_altitude-1, high=max_altitude+2)
			distances.append(distance)
		dataframe = pd.DataFrame({'relative_path':imagenumbers, 'altitude [m]':distances})
		distance_path = 'json_'
		distance_full_path = path_dummy_folder / distance_path
		distance_csv_path = distance_full_path / 'csv/ekf'
		if not distance_csv_path.exists():
			distance_csv_path.mkdir(parents=True)
		path_to_camera1_csv = distance_csv_path / 'auv_ekf_cam61003146.csv'
		path_to_camera2_csv = distance_csv_path / 'auv_ekf_cam61004444.csv'

		# create the csvs
		dataframe.to_csv(path_to_camera1_csv)
		dataframe.to_csv(path_to_camera2_csv)


		# create the folder where distance numpy files will be generated
		distance_folder = path_dummy_folder / 'distance'
		if not distance_folder.exists():
			distance_folder.mkdir(parents=True)


		# create the folder where distance numpy files will be generated
		distance_folder = path_dummy_folder / 'distance'
		if not distance_folder.exists():
			distance_folder.mkdir(parents=True)

		
		# set attributes
		self.distance_metric = 'altitude'
		self.path_processed = path_dummy_folder
		self.distance_path = distance_path
		self.distance_folder = distance_folder
		self._camera_image_file_list = 'none'
		self._imagelist = imagelist
		self.image_height = BIOCAM_IMG_HEIGHT
		self.image_width = BIOCAM_IMG_WIDTH

		self.altitude_min = min_altitude
		self.altitude_max = max_altitude
		

		# set cameras list
		self.cameras = ['cam61003146', 'cam61004444']
		self.compare_shape = [BIOCAM_IMG_HEIGHT, BIOCAM_IMG_WIDTH]




	def run_distance_matrix_generation(self):
		# create Corrector object
		corrector = Corrector(True)

		# set Corrector:: attributes
		corrector.distance_metric = 'altitude'
		corrector.path_processed = self.path_processed
		corrector.distance_path = self.distance_path
		corrector._camera_image_file_list = 'none'
		corrector.image_height = self.image_height
		corrector.image_width = self.image_width
		corrector._imagelist = self._imagelist
		corrector.distance_matrix_numpy_folder = self.distance_folder
		corrector.altitude_min = self.altitude_min
		corrector.altitude_max = self.altitude_max

		for camera in self.cameras:
			# set camera name
			corrector.camera_name = camera
			# test feature for chosen camera
			corrector.generate_distance_matrix()
			self.assertEqual(len(corrector._imagelist), 
					len(corrector.distance_matrix_numpy_filelist),
					'Length of distance matrix filelist does not match with imagelist')
			for idx in corrector.altitude_based_filtered_indices:
				distance_matrix = np.load(corrector.distance_matrix_numpy_filelist[idx])
				self.assertEqual(distance_matrix.shape[0], self.compare_shape[0], 'Dimension mismatch: height')
				self.assertEqual(distance_matrix.shape[1], self.compare_shape[1], 'Dimension mismatch: width')
				out_of_range = (np.abs(distance_matrix) < corrector.altitude_min).any() or (np.abs(distance_matrix) > corrector.altitude_max).any()
				self.assertEqual(out_of_range, False, 'distance matrix values out of altitude bounds error')

	def test_acfr_distance_matrix_generation(self):
		self.createACFRDummyDataset()
		self.run_distance_matrix_generation()		

	
	def test_sx3_distance_matrix_generation(self):
		self.createSX3DummyDataset()
		self.run_distance_matrix_generation()


	def test_biocam_distance_matrix_generation(self):
		self.createBiocamDummyDataset()
		self.run_distance_matrix_generation()
	
    		
if __name__ == '__main__':
    unittest.main()







# this is the class file for implementing the various correction algorithms
# IMPORT --------------------------------
# all imports go here 
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import trange
import cv2
import imageio
import os
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder
from auv_nav.tools.console import Console
from auv_nav.camera_system import *
import yaml

# -----------------------------------------

class Corrector:
	def __init__(self, camera, path=None):
		self.output_dir_path = None
		self.bayer_numpy_dir_path = None
		self.output_parameters_path = None
		self._camera = camera

		if path is not None:
			self.path_raw = get_raw_folder(path)
			self.path_processed = get_processed_folder(path)
			self.path_config = get_config_folder(path)

	def load_correct_images_yaml_file(self):
		correct_images_path = self.path_config / 'correct_images.yaml'
		with correct_images_path.open('r') as f:
			self.correct_images_params = yaml.safe_load(f)

	def load_generic_parameters(self):
		self.correction_method = self.correct_images_params['method']
		if self.correction_method == 'colour_correction':
			self.distance_metric = self.correct_images_params['colour_correction']['distance_metric']
			self.distance_path = self.correct_images_params['colour_correction']['metric_path']
			self.altitude_max = self.correct_images_params['colour_correction']['altitude_filter']['max_m']
			self.altitude_min = self.correct_images_params['colour_correction']['altitude_filter']['min_m']
			self.smoothing = self.correct_images_params['colour_correction']['smoothing']
			self.window_size = self.correct_images_params['colour_correction']['window_size']
			self.outlier_rejection = self.correct_images_params['colour_correction']['curve_fitting_outlier_rejection']
		self.cameras = self.correct_images_params['cameras']
		self.undistort = self.correct_images_params['output_settings']['undistort']
		self.output_format = self.correct_images_params['output_settings']['compression_parameter']

	def load_camera_specific_parameters(self):
		idx = [i for i, camera in enumerate(self.cameras) if camera.get('camera_name') == self._camera.name]
		self._camera_image_file_list = self.cameras[idx[0]].get('image_file_list')
		if self.correction_method == 'colour_correction':
			self._camera_correction = self.cameras[idx[0]].get('colour_correction')
		if self.correction_method == 'manual_balance':
			self._camera_correction = self.cameras[idx[0]].get('manual_balance')

	def get_imagelist(self):
		# TODO:
		# 1. get imagelist for given camera object
		if self._camera_image_file_list == 'None':
			self._imagelist = self._camera.image_list
		else:
			path_file_list = self.path_config / self._camera_image_file_list
			with path_file_list.open('r') as f:
				imagenames = f.readlines()
			new_image_list = [imagepath for imagepath in self._imagelist if Path(imagepath).name in imagenames]
			self._imagelist = new_image_list

	def generate_bayer_numpy_filelist(self):
		# create output directory path
		image_path = self._imagelist[0]
		output_dir_path = get_processed_folder(image_path)
		output_dir_path = output_dir_path / 'attenuation_correction'
		if not output_dir_path.exists():
			output_dir_path.mkdir(parents=True)
		self.output_dir_path = output_dir_path
		bayer_numpy_folder_name = 'bayer_' + self._camera.name
		self.bayer_numpy_dir_path = self.output_dir_path / bayer_numpy_folder_name
		if not self.bayer_numpy_dir_path.exists():
			self.bayer_numpy_dir_path.mkdir(parents=True)

		# generate bayer numpy filelist from imagelst
		bayer_numpy_filelist = []
		for imagepath in self._imagelist:
			imagepath_ = Path(imagepath)
			bayer_file_stem = imagepath_.stem
			bayer_file_path = self.bayer_numpy_dir_path / str(bayer_file_stem + ".npy")
			bayer_numpy_filelist.append(bayer_file_path)
		return bayer_numpy_filelist

	def generate_bayer_numpyfiles(self, bayer_numpy_filelist):
		# create numpy files as per bayer_numpy_filelist
		if self._camera.extension == 'tif':
			# write numpy files for corresponding bayer images
			for idx in trange(len(self._imagelist)):
				tmp_tif = imageio.imread(self._imagelist[idx])
				tmp_npy = np.zeros([self.image_height, self.image_width], np.uint16)
				tmp_npy[:, :] = np.array(tmp_tif, np.uint16)
				np.save(bayer_numpy_filelist[idx], tmp_npy)
		if self._camera.extension == 'raw':
			# create numpy files as per bayer_numpy_filelist
			raw_image_size = np.fromfile(str(self._imagelist[0]), dtype=np.uint8)
			binary_data = np.zeros(len(self._imagelist), raw_img_for_size.shape[0], dtype=raw_img_for_size.dtype)
			for idx in trange(len(self._imagelist)):
				binary_data[idx, :] = np.fromfile(str(self._imagelist[idx]), dtype=raw_image_size.dtype)
			image_raw = joblib.Parallel(n_jobs=2)(joblib.delayed
				(self.load_xviii_bayer_from_binary)(binary_data, self.image_height, self.image_width)
				for idx in range(len(self._imagelist)))
			
			for idx in trange(len(self._imagelist)):
				#binary_data = 
				#bayer_image_raw = self.load_xviii_bayer_from_binary(binary_data, self.image_height, self.image_width)
				np.save(bayer_numpy_filelist[idx], image_raw[idx, :])
		Console.info('Image numpy files written successfully')

	def load_xviii_bayer_from_binary(self, binary_data, image_height, image_width):
		"""
		Load bayer data of Xviii camera image from raw binary data.
		:param xviii_binary_data: raw binary of xviii image. Should be loaded by 'np.fromfile('path_of_xviii_raw_data(.raw)')'
		:return: bayer data of xviii image
		"""
		img_h = image_height
		img_w = image_width
		bayer_img = np.zeros((img_h, img_w), dtype=np.uint32)

		# https://github.com/ocean-perception/image_conversion/blob/master/src/xviii_demosaic.cpp
		# read raw data and put them into bayer patttern.
		count = 0
		for i in range(0, img_h, 1):
			for j in range(0, img_w, 4):
				chunk = binary_data[count : count + 12]
				bayer_img[i, j] = (
				(chunk[3] & 0xFF) << 16 | ((chunk[2] & 0xFF) << 8) | (chunk[1] & 0xFF)
				)
				bayer_img[i, j + 1] = (
				((chunk[0] & 0xFF) << 16) | ((chunk[7] & 0xFF) << 8) | (chunk[6] & 0xFF)
				)
				bayer_img[i, j + 2] = (
				((chunk[5] & 0xFF) << 16) | ((chunk[4] & 0xFF) << 8) | (chunk[11] & 0xFF)
				)
				bayer_img[i, j + 3] = (
				((chunk[10] & 0xFF) << 16) | ((chunk[9] & 0xFF) << 8) | (chunk[8] & 0xFF)
				)
				count += 12

		return bayer_img

	def get_image_properties(self):
		# TODO:
		# 1. read a single image from the imagelist
		image_path = self._imagelist[0]
		
		# read tiff
		if self._camera.extension == 'tif':
			image_matrix = imageio.imread(image_path)
			image_shape = image_matrix.shape
			self.image_height = image_shape[0]
			self.image_width = image_shape[1]
			if len(image_shape) == 3:
				self.image_channels = 3
			else:
				self.image_channels = 1

		# read raw
		if self._camera.extension == 'raw':
			self.image_height = 1024
			self.image_width = 1280
			self.image_channels = 1

		return self.image_height, self.image_width, self.image_channels

	def generate_distance_matrix(self):
		# TODO:
		# 0. create output distance_matrix_numpy_file paths
		distance_matrix_numpy_folder_name = 'distance_matrix_' + self._camera.name
		self.distance_matrix_numpy_folder_path = self.output_dir_path / distance_matrix_numpy_folder_name
		if not self.distance_matrix_numpy_folder_path.exists():
			self.distance_matrix_numpy_folder_path.mkdir(parents=True)
		Console.info('paths setup')
		# 1. read altitude / depth map depending on distance_metric
		if self.distance_metric == 'none':
			return None
		elif self.distance_metric == 'altitude':
			full_metric_path = self.path_processed / self.distance_path
			full_metric_path = full_metric_path / 'csv' / 'ekf'
			metric_file = 'auv_ekf_' + self._camera.name + '.csv'
			metric_file_path = full_metric_path / metric_file
			# read list of altitudes
			dataframe = pd.read_csv(metric_file_path)
			imagenames = [Path(imagepath).name for imagepath in self._imagelist]
			#print(imagenames)
			selected_dataframe = dataframe.loc[dataframe['Imagenumber'].isin(imagenames)]
			distancelist = selected_dataframe[' Altitude [m]'] 
		elif self.distance_metric == 'depth_map':
			# TODO: get depth map from metric path
			# TODO: select the depth map for images in self._imagelist
			print('get path to depth map')
		Console.info('Distance parameters loaded')
		
		# 2. filter images based on min/ max altitude match
		filtered_dataframe = selected_dataframe.loc[(selected_dataframe[' Altitude [m]'] >= self.altitude_min) & (selected_dataframe[' Altitude [m]'] < self.altitude_max)]
		# create filtered imagelist based on altitude filtering
		self.filtered_imagenames = filtered_dataframe['Imagenumber']
		self.filtered_imagelist = [imagepath for imagepath in self._imagelist if Path(imagepath).name in self.filtered_imagenames]
		distancelist = filtered_dataframe[' Altitude [m]']
		Console.info('images filtered based on min and max altitude')
	
		# 3. create distance matrix [image_width, image_height]
		distance_matrix = np.empty((self.image_height, self.image_width))
		distance_matrix_numpy_filelist = []
		distance_matrix_numpy_folder = self.output_dir_path / distance_matrix_numpy_folder_name
		if not distance_matrix_numpy_folder.exists():
			distance_matrix_numpy_folder.mkdir(parents=True)
		#print(self.filtered_imagenames)
		#print(distance_matrix_numpy_folder)

		for idx in trange(len(self.filtered_imagenames)):
			#for i in range(self.image_height):
			#	for j in range(self.image_width):
			#		distance_matrix[i, j] = distancelist[idx]
			distance_matrix.fill(distancelist[idx])
			distance_matrix_numpy_file = 'distance_' + str(idx) + '.npy'
			distance_matrix_numpy_file_path = distance_matrix_numpy_folder / distance_matrix_numpy_file
			distance_matrix_numpy_filelist.append(distance_matrix_numpy_file_path)
			
			# create the distance matrix numpy file
			np.save(distance_matrix_numpy_file_path, distance_matrix)
		Console.info('Distance matrix numpy files written successfully')
		return distance_matrix_numpy_filelist

	
	def calculate_attenuation_parameters(self, outlier_filter):
		# TODO:
		# 1. calculate parameters slow if outlier_filter == True
		# 2. calculate parameters fast if outlier_filter == False
		pass
	'''

	def process_correction(self):
		# 0. read correct_images.yaml file


		# 1.
		self.get_imagelist()

		self.channels, self.size, self.output_format = self.get_image_properties()
		# 2.
		self.generate_correction_parameters_per_camera()
		# 3.
		self.image_matrix = self.apply_corrections()
		# 4.
		if self.channels == 1:
		self.debayer(self.image_matrix, self.pattern)
		# 5.
		self.distortion_correction(self.image_matrix)

		# 6.
		if self.output_format == 'jpg':
		self.gamma_correction(self.image_matrix)
		# 7.
		self.write_output_images()


	

	

	def generate_correction_parameters_per_camera(self):
		# TODO :
		# 1. read from correct_config.yaml
		# 2. get correction parameters specific to given camera object
		# 3. assign class variables values from correction parametsr read 
		# 4. read correction_method from correct_config.yaml
		if correction_method == 'colour_correction':
			self.calculate_distance_based_correction_parameters()
		else:
			self.load_static_correction_parameters()

		pass

	def calculate_distance_based_correction_parameters(self):
		# TODO:
		# 1. generate distance matrix
		self.generate_distance_matrix()

		# 2. check for outlier filtering and call calculate_attenuation_parameters
		self.calculate_attenuation_parameters(outlier_filter=False)
		pass


	

	def load_static_correction_parameters(self):
		# TODO:
		# 1. read static correction parameters specific to given camera object

	def apply_corrections(self):
		# TODO :
		# 1. apply corrections to images in imagelist
		pass

	def debayer(self, image_matrix, pattern):
		# TODO :
		# 1. debayer the images given in matrix
		# 2. generate the image mean and std dev
		pass

	def distortion_correction(self, image_matrix):
		# TODO:
		# 1. read calibration parameters from calibration path
		# 2. correct image_matrix for distortion
		pass

	def gamma_correction(self, image_matrix):
		# TODO:
		# 1. gamma correct image_matrix

	def write_output_images(self):
		# TODO:
		# 1. write image_matrix to output images
		# 2. write correction parameters to output files
		pass

	'''
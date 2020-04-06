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
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.console import Console
from draft_camera_system import *
# -----------------------------------------


class Corrector:
	def __init__(self, camera, path_processed=None):
		self.output_dir_path = None
		self.bayer_dir_path = None
		self.output_parameters_path = None

		self.camera = camera

		if path_processed is not None:
			self.path_processed = path_processed
		'''
		self.correct_params = camera.get_correction_parameters()
		self.correct_method = camera.get_method()
		self.undistort = camera.get_undistort_check()
		self.output_image_format = camera.get_output_format()
		self.camera_name = camera.get_name()
		self.image_extension = camera.get_extension()
		self.image_bayer_pattern = camera.get_pattern()
		self.imagelist = camera.get_imagelist()
		
			self.altitude_max = correct_params.altitude_max
			self.altitude_min = correct_params.altitude_min
			self.sampling_method = correct_params.sampling_method
			self.window_edge_length = correct_params.window_edge_length
			self.distance_metric = correct_params.distance_metric
			self.target_mean = correct_params.target_mean
			self.target_std = correct_params.target_std
			self.apply_distortion_correction = correct_params.apply_distortion_correction
		'''
	def load_correction_parameters(self):
		# load imagelist
		self.imagelist = self.camera.get_imagelist()

		# check for grey world correction or color balance
		self.method = self.camera.get_method()

		if self.method == 'colour_correction':

			# read in color correction parameters
			self.read_color_correction_parameters()

			# read altitudes / depth maps
			self.read_distance_matrix()

			# create output folders for bayer_np files / attenuation_parameters / corrected images
			#self.create_output_folders()

			# write bayer numpy files
			#self.write_bayer_np_files()


		elif self.method == 'manual_balance':
			print('manual_balance')
			# read in static correction parameters
			#self.read_manual_balance_parameters()

	def read_color_correction_parameters(self):

		camera_correction_parameters = self.camera.get_correction_parameters()
		self.distance_metric = camera_correction_parameters['distance_metric']
		self.metric_path = camera_correction_parameters['metric_path']
		self.altitude_max = camera_correction_parameters['altitude_filter']['max_m']
		self.altitude_min = camera_correction_parameters['altitude_filter']['min_m']
		self.outlier_filter = camera_correction_parameters['outlier_filter']
		self.smoothing = camera_correction_parameters['smoothing']
		self.window_size = camera_correction_parameters['window_size']
		self.curve_fitting_outlier_rejection = camera_correction_parameters['curve_fitting_outlier_rejection']
		self.target_mean = camera_correction_parameters['brightness']
		self.target_std = camera_correction_parameters['contrast']

	def read_manual_balance_parameters(self):
		camera_correction_parameters = self.camera.get_correction_parameters()
		

	def read_distance_matrix(self):
		# create distance matrix
		distance_matrix = np.empty(())

		if self.distance_metric == 'none':
			self.distance_matrix = distance_matrix

		elif self.distance_metric == 'altitude':
			self.get_image_altitude_dataframe()

			# read image size
			image = self.image_altitude_dataframe['Imagenumber'][0]
			image_absolute_path = self.camera.get_absolute_path() / image
			print(image_absolute_path)
			self.image_size = imageio.imread(image_absolute_path)
			tmp = self.image_size.shape
			self.a = tmp[0]
			self.b = tmp[1]
			if len(tmp) == 3:
				self.image_channels = tmp[2]
			else:
				self.image_channels = 1
			print('image size: ', self.a, self.b)
			print('image channels: ', self.image_channels)

			# create a distance matrix
			self.distance_matrix = np.empty((self.a, self.b))
		
			altitude_list = self.image_altitude_dataframe['Altitude']
			for altitude in altitude_list:
				tmp_matrix = np.empty((self.a, self.b))
				for i in range(self.a):
					for j in range(self.b):
						tmp_matrix[i, j] = altitude
				if self.distance_matrix.size == self.a * self.b:
					self.distance_matrix = tmp_matrix 
				else:
					self.distance_matrix = np.concatenate((self.distance_matrix, tmp_matrix), axis=0)

			print(self.distance_matrix)

		elif self.distance_metric == 'depth_map':
			print('depth_map')
		else:
			Console.quit('Chosen distance_metric is not understood...')

	def get_image_altitude_dataframe(self):
		#>>> a = [1, 2, 9, 3, 8] 
		#>>> b = [1, 9, 1] 
		#>>> [a.index(item) for item in b]

		auv_nav_csv = 'auv_ekf_' + self.camera.get_name() + '.csv'
		path_altitude_csv = self.path_processed / self.metric_path / 'csv'
		path_altitude_csv = path_altitude_csv / 'ekf'
		path_altitude_csv = path_altitude_csv / auv_nav_csv
		auvnav_dataframe = pd.read_csv(path_altitude_csv)
		selected_dataframe = auvnav_dataframe[auvnav_dataframe['Imagenumber'].isin(self.imagelist)]
		filtered_dataframe = self.filter_image_altitude_dataframe(selected_dataframe)
		image_list = filtered_dataframe['Imagenumber']
		altitude_list = filtered_dataframe[' Altitude [m]']
		self.image_altitude_dataframe = pd.DataFrame({'Imagenumber':image_list, 'Altitude':altitude_list})

	def filter_image_altitude_dataframe(self, selected_dataframe):
		selected_dataframe[(selected_dataframe[' Altitude [m]'] >= self.altitude_min) & (selected_dataframe[' Altitude [m]'] < self.altitude_max)]
		return selected_dataframe

	#def get_bayer_np_filelist(self):


	'''

	def calculate_attenuation_parameters(self):
		# check if outlier filter is true
		if self.outlier_filter is True:
			# TODO compute attenuation parameters with 'mean_trimmed' smoothing method
		else:
			# TODO compute attenuation parameters with 'mean' / 'median' smoothing method

	def apply_color_corrections(self):
		pass

	def debayer_images(self, image_list, pattern):
		if not pattern == 'None':
			# source images are bayer. Debayer corrected images.
			# TODO Debayering code
			# return debayered images
		else:	
			# source images are debayered. No need to debayer corrected images.
			return image_list

	def distortion_correction(self, image_list, undistort):
		# distortion correction
		if undistort is True:
			# perform distortion correction
			# TODO distortion code
			# return corrected_image_list

		else:
			# no distortion correction
			return image_list

	def gamma_correction(self, image_list):
		# TODO code for gamma correction
		# return corrected_image_list

	def write_developed_images(self, image_list, output_format):
		# check for output format
		if output_format == 'jpg':
			# perform gamma correction default
			self.corrected_image_list = self.gamma_correction(image_list)
		else:
			# if png file format, perform no gamma correction default
			# TODO code for writing corrected images to output folders

	def execute_correction_pipeline(self):
		# load correction parameters 
		self.load_correction_parameters()

		# calculate attenuation parameters with / without outlier filtering
		self.calculate_attenuation_parameters()

		# apply corrections
		self.corrected_image_list = self.apply_color_corrections()

		# debayer / do not debayer
		pattern = camera.get_pattern()
		self.corrected_image_list = self.debayer_images(self.corrected_image_list, pattern)

		# correct / not correct for distortions
		undistort = camera.get_undistort_check()
		self.corrected_image_list = self.distortion_correction(self.corrected_image_list, undistort)

		# write corrected images to output folders with / without gamma correction
		output_format = camera.get_output_format()
		self.write_developed_images(self.corrected_image_list, output_format)
	


# developing the associated functions to support the pipeline defined above

	def read_imagename_altitudes(self, camera):
		auv_nav_csv = 'auv_ekf_' + camera.get_name() + '.csv'
		path_altitude_csv = self.path_processed / self.json_path / 'csv'
		path_altitude_csv = path_altitude_csv / 'ekf'
		path_altitude_csv = path_altitude_csv / auv_nav_csv
		auvnav_dataframe = pd.read_csv(path_altitude_csv)
		imagelist = auvnav_dataframe['Imagenumber']
		altitude_list = auvnav_dataframe[' Altitude [m]']
		imagename_altitude_dataframe = pd.DataFrame({'Imagenumber':imagelist, 'Altitude':altitude_list})
		return imagename_altitude_dataframe

	def read_imagename_depthmap(self, camera):
		pass

	def get_filtered_image_pathlist(self, camera, imagename_altitude_dataframe):
		imagelist = imagename_altitude_dataframe['Imagenumber']
		altitude_list = imagename_altitude_dataframe['Altitude']
		imagename_altitude_dataframe[(imagename_altitude_dataframe['Altitude'] >= self.altitude_min) & (imagename_altitude_dataframe['Altitude'] < self.altitude_max)]
		filtered_image_list = imagename_altitude_dataframe['Imagenumber']
		filtered_image_path_list = []
		for image in filtered_image_list:
			image_path = Path(camera.get_path()) / image
			filtered_image_path_list.append(image_path)
		
		return filtered_image_path_list

	def get_np_filelist(self, camera, camera_system, filtered_image_path_list):
		np_file_list = []

		for image_path in filtered_image_path_list:
			if camera_system == 'acfr_standard':
				image_path_parent = Path(image_path).resolve().parent
			#print(image_path_parent)
			if camera_system == 'seaxerocks3':
				image_path_parent = Path(image_path).parent.parent.parent
			self.output_dir_path = get_processed_folder(image_path_parent)
			self.output_dir_path = self.output_dir_path / "attenuation_correction"
			if not self.output_dir_path.exists():
				self.output_dir_path.mkdir(parents=True)
			bayer_folder_name = 'bayer' + camera.get_name()
			self.bayer_dir_path = self.output_dir_path / bayer_folder_name
			if not self.bayer_dir_path.exists():
				self.bayer_dir_path.mkdir(parents=True)
			bayer_file_stem = image_path.stem
			np_file_path = self.bayer_dir_path / str(bayer_file_stem + ".npy")
			np_file_list.append(np_file_path)

		return np_file_list
		
	def write_np_files(self, camera, np_file_list, filtered_image_path_list):		
		# load images
		if camera.get_extension() == 'tif' or camera.get_extension() == 'tiff':
			print('generating numpy files for source bayer images')
			
			# read image size

			tmp_tif_for_size = imageio.imread(filtered_image_path_list[0])
			a = tmp_tif_for_size.shape[0]
			b = tmp_tif_for_size.shape[1]
			
			# write numpy files for the source bayer images
			for image_idx in trange(len(filtered_image_path_list)):
				tmp_tif = imageio.imread(filtered_image_path_list[image_idx])
				tmp_npy = np.zeros([a, b], np.uint16)
				tmp_npy[:, :] = np.array(tmp_tif, np.uint16)
				np.save(np_file_list[image_idx], tmp_npy)
	'''        
	

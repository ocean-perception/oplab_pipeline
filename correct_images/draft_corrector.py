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
# -----------------------------------------


class Corrector:
	def __init__(self, path_processed=None, correct_params=None):
		if correct_params is not None:
			self.auv_nav_path = correct_params.auv_nav_path
			self.image_type = correct_params.image_type
			self.src_img_index = correct_params.src_img_index
			self.altitude_max = correct_params.altitude_max
			self.altitude_min = correct_params.altitude_min
			self.sampling_method = correct_params.sampling_method
			self.median_filter_kernel_size = correct_params.median_filter_kernel_size
			self.target_mean = correct_params.target_mean
			self.target_std = correct_params.target_std
			self.debayer_option = correct_params.debayer_option
			self.bayer_pattern = correct_params.bayer_pattern
			self.dst_file_format = correct_params.dst_file_format
			self.apply_attenuation_correction = correct_params.apply_attenuation_correction
			self.apply_gamma_correction = correct_params.apply_gamma_correction
			self.apply_distortion_correction = correct_params.apply_distortion_correction
			self.camera_parameter_file_path = correct_params.camera_parameter_file_path
		if path_processed is not None:
			self.path_processed = path_processed

	def read_imagename_altitudes(self, camera):
		# create the string name for the auv nav csv folder
		auv_nav_csv = 'auv_ekf_' + camera.get_name() + '.csv'
		path_altitude_csv = self.path_processed / self.auv_nav_path / 'csv'
		path_altitude_csv = path_altitude_csv / 'ekf'
		path_altitude_csv = path_altitude_csv / auv_nav_csv
		auvnav_dataframe = pd.read_csv(path_altitude_csv)
		imagelist = auvnav_dataframe['Imagenumber']
		altitude_list = auvnav_dataframe[' Altitude [m]']
		imagename_altitude_dataframe = pd.DataFrame({'Imagenumber':imagelist, 'Altitude':altitude_list})
		return imagename_altitude_dataframe

	def write_bayer_image(self, camera, imagename_altitude_dataframe):
		imagelist = imagename_altitude_dataframe['Imagenumber']
		altitude_list = imagename_altitude_dataframe['Altitude']
		imagename_altitude_dataframe[(imagename_altitude_dataframe['Altitude'] >= self.altitude_min) & (imagename_altitude_dataframe['Altitude'] < self.altitude_max)]
		filtered_image_list = imagename_altitude_dataframe['Imagenumber']

		filtered_image_path_list = []
		bayer_file_list = []
		for image in filtered_image_list:
			image_path = Path(camera.get_path()) / image
			image_path_parent = Path(image_path).parent.parent
			output_dir_path = get_processed_folder(image_path_parent)
			
			#print(image_path)
			filtered_image_path_list.append(image_path)
			output_dir_path = output_dir_path / "attenuation_correction"
			if not output_dir_path.exists():
				output_dir_path.mkdir(parents=True)
			bayer_folder_name = 'bayer' + camera.get_name()
			bayer_dir_path = output_dir_path / bayer_folder_name
			if not bayer_dir_path.exists():
				bayer_dir_path.mkdir(parents=True)
			bayer_file_stem = image_path.stem
			bayer_file_path = bayer_dir_path / str(bayer_file_stem + ".npy")
			bayer_file_list.append(bayer_file_path)
		
				
		# load images
		if self.image_type == 'tif' or self.image_type == 'tiff':
			print('generating bayer images')
			# read image size
			
			tmp_tif_for_size = imageio.imread(filtered_image_path_list[0])
			a = tmp_tif_for_size.shape[0]
			b = tmp_tif_for_size.shape[1]
			# write bayer images
			for image_idx in trange(len(filtered_image_path_list)):
				tmp_tif = imageio.imread(filtered_image_path_list[image_idx])
				tmp_npy = np.zeros([a, b], np.uint16)
				tmp_npy[:, :] = np.array(tmp_tif, np.uint16)
				np.save(bayer_file_list[image_idx], tmp_npy)
        
		return imagename_altitude_dataframe, bayer_file_list

	def calculate_attenuation_parameters(self, camera, imagename_altitude_dataframe):
		pass


	'''

	def Execute(self):
		
		# resolve paths for attenuation paremeters and output directories
		cameras = camera_system.cameras
		for camera in cameras:
			camera_name = camera.name
			bayer_pattern = camera.pattern
			image_type = camera.extension

			# Resolve path to camera images
			# TODO ----------------------
			# --------------------------

			# Compute attenuation correction coefficients

			self.calculate_attenuation_correction() # entire parse code (restructured) goes in here
			# TODO details on the data structure to read attenuation params
			# ....................

			# Apply corrections as opted by user through correct_config.yaml file
			for bayer_image in bayer_image_list:
				if self.attenuation_correct == True:
					corrected_bayer_image = self.attenuation_correction_bayer(bayer_image)
				else:
					self.pixel_stat(bayer_image)
				corrected_rgb_img = self.debayer(bayer_image, bayer_pattern, image_type) # Debayering
				
				if self.distortion_correct == True:
					map_x, map_y = self.calc_distortion_mapping(camera_parameters_path, bayer_img_size)
					corrected_rgb_img = self.correct_distortion(corrected_rgb_img, map_x, map_y)

				if self.gamma_correct == True:
					corrected_rgb_img = self.gamma_correct(corrected_rgb_img)
				self.write_image(dest_path)

		self.write_config()

	def calculate_attenuation_correction(self):
		pass

	def attenuation_correction_bayer(self, bayer_image):
		pass

	def pixel_stat(self, bayer_image):
		pass

	def debayer(bayer_image):
		pass

	def calc_distortion_mapping(self, camera_parameters_path, bayer_img_size):
		pass

	def correct_distortion(self, corrected_rgb_img, map_x, map_y):
		pass

	def gamma_correct(self, corrected_rgb_img):
		pass

	def write_image(self, dest_path):
		pass

	def write_config():
		pass

	'''





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
from correct_images.parser import *
import yaml
import joblib
import sys
import uuid
# -----------------------------------------

class Corrector:
	def __init__(self, force, camera=None, correct_config=None, path=None):
		self._camera = camera
		self._correct_config = correct_config
		if path is not None:
			self.path_raw = get_raw_folder(path)
			self.path_processed = get_processed_folder(path)
			self.path_config = get_config_folder(path)
		self.force = force


	# store into object correction parameters relevant to both all cameras in the system
	def load_generic_config_parameters(self):
		self.correction_method = self._correct_config.method
		if self.correction_method == 'colour_correction':
			self.distance_metric = self._correct_config.color_correction.distance_metric
			self.distance_path = self._correct_config.color_correction.metric_path
			self.altitude_max = self._correct_config.color_correction.altitude_max
			self.altitude_min = self._correct_config.color_correction.altitude_min
			self.smoothing = self._correct_config.color_correction.smoothing
			self.window_size = self._correct_config.color_correction.window_size
			self.outlier_rejection = self._correct_config.color_correction.outlier_reject
		self.cameraconfigs = self._correct_config.configs.camera_configs
		self.undistort = self._correct_config.output_settings.undistort_flag
		self.output_format = self._correct_config.output_settings.compression_parameter

	
	# create directories for storing intermediate image and distance_matrix numpy files,
	# correction parameters and corrected output images 
	def create_output_directories(self):
		# create output directory path
		image_path = Path(self._imagelist[0]).resolve()
		image_parent_path = image_path.parents[0]
		output_dir_path = get_processed_folder(image_path)
		self.output_dir_path = output_dir_path / 'attenuation_correction'
		if not self.output_dir_path.exists():
			self.output_dir_path.mkdir(parents=True)

		
		# create path for image numpy files
		bayer_numpy_folder_name = 'bayer_' + self._camera.name
		self.bayer_numpy_dir_path = self.output_dir_path / bayer_numpy_folder_name
		if not self.bayer_numpy_dir_path.exists():
			self.bayer_numpy_dir_path.mkdir(parents=True)
		
		# create path for distance matrix numpy files
		distance_matrix_numpy_folder_name = 'distance_matrix_' + self._camera.name
		self.distance_matrix_numpy_folder = self.output_dir_path / distance_matrix_numpy_folder_name
		if not self.distance_matrix_numpy_folder.exists():
			self.distance_matrix_numpy_folder.mkdir(parents=True)

		# create path for parameters files
		attenuation_parameters_folder_name = 'attenuation_params_' + self._camera.name
		self.attenuation_parameters_folder = self.output_dir_path / attenuation_parameters_folder_name
		if not self.attenuation_parameters_folder.exists():
			self.attenuation_parameters_folder.mkdir(parents=True)
		elif self.force == False:
			Console.quit('Overwrite parameters with a Force command...')

		# create path for output images
		output_images_folder_name = 'developed_' + self._camera.name
		self.output_images_folder = self.output_dir_path / output_images_folder_name
		if not self.output_images_folder.exists():
			self.output_images_folder.mkdir(parents=True)
		elif self.force == False:
			Console.quit('Overwrite images with a Force command...')
		Console.info('Output directories created...')

	

	# store into object correction paramters specific to the current camera 
	def load_camera_specific_config_parameters(self):
		idx = [i for i, cameraconfig in enumerate(self.cameraconfigs) if cameraconfig.camera_name == self._camera.name]
		
		self._camera_image_file_list = self.cameraconfigs[idx[0]].imagefilelist
		if self.correction_method == 'colour_correction':
			self.brightness = self.cameraconfigs[idx[0]].brightness
			self.contrast = self.cameraconfigs[idx[0]].contrast
		elif self.correction_method == 'manual_balance':
			self.subtractors_rgb = self.cameraconfigs[idx[0]].subtractors_rgb
			self.color_correct_matrix_rgb = self.cameraconfigs[idx[0]].color_correct_matrix_rgb


	# load imagelist: output is same as camera.imagelist unless a smaller filelist is specified by the user 
	def get_imagelist(self):
		# TODO:
		# 1. get imagelist for given camera object
		if self._camera_image_file_list == 'none':
			self._imagelist = self._camera.image_list
		else:
			path_file_list = self.path_config / self._camera_image_file_list
			with path_file_list.open('r') as f:
				imagenames = f.readlines()
			new_image_list = [imagepath for imagepath in self._imagelist if Path(imagepath).name in imagenames]
			self._imagelist = new_image_list
	
	

	# store image dimensiions and channels into object
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

	
	# save a set of distance matrix numpy files
	def generate_distance_matrix(self):
		
		# read altitude / depth map depending on distance_metric
		if self.distance_metric == 'none':
			return None
		elif self.distance_metric == 'altitude':
			full_metric_path = self.path_processed / self.distance_path
			full_metric_path = full_metric_path / 'csv' / 'ekf'
			metric_file = 'auv_ekf_' + self._camera.name + '.csv'
			metric_file_path = full_metric_path / metric_file
			# read list of altitudes
			dataframe = pd.read_csv(metric_file_path)
			if self._camera.extension == 'raw':
				imagenames = [Path(imagepath).stem for imagepath in self._imagelist]
				imagenumbers = []
				for imagename in imagenames:
					imagenumbers.append(int(imagename))
				selected_dataframe = dataframe.loc[dataframe['Imagenumber'].isin(imagenumbers)]
			else:
				imagenames = [Path(imagepath).name for imagepath in self._imagelist]
				selected_dataframe = dataframe.loc[dataframe['Imagenumber'].isin(imagenames)]
			selected_distancelist = selected_dataframe[' Altitude [m]']
			
			# create a list of paths to selected images from the full imagelist provided by camera system class
			selected_imagelist = []
			if self._camera.extension == 'raw':
				imagenumbers = selected_dataframe['Imagenumber']
				for image_number in imagenumbers:
					image_number_string = str(image_number)
					image_name_length = len(image_number_string)
					zero_padding_length = 7 - image_name_length
					imagename = image_number_string
					for i in range(zero_padding_length):
						imagename = '0' + imagename
					selected_imagelist.append(imagename)
			else:
				selected_imagelist = selected_dataframe['Imagenumber']	
			
			selected_image_pathlist = []
			image_indices = [imagenames.index(item) for item in selected_imagelist]
			for idx in image_indices:
				selected_image_pathlist.append(self._imagelist[idx])

		elif self.distance_metric == 'depth_map':
			# TODO: get depth map from metric path
			# TODO: select the depth map for images in self._imagelist
			print('get path to depth map')
		
		# update imagelist with the selected image pat list
		self._imagelist = selected_image_pathlist
		Console.info('Distance parameters loaded')

		# Create distance matrix
		distance_matrix = np.empty((self.image_height, self.image_width))
		self.distance_matrix_numpy_filelist = []
		
		for idx in trange(len(selected_distancelist)):
			distance_matrix.fill(selected_distancelist[idx])
			imagename = Path(self._imagelist[idx]).name
			distance_matrix_numpy_file = imagename + '.npy'
			distance_matrix_numpy_file_path = self.distance_matrix_numpy_folder / distance_matrix_numpy_file
			self.distance_matrix_numpy_filelist.append(distance_matrix_numpy_file_path)
			
			# create the distance matrix numpy file
			np.save(distance_matrix_numpy_file_path, distance_matrix)
		Console.info('Distance matrix numpy files written successfully')

		# Filter numpy images for calculating attenuation parameters based on altitude range
		
		self.filtered_indices = [index for index, distance in enumerate(selected_distancelist) if distance < self.altitude_max and distance > self.altitude_min]
		Console.info('Images filtered as per altitude range...')

		
	# create a list of image numpy files to be written to disk
	def generate_bayer_numpy_filelist(self, image_pathlist):
		# generate numpy filelist from imagelst
		self.bayer_numpy_filelist = []
		for imagepath in image_pathlist:
			imagepath_ = Path(imagepath)
			bayer_file_stem = imagepath_.stem
			bayer_file_path = self.bayer_numpy_dir_path / str(bayer_file_stem + ".npy")
			self.bayer_numpy_filelist.append(bayer_file_path)

	
	# write the intermediate image numpy files to disk
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
			raw_image_for_size = np.fromfile(str(self._imagelist[0]), dtype=np.uint8)
			binary_data = np.zeros((len(self._imagelist), raw_image_for_size.shape[0]), dtype=raw_image_for_size.dtype)
			
			for idx in trange(len(self._imagelist)):
				binary_data[idx, :] = np.fromfile(str(self._imagelist[idx]), dtype=raw_image_for_size.dtype)
			image_raw = joblib.Parallel(n_jobs=10)(joblib.delayed
				(load_xviii_bayer_from_binary)(binary_data[idx, :], self.image_height, self.image_width)
				for idx in trange(len(self._imagelist)))
			
			for idx in trange(len(self._imagelist)):
				np.save(bayer_numpy_filelist[idx], image_raw)
		Console.info('Image numpy files written successfully')

	
	# compute correction parameters either for attenuation correction or static correction of images 
	def generate_correction_parameters(self):
		
		# create an empty matrix to store image correction parameters
		# parameters can be color attenuation correction or
		# manual balance parameters
		self.image_correction_parameters = np.empty((self.image_channels, self.image_height,
													self.image_width))
		
		if self.correction_method == 'colour_correction':
			
			# create image and distance_matrix memmaps
			# based on altitude filtering
			image_memmap = load_memmap_from_numpyfilelist(self.bayer_numpy_filelist)
			distance_memmap = load_memmap_from_numpyfilelist(self.distance_matrix_numpy_filelist)
			# --- TODO ----
			#----------------------------------------

			
			# calculate mean, std for image and distance matrices
			self.raw_image_mean, self.raw_image_std = mean_std(image_memmap, True)
			self.distance_mean = mean_std(image_memmap, False)
			target_altitude = self.distance_mean

			
			# compute histogram of intensities with respect to altitude range(min, max)
			mean_image_samples_per_bin = np.array([])
			mean_distance_samples_per_bin = np.array([])
			# -----TODO:
			#--------------------------------------------

			for i in range(self.image_channels):
				# compute the attenuation regression parameters
				self.attenuation_parameters = np.empty((self.image_height, self.image_width, 3))
				self.attenuation_parameters = curve_fitting(mean_image_samples_per_bin,
											mean_distance_samples_per_bin)

				# calculate the attenuation gain values
				
				self.image_correction_parameters[i] = self.calculate_attenuation_gains(target_altitude)
			
		
		elif self.correction_method == 'manual_balance':
			static_correction_parameters = np.array((self.image_height, self.image_width, 
												self.image_channels))
			# generate static correction values for each pixel each channel
			# TODO------ 
			self.image_correction_parameters = static_correction_parameters
	

	# compute gain values for each pixel for a targeted altitide using the attenuation parameters
	def calculate_attenuation_gains(self, target_altitude):
		attenuation_gains =  np.empty((self.image_channels, self.image_height,
													self.image_width))
		# TODO -------
		return attenuation_gains


	# execute the corrections of images using the gain values in case of attenuation correction or static color balance
	def process_correction(self):
		for idx in trange(0, len(self.bayer_numpy_filelist)):
			# load numpy image and distance files
			image = []
			distance = []
			# image = np.load(self.bayer_numpy_filelist[idx])
			# distance = np.load(self.distance_matrix_numpy_filelist[idx])

			# debayer images of source images are bayer ones
			if not self._camera.type == 'grayscale':
				# debayer images
				image_rgb = self.debayer(image, self._camera.type)
			
			# apply corrections
			image_rgb = self.apply_corrections(image_rgb, distance, self.brightness, self.contrast)

			# apply distortion corrections
			image_rgb = self.distortion_correct(image_rgb)

			# apply gamma corrections to rgb images
			image_rgb = self.gamma_correct(image_rgb)
			
			# write to output files
			self.write_output_image(image_rgb, self.output_format)		



	# apply corrections on each image using the correction paramters for targeted brightness and contrast
	def apply_corrections(self, image, distance, brightness, contrast):
		# TODO :
		# ------------------------
		return image

	
	# convert bayer image to RGB based 
	# on the bayer pattern for the camera
	def debayer(self, image, pattern):
		# TODO :
		# ---------------------
		return image

	

	# correct image for distortions using
	# camera calibration parameters
	def distortion_correct(self, image):
		# TODO:
		# ----------------------
		return image

	
	# gamma corrections for image
	def gamma_correct(self, image):
		# TODO:
		# -----------------------------
		return image

	
	# save processed image in an output file with
	# given output format
	def write_output_image(self, image, format):
		# TODO:
		# ----------------------------------
		pass

	# NON-MEMBER FUNCTIONS:
	# ------------------------------------------------------------------------------

# read binary raw image files for xviii camera
def load_xviii_bayer_from_binary(binary_data, image_height, image_width):
	
	img_h = image_height
	img_w = image_width
	bayer_img = np.zeros((img_h, img_w), dtype=np.uint32)

	# read raw data and put them into bayer patttern.
	count = 0
	for i in range(0, img_h, 1):
		for j in range(0, img_w, 4):
			chunk = binary_data[count : count + 12]
			bayer_img[i, j] = (
			((chunk[3] & 0xFF) << 16) | ((chunk[2] & 0xFF) << 8) | (chunk[1] & 0xFF)
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


# store into memmaps the distance and image numpy files
def load_memmap_from_numpyfilelist(numpyfilelist):
	message = 'loading binary files into memmap...'
	#image = np.load(str(numpyfilelist[0]))
	list_shape = [len(numpyfilelist)]
	#list_shape = list_shape + list(image.shape)
	filename_map = 'memmap_' + str(uuid.uuid4()) + '.map'
	memmap_ = np.memmap(filename=filename_map, mode='w+', shape=tuple(list_shape),
		dtype=np.float32)
	#for idx in trange(0, len(numpyfilelist), ascii=True, desc=message):
	#	memmap_[idx, ...] = np.load(numpyfilelist[idx])

	return filename_map, memmap_


# calculate the mean and std of an image
def mean_std(data, calculate_std=True):
	'''
	if calculate_std:
		return np.mean(data), np.std(data)
	else:
		return np.mean(data)
	'''
	return None, None


# compute attenuation correction parameters through regression
def curve_fitting(imagelist, distancelist):
	parameters = np.array([])
	# TODO
	# -----------------------------
	return parameters


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
	def __init__(self, camera, path_processed=None, path_config=None):
		self.output_dir_path = None
		self.bayer_numpy_dir_path = None
		self.output_parameters_path = None
		self.camera = camera

		if path_processed is not None:
			self.path_processed = path_processed

		if path_config is not None:
			self.path_config = path_config

	def process_correction(self):
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


	def get_imagelist(self):
		# TODO:
		# 1. get imagelist for given camera object
		pass

	def get_image_properties(self):
		# TODO:
		# 1. read a single image from the imagelist 
		# 2. set to class variables - image channels, image size
		pass


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
	
	def generate_distance_matrix(self):
		# TODO:
		# 1. read altitude / depth map depending on distance_metric
		# 2. filter images based on min/ max altitude match
		# 3. create distance matrix [image_width, image_height]
		pass

	def calculate_attenuation_parameters(self, outlier_filter):
		# TODO:
		# 1. calculate parameters slow if outlier_filter == True
		# 2. calculate parameters fast if outlier_filter == False
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


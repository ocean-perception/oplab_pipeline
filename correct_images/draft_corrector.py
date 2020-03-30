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
from draft_camera_system import *
# -----------------------------------------


class Corrector:
	def __init__(self, path_processed=None, correct_params=None):
		self.output_dir_path = None
		self.bayer_dir_path = None

		if correct_params is not None:
			self.json_path = correct_params.json_path
			self.altitude_max = correct_params.altitude_max
			self.altitude_min = correct_params.altitude_min
			self.sampling_method = correct_params.sampling_method
			self.window_edge_length = correct_params.window_edge_length
			self.distance_metric = correct_params.distance_metric
			self.target_mean = correct_params.target_mean
			self.target_std = correct_params.target_std
			self.apply_distortion_correction = correct_params.apply_distortion_correction
			
		if path_processed is not None:
			self.path_processed = path_processed

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
        
	def calculate_attenuation_parameters(self, camera, imagename_altitude_dataframe):
		pass


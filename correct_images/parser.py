 # -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import yaml
from oplab import Console

class ColourCorrection:
    def __init__(self, node):
        self.distance_metric = node['distance_metric']
        self.metric_path = node['metric_path']
        self.altitude_max = node['altitude_filter']['max_m']
        self.altitude_min = node['altitude_filter']['min_m']
        self.smoothing = node['smoothing']
        self.window_size = node['window_size']
        self.outlier_reject = node['curve_fitting_outlier_rejection']

class Config:
	def __init__(self, camera_name, imagefilelist, 
		brightness, contrast, subtractors_rgb, color_correct_matrix_rgb):
		self.camera_name = camera_name
		self.imagefilelist = imagefilelist
		self.brightness = brightness
		self.contrast = contrast
		self.subtractors_rgb = subtractors_rgb
		self.color_correct_matrix_rgb = color_correct_matrix_rgb

class CameraConfigs:
    def __init__(self, node):
        self.camera_configs = []
        self.num_cameras = len(node)
        for i in range(self.num_cameras):
            self.camera_configs.append(Config(node[i]['camera_name'],
            node[i]['image_file_list'], 
            node[i]['colour_correction']['brightness'],
            node[i]['colour_correction']['contrast'], 
            node[i]['manual_balance']['subtractors_rgb'],
            node[i]['manual_balance']['colour_correction_matrix_rgb']))

class OutputSettings:
	def __init__(self, node):
		self.undistort_flag = node['undistort']
		self.compression_parameter = node['compression_parameter']

class CorrectConfig:
    def __init__(self, filename=None):
        if filename is None:
            return
        with filename.open('r') as stream:
            data = yaml.safe_load(stream)

        if 'version' not in data:
            Console.error('It seems you are using an old correct_images.yaml. You will have to delete it and run this software again.')
            Console.error('Delete the file with:')
            Console.error('    rm ', filename)
            Console.quit('Wrong correct_images.yaml format')
        self.version = data['version']
        self.method = data['method']
        node = data['colour_correction']
        self.color_correction = ColourCorrection(node)
        node = data['cameras']
        self.configs = CameraConfigs(node)
        node = data['output_settings']
        self.output_settings = OutputSettings(node)
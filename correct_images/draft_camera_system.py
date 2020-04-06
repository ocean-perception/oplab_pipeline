# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
"""
# imports 

from numpy import loadtxt
import os

# Camera class
class Camera():
    def __init__(self, name, pattern, extension, relative_path, absolute_path, path_imagelist, method, correction_parameters, undistort, output_format):
        self.name = name                                   # name of camera                           
        self.pattern = pattern                             # bayer pattern
        self.extension = extension                         # image extension
        self.relative_path = relative_path                 # path relative to mission.yaml file
        self.absolute_path = absolute_path                 # absolute path to images
        self.path_imagelist = path_imagelist               # path to image filelist
        self.method = method                               # correction method: grayworld / manual balance
        self.correction_parameters = correction_parameters # correction parameters for each camera
        self.undistort = undistort                         # flag to perform distortion correction
        self.output_format = output_format                 # format of output images

    def get_name(self):
        return self.name

    def get_pattern(self):
        return self.pattern

    def get_extension(self):
        return self.extension

    def get_relative_path(self):
        return self.relative_path

    def get_absolute_path(self):
        return self.absolute_path

    def get_correction_parameters(self):
        return self.correction_parameters

    def get_imagelist(self):
        # Imagelist: list of imagenumbers found in filelist / image directory 
        # TODO: read imagenumbers from the filelist / iamge directory if filelist is not defined  
        return imagelist

    def get_method(self):
        return self.method

    def get_undistort_check(self):
        return self.undistort

    def get_output_format(self):
        return self.output_format

# Camerasystem Class
class CameraSystem:
    def __init__(self, path_raw_folder, mission_parameters, correction_parameters):
        self.path_raw = path_raw_folder
        self.cameras = []

    # instantiate camera class using parameters read from missin and config.yaml files
    def setup_cameras(self):
        # TODO set up list of cameras within the camera system 
        return self.cameras
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
        self.name = name                            
        self.pattern = pattern
        self.extension = extension
        self.relative_path = relative_path
        self.absolute_path = absolute_path
        self.path_imagelist = path_imagelist
        self.method = method
        self.correction_parameters = correction_parameters
        self.undistort = undistort
        self.output_format = output_format

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
        if self.path_imagelist is not None:
            with self.path_imagelist.open('r') as f:
                imagelist = f.readlines()
        else:
            imagelist = []
            for root, dirs, files in os.walk(self.absolute_path):
                for filename in files:
                    imagelist.append(filename)
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
        self.camera_system = mission_parameters.image_system
        self.cameras_from_mission = mission_parameters.cameras_mission
        self.cameras_from_config = correction_parameters.cameras_config
        self.undistort = correction_parameters.undistort
        self.output_format = correction_parameters.output_image_format
        self.cameras = []

    # read parameters from mission.yaml and config.yaml and fill up Camera class
    def read_camera_parameters_from_mission(self):
        self.relative_pathlist = []
        self.patternlist = []
        self.extensionlist = []

        camera_names_mission = []
        for camera in self.cameras_from_mission:
            camera_names_mission.append(camera.get('name'))
        
        for camera in self.cameras_from_config:
            camera_name = camera.get('camera_name')
            index = [index for index,camera_name_ in enumerate(camera_names_mission) if camera_name_ == camera_name]
            relative_path = self.cameras_from_mission[index[0]].get('path')
            pattern = self.cameras_from_mission[index[0]].get('type')
            extension = self.cameras_from_mission[index[0]].get('extension')

            self.relative_pathlist.append(relative_path)
            self.patternlist.append(pattern)
            self.extensionlist.append(extension)

    # instantiate camera class using parameters read from missin and config.yaml files
    def setup_cameras(self):
        self.read_camera_parameters_from_mission()
        for index in range(len(self.relative_pathlist)):
            name = self.cameras_from_config[index].get('camera_name')
            relative_path = self.relative_pathlist[index]
            absolute_path = self.path_raw / relative_path
            pattern = self.patternlist[index]
            extension = self.extensionlist[index]
            imagelist = self.cameras_from_config[index].get('image_file_list')
            if imagelist == 'None':
                path_imagelist = None
            else:
                path_imagelist = path / imagelist
            method = self.cameras_from_config[index].get('method')
            if method == 'colour_correction':
                correction_parameters = self.cameras_from_config[index].get('colour_correction')
            else:
                correction_parameters = self.cameras_from_config[index].get('manual_balance')
            self.cameras.append(Camera(name, pattern, extension, relative_path, absolute_path, path_imagelist, method, correction_parameters, self.undistort, self.output_format))

        return self.cameras
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, University of Southampton
All rights reserved.
"""

class Pattern:
    MONO = 'mono'
    BGGR = 'bggr'
    RGGB = 'rggb'
    GRGB = 'grgb'
   
class Extension:
    TIF = 'tif'
    JPG = 'jpg'
    RAW = 'raw'

class Camera():
    def __init__(self, name, pattern, extension, path=None):
        self.name = name
        self.pattern = pattern
        self.extension = extension
        self.path = path

    def get_name(self):
        return self.name

    def get_pattern(self):
        return self.pattern

    def get_extension(self):
        return self.extension

    def get_path(self):
        return self.path

class CameraSystem:
    def __init__(self, path_raw_folder):
        self.path_raw = path_raw_folder
        self.cameras = []

    def set_image_paths(self, path_list):
        if isinstance(self, AcfrSystem):
            path = path_list[0].get('path')
            for idx in range(len(self.cameras)):
                self.cameras[idx].path = self.path_raw / path
        if isinstance(self, Sx3System):
            for idx in range(len(self.cameras)):
                self.cameras[idx].path = self.path_raw / path_list[idx].get('path')
        if isinstance(self, BiocamSystem):
            for idx in range(len(self.cameras)):
                self.cameras[idx].path = self.path_raw / path_list[idx].get('path')
    def get_cameras(self):
        return self.cameras

class BiocamSystem(CameraSystem):
    def __init__(self, path_raw_folder):
        self.path_raw = path_raw_folder
        self.cameras = []
        self.cameras.append(Camera('cam61003146', Pattern.BGGR, Extension.TIF))
        self.cameras.append(Camera('cam61004444', Pattern.MONO, Extension.TIF))
        self.cameras.append(Camera('cam61004444_laser', Pattern.MONO, Extension.JPG))

class Sx3System(CameraSystem):
    def __init__(self, path_raw_folder):
        self.path_raw = path_raw_folder
        self.cameras = []
        self.cameras.append(Camera('Cam51707923', Pattern.RGGB, Extension.TIF))
        self.cameras.append(Camera('Cam51707925', Pattern.RGGB, Extension.TIF))
        self.cameras.append(Camera('LM165', Pattern.MONO, Extension.TIF))

class AcfrSystem(CameraSystem):
    def __init__(self, path_raw_folder):
        self.path_raw = path_raw_folder
        self.cameras = []
        self.cameras.append(Camera('LC', Pattern.GRGB, Extension.TIF))
        self.cameras.append(Camera('RC', Pattern.GRGB, Extension.TIF))  
        

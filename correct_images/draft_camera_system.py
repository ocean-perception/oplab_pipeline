# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, University of Southampton
All rights reserved.
"""
'''
class Pattern:
    MONO = 'mono'
    BGGR = 'bggr'
    RGGB = 'rggb'
    GRGB = 'grgb'
   
class Extension:
    TIF = 'tif'
    JPG = 'jpg'
    RAW = 'raw'
'''

class Camera():
    def __init__(self, name, format_, pattern, extension, image_path=None):
        self.name = name
        self.format = format_
        self.pattern = pattern
        self.extension = extension
        self.path = image_path
        

    def get_name(self):
        return self.name

    def get_format(self):
        return self.format

    def get_pattern(self):
        return self.pattern

    def get_extension(self):
        return self.extension

    def get_path(self):
        return self.path

    


class CameraSystem:
    def __init__(self, path_raw_folder, mission_parameters, correct_parameters):
        self.path_raw = path_raw_folder
        self.mission = mission_parameters
        self.correct_parameters = correct_parameters
        self.cameras = []

    def read_cameras(self):
        for camera in self.mission.cameras:
            name = camera.get('name')
            path = camera.get('path')
            image_path = self.path_raw / path
            pattern = self.correct_parameters.bayer_pattern
            format_ = self.mission.format
            extension = self.correct_parameters.image_type
            self.cameras.append(Camera(name, format_, pattern, extension, image_path))
        return self.cameras



        '''
        if format == 'biocam':
            # Treat Biocam
            self._system = BiocamSystem(path, mission)
        elif format == 'seaxerocks_3':
            # Treat SX3 images
            self._system = Sx3System(path, mission)
        elif format=='acfr_standard':
            # Treat Unagi images
            self._system = AcfrSystem(path, mission)
        '''
    '''
    @property
    def pattern(self):
        return self._system.pattern

    @property
    def num_cameras(self):
        return self._system.num_cameras

    @property
    def extension(self):
        return self._system.extension
    '''
'''

class BiocamSystem():
    def __init__(self, path, mission):
        self.cameras = []
        self.cameras.append(Camera('cam61003146', Pattern.BGGR, Extension.TIF))
        self.cameras.append(Camera('cam61004444', Pattern.MONO, Extension.TIF))
        self.cameras.append(Camera('cam61004444_laser', Pattern.MONO, Extension.JPG))
        # TODO get list of images
        # TODO get navigation filename (data will be generic)


class Sx3System():
    def __init__(self, path, mission):
        self.cameras = []
        self.cameras.append(Camera('Cam51707923', Pattern.RGGB, Extension.TIF))
        self.cameras.append(Camera('Cam51707925', Pattern.RGGB, Extension.TIF))
        self.cameras.append(Camera('LM165', Pattern.MONO, 'tif'))
        # TODO get list of images
        # TODO get navigation filename (data will be generic)


class AcfrSystem():
    def __init__(self, path, mission):
        self.cameras = []
        self.cameras.append(Camera('LC', Pattern.GRGB, Extension.TIF))
        self.cameras.append(Camera('RC', Pattern.GRGB, Extension.TIF))  # mono?
        # TODO get list of images
        # TODO get navigation filename (data will be generic)
'''
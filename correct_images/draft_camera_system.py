# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, University of Southampton
All rights reserved.
"""

class Camera():
    def __init__(self, name, pattern, extension, navigation=None):
        self.name = name
        self.extension = pattern
        self.pattern = extension
        self.navigation = navigation


class BiocamSystem():
    def __init__(self, path, mission):
        self.cameras = []
        self.cameras.append(Camera('cam61003146', 'bggr', 'tif'))
        self.cameras.append(Camera('cam61004444', 'mono', 'tif'))
        self.cameras.append(Camera('cam61004444_laser', 'mono', 'jpg'))
        # TODO get list of images
        # TODO get navigation filename (data will be generic)


class Sx3System():
    def __init__(self, path, mission):
        self.cameras = []
        self.cameras.append(Camera('Cam51707923', 'rggb', 'tif'))
        self.cameras.append(Camera('Cam51707925', 'rggb', 'tif'))
        self.cameras.append(Camera('LM165', 'mono', 'tif'))
        # TODO get list of images
        # TODO get navigation filename (data will be generic)


class AcfrSystem():
    def __init__(self, path, mission):
        self.cameras = []
        self.cameras.append(Camera('LC', 'grgb', 'tif'))
        self.cameras.append(Camera('RC', 'grgb', 'tif'))  # mono?
        # TODO get list of images
        # TODO get navigation filename (data will be generic)


class CameraSystem:
    def __init__(self, path, mission):
        if format == 'biocam':
            # Treat Biocam
            self._system = BiocamSystem(path, mission)
        elif format == 'seaxerocks_3':
            # Treat SX3 images
            self._system = Sx3System(path, mission)
        elif format=='acfr_standard':
            # Treat Unagi images
            self._system = AcfrSystem(path, mission)

    @property
    def pattern(self):
        return self._system.pattern

    @property
    def num_cameras(self):
        return self._system.num_cameras

    @property
    def extension(self):
        return self._system.extension

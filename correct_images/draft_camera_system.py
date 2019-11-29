# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, University of Southampton
All rights reserved.
"""

class BiocamSystem:
    def __init__(self, path, mission):
        self.num_cameras = 2
        self.pattern = 'bggr'
        self.extension = 'tif'
        self.navigation = ['']*self.num_cameras
        # TODO get list of images
        # TODO get navigation filename (data will be generic)


class Sx3System:
    def __init__(self, path, mission):
        self.num_cameras = 3
        self.pattern = 'rggb'
        self.extension = 'tif'
        self.navigation = ['']*self.num_cameras
        # TODO get list of images
        # TODO get navigation filename (data will be generic)


class AcfrSystem:
    def __init__(self, path, mission):
        self.num_cameras = 2
        self.pattern = 'grgb'
        self.extension = 'tif'
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

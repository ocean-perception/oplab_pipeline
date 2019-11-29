# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, University of Southampton
All rights reserved.
"""

# This file should implement the different correction algorithms with
# a common function name and common input/ouput system, so that we can
# easily swap from one to another. We can pass the mission and configuration
# classes to handle any extra parameters we would need.

class DebayerCorrection:
    def __init__(self, camera_system, configuration):
        pass


class PixelStatsCorrection:
    def __init__(self, camera_system, configuration):
        pass


class AttenuationCorrection:
    def __init__(self, camera_system, configuration):
        pass

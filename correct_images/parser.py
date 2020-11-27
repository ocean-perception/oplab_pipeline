# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import yaml
import numpy as np
from oplab import Console


class ColourCorrection:
    """ class ColourCorrection creates an object for generic colour correction parameters from correct_images.yaml file

    Attributes
    -----------
    distance_metric : string
        what mode of distance values will be used
    metric_path : string
        path to the file containing distance values
    altitude_max : int
        maximum permissible height for an image for use in calculating attenuation coefficients
    altitude_min : int
        minimum permissible height for an image for use in calculating attenuation coefficients
    smoothing : string
        method of sampling intensity values
    window_size : int
        control how noisy the parameters can be
    outlier_reject : bool
        flag for choosing filtering image outliers
    """

    def __init__(self, node):
        """ __init__ is the constructor function

        Parameters
        -----------
        node : cdict
            dictionary object for an entry in correct_images.yaml file
        """
        self.distance_metric = node["distance_metric"]
        self.metric_path = node["metric_path"]
        self.altitude_max = node["altitude_filter"]["max_m"]
        self.altitude_min = node["altitude_filter"]["min_m"]
        self.smoothing = node["smoothing"]
        self.window_size = node["window_size"]
        self.outlier_reject = node["curve_fitting_outlier_rejection"]


class CameraConfig:
    """ class Config creates an object for camera specific configuration parameters from correct_images.yaml file

    Attributes
    -----------
    camera_name : string
        name of camera
    imagefilelist : list
        list of paths to images provided by user
    brightness : int
        target mean for the image intensities
    contrast : int
        target std for the image intensities
    subtractors_rgb : matrix
        parameters for manual balance of images
    color_correct_matrix_rgb : matrix
        parameters for manual balance of images
    """

    def __init__(self, node):
        """ __init__ is the constructor function

        Parameters
        -----------
        node : cdict
            dictionary object for an entry in correct_images.yaml file
        """

        camera_name = node["camera_name"]

        imagefilelist_parse = 'none'
        imagefilelist_process = 'none'
        if 'image_file_list' in node:
            imagefilelist_parse = node.get("image_file_list", {}).get("parse", 'none')
            imagefilelist_process = node.get("image_file_list", {}).get("process", 'none')

        brightness = 30.
        contrast = 3.
        if 'colour_correction' in node:
            brightness = node.get("colour_correction", {}).get("brightness", brightness)
            contrast = node.get("colour_correction", {}).get("contrast", contrast)

        subtractors_rgb = np.array([0, 0, 0])
        color_gain_matrix_rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if 'manual_balance' in node:
            subtractors_rgb = node.get("manual_balance", {}).get("subtractors_rgb", subtractors_rgb)
            color_gain_matrix_rgb = node.get("manual_balance", {}).get("colour_gain_matrix_rgb", color_gain_matrix_rgb)

        self.camera_name = camera_name
        self.imagefilelist_parse = imagefilelist_parse
        self.imagefilelist_process = imagefilelist_process
        self.brightness = brightness
        self.contrast = contrast
        self.subtractors_rgb = subtractors_rgb
        self.color_gain_matrix_rgb = color_gain_matrix_rgb


class CameraConfigs:

    """ class OutputSettings creates an object for output_settings parameters from the correct_images.yaml file
    
    Attributes
    ----------
    camera_configs : Config object
        object stores camera specific configuration parameters 
    num_cameras : int
        number of cameras in the system
    """

    def __init__(self, node):
        """ __init__ is the constructor function

        Parameters
        -----------
        node : cdict
            dictionary object for an entry in correct_images.yaml file
        """
        self.camera_configs = []
        self.num_cameras = len(node)
        for i in range(self.num_cameras):
            self.camera_configs.append(CameraConfig(node[i]))


class OutputSettings:

    """ class OutputSettings creates an object for output_settings parameters from the correct_images.yaml file
    
    Attributes
    ----------
    undistort_flag : bool
        flag denotes if images need to be corrected for distortion
    compression_parameter : str
        output format in which images need to be saved
    """

    def __init__(self, node):

        """ __init__ is the constructor function

        Parameters
        -----------
        node : cdict
            dictionary object for an entry in correct_images.yaml file
        """

        self.undistort_flag = node["undistort"]
        self.compression_parameter = node["compression_parameter"]




class RescaleImage:

    def __init__(
        self,
        camera_name,
        path,
        distance_path,
        interpolate_method,
        target_pixel_size,
        maintain_pixels,
        output_folder,
    ):

        self.camera_name = camera_name
        self.path = path
        self.distance_path = distance_path
        self.interpolate_method = interpolate_method
        self.target_pixel_size = target_pixel_size / 100
        self.maintain_pixels = maintain_pixels
        self.output_folder = output_folder



class CameraRescale:
    def __init__(self, node):
        """ __init__ is the constructor function

        Parameters
        -----------
        node : cdict
            dictionary object for an entry in correct_images.yaml file
        """
        self.rescale_cameras = []
        self.num_cameras = len(node)
        for i in range(self.num_cameras):
            self.rescale_cameras.append(
                RescaleImage(
                    node[i]["camera_name"],
                    node[i]["path"],
                    node[i]["distance_path"],
                    node[i]["interpolate_method"],
                    node[i]["target_pixel_size"],
                    node[i]["maintain_pixels"],
                    node[i]["output_folder"],
                )
            )


class CorrectConfig:
    """ class CorrectConfig creates an object from the correct_images.yaml file
    
    Attributes
    ----------
    version : str
        version of the correct_images.yaml
    method : str
        method of correction to be used
    color_correction : ColourCorrection Object
        object contains parameters for colour based corrections
    configs : CameraConfigs Object
        object contains camera specific configuration parameters 
    output_settings : OutputSettings Object
        object contains parameters for output settings

    """
    def __init__(self, filename=None):

        """ __init__ is the constructor function

        Parameters
        -----------
        filename : Path
            path to the correct_images.yaml file
        """

        if filename is None:
            return
        with filename.open("r") as stream:
            data = yaml.safe_load(stream)

        if "version" not in data:
            Console.error(
                "It seems you are using an old correct_images.yaml.",
                "You will have to delete it and run this software again."
            )
            Console.error("Delete the file with:")
            Console.error("    rm ", filename)
            Console.quit("Wrong correct_images.yaml format")
        self.version = data["version"]

        valid_methods = ['manual_balance',
                         'colour_correction']
        self.method = data["method"]

        if self.method not in valid_methods:
            Console.quit("The method requested (", self.method,
                         ") is not supported or implemented. The valid methods",
                         "are:", ' '.join(m for m in valid_methods))

        node = data["colour_correction"]
        self.color_correction = ColourCorrection(node)
        node = data["cameras"]
        self.configs = CameraConfigs(node)
        node = data["output_settings"]
        self.output_settings = OutputSettings(node)
        node = data["rescale"]
        self.camerarescale = CameraRescale(node)

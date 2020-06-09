# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

# this is the class file for implementing the various correction algorithms
# IMPORT --------------------------------
# all imports go here
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import trange
from tqdm import tqdm
import cv2
import imageio
import os
from oplab import get_raw_folder
from oplab import get_processed_folder
from oplab import get_config_folder
from oplab import Console
from oplab import CameraSystem
from correct_images.parser import CorrectConfig
from oplab.camera_models import MonoCamera
import yaml
import joblib
import sys
import uuid
import datetime
from scipy import optimize

# -----------------------------------------


class Corrector:
    def __init__(self, force, camera=None, correct_config=None, path=None):
        """Constructor for the Corrector class

        Parameters
        ----------
        force : bool
            to indicate an overwrite for existing parameters or images
        camera : CameraEntry
            camera object for which the processing is expected
        correct_config : CorrectConfig
            correct config object storing all the correction configuration parameters
        path : Path
            path to the dive folder where image directory is present
        

        """

        self._camera = camera
        self._correct_config = correct_config
        if path is not None:
            self.path_raw = get_raw_folder(path)
            self.path_processed = get_processed_folder(path)
            self.path_config = get_config_folder(path)
        self.force = force

        # setup the corrector instance

    def setup(self):
        """Setup the corrector object by loading required
        configuration parameters and creating output directories
        
        Returns
        -------
        int
            Returns 0 if successful, -1 otherwise.
        """

        self.load_generic_config_parameters()
        ret = self.load_camera_specific_config_parameters()
        if ret < 0:
            return -1
        else:
            self.get_imagelist()
            self.create_output_directories()
            self.generate_distance_matrix()
            self.generate_bayer_numpy_filelist(self._imagelist)
            self.generate_bayer_numpyfiles(self.bayer_numpy_filelist)
            return 0

        # store into object correction parameters relevant to both all cameras in the system

    def load_generic_config_parameters(self):
        """Load general configuration parameters"""

        self.correction_method = self._correct_config.method
        if self.correction_method == "colour_correction":
            self.distance_metric = self._correct_config.color_correction.distance_metric
            self.distance_path = self._correct_config.color_correction.metric_path
            self.altitude_max = self._correct_config.color_correction.altitude_max
            self.altitude_min = self._correct_config.color_correction.altitude_min
            self.smoothing = self._correct_config.color_correction.smoothing
            self.window_size = self._correct_config.color_correction.window_size
            self.outlier_rejection = (
                self._correct_config.color_correction.outlier_reject
            )
        self.cameraconfigs = self._correct_config.configs.camera_configs
        self.undistort = self._correct_config.output_settings.undistort_flag
        self.output_format = self._correct_config.output_settings.compression_parameter

        # create directories for storing intermediate image and distance_matrix numpy files,
        # correction parameters and corrected output images

    def create_output_directories(self):
        """Handle the creation of output directories"""

        # create output directory path
        image_path = Path(self._imagelist[0]).resolve()
        image_parent_path = image_path.parents[0]
        output_dir_path = get_processed_folder(image_parent_path)
        self.output_dir_path = output_dir_path / "attenuation_correction"
        if not self.output_dir_path.exists():
            self.output_dir_path.mkdir(parents=True)

        # create path for image numpy files
        bayer_numpy_folder_name = "bayer_" + self._camera.name
        self.bayer_numpy_dir_path = self.output_dir_path / bayer_numpy_folder_name
        if not self.bayer_numpy_dir_path.exists():
            self.bayer_numpy_dir_path.mkdir(parents=True)

        # create path for distance matrix numpy files
        distance_matrix_numpy_folder_name = "distance_" + self._camera.name
        self.distance_matrix_numpy_folder = (
            self.output_dir_path / distance_matrix_numpy_folder_name
        )
        if not self.distance_matrix_numpy_folder.exists():
            self.distance_matrix_numpy_folder.mkdir(parents=True)

        # create path for parameters files
        attenuation_parameters_folder_name = "params_" + self._camera.name
        self.attenuation_parameters_folder = (
            self.output_dir_path / attenuation_parameters_folder_name
        )
        if not self.attenuation_parameters_folder.exists():
            self.attenuation_parameters_folder.mkdir(parents=True)
        else:
            dir_temp = self.attenuation_parameters_folder
            file_list = list(dir_temp.glob("*.npy"))
            if len(file_list) > 0:
                if self.force == False:
                    Console.quit(
                        "Processed files exist. If you want to overwrite them, run correct_images with the force overwrite flag (-F)"
                    )

        # create path for memmap files
        memmap_folder_name = "memmaps_" + self._camera.name
        self.memmap_folder = self.output_dir_path / memmap_folder_name
        if not self.memmap_folder.exists():
            self.memmap_folder.mkdir(parents=True)

        # create path for output images
        output_images_folder_name = "developed_" + self._camera.name
        self.output_images_folder = self.output_dir_path / output_images_folder_name
        if not self.output_images_folder.exists():
            self.output_images_folder.mkdir(parents=True)
        else:
            dir_temp = self.output_images_folder
            file_list = list(dir_temp.glob("*.*"))
            if len(file_list) > 0:
                if self.force == False:
                    Console.quit("Overwrite images with a Force command...")
        Console.info("Output directories created or existing...")

    # store into object correction paramters specific to the current camera
    def load_camera_specific_config_parameters(self):
        """Load camera specific configuration parameters

        Returns
        -------
        int
            Returns 0 if successful, -1 otherwise
        """

        idx = [
            i
            for i, cameraconfig in enumerate(self.cameraconfigs)
            if cameraconfig.camera_name == self._camera.name
        ]
        if len(idx) > 0:
            self._camera_image_file_list = self.cameraconfigs[idx[0]].imagefilelist
            if self.correction_method == "colour_correction":
                self.brightness = self.cameraconfigs[idx[0]].brightness
                self.contrast = self.cameraconfigs[idx[0]].contrast
            elif self.correction_method == "manual_balance":
                self.subtractors_rgb = self.cameraconfigs[idx[0]].subtractors_rgb
                self.color_correct_matrix_rgb = self.cameraconfigs[
                    idx[0]
                ].color_correct_matrix_rgb
            image_properties = self._camera.image_properties
            self.image_height = image_properties[0]
            self.image_width = image_properties[1]
            self.image_channels = image_properties[2]
            self.camera_name = self._camera.name
            self._type = self._camera.type
            return 0
        else:
            return -1

        # load imagelist: output is same as camera.imagelist unless a smaller filelist is specified by the user

    def get_imagelist(self):
        """Generate list of source images"""
        
        if self.distance_path == "json_renav_*":
            Console.info(
                "Picking first JSON folder as the default path to auv_nav csv files..."
            )
            dir_ = self.path_processed
            json_list = list(dir_.glob("json_*"))
            self.distance_path = json_list[0]

        full_metric_path = self.path_processed / self.distance_path
        full_metric_path = full_metric_path / "csv" / "ekf"
        metric_file = "auv_ekf_" + self.camera_name + ".csv"
        self.altitude_path = full_metric_path / metric_file


        # get imagelist for given camera object
        if self._camera_image_file_list == "none":
            self._imagelist = self._camera.image_list
        # get imagelist from user provided filelist
        else:
            path_file_list = Path(self.path_config) / self._camera_image_file_list
            trimmed_csv_file = 'trimmed_csv_' + self._camera.name + '.csv'
            self.trimmed_csv_path = Path(self.path_config) / trimmed_csv_file

            if not self.altitude_path.exists():
                message = "Path to " + metric_file + " does not exist..."
                Console.quit(message)
            else:
                # create trimmed csv based on user's provided list of images
                trim_csv_files(path_file_list, self.altitude_path, self.trimmed_csv_path)

            # read trimmed csv filepath
            dataframe = pd.read_csv(self.trimmed_csv_path)
            user_imagepath_list = dataframe["relative_path"]
            user_imagenames_list = [Path(image).name for image in user_imagepath_list]
            self._imagelist = [
                item
                for item in self._camera.image_list
                for image in user_imagenames_list
                if Path(item).name == image
            ]

        # save a set of distance matrix numpy files

    def generate_distance_matrix(self):
        """Generate distance matrix numpy files and save them"""

        # create empty distance matrix and list to store paths to the distance numpy files
        distance_matrix = np.empty((self.image_height, self.image_width))
        self.distance_matrix_numpy_filelist = []
        self.altitude_based_filtered_indices = []

        # read altitude / depth map depending on distance_metric
        if self.distance_metric == "none":
            Console.info("Null distance matrix created")

        elif self.distance_metric == "depth_map":
            path_depth = self.path_processed / self.distance_path
            if not path_depth.exists():
                Console.quit("Depth maps not found...")
            else:
                Console.info("Path to depth maps found...")
                depth_map_list = list(path_depth.glob("*.npy"))
                self.distance_matrix_numpy_filelist = [
                    Path(item)
                    for item in depth_map_list
                    for image_path in self._imagelist
                    if Path(item).stem in Path(image_path).stem
                ]
                Console.info("Filtering depth maps...")
                for idx in trange((len(self.distance_matrix_numpy_filelist))):
                    depth_array = np.load(self.distance_matrix_numpy_filelist[idx])
                    min_depth = depth_array.min()
                    max_depth = depth_array.max()
                    if min_depth > self.altitude_max or max_depth < self.altitude_min:
                        continue
                    else:
                        self.altitude_based_filtered_indices.append(idx)
                if len(self.altitude_based_filtered_indices) < 3:
                    Console.quit(
                        "Insufficient number of images to compute attenuation parameters..."
                    )

        elif self.distance_metric == "altitude":
            # check if user provides a filelist
            if self._camera_image_file_list == "none":
                distance_csv_path = self.altitude_path
            else:
                distance_csv_path = (
                    Path(self.path_config) / self.trimmed_csv_path
                )

            # read dataframe for corresponding distance csv path
            dataframe = pd.read_csv(distance_csv_path)
            distance_list = dataframe["altitude [m]"]

            for idx in trange(len(distance_list)):
                distance_matrix.fill(distance_list[idx])
                imagename = Path(self._imagelist[idx]).stem
                distance_matrix_numpy_file = imagename + ".npy"
                distance_matrix_numpy_file_path = (
                    self.distance_matrix_numpy_folder / distance_matrix_numpy_file
                )
                self.distance_matrix_numpy_filelist.append(
                    distance_matrix_numpy_file_path
                )

                # create the distance matrix numpy file
                np.save(distance_matrix_numpy_file_path, distance_matrix)

            Console.info("Distance matrix numpy files written successfully")

            self.altitude_based_filtered_indices = [
                index
                for index, distance in enumerate(distance_list)
                if distance < self.altitude_max and distance > self.altitude_min
            ]
            Console.info(
                len(self.altitude_based_filtered_indices),
                "Images filtered as per altitude range...",
            )
            if len(self.altitude_based_filtered_indices) < 3:
                Console.quit(
                    "Insufficient number of images to compute attenuation parameters..."
                )

        # create a list of image numpy files to be written to disk

    def generate_bayer_numpy_filelist(self, image_pathlist):
        """Generate list of paths to image numpy files
        
        Parameters
        ----------
        image_pathlist : list
            list of paths to source images
        """

        # generate numpy filelist from imagelst
        self.bayer_numpy_filelist = []
        for imagepath in image_pathlist:
            imagepath_temp = Path(imagepath)
            bayer_file_stem = imagepath_temp.stem
            bayer_file_path = self.bayer_numpy_dir_path / str(bayer_file_stem + ".npy")
            self.bayer_numpy_filelist.append(bayer_file_path)

        # write the intermediate image numpy files to disk

    def generate_bayer_numpyfiles(self, bayer_numpy_filelist):

        """Generates image numpy files
        
        Parameters
        ----------
        bayer_numpy_filelist : list
            list of paths to image numpy files
        """

        # create numpy files as per bayer_numpy_filelist
        if self._camera.extension == "tif":
            # write numpy files for corresponding bayer images
            for idx in trange(len(self._imagelist)):
                tmp_tif = imageio.imread(self._imagelist[idx])
                tmp_npy = np.array(tmp_tif, np.uint16)
                np.save(bayer_numpy_filelist[idx], tmp_npy)
        if self._camera.extension == "raw":
            # create numpy files as per bayer_numpy_filelist
            raw_image_for_size = np.fromfile(str(self._imagelist[0]), dtype=np.uint8)
            binary_data = np.zeros(
                (len(self._imagelist), raw_image_for_size.shape[0]),
                dtype=raw_image_for_size.dtype,
            )
            for idx in range(len(self._imagelist)):
                binary_data[idx] = np.fromfile(
                    str(self._imagelist[idx]), dtype=raw_image_for_size.dtype
                )
            Console.info("Writing RAW images to numpy...")

            image_raw = joblib.Parallel(n_jobs=-2, verbose=3)(
                [
                    joblib.delayed(load_xviii_bayer_from_binary)(
                        binary_data[idx, :], self.image_height, self.image_width
                    )
                    for idx in trange(len(self._imagelist))
                ]
            )
            for idx in trange(len(self._imagelist)):
                np.save(bayer_numpy_filelist[idx], image_raw[idx])

        Console.info("Image numpy files written successfully...")

    # compute correction parameters either for attenuation correction or static correction of images
    def generate_attenuation_correction_parameters(self):
        """Generates image stats and attenuation coeffiecients and saves the parameters for process"""

        # create empty matrices to store image correction parameters
        self.image_raw_mean = np.empty(
            (self.image_channels, self.image_height, self.image_width)
        )
        self.image_raw_std = np.empty(
            (self.image_channels, self.image_height, self.image_width)
        )

        self.image_attenuation_parameters = np.empty(
            (self.image_channels, self.image_height, self.image_width, 3)
        )
        self.image_corrected_mean = np.empty(
            (self.image_channels, self.image_height, self.image_width)
        )
        self.image_corrected_std = np.empty(
            (self.image_channels, self.image_height, self.image_width)
        )

        self.correction_gains = np.empty(
            (self.image_channels, self.image_height, self.image_width)
        )

        # compute correction parameters if distance matrix is generated
        if len(self.distance_matrix_numpy_filelist) > 0:
            # create image and distance_matrix memmaps
            # based on altitude filtering
            filtered_image_numpy_filelist = []
            filtered_distance_numpy_filelist = []

            for idx in self.altitude_based_filtered_indices:
                filtered_image_numpy_filelist.append(self.bayer_numpy_filelist[idx])
                filtered_distance_numpy_filelist.append(
                    self.distance_matrix_numpy_filelist[idx]
                )

            # delete existing memmap files
            memmap_files_path = self.memmap_folder.glob("*.map")
            for file in memmap_files_path:
                if file.exists():
                    file.unlink()

            image_memmap_path, image_memmap = load_memmap_from_numpyfilelist(
                self.memmap_folder, filtered_image_numpy_filelist
            )
            distance_memmap_path, distance_memmap = load_memmap_from_numpyfilelist(
                self.memmap_folder, filtered_distance_numpy_filelist
            )

            for i in range(self.image_channels):

                if self.image_channels == 1:
                    image_memmap_per_channel = image_memmap
                else:
                    image_memmap_per_channel = image_memmap[:, :, :, i]
                # calculate mean, std for image and target_altitude
                raw_image_mean, raw_image_std = mean_std(image_memmap_per_channel)
                self.image_raw_mean[i] = raw_image_mean
                self.image_raw_std[i] = raw_image_std

                target_altitude = mean_std(distance_memmap, False)

                # compute the mean distance for each image
                [n, a, b] = distance_memmap.shape
                distance_vector = distance_memmap.reshape((n, a * b))
                mean_distance_array = distance_vector.mean(axis=1)

                # compute histogram of distance with respect to altitude range(min, max)
                bin_band = 0.1
                hist_bounds = np.arange(self.altitude_min, self.altitude_max, bin_band)
                idxs = np.digitize(mean_distance_array, hist_bounds)

                bin_images_sample_list = []
                bin_distances_sample_list = []

                for idx_bin in trange(1, hist_bounds.size):
                    tmp_idxs = np.where(idxs == idx_bin)[0]
                    if len(tmp_idxs) > 0:
                        bin_images = image_memmap_per_channel[tmp_idxs]
                        bin_distances = distance_memmap[tmp_idxs]
                        if self.smoothing == "mean":
                            bin_images_sample = np.mean(bin_images, axis=0)
                            bin_distances_sample = np.mean(bin_distances, axis=0)
                        elif self.smoothing == "mean_trimmed":
                            bin_images_sample = image_mean_std_trimmed(bin_images)
                            bin_distances_sample = np.mean(bin_distances, axis=0)
                        elif self.smoothing == "median":
                            bin_images_sample = np.median(bin_images, axis=0)
                            bin_distances_sample = np.mean(bin_distances, axis=0)

                        del bin_images
                        del bin_distances

                        bin_images_sample_list.append(bin_images_sample)
                        bin_distances_sample_list.append(bin_distances_sample)

                images_for_attenuation_calculation = np.array(bin_images_sample_list)
                distances_for_attenuation_calculation = np.array(
                    bin_distances_sample_list
                )
                images_for_attenuation_calculation = images_for_attenuation_calculation.reshape(
                    [len(bin_images_sample_list), self.image_height * self.image_width]
                )
                distances_for_attenuation_calculation = distances_for_attenuation_calculation.reshape(
                    [
                        len(bin_distances_sample_list),
                        self.image_height * self.image_width,
                    ]
                )

                # calculate attenuation parameters per channel
                attenuation_parameters = self.calculate_attenuation_parameters(
                    images_for_attenuation_calculation,
                    distances_for_attenuation_calculation,
                    self.image_height,
                    self.image_width,
                )

                self.image_attenuation_parameters[i] = attenuation_parameters

                # compute correction gains per channel
                Console.info("Computing correction gains...")
                correction_gains = self.calculate_correction_gains(
                    target_altitude, attenuation_parameters
                )
                self.correction_gains[i] = correction_gains

                # apply gains to images
                Console.info("Applying attenuation corrections to images...")
                image_memmap_per_channel = self.apply_attenuation_corrections(
                    image_memmap_per_channel,
                    distance_memmap,
                    attenuation_parameters,
                    correction_gains,
                )

                # calculate corrected mean and std per channel
                image_corrected_mean, image_corrected_std = mean_std(
                    image_memmap_per_channel
                )
                self.image_corrected_mean[i] = image_corrected_mean
                self.image_corrected_std[i] = image_corrected_std

            Console.info("Correction parameters generated for all channels...")

            attenuation_parameters_file = (
                self.attenuation_parameters_folder / "attenuation_parameters.npy"
            )
            correction_gains_file = (
                self.attenuation_parameters_folder / "correction_gains.npy"
            )
            image_corrected_mean_file = (
                self.attenuation_parameters_folder / "image_corrected_mean.npy"
            )
            image_corrected_std_file = (
                self.attenuation_parameters_folder / "image_corrected_std.npy"
            )

            # save parameters for process
            np.save(attenuation_parameters_file, self.image_attenuation_parameters)
            np.save(correction_gains_file, self.correction_gains)
            np.save(image_corrected_mean_file, self.image_corrected_mean)
            np.save(image_corrected_std_file, self.image_corrected_std)

        # compute only the raw image mean and std if distance matrix is null
        if len(self.distance_matrix_numpy_filelist) == 0:
            image_memmap_path, image_memmap = load_memmap_from_numpyfilelist(
                self.memmap_folder, self.bayer_numpy_filelist
            )
            for i in range(self.image_channels):
                if self.image_channels == 1:
                    image_memmap_per_channel = image_memmap
                else:
                    image_memmap_per_channel = image_memmap[:, :, :, i]

                # calculate mean, std for image and target_altitude
                raw_image_mean, raw_image_std = mean_std(image_memmap_per_channel)
                self.image_raw_mean[i] = raw_image_mean
                self.image_raw_std[i] = raw_image_std
        image_raw_mean_file = self.attenuation_parameters_folder / "image_raw_mean.npy"
        image_raw_std_file = self.attenuation_parameters_folder / "image_raw_std.npy"

        np.save(image_raw_mean_file, self.image_raw_mean)
        np.save(image_raw_std_file, self.image_raw_std)

        Console.info("Correction parameters saved...")

    # calculate image attenuation parameters
    def calculate_attenuation_parameters(
        self, images, distances, image_height, image_width
    ):
        """Compute attenuation parameters for all images

        Parameters
        -----------
        images : numpy.ndarray
            image memmap reshaped as a vector
        distances : numpy.ndarray
            distance memmap reshaped as a vector
        image_height : int
            height of an image
        image_width : int
            width of an image
        
        Returns
        -------
        numpy.ndarray
            attenuation_parameters
        """

        Console.info("Start curve fitting...")

        results = joblib.Parallel(n_jobs=-2, verbose=3)(
            [
                joblib.delayed(curve_fitting)(distances[:, i_pixel], images[:, i_pixel])
                for i_pixel in trange(image_height * image_width)
            ]
        )

        attenuation_parameters = np.array(results)
        attenuation_parameters = attenuation_parameters.reshape(
            [self.image_height, self.image_width, 3]
        )
        return attenuation_parameters

    # compute gain values for each pixel for a targeted altitide using the attenuation parameters
    def calculate_correction_gains(self, target_altitude, attenuation_parameters):
        """Compute correction gains for an image

        Parameters
        -----------
        target_altitude : numpy.ndarray
            target distance for which the images will be corrected
        attenuation_parameters : numpy.ndarray
            attenuation coefficients

        Returns
        -------
        numpy.ndarray
            The correction gains
        """

        attenuation_parameters = attenuation_parameters.squeeze()
        return (
            attenuation_parameters[:, :, 0]
            * np.exp(attenuation_parameters[:, :, 1] * target_altitude)
            + attenuation_parameters[:, :, 2]
        )

    def apply_attenuation_corrections(
        self, image_memmap, distance_memmap, attenuation_parameters, gains
    ):
        """Apply attenuation corrections to an image memmap

        Parameters
        -----------
        image_memmap : numpy.ndarray
            input image memmap
        distance_memmap : numpy.ndarray
            input distance memmap
        attenuation_parameters : numpy.ndarray
            attenuation coefficients
        gains : numpy.ndarray
            gain values for the image

        
        Returns
        -------
        numpy.ndarray
            Resulting images after applying attenuation correction
        """

        for i_img in trange(image_memmap.shape[0]):
            # memmap data can not be updated in joblib .
            image_memmap[i_img, ...] = self.apply_atn_crr_2_img(
                image_memmap[i_img, ...],
                distance_memmap[i_img, ...],
                attenuation_parameters,
                gains,
            )
        return image_memmap

    def apply_atn_crr_2_img(self, img, altitude, atn_crr_params, gain):
        """ apply attenuation coefficents to an input image

        Parameters
        -----------
        img : numpy.ndarray
            input image
        altitude : 
            distance matrix corresponding to the image
        atn_crr_params : numpy.ndarray
            attenuation coefficients
        gain : numpy.ndarray
            gain value for the image
        
        Returns
        -------
        numpy.ndarray
            Corrected image
        """

        atn_crr_params = atn_crr_params.squeeze()
        img = (
            (
                gain
                / (
                    atn_crr_params[:, :, 0] * np.exp(atn_crr_params[:, :, 1] * altitude)
                    + atn_crr_params[:, :, 2]
                )
            )
            * img
        ).astype(np.float32)
        return img

    # execute the corrections of images using the gain values in case of attenuation correction or static color balance
    def process_correction(self, test_phase=False):
        """Execute series of corrections for a set of input images

        Parameters
        -----------
        idx : int
            index to the list of image numpy files
        test_phase : bool
            argument needed to indicate function is being called for unit testing
        """

        # check for calibration file if distortion correction needed
        if self.undistort:
            camera_params_folder = Path(self.path_processed).parents[0] / "calibration"
            camera_params_filename = "mono" + self.camera_name + ".yaml"
            camera_params_file_path = camera_params_folder / camera_params_filename

            if not camera_params_file_path.exists():
                Console.info("Calibration file not found...")
                self.undistort = False
            else:
                Console.info("Calibration file found...")
                self.camera_params_file_path = camera_params_file_path

        Console.info("Processing images for color , distortion, gamma corrections...")

        joblib.Parallel(n_jobs=-2, verbose=3)(
            joblib.delayed(self.process_image)(idx, test_phase)
            for idx in trange(0, len(self.bayer_numpy_filelist))
        )

        Console.info("Processing of images is completed...")

    def process_image(self, idx, test_phase):
        """Execute series of corrections for an image

        Parameters
        -----------
        idx : int
            index to the list of image numpy files
        test_phase : bool
            argument needed to indicate function is being called for unit testing
        """

        # load numpy image and distance files
        image = np.load(self.bayer_numpy_filelist[idx])
        if len(self.distance_matrix_numpy_filelist) > 0:
            distance = np.load(self.distance_matrix_numpy_filelist[idx])
        else:
            distance = None

        # apply corrections
        # Console.info('Correcting images to targetted mean and std...')
        if self.correction_method == "colour_correction":
            image = self.apply_distance_based_corrections(
                image, distance, self.brightness, self.contrast
            )
        elif self.correction_method == "manual_balance":
            image = self.apply_manual_balance(
                image, self.colour_correction_matrix_rgb, self.subtractors_rgb
            )

        # save corrected image back to numpy list for testing purposes
        if test_phase:
            np.save(self.bayer_numpy_filelist[idx], image)

        # debayer images of source images are bayer ones
        if not self._type == "grayscale":
            # debayer images
            image_rgb = self.debayer(image, self._type)
        else:
            image_rgb = image

        # apply distortion corrections
        if self.undistort:
            image_rgb = self.distortion_correct(image_rgb)

        # apply gamma corrections to rgb images
        image_rgb = self.gamma_correct(image_rgb)

        # write to output files
        image_filename = Path(self.bayer_numpy_filelist[idx]).stem
        image_rgb = image_rgb.astype(np.uint8)
        self.write_output_image(
            image_rgb, image_filename, self.output_images_folder, self.output_format
        )

    # apply corrections on each image using the correction paramters for targeted brightness and contrast
    def apply_distance_based_corrections(self, image, distance, brightness, contrast):
        """Apply attenuation corrections to images

        Parameters
        -----------
        image : numpy.ndarray
            image data to be debayered
        distance : numpy.ndarray
            distance values
        brightness : int
            target mean for output image
        contrast : int
            target std for output image

        Returns
        -------
        numpy.ndarray
            Corrected image
        """
        for i in range(self.image_channels):
            if self.image_channels == 3:
                intensities = image[:, :, i]
            else:
                intensities = image[:, :]
            if not distance is None:
                intensities = self.apply_atn_crr_2_img(
                    intensities,
                    distance,
                    self.image_attenuation_parameters[i],
                    self.correction_gains[i],
                )
                intensities = self.pixel_stat(
                    intensities,
                    self.image_corrected_mean[i],
                    self.image_corrected_std[i],
                    brightness,
                    contrast,
                )
            else:
                intensities = self.pixel_stat(
                    intensities,
                    self.image_raw_mean[i],
                    self.image_raw_std[i],
                    brightness,
                    contrast,
                )
            if self.image_channels == 3:
                image[:, :, i] = intensities
            else:
                image[:, :] = intensities

        return image

    def pixel_stat(self, img, img_mean, img_std, target_mean, target_std, dst_bit=8):
        """Generate target stats for images

        Parameters
        -----------
        img : numpy.ndarray
            image data to be corrected for target stats
        img_mean : int
            current mean
        img_std : int
            current std
        target_mean : int
            desired mean
        target_std : int
            desired std
        dst_bit : int
            destination bit depth

        Returns
        -------
        numpy.ndarray
            Corrected image
        """

        image = (((img - img_mean) / img_std) * target_std) + target_mean
        image = np.clip(image, 0, 2 ** dst_bit - 1)
        return image

    def apply_manual_balance(self, image):
        """Perform manual balance of input image

        Parameters
        -----------
        image : numpy.ndarray
            image data to be debayered

        Returns
        -------
        numpy.ndarray
            Corrected image
        """

        if self.image_channels == 3:
            # corrections for RGB images
            image = image.reshape((self.image_height * self.image_width, 3))
            for i in range(self.image_height * self.image_width):
                intensity_vector = image[i, :]
                gain_matrix = self.colour_correction_matrix_rgb
                intensity_vector = gain_matrix.dot(intensity_vector)
                intensity_vector = intensity_vector - self.subtractors_rgb
        else:
            # for B/W images, default values are the ones for red channel
            image = (
                image * self.colour_correction_matrix_rgb[0, 0]
                - self.subtractors_rgb[0, 0]
            )

        return image

    # convert bayer image to RGB based
    # on the bayer pattern for the camera
    def debayer(self, image, pattern):
        """Perform debayering of input image

        Parameters
        -----------
        image : numpy.ndarray
            image data to be debayered
        pattern : string
            bayer pattern

        Returns
        -------
        numpy.ndarray
            Debayered image
        """
        Console.info(pattern)
        image8 = image.astype(np.uint8)
        corrected_rgb_img = None
        if pattern == "rggb" or pattern == "RGGB":
            corrected_rgb_img = cv2.cvtColor(image8, cv2.COLOR_BAYER_RG2RGB)
        elif pattern == "grbg" or pattern == "GRBG":
            corrected_rgb_img = cv2.cvtColor(image8, cv2.COLOR_BAYER_GR2RGB)
        elif pattern == "bggr" or pattern == "BGGR":
            corrected_rgb_img = cv2.cvtColor(image8, cv2.COLOR_BAYER_BG2RGB)
        elif pattern == "gbrg" or pattern == "GBRG":
            corrected_rgb_img = cv2.cvtColor(image8, cv2.COLOR_BAYER_GB2RGB)
        elif pattern == "mono" or pattern == "MONO":
            return image
        else:
            Console.quit("Bayer pattern not supported (", pattern, ")")
        return corrected_rgb_img

    # correct image for distortions using camera calibration parameters
    def distortion_correct(self, image, dst_bit=8):
        """Perform distortion correction for images

        Parameters
        -----------
        image : numpy.ndarray
            image data to be corrected for distortion
        dst_bit : int
            target bitdepth for output image

        Returns
        -------
        numpy.ndarray
            Image
        """

        monocam = MonoCamera(camera_params_file_path)
        map_x, map_y = monocam.rectification_maps
        image = np.clip(image, 0, 2 ** dst_bit - 1)
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        return image

    # gamma corrections for image
    def gamma_correct(self, image, bitdepth=8):
        """ performs gamma correction for images
        Parameters
        -----------
        image : numpy.ndarray
            image data to be corrected for gamma
        bitdepth : int
            target bitdepth for output image


        Returns
        -------
        numpy.ndarray
            Image
        """
        image = np.divide(image, ((2 ** bitdepth - 1)))
        if all(i < 0.0031308 for i in image.flatten()):
            image = 12.92 * image
        else:
            image = 1.055 * np.power(image, (1 / 1.5)) - 0.055
        image = np.multiply(np.array(image), np.array(2 ** bitdepth - 1))
        image = np.clip(image, 0, 2 ** bitdepth - 1)
        return image

    # save processed image in an output file with
    # given output format
    def write_output_image(self, image, filename, dest_path, dest_format):
        """Write into output images

        Parameters
        -----------
        image : numpy.ndarray
            image data to be written
        filename : string
            name of output image file
        dest_path : Path
            path to the output folder
        dest_format : string
            output image format
        """

        file = filename + "." + dest_format
        file_path = Path(dest_path) / file
        imageio.imwrite(file_path, image)


# NON-MEMBER FUNCTIONS:
# ------------------------------------------------------------------------------


# read binary raw image files for xviii camera
def load_xviii_bayer_from_binary(binary_data, image_height, image_width):
    """Read XVIII binary images into bayer array

    Parameters
    -----------
    binary_data : numpy.ndarray
        binary image data from XVIII
    image_height : int
        image height
    image_width : int
        image width

    Returns
    --------
    numpy.ndarray
        Bayer image
    """

    img_h = image_height
    img_w = image_width
    bayer_img = np.zeros((img_h, img_w), dtype=np.uint32)

    # read raw data and put them into bayer patttern.
    count = 0
    for i in range(0, img_h, 1):
        for j in range(0, img_w, 4):
            chunk = binary_data[count : count + 12]
            bayer_img[i, j] = (
                ((chunk[3] & 0xFF) << 16) | ((chunk[2] & 0xFF) << 8) | (chunk[1] & 0xFF)
            )
            bayer_img[i, j + 1] = (
                ((chunk[0] & 0xFF) << 16) | ((chunk[7] & 0xFF) << 8) | (chunk[6] & 0xFF)
            )
            bayer_img[i, j + 2] = (
                ((chunk[5] & 0xFF) << 16)
                | ((chunk[4] & 0xFF) << 8)
                | (chunk[11] & 0xFF)
            )
            bayer_img[i, j + 3] = (
                ((chunk[10] & 0xFF) << 16)
                | ((chunk[9] & 0xFF) << 8)
                | (chunk[8] & 0xFF)
            )
            count += 12

    bayer_img = bayer_img / 1024
    return bayer_img


# store into memmaps the distance and image numpy files
def load_memmap_from_numpyfilelist(filepath, numpyfilelist):
    """Generate memmaps from numpy arrays
    
    Parameters
    -----------
    filepath : Path
        path to output memmap folder 
    numpyfilelist : list
        list of paths to numpy files

    Returns
    --------
    Path, numpy.ndarray
        memmap_path and memmap_handle
    """

    message = "loading binary files into memmap..."
    image = np.load(str(numpyfilelist[0]))
    list_shape = [len(numpyfilelist)]
    list_shape = list_shape + list(image.shape)

    filename_map = "memmap_" + str(uuid.uuid4()) + ".map"
    memmap_path = Path(filepath) / filename_map

    memmap_handle = np.memmap(
        filename=memmap_path, mode="w+", shape=tuple(list_shape), dtype=np.float32
    )
    Console.info("Loading memmaps from numpy files...")
    for idx in trange(0, len(numpyfilelist), ascii=True, desc=message):
        memmap_handle[idx, ...] = np.load(numpyfilelist[idx])

    return memmap_path, memmap_handle


# calculate the mean and std of an image
def mean_std(data, calculate_std=True):
    """Compute mean and std for image intensities and distance values

    Parameters
    -----------
    data : numpy.ndarray
        data series 
    calculate_std : bool
        denotes to compute std along with mean

    Returns
    --------
    numpy.ndarray
        mean and std of input image
    """

    [n, a, b] = data.shape

    data.reshape((n, a * b))

    ret_mean = data.mean(axis=0)
    if calculate_std:
        ret_std = data.std(axis=0)
        return ret_mean.reshape((a, b)), ret_std.reshape((a, b))

    else:
        return ret_mean.reshape((a, b))


def image_mean_std_trimmed(data, ratio_trimming=0.2, calculate_std=True):
    """Compute trimmed mean and std for image intensities using parallel computing

    Parameters
    -----------
    data : numpy.ndarray
        image intensities 
    ratio_trimming : float
        trim ratio
    calculate_std : bool
        denotes to compute std along with mean

    Returns
    --------
    numpy.ndarray
        Trimmed mean and std
    """

    [n, a, b] = data.shape
    ret_mean = np.zeros((a, b), np.float32)
    ret_std = np.zeros((a, b), np.float32)

    effective_index = [list(range(0, n))]

    message = "calculating mean and std of images " + datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if ratio_trimming <= 0:
        ret_mean = np.mean(data, axis=0)
        ret_std = np.std(data, axis=0)

    else:
        if calculate_std == False:
            for idx_a in trange(a, ascii=True, desc=message):
                results = joblib.Parallel(n_jobs=-2, verbose=3)(
                    [
                        joblib.delayed(calc_mean_and_std_trimmed)(
                            data[effective_index, idx_a, idx_b][0],
                            ratio_trimming,
                            calculate_std,
                        )
                        for idx_b in trange(b)
                    ]
                )
                ret_mean[idx_a, :] = np.array(results)[:, 0]
            return ret_mean

        else:
            for idx_a in trange(a, ascii=True, desc=message):
                results = joblib.Parallel(n_jobs=-2, verbose=3)(
                    [
                        joblib.delayed(calc_mean_and_std_trimmed)(
                            data[effective_index, idx_a, idx_b][0],
                            ratio_trimming,
                            calculate_std,
                        )
                        for idx_b in trange(b)
                    ]
                )
                ret_mean[idx_a, :] = np.array(results)[:, 0]
                ret_std[idx_a, :] = np.array(results)[:, 1]
            return ret_mean, ret_std


def calc_mean_and_std_trimmed(data, rate_trimming, calc_std=True):
    """Compute trimmed mean and std for image intensities

    Parameters
    -----------
    data : numpy.ndarray
        image intensities 
    rate_trimming : float
        trim ratio
    calc_std : bool
        denotes to compute std along with mean

    Returns
    --------
    numpy.ndarray
    """

    if rate_trimming <= 0:
        mean = np.mean(data)
        if calc_std:
            std = np.std(data)
    else:
        sorted_values = np.sort(data)
        idx_left_limit = int(len(data) * rate_trimming / 2.0)
        idx_right_limit = int(len(data) * (1.0 - rate_trimming / 2.0))

        mean = np.mean(sorted_values[idx_left_limit:idx_right_limit])
        std = 0

        if calc_std:
            std = np.std(sorted_values[idx_left_limit:idx_right_limit])

    return np.array([mean, std])


def exp_curve(x, a, b, c):
    """Compute exponent value with respect to a distance value

    Parameters
    -----------
    x : float
        distance value
    a : float
        coefficient
    b : float
        coefficient
    c : float
        coefficient

    Returns
    --------
    float
    """

    return a * np.exp(b * x) + c


def residual_exp_curve(params, x, y):
    """Compute residuals with respect to init params

    Parameters
    -----------
    params : numpy.ndarray
        array of init params
    x : numpy.ndarray
        array of distance values
    y : numpy.ndarray
        array of intensity values

    Returns
    --------
    numpy.ndarray
        residual
    """

    residual = exp_curve(x, params[0], params[1], params[2]) - y
    return residual


# compute attenuation correction parameters through regression
def curve_fitting(distancelist, intensitylist):
    """Compute attenuation coefficients with respect to distance values

    Parameters
    -----------
    distancelist : list
        list of distance values
    intensitylist : list
        list of intensity values

    Returns
    --------
    numpy.ndarray
        parameters
    """

    loss = "soft_l1"
    # loss='linear'
    method = "trf"
    # method='lm'
    bound_lower = [1, -np.inf, 0]
    bound_upper = [np.inf, 0, np.inf]

    altitudes = np.array(distancelist)
    intensities = np.array(intensitylist)

    flag_already_calculated = False
    min_cost = float("inf")
    c = 0

    idx_0 = int(len(intensities) * 0.3)
    idx_1 = int(len(intensities) * 0.7)

    b = (np.log((intensities[idx_0] - c) / (intensities[idx_1] - c))) / (
        altitudes[idx_0] - altitudes[idx_1]
    )
    a = (intensities[idx_1] - c) / np.exp(b * altitudes[idx_1])
    if a < 1 or b > 0 or np.isnan(a) or np.isnan(b):
        a = 1.01
        b = -0.01

    init_params = np.array([a, b, c])
    # tmp_params=None
    try:
        tmp_params = optimize.least_squares(
            residual_exp_curve,
            init_params,
            loss=loss,
            method=method,
            args=(altitudes, intensities),
            bounds=(bound_lower, bound_upper),
        )
        if tmp_params.cost < min_cost:
            min_cost = tmp_params.cost
            ret_params = tmp_params.x

    except (ValueError, UnboundLocalError) as e:
        Console.error("Value Error due to Overflow", a, b, c)
        ret_params = init_params
        Console.warn('Parameters calculated are unoptimised because of Value Error...')

    return ret_params


def remove_directory(folder):
    """Remove a specific directory recursively

    Parameters
    -----------
    folder : Path
        folder to be removed    
    """

    for p in folder.iterdir():
        if p.is_dir():
            remove_directory(p)
        else:
            p.unlink()
    folder.rmdir()



# functions used to create a trimmed auv_ekf_<camera_name>.csv file based on user's selection of images
def trim_csv_files(filelist, original_csv_path, trimmed_csv_path):
    """Trim csv files based on the list of images provided by user

    Parameters
    -----------
    filelist : user provided list of imagenames which need to be processed
    original_csv_path : path to the auv_ekf_<camera_name.csv>
    trimmed_csv_path : path to trimmed csv which needs to be created    
    """

    imagename_list=get_imagename_list(filelist)
    dataframe = pd.read_csv(original_csv_path)
    image_path_list = dataframe['relative_path']
    trimmed_path_list = [path 
                            for path in image_path_list
                            if Path(path).name in imagename_list ]
    trimmed_dataframe = dataframe.loc[dataframe['relative_path'].isin(trimmed_path_list) ]
    trimmed_dataframe.to_csv (trimmed_csv_path, index = False, header=True)



def get_imagename_list(imagename_list_filepath):
    """ get list of imagenames from the filelist provided by user

    Parameters
    -----------
    imagename_list_filepath : Path to the filelist provided in correct_images.yaml
    """

    with open(imagename_list_filepath, 'r') as imagename_list_file:
        imagename_list = imagename_list_file.read().splitlines()
    return imagename_list





    
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import os
import random
from datetime import datetime
from pathlib import Path

import imageio
import joblib
import numpy as np
import pandas as pd
from oplab import (
    Console,
    get_config_folder,
    get_processed_folder,
    get_raw_folder,
)
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

from correct_images import corrections
from correct_images.loaders import loader, depth_map
from correct_images.tools.file_handlers import (
    trim_csv_files,
    write_output_image,
)
from correct_images.tools.joblib_tqdm import tqdm_joblib
from correct_images.tools.memmap import (
    create_memmap,
    open_memmap,
)
from correct_images.tools.numerical import (
    RunningMeanStd,
    image_mean_std_trimmed,
    median_array,
    running_mean_std,
)

# -----------------------------------------


class Corrector:
    def __init__(
        self, force=False, camera=None, correct_config=None, path=None
    ):
        """Constructor for the Corrector class

        Parameters
        ----------
        force : bool
            to indicate an overwrite for existing parameters or images
        camera : CameraEntry
            camera object for which the processing is expected
        correct_config : CorrectConfig
            correct config object storing all the correction configuration
            parameters
        path : Path
            path to the dive folder where image directory is present
        """

        self.camera = camera
        self.correct_config = correct_config
        self.camera_image_list = None
        self.processed_image_list = None
        self.altitude_list = None
        self.depth_map_list = None

        # Members for folder paths
        self.output_dir_path = None
        self.bayer_numpy_dir_path = None
        self.attenuation_parameters_folder = None
        self.memmap_folder = None
        self.output_images_folder = None

        # Placeholders for process
        self.image_attenuation_parameters = None
        self.correction_gains = None
        self.image_corrected_mean = None
        self.image_corrected_std = None
        self.image_raw_mean = None
        self.image_raw_std = None

        # From get image list
        self.altitude_csv_path = None
        self.trimmed_csv_path = None
        self.camera_params_file_path = None

        if path is not None:
            self.path_raw = get_raw_folder(path)
            self.path_processed = get_processed_folder(path)
            self.path_config = get_config_folder(path)
        self.force = force

        self.user_specified_image_list = (
            None  # To be overwritten on parse/process
        )
        self.user_specified_image_list_parse = None
        self.user_specified_image_list_process = None

        if self.correct_config is not None:
            """Load general configuration parameters"""
            self.correction_method = self.correct_config.method
            if self.correction_method == "colour_correction":
                self.distance_metric = (
                    self.correct_config.color_correction.distance_metric
                )
                self.distance_path = (
                    self.correct_config.color_correction.metric_path
                )
                self.altitude_max = (
                    self.correct_config.color_correction.altitude_max
                )
                self.altitude_min = (
                    self.correct_config.color_correction.altitude_min
                )
                self.smoothing = self.correct_config.color_correction.smoothing
                self.window_size = (
                    self.correct_config.color_correction.window_size
                )
                self.outlier_rejection = (
                    self.correct_config.color_correction.outlier_reject
                )
            self.cameraconfigs = self.correct_config.configs.camera_configs
            self.undistort = self.correct_config.output_settings.undistort_flag
            self.output_format = (
                self.correct_config.output_settings.compression_parameter
            )

            # Load camera parameters
            cam_idx = self.get_camera_idx()
            self.camera_found = False
            if cam_idx is None:
                Console.info(
                    "Camera not included in correct_images.yaml. No",
                    "processing will be done for this camera.",
                )
                return
            else:
                self.camera_found = True
            self.user_specified_image_list_parse = self.cameraconfigs[
                cam_idx
            ].imagefilelist_parse
            self.user_specified_image_list_process = self.cameraconfigs[
                cam_idx
            ].imagefilelist_process

            if self.correction_method == "colour_correction":
                # Brighness and contrast are percentages of 255
                # e.g. brightness of 30 means 30% of 255 = 77
                self.brightness = float(self.cameraconfigs[cam_idx].brightness)
                self.contrast = float(self.cameraconfigs[cam_idx].contrast)
            elif self.correction_method == "manual_balance":
                self.subtractors_rgb = np.array(
                    self.cameraconfigs[cam_idx].subtractors_rgb
                )
                self.color_gain_matrix_rgb = np.array(
                    self.cameraconfigs[cam_idx].color_gain_matrix_rgb
                )
            image_properties = self.camera.image_properties
            self.image_height = image_properties[0]
            self.image_width = image_properties[1]
            self.image_channels = image_properties[2]
            self.camera_name = self.camera.name
            self._type = self.camera.type

            # Create output directories and needed attributes
            self.create_output_directories()

            # Define basic filepaths
            if self.correction_method == "colour_correction":
                self.attenuation_params_filepath = (
                    Path(self.attenuation_parameters_folder)
                    / "attenuation_parameters.npy"
                )
                self.correction_gains_filepath = (
                    Path(self.attenuation_parameters_folder)
                    / "correction_gains.npy"
                )
                self.corrected_mean_filepath = (
                    Path(self.attenuation_parameters_folder)
                    / "image_corrected_mean.npy"
                )
                self.corrected_std_filepath = (
                    Path(self.attenuation_parameters_folder)
                    / "image_corrected_std.npy"
                )
                self.raw_mean_filepath = (
                    Path(self.attenuation_parameters_folder)
                    / "image_raw_mean.npy"
                )
                self.raw_std_filepath = (
                    Path(self.attenuation_parameters_folder)
                    / "image_raw_std.npy"
                )

            # Define image loader
            # Use default loader
            self.loader = loader.Loader()
            self.loader.bit_depth = camera.bit_depth
            if self.camera.extension == "raw":
                self.loader.set_loader("xviii")
            else:
                self.loader.set_loader("default")

    def parse(self):
        # Set the user specified list if any
        self.user_specified_image_list = self.user_specified_image_list_parse

        # Read list of images
        self.get_imagelist()

        Console.info("Output directories created / existing...")

        if self.correction_method == "colour_correction":
            self.get_altitude_and_depth_maps()
            self.generate_attenuation_correction_parameters()
        elif self.correction_method == "manual_balance":
            Console.info("run process for manual_balance...")

    def process(self):
        # Set the user specified list if any
        self.user_specified_image_list = self.user_specified_image_list_process

        # Read list of images
        self.get_imagelist()

        # create target sub directores based on configurations defined in
        # correct_images.yaml
        if self.correction_method == "colour_correction":
            self.get_altitude_and_depth_maps()

            # read parameters from disk
            if self.attenuation_params_filepath.exists():
                self.image_attenuation_parameters = np.load(
                    self.attenuation_params_filepath
                )
            else:
                if self.distance_metric != "none":
                    Console.quit(
                        "Code does not find attenuation_parameters.npy...",
                        "Please run parse before process...",
                    )
            if self.correction_gains_filepath.exists():
                self.correction_gains = np.load(self.correction_gains_filepath)
            else:
                if self.distance_metric != "none":
                    Console.quit(
                        "Code does not find correction_gains.npy...",
                        "Please run parse before process...",
                    )
            if self.corrected_mean_filepath.exists():
                self.image_corrected_mean = np.load(
                    self.corrected_mean_filepath
                ).squeeze()
            else:
                if self.distance_metric != "none":
                    Console.quit(
                        "Code does not find image_corrected_mean.npy...",
                        "Please run parse before process...",
                    )
            if self.corrected_std_filepath.exists():
                self.image_corrected_std = np.load(
                    self.corrected_std_filepath
                ).squeeze()
            else:
                if self.distance_metric != "none":
                    Console.quit(
                        "Code does not find image_corrected_std.npy...",
                        "Please run parse before process...",
                    )
            if (
                self.raw_mean_filepath.exists()
                and self.distance_metric == "none"
            ):
                self.image_raw_mean = np.load(self.raw_mean_filepath).squeeze()
            elif self.distance_metric == "none":
                Console.quit(
                    "Code does not find image_raw_mean.npy...",
                    "Please run parse before process...",
                )
            if (
                self.raw_std_filepath.exists()
                and self.distance_metric == "none"
            ):
                self.image_raw_std = np.load(self.raw_std_filepath).squeeze()
            elif self.distance_metric == "none":
                Console.quit(
                    "Code does not find image_raw_std.npy...",
                    "Please run parse before process...",
                )
            Console.info("Correction parameters loaded...")
            Console.info("Running process for colour correction...")
        else:
            Console.info("Running process with manual colour balancing...")
        self.process_correction()

    # create directories for storing intermediate image and distance_matrix
    # numpy files, correction parameters and corrected output images
    def create_output_directories(self):
        """Handle the creation of output directories for each camera"""

        # create output directory path
        image_path = Path(self.camera.image_list[0]).resolve()
        image_parent_path = image_path.parent
        output_dir_path = get_processed_folder(image_parent_path)
        self.output_dir_path = output_dir_path / "attenuation_correction"
        if not self.output_dir_path.exists():
            self.output_dir_path.mkdir(parents=True)

        if self.correction_method == "colour_correction":
            # create path for parameters files
            attenuation_parameters_folder_name = "params_" + self.camera.name
            self.attenuation_parameters_folder = (
                self.output_dir_path / attenuation_parameters_folder_name
            )
            if not self.attenuation_parameters_folder.exists():
                self.attenuation_parameters_folder.mkdir(parents=True)

        # create path for output images
        output_images_folder_name = "developed_" + self.camera.name
        self.output_images_folder = (
            self.output_dir_path / output_images_folder_name
        )
        if not self.output_images_folder.exists():
            self.output_images_folder.mkdir(parents=True)

        # create folder name for  parameters based on correction method
        sub_directory_name = "unknown_sub_directory_name"
        output_folder_name = "unknown_output_folder_name"
        if self.correction_method == "colour_correction":
            if self.distance_metric == "none":
                sub_directory_name = "greyworld_corrected"
            elif self.distance_metric == "altitude":
                sub_directory_name = "altitude_corrected"
            elif self.distance_metric == "depth_map":
                sub_directory_name = "depth_map_corrected"

            output_folder_name = (
                "m"
                + str(int(self.brightness))
                + "_std"
                + str(int(self.contrast))
            )

            # appending params path with sub directory and output folder
            self.attenuation_parameters_folder = (
                self.attenuation_parameters_folder / sub_directory_name
            )
            if not self.attenuation_parameters_folder.exists():
                self.attenuation_parameters_folder.mkdir(parents=True)
            else:
                dir_temp = self.attenuation_parameters_folder
                file_list = list(dir_temp.glob("*.npy"))
                if len(file_list) > 0:
                    if not self.force:
                        Console.quit(
                            "Parameters exist for current configuration.",
                            "Run parse with Force (-F flag)...",
                        )
                    else:
                        Console.warn(
                            "Code will overwrite existing parameters for ",
                            "current configuration...",
                        )
        elif self.correction_method == "manual_balance":
            sub_directory_name = "manually_corrected"
            temp1 = str(datetime.now())
            temp2 = temp1.split(":")
            temp3 = temp2[0].split(" ")
            temp4 = temp3[1] + temp2[1]
            output_folder_name = "developed_" + temp4

        # appending developed images path with sub directory and output folder
        self.output_images_folder = (
            self.output_images_folder / sub_directory_name / output_folder_name
        )
        if not self.output_images_folder.exists():
            self.output_images_folder.mkdir(parents=True)
        else:
            dir_temp = self.output_images_folder
            file_list = list(dir_temp.glob("*.*"))
            if len(file_list) > 0:
                if not self.force:
                    Console.quit(
                        "Corrected images exist for current configuration. ",
                        "Run process with Force (-F flag)...",
                    )
                else:
                    Console.warn(
                        "Code will overwrite existing corrected images for ",
                        "current configuration...",
                    )

    def get_camera_idx(self):
        idx = [
            i
            for i, camera_config in enumerate(self.cameraconfigs)
            if camera_config.camera_name == self.camera.name
        ]
        if len(idx) > 0:
            return idx[0]
        else:
            Console.warn(
                "The camera",
                self.camera.name,
                "could not be found in the correct_images.yaml",
            )
            return None

    # load imagelist: output is same as camera.imagelist unless a smaller
    # filelist is specified by the user
    def get_imagelist(self):
        """Generate list of source images"""
        if self.correction_method == "colour_correction":
            if self.distance_path == "json_renav_*":
                Console.info(
                    "Picking first JSON folder as the default path to auv_nav",
                    " csv files...",
                )
                dir_ = self.path_processed
                json_list = list(dir_.glob("json_*"))
                if len(json_list) == 0:
                    Console.quit(
                        "No navigation solution could be found. Please run ",
                        "auv_nav parse and process first",
                    )
                self.distance_path = json_list[0]

            metric_path = self.path_processed / self.distance_path
            # Try if ekf exists:
            full_metric_path = metric_path / "csv" / "ekf"
            metric_file = "auv_ekf_" + self.camera_name + ".csv"
            if not full_metric_path.exists():
                full_metric_path = metric_path / "csv" / "dead_reckoning"
                metric_file = "auv_dr_" + self.camera_name + ".csv"
            self.altitude_csv_path = full_metric_path / metric_file

            # get imagelist for given camera object
            if self.user_specified_image_list == "none":
                self.camera_image_list = self.camera.image_list
            # get imagelist from user provided filelist
            else:
                path_file_list = (
                    Path(self.path_config) / self.user_specified_image_list
                )
                trimmed_csv_file = "trimmed_csv_" + self.camera.name + ".csv"
                self.trimmed_csv_path = (
                    Path(self.path_config) / trimmed_csv_file
                )

                if not self.altitude_csv_path.exists():
                    message = "Path to " + metric_file + " does not exist..."
                    Console.quit(message)
                else:
                    # create trimmed csv based on user's  list of images
                    trim_csv_files(
                        path_file_list,
                        self.altitude_csv_path,
                        self.trimmed_csv_path,
                    )

                # read trimmed csv filepath
                dataframe = pd.read_csv(self.trimmed_csv_path)
                user_imagepath_list = dataframe["relative_path"]
                user_imagenames_list = [
                    Path(image).name for image in user_imagepath_list
                ]
                self.camera_image_list = [
                    item
                    for item in self.camera.image_list
                    for image in user_imagenames_list
                    if Path(item).name == image
                ]
        elif self.correction_method == "manual_balance":
            self.camera_image_list = self.camera.image_list

        # save a set of distance matrix numpy files

    def get_altitude_and_depth_maps(self):
        """Generate distance matrix numpy files and save them"""
        # read altitude / depth map depending on distance_metric
        if self.distance_metric == "none":
            Console.info("Null distance matrix created")
            return

        # elif self.distance_metric == "altitude":
        # check if user provides a file list
        if self.user_specified_image_list == "none":
            distance_csv_path = Path(self.altitude_csv_path)
        else:
            distance_csv_path = Path(self.path_config) / self.trimmed_csv_path

        # Check if file exists
        if not distance_csv_path.exists():
            Console.quit(
                "The navigation CSV file is not present. Run auv_nav first."
            )

        # read dataframe for corresponding distance csv path
        dataframe = pd.read_csv(distance_csv_path)
        distance_list = dataframe["altitude [m]"].tolist()

        """
        if len(distance_list) != len(self.camera_image_list):
            Console.warn(
                "The number of images does not coincide with the altitude",
                "measurements.",
            )
            Console.info("Using image file paths from CSV instead.")
        """

        # Check images exist:
        valid_idx = []
        self.camera_image_list = []
        for idx, entry in enumerate(dataframe["relative_path"]):
            im_path = self.path_raw / entry
            if im_path.exists():
                valid_idx.append(idx)
        filtered_dataframe = dataframe.iloc[valid_idx]
        filtered_dataframe.reset_index(drop=True)
        distance_list = filtered_dataframe["altitude [m]"].tolist()
        self.camera_image_list = []
        self.altitude_list = []
        for _, row in filtered_dataframe.iterrows():
            alt = row["altitude [m]"]
            if alt > self.altitude_min and alt < self.altitude_max:
                self.camera_image_list.append(
                    self.path_raw / row["relative_path"]
                )
                self.altitude_list.append(alt)

        Console.info(
            len(self.altitude_list),
            "/",
            len(distance_list),
            "Images filtered as per altitude range...",
        )
        if len(self.altitude_list) < 3:
            Console.quit(
                "Insufficient number of images to compute attenuation ",
                "parameters...",
            )

        if self.distance_metric == "depth_map":
            path_depth = self.path_processed / "depth_map"
            if not path_depth.exists():
                Console.quit("Depth maps not found...")
            else:
                Console.info("Path to depth maps found...")
                depth_map_list = list(path_depth.glob("*.npy"))
                self.depth_map_list = [
                    Path(item)
                    for item in depth_map_list
                    for image_path in self.camera_image_list
                    if Path(image_path).stem in Path(item).stem
                ]

                if len(self.camera_image_list) != len(self.depth_map_list):
                    Console.quit(
                        "The number of images does not coincide with the ",
                        "number of depth maps.",
                    )

    # compute correction parameters either for attenuation correction or
    # static correction of images
    def generate_attenuation_correction_parameters(self):
        """Generates image stats and attenuation coefficients and saves the
        parameters for process"""

        # create empty matrices to store image correction parameters
        self.image_raw_mean = np.empty(
            (self.image_channels, self.image_height, self.image_width),
            dtype=np.float32,
        )
        self.image_raw_std = np.empty(
            (self.image_channels, self.image_height, self.image_width),
            dtype=np.float32,
        )
        self.image_attenuation_parameters = np.empty(
            (self.image_channels, self.image_height, self.image_width, 3),
            dtype=np.float32,
        )
        self.image_corrected_mean = np.empty(
            (self.image_channels, self.image_height, self.image_width),
            dtype=np.float32,
        )
        self.image_corrected_std = np.empty(
            (self.image_channels, self.image_height, self.image_width),
            dtype=np.float32,
        )
        self.correction_gains = np.empty(
            (self.image_channels, self.image_height, self.image_width),
            dtype=np.float32,
        )

        image_size_gb = (
            self.image_channels
            * self.image_height
            * self.image_width
            * 4.0
            / (1024.0 ** 3)
        )
        max_bin_size_gb = 50.0
        max_bin_size = int(max_bin_size_gb / image_size_gb)

        self.bin_band = 0.1
        hist_bins = np.arange(
            self.altitude_min, self.altitude_max, self.bin_band
        )

        images_fn, images_map = open_memmap(
            shape=(
                len(hist_bins),
                self.image_height * self.image_width,
                self.image_channels,
            ),
            dtype=np.float32,
        )

        distances_fn, distances_map = open_memmap(
            shape=(len(hist_bins), self.image_height * self.image_width),
            dtype=np.float32,
        )

        # Fill with NaN not to use empty bins
        # images_map.fill(np.nan)
        # distances_map.fill(np.nan)

        distance_vector = None

        if self.depth_map_list is not None:
            Console.info(
                "Computing depth map histogram with", hist_bins.size, " bins"
            )
            distance_vector = np.zeros((len(self.depth_map_list), 1))
            for i, dm_file in enumerate(self.depth_map_list):
                dm_np = depth_map.loader(
                    dm_file, self.image_width, self.image_height
                )
                distance_vector[i] = dm_np.mean(axis=1)
        elif self.altitude_list is not None:
            Console.info(
                "Computing altitude histogram with", hist_bins.size, " bins"
            )
            distance_vector = np.array(self.altitude_list)

        if distance_vector is not None:
            idxs = np.digitize(distance_vector, hist_bins)

            """
            for idx_bin in range(1, hist_bins.size):
                Console.info(
                    "Computing distance bin",
                    str(idx_bin) + "/" + str(hist_bins.size),
                )
                self.compute_distance_bin(
                    idxs,
                    idx_bin,
                    images_map,
                    distances_map,
                    max_bin_size,
                    max_bin_size_gb,
                    distance_vector,
                )
            """
            with tqdm_joblib(
                tqdm(
                    desc="Computing altitude histogram",
                    total=hist_bins.size - 1,
                )
            ):
                joblib.Parallel(n_jobs=-2, verbose=0)(
                    joblib.delayed(self.compute_distance_bin)(
                        idxs,
                        idx_bin,
                        images_map,
                        distances_map,
                        max_bin_size,
                        max_bin_size_gb,
                        distance_vector,
                    )
                    for idx_bin in range(1, hist_bins.size)
                )

            # calculate attenuation parameters per channel
            self.image_attenuation_parameters = (
                corrections.calculate_attenuation_parameters(
                    images_map,
                    distances_map,
                    self.image_height,
                    self.image_width,
                    self.image_channels,
                )
            )

            # delete memmap handles
            del images_map
            os.remove(images_fn)
            del distances_map
            os.remove(distances_fn)

            # Save attenuation parameter results.
            np.save(
                self.attenuation_params_filepath,
                self.image_attenuation_parameters,
            )

            corrections.save_attenuation_plots(
                self.attenuation_parameters_folder,
                attn=self.image_attenuation_parameters,
            )

            # compute correction gains per channel
            target_altitude = distance_vector.mean()
            Console.info(
                "Computing correction gains for target altitude",
                target_altitude,
            )
            self.correction_gains = corrections.calculate_correction_gains(
                target_altitude,
                self.image_attenuation_parameters,
                self.image_height,
                self.image_width,
                self.image_channels,
            )
            # Save correction gains
            np.save(self.correction_gains_filepath, self.correction_gains)

            corrections.save_attenuation_plots(
                self.attenuation_parameters_folder,
                gains=self.correction_gains,
            )

            # Useful if fails, to reload precomputed numpyfiles.
            # TODO offer as a new step.
            # self.image_attenuation_parameters = np.load(
            #    self.attenuation_params_filepath)
            # self.correction_gains = np.load(self.correction_gains_filepath)

            # apply gains to images
            Console.info("Applying attenuation corrections to images...")

            temp = self.loader(self.camera_image_list[0])  ## bitdepth?
            # runner = RunningMeanStd(temp.shape)

            memmap_filename, memmap_handle = open_memmap(
                shape=(
                    len(self.camera_image_list),
                    self.image_height,
                    self.image_width,
                    self.image_channels,
                ),
                dtype=np.float32,
            )

            for i in trange(len(self.camera_image_list)):
                # Load the image
                img = self.loader(self.camera_image_list[i])

                # Load the distance matrix
                if self.depth_map_list is None:
                    # Generate matrices on the fly
                    distance = distance_vector[i]
                    distance_mtx = np.empty(
                        (self.image_height, self.image_width)
                    )
                    distance_mtx.fill(distance)
                else:
                    distance_mtx = depth_map.loader(
                        self.depth_map_list[i],
                        self.image_width,
                        self.image_height,
                    )

                # Correct the image
                corrected_img = corrections.attenuation_correct(
                    img,
                    distance_mtx,
                    self.image_attenuation_parameters,
                    self.correction_gains,
                )
                # runner.compute(corrected_img)
                memmap_handle[i] = corrected_img.reshape(
                    self.image_height, self.image_width, self.image_channels
                )

            """
            image_corrected_mean = runner.mean.reshape(
                self.image_height, self.image_width, self.image_channels
            )
            image_corrected_std = runner.std.reshape(
                self.image_height, self.image_width, self.image_channels
            )
            """

            Console.info("Computing trimmed mean and std to corrected images")
            image_corrected_mean, image_corrected_std = image_mean_std_trimmed(
                memmap_handle
            )
            del memmap_handle
            os.remove(memmap_filename)

            # save parameters for process
            np.save(self.corrected_mean_filepath, image_corrected_mean)
            np.save(self.corrected_std_filepath, image_corrected_std)

            corrections.save_attenuation_plots(
                self.attenuation_parameters_folder,
                img_mean=image_corrected_mean,
                img_std=image_corrected_std,
            )
        else:
            Console.info(
                "No altitude or depth maps available. \
                Computing raw mean and std."
            )
            image_raw_mean, image_raw_std = running_mean_std(
                self.camera_image_list, self.loader
            )
            np.save(self.raw_mean_filepath, image_raw_mean)
            np.save(self.raw_std_filepath, image_raw_std)
            imageio.imwrite(
                Path(self.attenuation_parameters_folder)
                / "image_raw_mean.png",
                image_raw_mean,
            )
            imageio.imwrite(
                Path(self.attenuation_parameters_folder) / "image_raw_std.png",
                image_raw_std,
            )

            corrections.save_attenuation_plots(
                self.attenuation_parameters_folder,
                img_mean=image_raw_mean,
                img_std=image_raw_std,
            )

        Console.info("Correction parameters saved...")

    def compute_distance_bin(
        self,
        idxs,
        idx_bin,
        images_map,
        distances_map,
        max_bin_size,
        max_bin_size_gb,
        distance_vector,
    ):
        dimensions = [self.image_height, self.image_width, self.image_channels]
        tmp_idxs = np.where(idxs == idx_bin)[0]
        # Console.info("In bin", idx_bin, "there are", len(tmp_idxs), "images")
        if len(tmp_idxs) > 2:
            bin_images = [self.camera_image_list[i] for i in tmp_idxs]
            bin_distances_sample = None
            bin_images_sample = None

            # Random sample if memmap has to be created
            if self.smoothing != "mean" and len(bin_images) > max_bin_size:
                Console.info(
                    "Random sampling altitude bin to fit in",
                    max_bin_size_gb,
                    "Gb",
                )
                bin_images = random.sample(bin_images, max_bin_size)

            if self.depth_map_list is None:
                # Generate matrices on the fly
                distance_bin = distance_vector[tmp_idxs]
                distance_bin_sample = distance_bin.mean()
                if distance_bin_sample <= 0 or np.isnan(distance_bin_sample):
                    Console.warn(
                        "The mean distance is equal or lower than zero!"
                    )
                    Console.warn("Printing the entire vector:", distance_bin)
                    Console.warn("Printing the mean:", distance_bin_sample)
                    distance_bin_sample = (
                        self.altitude_min + self.bin_band * idx_bin
                    )

                bin_distances_sample = np.empty(
                    (self.image_height, self.image_width)
                )
                bin_distances_sample.fill(distance_bin_sample)
            else:
                bin_distances = [self.depth_map_list[i] for i in tmp_idxs]
                bin_distances_sample = running_mean_std(
                    bin_distances, loader=self.loader
                )[0]

            if self.smoothing == "mean":
                bin_images_sample = running_mean_std(
                    bin_images, loader=self.loader
                )[0]
            elif self.smoothing == "mean_trimmed":
                memmap_filename, memmap_handle = create_memmap(
                    bin_images,
                    dimensions,
                    loader=self.loader,
                )
                bin_images_sample = image_mean_std_trimmed(memmap_handle)[0]
                del memmap_handle
                os.remove(memmap_filename)
            elif self.smoothing == "median":
                memmap_filename, memmap_handle = create_memmap(
                    bin_images,
                    dimensions,
                    loader=self.loader,
                )
                bin_images_sample = median_array(memmap_handle)
                del memmap_handle
                os.remove(memmap_filename)

            fig = plt.figure()
            plt.imshow(bin_images_sample)
            plt.colorbar()
            plt.title("Image bin " + str(idx_bin))
            plt.savefig(
                Path(self.attenuation_parameters_folder)
                / ("bin_images_sample_" + str(idx_bin) + ".png"),
                dpi=600,
            )
            plt.close(fig)

            fig = plt.figure()
            plt.imshow(bin_distances_sample)
            plt.colorbar()
            plt.title("Distance bin " + str(idx_bin))
            plt.savefig(
                Path(self.attenuation_parameters_folder)
                / ("bin_distances_sample_" + str(idx_bin) + ".png"),
                dpi=600,
            )
            plt.close(fig)

            images_map[idx_bin] = bin_images_sample.reshape(
                [self.image_height * self.image_width, self.image_channels]
            )
            distances_map[idx_bin] = bin_distances_sample.reshape(
                [self.image_height * self.image_width]
            )

    # execute the corrections of images using the gain values in case of
    # attenuation correction or static color balance
    def process_correction(self):
        """Execute series of corrections for a set of input images"""

        # check for calibration file if distortion correction needed
        if self.undistort:
            camera_params_folder = (
                Path(self.path_processed).parents[0] / "calibration"
            )
            camera_params_filename = "mono" + self.camera_name + ".yaml"
            camera_params_file_path = (
                camera_params_folder / camera_params_filename
            )

            if not camera_params_file_path.exists():
                Console.info("Calibration file not found...")
                self.undistort = False
            else:
                Console.info("Calibration file found...")
                self.camera_params_file_path = camera_params_file_path

        Console.info(
            "Processing",
            len(self.camera_image_list),
            "images for color, distortion, gamma corrections...",
        )

        with tqdm_joblib(
            tqdm(desc="Correcting images", total=len(self.camera_image_list))
        ):
            self.processed_image_list = joblib.Parallel(n_jobs=-2, verbose=0)(
                joblib.delayed(self.process_image)(idx)
                for idx in range(0, len(self.camera_image_list))
            )

        # write a filelist.csv containing image filenames which are processed
        image_files = []
        for path in self.processed_image_list:
            if path is not None:
                image_files.append(Path(path).name)
        dataframe = pd.DataFrame(image_files)
        filelist_path = self.output_images_folder / "filelist.csv"
        dataframe.to_csv(filelist_path)
        Console.info("Processing of images is completed...")

    def process_image(self, idx):
        """Execute series of corrections for an image

        Parameters
        -----------
        idx : int
            index to the list of image numpy files
        """

        # load image and convert to float
        image = self.loader(self.camera_image_list[idx])
        image_rgb = None
        # print("loader:", image.dtype, np.max(image), np.min(image))

        # apply corrections
        if self.correction_method == "colour_correction":
            distance_matrix = None
            if self.distance_metric == "depth_map":
                distance_matrix = depth_map.loader(
                    self.depth_map_list[idx],
                    self.image_width,
                    self.image_height,
                )
            elif self.distance_metric == "altitude":
                if idx > len(self.altitude_list) - 1:
                    Console.quit(
                        "The image index does not coincide with the",
                        "available indices in the altitude vector",
                    )
                    return None
                distance = self.altitude_list[idx]
                distance_matrix = np.empty(
                    (self.image_height, self.image_width)
                )
                distance_matrix.fill(distance)
            if distance_matrix is not None:
                image = corrections.attenuation_correct(
                    image,
                    distance_matrix,
                    self.image_attenuation_parameters,
                    self.correction_gains,
                )

                # print('attenuation_correct:', image.dtype, np.max(image), np.min(image))

            if distance_matrix is not None:
                image = corrections.pixel_stat(
                    image,
                    self.image_corrected_mean,
                    self.image_corrected_std,
                    self.brightness,
                    self.contrast,
                )
                # print('pixel_stat:', image.dtype, np.max(image), np.min(image))
            else:
                image = corrections.pixel_stat(
                    image,
                    self.image_raw_mean,
                    self.image_raw_std,
                    self.brightness,
                    self.contrast,
                )
            if self._type != "grayscale" and self._type != "rgb":
                # debayer images
                image_rgb = corrections.debayer(image, self._type)
            else:
                image_rgb = image

        elif self.correction_method == "manual_balance":
            if not self._type == "grayscale":
                # debayer images
                image_rgb = corrections.debayer(image, self._type)
            else:
                image_rgb = image
            image_rgb = corrections.manual_balance(
                image_rgb, self.color_gain_matrix_rgb, self.subtractors_rgb
            )

        # apply distortion corrections
        if self.undistort:
            image_rgb = corrections.distortion_correct(
                self.camera_params_file_path, image_rgb
            )

        # apply gamma corrections to rgb images for colour correction
        if self.correction_method == "colour_correction":
            image_rgb = corrections.gamma_correct(image_rgb)

        # apply scaling to 8 bit and format image to unit8
        image_rgb *= 255
        image_rgb = image_rgb.clip(0, 255).astype(np.uint8)
        # print('clip:', image_rgb.dtype, np.max(image_rgb), np.min(image_rgb))

        # write to output files
        image_filename = Path(self.camera_image_list[idx]).stem
        return write_output_image(
            image_rgb,
            image_filename,
            self.output_images_folder,
            self.output_format,
        )

# -*- coding: utf-8 -*-
"""
Copyright (c) 2023, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import copy
import os
import random
from pathlib import Path

try:
    # Try using the v2 API directly to avoid a warning from imageio >= 2.16.2
    from imageio.v2 import imwrite
except ImportError:
    from imageio import imwrite

import joblib
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

# fmt: off
from auv_nav.tools.time_conversions import read_timezone
from correct_images import corrections
from correct_images.loaders import depth_map, loader
from correct_images.tools.file_handlers import trim_csv_files, write_output_image
from correct_images.tools.joblib_tqdm import tqdm_joblib
from correct_images.tools.memmap import create_memmap, open_memmap
from correct_images.tools.numerical import (
    RunningMeanStd,
    image_mean_std_trimmed,
    median_array,
    ransac_mean_std,
    running_mean_std,
)
from oplab import (
    Console,
    Mission,
    get_config_folder,
    get_processed_folder,
    get_raw_folder,
)

# fmt: on
matplotlib.use("Agg")


def try_remove(filename, verbose=False):
    if filename is not None:
        try:
            os.remove(filename)
            if verbose:
                Console.info("Memmap", filename, "deleted!")
        except PermissionError:
            Console.warn(
                "Unable to remove distances memmap file",
                filename,
                ". Please delete the file manually.",
            )
        except FileNotFoundError:
            pass


# -----------------------------------------
def copy_file_if_exists(original_file: Path, dest_dir: Path):
    """Copy a file if it exists.

    Parameters
    ----------
    original_file : Path
        Source file to copy from
    dest_dir : Path
        Destination directory to copy the file to
    """
    if not original_file.exists():
        return
    fname = original_file.name
    dest_file = dest_dir / fname
    original_file.copy(dest_file)


class Corrector:
    def __init__(
        self,
        mode,
        force=False,
        suffix=None,
        camera=None,
        correct_config=None,
        path=None,
    ):
        """Constructor for the Corrector class

        Parameters
        ----------
        mode : str
            mode of the corrector. Can be "parse" or "process"
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
        self.camera_image_list = []
        self.processed_image_list = []
        self.altitude_list = []
        self.depth_map_list = []

        # Members for folder paths
        self.output_dir_path = None
        self.attenuation_parameters_folder = None
        self.output_images_folder = None
        if suffix is not None and suffix != "":
            self.suffix = suffix
        else:
            self.suffix = None

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

        self.memmaps_to_remove = []

        assert mode in ["parse", "process"]
        self.mode = mode
        self.force = force

        # if path is None then user must define it externally and call set_path
        if path is not None:
            # Set the path for the corrector
            self.set_path(path)
        else:
            return

        if self.correct_config is not None:
            # explicit call to load the correct_config
            self.load_configuration(self.correct_config)

    def cleanup(self):
        Console.info("Check if any memmaps have not been deleted yet, and delete if so")
        for fn in self.memmaps_to_remove:
            try_remove(fn, verbose=True)

    def set_path(self, path):
        """Set the path for the corrector"""
        # The path is expected to be non-empty, so we no longer check for it
        self.path_raw = get_raw_folder(path)
        self.path_processed = get_processed_folder(path)
        self.path_config = get_config_folder(path)

        self.mission = Mission(self.path_raw / "mission.yaml")

        self.user_specified_image_list = None  # To be overwritten on parse/process
        self.user_specified_image_list_parse = None
        self.user_specified_image_list_process = None

        self.camera_name = self.camera.name
        # Find the camera topic corresponding to the name
        for camera in self.mission.image.cameras:
            if camera.name == self.camera_name and camera.topic is not None:
                self.camera.topic = camera.topic
                break
        self._type = self.camera.type
        # Load camera configuration
        image_properties = self.camera.image_properties
        self.image_height = image_properties[0]
        self.image_width = image_properties[1]
        self.image_channels = image_properties[2]

    # NOTE: we could use an implicit version (correct_config already stored as member)
    # but in this implementation we force explicit (argument required) call to load_configuration()
    def load_configuration(self, correct_config=None):
        if correct_config is None:  # nothing to do here, we expect an explicit call
            Console.warn("No correct_config provided. Skipping load_configuration()")
            return

        self.correct_config = correct_config

        """Load general configuration parameters"""
        self.correction_method = self.correct_config.method
        # if self.correction_method == "colour_correction":
        self.distance_metric = self.correct_config.color_correction.distance_metric
        self.distance_path = self.correct_config.color_correction.metric_path
        self.parse_altitude_min = (
            self.correct_config.color_correction.parse_altitude_min
        )
        self.parse_altitude_max = (
            self.correct_config.color_correction.parse_altitude_max
        )
        self.process_altitude_min = (
            self.correct_config.color_correction.process_altitude_min
        )
        self.process_altitude_max = (
            self.correct_config.color_correction.process_altitude_max
        )
        self.smoothing = self.correct_config.color_correction.smoothing
        self.window_size = self.correct_config.color_correction.window_size
        self.cameraconfigs = self.correct_config.configs.camera_configs
        self.undistort = self.correct_config.output_settings.undistort_flag
        self.output_format = self.correct_config.output_settings.compression_parameter

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
            self.subtractors_rgb = np.array(self.cameraconfigs[cam_idx].subtractors_rgb)
            self.color_gain_matrix_rgb = np.array(
                self.cameraconfigs[cam_idx].color_gain_matrix_rgb
            )

        # Create output directories and needed attributes
        self.create_output_directories()

        # Define basic filepaths
        if self.correction_method == "colour_correction":
            p = Path(self.attenuation_parameters_folder)
            self.images_map_filepath = p / "images_map.npy"
            self.distances_map_filepath = p / "distances_map.npy"
            self.attenuation_plot_filepath = p / "attenuation_plot.png"
            self.attenuation_params_filepath = p / "attenuation_parameters.npy"
            self.correction_gains_filepath = p / "correction_gains.npy"
            self.corrected_mean_filepath = p / "image_corrected_mean.npy"
            self.corrected_std_filepath = p / "image_corrected_std.npy"
            self.raw_mean_filepath = p / "image_raw_mean.npy"
            self.raw_std_filepath = p / "image_raw_std.npy"

        # Define image loader
        # Use default loader
        self.loader = loader.Loader()
        self.loader.bit_depth = self.camera.bit_depth
        self._type = self.camera.type
        if self.camera.extension == "raw":
            self.loader.set_loader("xviii")
        elif self.camera.extension == "bag":
            self.loader.set_loader("rosbag")
            tz_offset_s = (
                read_timezone(self.mission.image.timezone) * 60
                + self.mission.image.timeoffset
            )
            self.loader.tz_offset_s = tz_offset_s
        else:
            self.loader.set_loader("default")

    def parse(self, path_list, correct_config_list):
        # both path_list and correct_config_list are assumed to be valid + equivalent
        for i in range(len(path_list)):  # for each dive
            path = path_list[i]
            correct_config = correct_config_list[i]

            Console.info("Parsing dive:", path)
            # Console.info("Setting path")
            self.set_path(path)  # update the dive path

            # Console.info("Loading configuration")
            self.load_configuration(correct_config)  # load the dive config
            # Set the user specified list - if any
            self.user_specified_image_list = self.user_specified_image_list_parse
            # Update list of images by appending user-defined list
            # TODO: this list must be populated from AFTER loading the configuration and BEFORE getting image list
            self.get_imagelist("parse")

        if (
            len(self.altitude_list) < 3
            and self.correction_method == "colour_correction"
        ):
            Console.quit(
                "Insufficient number of images to compute attenuation",
                "parameters",
            )

        # Show the total number of images after filtering + merging the dives. It should match the sum of the filtered images of each dive.
        if len(path_list) > 1:
            Console.info(
                "Total number of images after merging dives:",
                len(self.camera_image_list),
            )

        Console.info("Output directories created / existing")

        if self.correction_method == "colour_correction":
            self.get_altitude_and_depth_maps()
            self.generate_attenuation_correction_parameters()

            for i in range(len(path_list)):  # for each dive
                path = get_processed_folder(path_list[i])
                attn_dir = Path(self.attenuation_parameters_folder)
                relative_folder = attn_dir.relative_to(self.path_processed)
                dest_dir = path / relative_folder
                if dest_dir == attn_dir:
                    # Do not copy over the original files
                    continue
                copy_file_if_exists(self.attenuation_params_filepath, dest_dir)
                copy_file_if_exists(self.correction_gains_filepath, dest_dir)
                copy_file_if_exists(self.corrected_mean_filepath, dest_dir)
                copy_file_if_exists(self.corrected_std_filepath, dest_dir)
                copy_file_if_exists(self.raw_mean_filepath, dest_dir)
                copy_file_if_exists(self.raw_std_filepath, dest_dir)
        elif self.correction_method == "manual_balance":
            Console.info("run process for manual_balance")

    def process(self):
        # Set the user specified list if any
        self.user_specified_image_list = self.user_specified_image_list_process

        # Read list of images
        self.get_imagelist("process")

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
                if self.distance_metric != "uniform":
                    Console.quit(
                        "Code does not find attenuation_parameters.npy.",
                        "Please run parse before process.",
                    )
            if self.correction_gains_filepath.exists():
                self.correction_gains = np.load(self.correction_gains_filepath)
            else:
                if self.distance_metric != "uniform":
                    Console.quit(
                        "Code does not find correction_gains.npy.",
                        "Please run parse before process.",
                    )
            if self.corrected_mean_filepath.exists():
                self.image_corrected_mean = np.load(
                    self.corrected_mean_filepath
                ).squeeze()
            else:
                if self.distance_metric != "uniform":
                    Console.quit(
                        "Code does not find image_corrected_mean.npy.",
                        "Please run parse before process.",
                    )
            if self.corrected_std_filepath.exists():
                self.image_corrected_std = np.load(
                    self.corrected_std_filepath
                ).squeeze()
            else:
                if self.distance_metric != "uniform":
                    Console.quit(
                        "Code does not find image_corrected_std.npy...",
                        "Please run parse before process...",
                    )
            if self.raw_mean_filepath.exists() and self.distance_metric == "uniform":
                self.image_raw_mean = np.load(self.raw_mean_filepath).squeeze()
            elif self.distance_metric == "uniform":
                Console.quit(
                    "Code does not find image_raw_mean.npy...",
                    "Please run parse before process...",
                )
            if self.raw_std_filepath.exists() and self.distance_metric == "uniform":
                self.image_raw_std = np.load(self.raw_std_filepath).squeeze()
            elif self.distance_metric == "uniform":
                Console.quit(
                    "Code does not find image_raw_std.npy...",
                    "Please run parse before process...",
                )
            Console.info("Correction parameters loaded")
            Console.info("Running process for colour correction...")
        else:
            Console.info("Running process with manual colour balancing...")
        self.process_correction()

    # create directories for storing intermediate image and distance_matrix
    # numpy files, correction parameters and corrected output images
    def create_output_directories(self):
        """Handle the creation of output directories for each camera"""
        self.output_dir_path = self.path_processed / "image"
        self.output_dir_path /= self.camera_name

        # Create output directories depending on the correction method
        parameters_folder_str = "params_"
        developed_folder_str = "corrected_"

        if self.correction_method == "colour_correction":
            parameters_folder_str += self.distance_metric
            developed_folder_str += self.distance_metric
            developed_folder_str += (
                "_mean_" + str(int(self.brightness)) + "_std_" + str(int(self.contrast))
            )
        elif self.correction_method == "manual_balance":
            parameters_folder_str += "manual"
            developed_folder_str += "manual"
            developed_folder_str += (
                "_gain_"
                + str(self.color_gain_matrix_rgb[0, 0])
                + "_"
                + str(self.color_gain_matrix_rgb[1, 1])
                + "_"
                + str(self.color_gain_matrix_rgb[2, 2])
                + "_sub_"
                + str(self.subtractors_rgb[0])
                + "_"
                + str(self.subtractors_rgb[1])
                + "_"
                + str(self.subtractors_rgb[2])
            )
        # Accept suffixes for the output directories
        if self.suffix:
            parameters_folder_str += "_" + self.suffix
            developed_folder_str += "_" + self.suffix

        self.attenuation_parameters_folder = (
            self.output_dir_path / parameters_folder_str
        )
        self.output_images_folder = self.output_dir_path / developed_folder_str

        if self.mode == "parse":
            if not self.attenuation_parameters_folder.exists():
                self.attenuation_parameters_folder.mkdir(parents=True)
            else:
                file_list = list(self.attenuation_parameters_folder.glob("*.npy"))
                if len(file_list) > 0:
                    if not self.force:
                        Console.quit(
                            "Parameters exist for current configuration.",
                            "Run parse with Force (-F flag)...",
                        )
                    else:
                        Console.warn(
                            "Code will overwrite existing parameters for",
                            "current configuration...",
                        )

        if self.mode == "process":
            if not self.output_images_folder.exists():
                self.output_images_folder.mkdir(parents=True)
            else:
                file_list = list(self.output_images_folder.glob("*.*"))
                if len(file_list) > 0:
                    if not self.force:
                        Console.quit(
                            "Corrected images exist for current configuration.",
                            "Run process with Force (-F flag)...",
                        )
                    else:
                        Console.warn(
                            "Code will overwrite existing corrected images for",
                            "current configuration...",
                        )

    def get_camera_idx(self):
        idx = [
            i
            for i, camera_config in enumerate(self.cameraconfigs)
            if camera_config.camera_name == self.camera_name
        ]
        if len(idx) > 0:
            return idx[0]
        else:
            Console.warn(
                "The camera",
                self.camera_name,
                "could not be found in the correct_images.yaml",
            )
            return None

    # load imagelist: output is same as camera.imagelist unless a smaller
    # filelist is specified by the user
    def get_imagelist(self, mode):
        """Generate list of source images"""
        # Store a copy of the currently stored image list in the Corrector object
        _original_image_list = self.camera_image_list
        _original_altitude_list = self.altitude_list
        # Replaces Corrector object's image_list with the camera image list
        # OLD: self.camera_image_list = self.camera.image_list # <---- Now this is done at the end (else condition)
        self.camera_image_list = []
        self.altitude_list = []

        # If using colour_correction, we need to read in the navigation
        if (
            self.correction_method == "colour_correction"
            or self.camera.extension == "bag"
        ):
            assert mode in ["parse", "process"]
            if mode == "parse":
                altitude_min = self.parse_altitude_min
                altitude_max = self.parse_altitude_max
            elif mode == "process":
                altitude_min = self.process_altitude_min
                altitude_max = self.process_altitude_max

            # Deal with bagfiles:
            # if the extension is bag, the list is a list of bagfiles
            if self.camera.extension == "bag":
                self.camera.bagfile_list = copy.deepcopy(self.camera.image_list)
                Console.info("Setting camera topic to", self.camera.topic)
                self.loader.set_bagfile_list_and_topic(
                    self.camera.bagfile_list, self.camera.topic
                )

            if self.distance_path == "json_renav_*":
                Console.info(
                    "Picking first JSON folder as the default path to auv_nav",
                    "csv files...",
                )
                dir_ = self.path_processed
                json_list = list(dir_.glob("json_*"))
                if len(json_list) == 0:
                    Console.quit(
                        "No navigation solution could be found. Please run ",
                        "auv_nav parse and process first",
                    )
                self.distance_path = json_list[0]
                Console.info("JSON:", self.distance_path)
            metric_path = self.path_processed / self.distance_path
            # Try if ekf exists:
            full_metric_path = metric_path / "csv" / "ekf"
            metric_file = "auv_ekf_" + self.camera_name + ".csv"

            if not full_metric_path.exists():
                full_metric_path = metric_path / "csv" / "dead_reckoning"
                metric_file = "auv_dr_" + self.camera_name + ".csv"
            self.altitude_csv_path = full_metric_path / metric_file

            # Check if file exists
            if not self.altitude_csv_path.exists():
                Console.quit(
                    "Cannot find altitude csv file expected to be save at ",
                    self.altitude_csv_path,
                    ". Run auv_nav first.",
                )

            # read dataframe for corresponding distance csv path
            dataframe = pd.read_csv(self.altitude_csv_path)

            # get imagelist for given camera object
            if self.user_specified_image_list != "none":
                path_file_list = Path(self.path_config) / self.user_specified_image_list
                trimmed_csv_file = "trimmed_csv_" + self.camera_name + ".csv"
                self.trimmed_csv_path = Path(self.path_config) / trimmed_csv_file

                if not self.altitude_csv_path.exists():
                    message = "Path to " + metric_file + " does not exist..."
                    Console.quit(message)
                else:
                    # create trimmed csv based on user's  list of images
                    dataframe = trim_csv_files(
                        path_file_list,
                        self.altitude_csv_path,
                        self.trimmed_csv_path,
                    )

            # Check images exist:
            valid_idx = []
            if "relative_path" not in dataframe:
                Console.error("CSV FILE:", self.altitude_csv_path)
                Console.quit(
                    "Your CSV navigation file does not have a relative_path column"
                )
            for idx, entry in enumerate(dataframe["relative_path"]):
                if self.camera.extension == "bag":
                    valid_idx.append(idx)
                else:
                    im_path = self.path_raw / entry
                    if im_path.exists():
                        valid_idx.append(idx)
            filtered_dataframe = dataframe.iloc[valid_idx]
            filtered_dataframe.reset_index(drop=True)
            # WARNING: if the column does not contain any 'None' entry, it will be
            # parsed as float, and the .str() accesor will fail
            filtered_dataframe["altitude [m]"] = filtered_dataframe[
                "altitude [m]"
            ].astype(str)
            filtered_dataframe = filtered_dataframe[
                ~filtered_dataframe["altitude [m]"].str.contains("None")
            ]  # drop rows with None altitude
            distance_list = filtered_dataframe["altitude [m]"].tolist()
            for _, row in filtered_dataframe.iterrows():
                alt = float(row["altitude [m]"])
                if alt > altitude_min and alt < altitude_max:
                    if self.camera.extension == "bag":
                        self.camera_image_list.append(row["timestamp [s]"])
                    else:
                        self.camera_image_list.append(
                            self.path_raw / row["relative_path"]
                        )
                    self.altitude_list.append(alt)

            if len(distance_list) == 0:
                Console.error("No images exist / can be found!")
                Console.error(
                    "Check the file",
                    self.altitude_csv_path,
                    "and make sure that the 'relative_path' column points to",
                    "existing images relative to the raw mission folder (e.g.",
                    self.path_raw,
                    ")",
                )
                Console.error("You may need to reprocess the dive with auv_nav")
                Console.quit("No images were found.")

            # WARNING: what happens in a multidive setup when the current dive has no images (but the rest of the dive does)?
            Console.info(
                len(self.altitude_list),
                "of",
                len(distance_list),
                "images remaining after applying altitude filter",
            )
        else:
            # Copy the images list from the camera
            self.camera_image_list = self.camera.image_list

            # Join the current image list with the original image list (copy)
            self.camera_image_list.extend(_original_image_list)
            # Show size of the extended image list
            # Informative for message for multidive
            Console.info(
                "The camera image list has", len(self.camera_image_list), "entries"
            )
            # Join the current image list with the original image list (copy)
            self.altitude_list.extend(_original_altitude_list)

    def get_altitude_and_depth_maps(self):
        """Generate distance matrix numpy files and save them"""
        # read altitude / depth map depending on distance_metric
        if self.distance_metric == "uniform":
            Console.info("Null distance matrix created")
            self.depth_map_list = []
            return
        elif self.distance_metric == "altitude":
            Console.info("Null distance matrix created")
            self.depth_map_list = []
            return
        elif self.distance_metric == "depth_map":
            # Load depth maps
            path_depth = self.path_processed / "3d_reconstruction" / "depth_maps"
            if not path_depth.exists():
                Console.quit(
                    "Depth maps folder", path_depth, "does not exist."
                )
            images_to_drop = []
            for img_idx, image_path in enumerate(self.camera_image_list):
                p = path_depth / os.path.relpath(image_path, self.path_raw)
                p = p.with_name(p.stem + "_depthmap.npy")
                if p.exists():
                    self.depth_map_list.append(p)
                else:
                    images_to_drop.append(img_idx)

            # Drop images that do not have a corresponding depth map
            if len(images_to_drop) > 0:
                if len(images_to_drop) == len(self.camera_image_list):
                    Console.quit(
                        "No depth maps found in", path_depth,
                        "\nLast path checked:", p
                    )
                Console.info(
                    "Dropping", len(images_to_drop), "of", len(self.camera_image_list),
                    "images for which no depth maps exists."
                )
                for idx in sorted(images_to_drop, reverse=True):
                    del self.camera_image_list[idx]
            assert(len(self.camera_image_list) == len(self.depth_map_list))

            Console.info("Depth maps loaded")
            return

    # compute correction parameters either for attenuation correction or
    # static correction of images
    def generate_attenuation_correction_parameters(self):
        """Generates image stats and attenuation coefficients and saves the
        parameters for process"""

        Console.info("Generating image stats and attenuation coefficients...")
        if len(self.altitude_list) < 3:
            Console.quit(
                "Insufficient number of images to compute attenuation ",
                "parameters...",
            )

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
            / (1024.0**3)
        )
        max_bin_size_gb = 50.0
        max_bin_size = int(max_bin_size_gb / image_size_gb)

        self.bin_band = 0.1
        hist_bins = np.arange(
            self.parse_altitude_min,
            self.parse_altitude_max + 0.5 * self.bin_band,
            self.bin_band,
        )

        distance_vector = None

        if self.depth_map_list and self.distance_metric == "depth_map":
            Console.info("Computing depth map histogram with", hist_bins.size, "bins")

            distance_vector = np.zeros((len(self.depth_map_list), 1))
            for i, dm_file in enumerate(self.depth_map_list):
                dm_np = depth_map.loader(dm_file, self.image_width, self.image_height)
                distance_vector[i] = dm_np.mean()

        elif self.altitude_list and self.distance_metric == "altitude":
            Console.info("Computing altitude histogram with", hist_bins.size, "bins")
            distance_vector = np.array(self.altitude_list)

        if distance_vector is not None:
            idxs = np.digitize(distance_vector, hist_bins) - 1

            # Display histogram in console
            for idx_bin in range(hist_bins.size - 1):
                tmp_idxs = np.where(idxs == idx_bin)[0]
                Console.info(
                    "  Bin",
                    format(idx_bin, "02d"),
                    "(",
                    round(hist_bins[idx_bin], 1),
                    "m < x <",
                    round(hist_bins[idx_bin + 1], 1),
                    "m):",
                    len(tmp_idxs),
                    "images",
                )

            # Watch out: need to substract 1 to get the correct number of bins
            # because the last bin is not included in the range
            images_fn, images_map = open_memmap(
                shape=(
                    len(hist_bins) - 1,
                    self.image_height * self.image_width,
                    self.image_channels,
                ),
                dtype=np.float32,
            )
            self.memmaps_to_remove.append(images_fn)

            distances_fn, distances_map = open_memmap(
                shape=(len(hist_bins) - 1, self.image_height * self.image_width),
                dtype=np.float32,
            )
            self.memmaps_to_remove.append(distances_fn)

            with tqdm_joblib(
                tqdm(
                    desc="Computing altitude histogram",
                    total=hist_bins.size - 1,
                )
            ):
                joblib.Parallel(n_jobs=-2, verbose=0)(
                    # Do not prefer="threads" as in certain cases this can cause the
                    # program to crash when calling plt.imshow() in compute_distance_bin
                    joblib.delayed(self.compute_distance_bin)(
                        idxs,
                        idx_bin,
                        images_map,
                        distances_map,
                        max_bin_size,
                        max_bin_size_gb,
                        distance_vector,
                    )
                    for idx_bin in range(hist_bins.size - 1)
                )

            # Save images map and distances map
            np.save(self.images_map_filepath, images_map)
            np.save(self.distances_map_filepath, distances_map)

            # calculate attenuation parameters per channel
            self.image_attenuation_parameters = (
                corrections.calculate_attenuation_parameters(
                    images_map,
                    distances_map,
                    self.image_height,
                    self.image_width,
                    self.image_channels,
                    self.attenuation_parameters_folder,
                )
            )

            self.plot_all_attenuation_curves(images_map, distances_map)

            # delete memmap handles
            del images_map
            del distances_map
            try_remove(images_fn)
            try_remove(distances_fn)

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
            Console.info("Saving correction gains")
            # Save correction gains
            np.save(self.correction_gains_filepath, self.correction_gains)

            corrections.save_attenuation_plots(
                self.attenuation_parameters_folder,
                gains=self.correction_gains,
            )

            # Useful if fails, to reload precomputed numpyfiles.
            # TODO: offer as a new step.
            # self.image_attenuation_parameters = np.load(
            #   self.attenuation_params_filepath)
            # self.correction_gains = np.load(self.correction_gains_filepath)

            # apply gains to images
            Console.info("Applying attenuation corrections to images...")
            image_properties = [
                self.image_height,
                self.image_width,
                self.image_channels,
            ]
            runner = RunningMeanStd(image_properties)

            """
            memmap_filename, memmap_handle = open_memmap(
                shape=(
                    len(self.camera_image_list),
                    self.image_height,
                    self.image_width,
                    self.image_channels,
                ),
                dtype=np.float32,
            )
            """

            # DEBUG: can be removed
            # Console.error("depth_map_list size", len(self.depth_map_list))
            # Console.error("camera_image_list size", len(self.camera_image_list))
            ###################################################################################################
            for i in trange(len(self.camera_image_list)):
                # Load the image
                img = self.loader(self.camera_image_list[i])

                # Load the distance matrix
                if not self.depth_map_list:
                    # TODO: Show the depth_map creation
                    # if self.depth_map_list is None:
                    # Generate matrices on the fly
                    distance = distance_vector[i]
                    distance_mtx = np.empty((self.image_height, self.image_width))
                    distance_mtx.fill(distance)
                else:
                    distance_mtx = depth_map.loader(
                        self.depth_map_list[i],
                        self.image_width,
                        self.image_height,
                    )
                # TODO: Show the size of the produced distance_mtx
                # Correct the image
                corrected_img = corrections.attenuation_correct(
                    img,
                    distance_mtx,
                    self.image_attenuation_parameters,
                    self.correction_gains,
                )
                # TODO: Inspect the corrected image after attenuation correction
                # Before calling compute, let's show the corrected_img dimensions
                # Console.error("corrected_img.shape", corrected_img.shape)
                runner.compute(corrected_img)
                """
                memmap_handle[i] = corrected_img.reshape(
                    self.image_height, self.image_width, self.image_channels
                )
                """
            # Compute mean and std using RANSAC
            """
            image_corrected_mean_ransac, image_corrected_std_ransac = ransac_mean_std(
                memmap_handle, max_iterations=1000, threshold=0.1, sample_size=2
            )
            # This is the last time we need the memmap_handle.
            # We probably should call try_remove(memmap_filename) here.

            image_corrected_mean_ransac = image_corrected_mean_ransac.reshape(
                self.image_height, self.image_width, self.image_channels
            )
            image_corrected_std_ransac = image_corrected_std_ransac.reshape(
                self.image_height, self.image_width, self.image_channels
            )
            """

            image_corrected_mean = runner.mean.reshape(
                self.image_height, self.image_width, self.image_channels
            )
            image_corrected_std = runner.std.reshape(
                self.image_height, self.image_width, self.image_channels
            )

            # save parameters for process
            np.save(
                self.corrected_mean_filepath, image_corrected_mean
            )  # TODO: make member
            np.save(
                self.corrected_std_filepath, image_corrected_std
            )  # TODO: make member

            """
            np.save(
                self.corrected_mean_filepath.with_name(
                    "image_corrected_mean_ransac.npy"
                ),
                image_corrected_mean_ransac,
            )  # TODO: make member
            np.save(
                self.corrected_std_filepath.with_name("image_corrected_std_ransac.npy"),
                image_corrected_std_ransac,
            )  # TODO: make member
            """

            corrections.save_attenuation_plots(
                self.attenuation_parameters_folder,
                img_mean=image_corrected_mean,
                img_std=image_corrected_std,
            )

            """
            corrections.save_attenuation_plots(
                self.attenuation_parameters_folder / "ransac",
                img_mean=image_corrected_mean_ransac,
                img_std=image_corrected_std_ransac,
            )
            """

        else:
            Console.info(
                "No altitude or depth maps available.",
                "Computing raw mean and std.",
            )

            image_raw_mean, image_raw_std = running_mean_std(
                self.camera_image_list, self.loader
            )
            np.save(self.raw_mean_filepath, image_raw_mean)
            np.save(self.raw_std_filepath, image_raw_std)

            # image_raw_mean = np.load(self.raw_mean_filepath)
            # image_raw_std = np.load(self.raw_std_filepath)

            ch = image_raw_mean.shape[0]
            if ch == 3:
                image_raw_mean = image_raw_mean.transpose((1, 2, 0))
                image_raw_std = image_raw_std.transpose((1, 2, 0))

            imwrite(
                Path(self.attenuation_parameters_folder) / "image_raw_mean.png",
                image_raw_mean,
            )
            imwrite(
                Path(self.attenuation_parameters_folder) / "image_raw_std.png",
                image_raw_std,
            )

            corrections.save_attenuation_plots(
                self.attenuation_parameters_folder,
                img_mean=image_raw_mean,
                img_std=image_raw_std,
            )

        Console.info("Correction parameters saved")

    def plot_all_attenuation_curves(self, images_map, distances_map):
        fig = plt.figure()

        images_map[images_map == 0] = np.NaN
        distances_map[distances_map == 0] = np.NaN

        pixel_range = range(
            0,
            self.image_height * self.image_width,
            (self.image_height * self.image_width) // 100,
        )
        for i_channel in range(self.image_channels):
            with tqdm(desc="Attenuation plot", total=len(pixel_range)) as pbar:
                for i_pixel in pixel_range:
                    i_pixel_height = i_pixel // self.image_width
                    i_pixel_width = i_pixel % self.image_width
                    p0, p1, p2 = self.image_attenuation_parameters[
                        i_channel, i_pixel_height, i_pixel_width
                    ]
                    xs = np.arange(2, 10, 0.1)
                    plt.plot(
                        xs,
                        p0 * np.exp(p1 * xs) + p2,
                        color="black",
                        alpha=0.1,
                    )
                    plt.plot(
                        distances_map[:, i_pixel],
                        images_map[:, i_pixel, i_channel],
                        ",",
                        color="blue",
                        alpha=0.1,
                    )
                    pbar.update(1)
        plt.savefig(self.attenuation_plot_filepath, dpi=600)
        plt.close(fig)

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

        if len(tmp_idxs) >= 10:
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

            if not self.depth_map_list:
                # Generate matrices on the fly
                distance_bin = distance_vector[tmp_idxs]
                distance_bin_sample = distance_bin.mean()
                if distance_bin_sample <= 0 or np.isnan(distance_bin_sample):
                    Console.warn("The mean distance is equal or lower than zero!")
                    Console.warn("Printing the entire vector:", distance_bin)
                    Console.warn("Printing the mean:", distance_bin_sample)
                    distance_bin_sample = self.altitude_min + self.bin_band * idx_bin

                bin_distances_sample = np.empty((self.image_height, self.image_width))
                bin_distances_sample.fill(distance_bin_sample)
            else:
                bin_distances = [self.depth_map_list[i] for i in tmp_idxs]
                bin_distances_sample = running_mean_std(
                    bin_distances,
                    loader=depth_map.loader,
                    width=self.image_width,
                    height=self.image_height,
                    ignore_zeroes=True,
                )[0]
                distance_bin_sample = bin_distances_sample.mean()

            if self.smoothing == "mean":
                bin_images_sample = running_mean_std(bin_images, loader=self.loader)[0]
            elif self.smoothing == "mean_trimmed":
                memmap_filename, memmap_handle = create_memmap(
                    bin_images,
                    dimensions,
                    loader=self.loader,
                )
                self.memmaps_to_remove.append(memmap_filename)
                bin_images_sample = image_mean_std_trimmed(memmap_handle)[0]
                del memmap_handle
                try_remove(memmap_filename)
            elif self.smoothing == "median":
                memmap_filename, memmap_handle = create_memmap(
                    bin_images,
                    dimensions,
                    loader=self.loader,
                )
                self.memmaps_to_remove.append(memmap_filename)
                bin_images_sample = median_array(memmap_handle)
                del memmap_handle
                try_remove(memmap_filename)
            elif self.smoothing == "ransac_mean":
                memmap_filename, memmap_handle = create_memmap(
                    bin_images,
                    dimensions,
                    loader=self.loader,
                )
                self.memmaps_to_remove.append(memmap_filename)
                bin_images_sample = ransac_mean_std(
                    memmap_handle, max_iterations=1000, threshold=0.1, sample_size=2
                )[0]
                del memmap_handle
                try_remove(memmap_filename)
            else:
                raise ValueError("Smoothing method not recognized")

            base_path = Path(self.attenuation_parameters_folder)

            fig = plt.figure()
            plt.imshow(bin_images_sample)
            plt.colorbar()
            plt.title(f"Image bin {idx_bin}, altitude: {distance_bin_sample:.2f}")
            fig_name = base_path / (
                f"bin_image_{idx_bin:02}_{distance_bin_sample:05.2f}m.png"
            )
            # Console.info("Saved figure at", fig_name)
            plt.savefig(fig_name, dpi=600)
            plt.close(fig)

            fig = plt.figure()
            plt.imshow(bin_distances_sample)
            plt.colorbar()
            plt.title(f"Distance bin {idx_bin}, altitude: {distance_bin_sample:.2f}")
            fig_name = base_path / (
                f"bin_distance_sample_{idx_bin:02}_{distance_bin_sample:05.2f}m.png"
            )
            # Console.info("Saved figure at", fig_name)
            plt.savefig(fig_name, dpi=600)
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
            camera_params_folder = Path(self.path_processed).parents[0] / "calibration"
            camera_params_filename = "mono" + self.camera_name + ".yaml"
            camera_params_file_path = camera_params_folder / camera_params_filename

            if not camera_params_file_path.exists():
                Console.info("Calibration file not found")
                self.undistort = False
            else:
                Console.info("Calibration file found")
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

        # write a filelist.csv containing image filenames and navigation
        image_names = []
        for path in self.processed_image_list:
            if path is not None:
                image_names.append(Path(path).stem)
        d = {"image_names": image_names, "relative_path": self.processed_image_list}
        dataframe = pd.DataFrame(d)

        orig_dataframe = pd.read_csv(self.altitude_csv_path)
        orig_dataframe["raw_relative_path"] = orig_dataframe["relative_path"]

        # Merge the two dataframes when image filenames are the same
        output_df = orig_dataframe.copy()
        for index, row in output_df.iterrows():
            current_image_name = Path(row["relative_path"]).stem
            # find current_image_name in dataframe
            res = dataframe.loc[dataframe["image_names"] == current_image_name]
            if not res.empty:
                # update relative_path
                output_df.at[index, "relative_path"] = Path(
                    res["relative_path"].values[0]
                ).name
                # Check if path is absolute:
                if Path(row["relative_path"]).is_absolute():
                    # update raw_relative_path to relative
                    output_df.at[index, "raw_relative_path"] = Path(
                        row["relative_path"]
                    ).relative_to(self.path_raw)
            else:
                # remove row
                output_df.drop(index, inplace=True)
        filelist_path = self.output_images_folder / "filelist.csv"
        output_df.to_csv(filelist_path)
        Console.info("Processing of images is completed")

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
                distance_matrix = np.empty((self.image_height, self.image_width))
                distance_matrix.fill(distance)
            if distance_matrix is not None:
                image = corrections.attenuation_correct(
                    image,
                    distance_matrix,
                    self.image_attenuation_parameters,
                    self.correction_gains,
                )

            if distance_matrix is not None:
                image = corrections.pixel_stat(
                    image,
                    self.image_corrected_mean,
                    self.image_corrected_std,
                    self.brightness,
                    self.contrast,
                )
            else:
                image = corrections.pixel_stat(
                    image,
                    self.image_raw_mean,
                    self.image_raw_std,
                    self.brightness,
                    self.contrast,
                )
            if (
                self._type != "grayscale"
                and self._type != "rgb"
                and self._type != "bgr"
            ):
                # debayer images
                image_rgb = corrections.debayer(image, self._type)
            elif self._type != "bgr" and self._type != "grayscale":
                image_rgb = image.copy()
                image_rgb[:, :, [0, 1, 2]] = image_rgb[:, :, [2, 1, 0]]
            else:
                image_rgb = image

        elif self.correction_method == "manual_balance":
            if (
                self._type != "grayscale"
                and self._type != "rgb"
                and self._type != "bgr"
            ):
                # debayer images
                image_rgb = corrections.debayer(image, self._type)
            elif self._type != "bgr":
                image_rgb = image.copy()
                image_rgb[:, :, [0, 1, 2]] = image_rgb[:, :, [2, 1, 0]]
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
        try:
            if self.camera.extension == "bag":
                image_filename = (
                    self.camera_name + "_" + str(self.camera_image_list[idx])
                )
            else:
                image_filename = Path(self.camera_image_list[idx]).stem
        except FileNotFoundError:
            image_filename = self.camera_name + "_" + str(self.camera_image_list[idx])
        return write_output_image(
            image_rgb,
            image_filename,
            self.output_images_folder,
            self.output_format,
        )

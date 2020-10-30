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
import imageio
from oplab import get_raw_folder
from oplab import get_processed_folder
from oplab import get_config_folder
from oplab import Console
from correct_images.parser import CorrectConfig
from correct_images import corrections
from correct_images.tools.numerical import image_mean_std_trimmed, mean_std
from correct_images.tools.file_handlers import write_output_image
from correct_images.tools.file_handlers import load_memmap_from_numpyfilelist
from correct_images.tools.file_handlers import trim_csv_files
from correct_images.tools.joblib_tqdm import tqdm_joblib
from correct_images.camera_specific import xviii
import joblib
import shutil
import datetime
from datetime import datetime
from skimage.transform import resize


# -----------------------------------------


class Corrector:
    def __init__(self, force=False, camera=None, correct_config=None, path=None):
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

        self.camera = camera
        self.correct_config = correct_config

        if path is not None:
            self.path_raw = get_raw_folder(path)
            self.path_processed = get_processed_folder(path)
            self.path_config = get_config_folder(path)
        self.force = force

        self.user_specified_image_list = None  # To be overwritten on parse/process

        if self.correct_config is not None:
            """Load general configuration parameters"""
            self.correction_method = self.correct_config.method
            if self.correction_method == "colour_correction":
                self.distance_metric = self.correct_config.color_correction.distance_metric
                self.distance_path = self.correct_config.color_correction.metric_path
                self.altitude_max = self.correct_config.color_correction.altitude_max
                self.altitude_min = self.correct_config.color_correction.altitude_min
                self.smoothing = self.correct_config.color_correction.smoothing
                self.window_size = self.correct_config.color_correction.window_size
                self.outlier_rejection = (
                    self.correct_config.color_correction.outlier_reject
                )
            self.cameraconfigs = self.correct_config.configs.camera_configs
            self.undistort = self.correct_config.output_settings.undistort_flag
            self.output_format = self.correct_config.output_settings.compression_parameter

            # Load camera parameters
            cam_idx = self.get_camera_idx()
            self.camera_found = False
            if cam_idx is None:
                Console.info("Camera not included in correct_images.yaml. No processing will be done for this camera.")
                return
            else:
                self.camera_found = True
            self.user_specified_image_list_parse = self.cameraconfigs[cam_idx].imagefilelist_parse
            self.user_specified_image_list_process = self.cameraconfigs[cam_idx].imagefilelist_process
            self.camera_image_list = None
            if self.correction_method == "colour_correction":
                # Brighness and contrast are percentages of 255
                # e.g. brightness of 30 means 30% of 255 = 77
                self.brightness = float(self.cameraconfigs[cam_idx].brightness)*2.55
                self.contrast = float(self.cameraconfigs[cam_idx].contrast)*2.55
            elif self.correction_method == "manual_balance":
                self.subtractors_rgb = np.array(self.cameraconfigs[cam_idx].subtractors_rgb)
                self.color_gain_matrix_rgb = np.array(self.cameraconfigs[
                                                        cam_idx
                                                    ].color_gain_matrix_rgb)
            image_properties = self.camera.image_properties
            self.image_height = image_properties[0]
            self.image_width = image_properties[1]
            self.image_channels = image_properties[2]
            self.camera_name = self.camera.name
            self._type = self.camera.type

        # Members for folder paths
        self.output_dir_path = None
        self.bayer_numpy_dir_path = None
        self.distance_matrix_numpy_folder = None
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

        # Distance matrix related
        self.distance_matrix_numpy_filelist = None
        self.altitude_based_filtered_indices = None

        self.bayer_numpy_filelist = None

        self.camera_params_file_path = None

    def parse(self):
        if not self.camera_found:
            return
        self.user_specified_image_list = self.user_specified_image_list_parse
        self.setup('parse')
        if self.correction_method == "colour_correction":
            self.generate_attenuation_correction_parameters()
            return True
        elif self.correction_method == "manual_balance":
            Console.info('run process for manual_balance...')
            return False
        self.cleanup()

    def process(self):
        if not self.camera_found:
            return
        self.user_specified_image_list = self.user_specified_image_list_process
        self.setup('process')
        if self.correction_method == "colour_correction":
            filepath_attenuation_params = Path(
                self.attenuation_parameters_folder) / "attenuation_parameters.npy"
            filepath_correction_gains = Path(self.attenuation_parameters_folder) / "correction_gains.npy"
            filepath_corrected_mean = Path(self.attenuation_parameters_folder) / "image_corrected_mean.npy"
            filepath_corrected_std = Path(self.attenuation_parameters_folder) / "image_corrected_std.npy"
            filepath_raw_mean = Path(self.attenuation_parameters_folder) / "image_raw_mean.npy"
            filepath_raw_std = Path(self.attenuation_parameters_folder) / "image_raw_std.npy"

            # read parameters from disk
            if filepath_attenuation_params.exists():
                self.image_attenuation_parameters = np.load(
                    filepath_attenuation_params
                )
            else:
                if self.distance_metric == "altitude" or self.distance_metric == "depth_map":
                    Console.quit(
                        "Code does not find attenuation_parameters.npy...Please run parse before process...")
            if filepath_correction_gains.exists():
                self.correction_gains = np.load(filepath_correction_gains)
            else:
                if self.distance_metric == "altitude" or self.distance_metric == "depth_map":
                    Console.quit("Code does not find correction_gains.npy...Please run parse before process...")
            if filepath_corrected_mean.exists():
                self.image_corrected_mean = np.load(filepath_corrected_mean)
            else:
                if self.distance_metric == "altitude" or self.distance_metric == "depth_map":
                    Console.quit("Code does not find image_corrected_mean.npy...Please run parse before process...")
            if filepath_corrected_std.exists():
                self.image_corrected_std = np.load(filepath_corrected_std)
            else:
                if self.distance_metric == "altitude" or self.distance_metric == "depth_map":
                    Console.quit("Code does not find image_corrected_std.npy...Please run parse before process...")
            if filepath_raw_mean.exists():
                self.image_raw_mean = np.load(filepath_raw_mean)
            else:
                if self.distance_metric == "altitude" or self.distance_metric == "depth_map" or self.distance_metric == "none":
                    Console.quit("Code does not find image_raw_mean.npy...Please run parse before process...")
            if filepath_raw_std.exists():
                self.image_raw_std = np.load(filepath_raw_std)
            else:
                if self.distance_metric == "altitude" or self.distance_metric == "depth_map" or self.distance_metric == "none":
                    Console.quit("Code does not find image_raw_std.npy...Please run parse before process...")
            Console.info('Correction parameters loaded...')
            Console.info('Running process for colour correction...')
        else:
            Console.info('Running process with manual colour balancing...')
        self.process_correction()
        self.cleanup()

    def cleanup(self):
        # TODO(MMC) was done before for each camera (a single corrector instance)
        # remove memmaps
        Console.info("Removing memmaps...")
        memmap_files_path = self.memmap_folder.glob("*.map")
        for file in memmap_files_path:
            if file.exists():
                file.unlink()
        # print("-----------------------------------------------------")

        # remove bayer image numpy files
        try:
            shutil.rmtree(self.bayer_numpy_dir_path, ignore_errors=True)
        except OSError as err:
            Console.warn("could not delete the folder for image numpy files...")
            Console.warn("OS error: {0}".format(err))

        # remove distance matrix numpy files
        try:
            shutil.rmtree(self.distance_matrix_numpy_folder, ignore_errors=True)
        except OSError as err:
            Console.warn("Could not delete the folder for image numpy files")
            Console.warn("OS error: {0}".format(err))

    def setup(self, phase):
        """Setup the corrector object by loading required
        configuration parameters and creating output directories
        
        Parameters
        ----------
        phase
            indicates whether the setup is for a parse or process phase of correct_images

        Returns
        -------
        int
            Returns 0 if successful, -1 otherwise.
        """

        self.get_imagelist()
        self.create_output_directories(phase)
        if self.correction_method == "colour_correction":
            self.generate_distance_matrix(phase)
        self.generate_bayer_numpy_filelist(self.camera_image_list)
        self.generate_bayer_numpyfiles(self.bayer_numpy_filelist)
        return 0

    # create directories for storing intermediate image and distance_matrix numpy files,
    # correction parameters and corrected output images
    def create_output_directories(self, phase):
        """Handle the creation of output directories for each camera

        Parameters
        -----------
        phase
            indicates whether the setup is for a parse or process phase of correct_images
        """

        if phase == "parse" or phase == "process":
            # create output directory path
            image_path = Path(self.camera_image_list[0]).resolve()
            image_parent_path = image_path.parents[0]
            output_dir_path = get_processed_folder(image_parent_path)
            self.output_dir_path = output_dir_path / "attenuation_correction"
            if not self.output_dir_path.exists():
                self.output_dir_path.mkdir(parents=True)

            # create path for image numpy files
            bayer_numpy_folder_name = "bayer_" + self.camera.name
            self.bayer_numpy_dir_path = self.output_dir_path / bayer_numpy_folder_name
            if not self.bayer_numpy_dir_path.exists():
                self.bayer_numpy_dir_path.mkdir(parents=True)

            if self.correction_method == "colour_correction":
                # create path for distance matrix numpy files
                distance_matrix_numpy_folder_name = "distance_" + self.camera.name
                self.distance_matrix_numpy_folder = (
                        self.output_dir_path / distance_matrix_numpy_folder_name
                )
                if not self.distance_matrix_numpy_folder.exists():
                    self.distance_matrix_numpy_folder.mkdir(parents=True)

                # create path for parameters files
                attenuation_parameters_folder_name = "params_" + self.camera.name
                self.attenuation_parameters_folder = (
                        self.output_dir_path / attenuation_parameters_folder_name
                )
                if not self.attenuation_parameters_folder.exists():
                    self.attenuation_parameters_folder.mkdir(parents=True)

                # create path for memmap files
                memmap_folder_name = "memmaps_" + self.camera.name
                self.memmap_folder = self.output_dir_path / memmap_folder_name
                if not self.memmap_folder.exists():
                    self.memmap_folder.mkdir(parents=True)

        if phase == "process":
            # create path for output images
            output_images_folder_name = "developed_" + self.camera.name
            self.output_images_folder = self.output_dir_path / output_images_folder_name
            if not self.output_images_folder.exists():
                self.output_images_folder.mkdir(parents=True)

        # create target sub directores based on configurations defined in correct_images.yaml
        self.create_subdirectories(phase)
        Console.info("Output directories created / existing...")

    def create_subdirectories(self, phase):

        """Handle the creation of sub directories corresponding to a particular configuration selected
        through correct_images.yaml file

        Parameters
        -----------
        phase
            indicates whether the setup is for a parse or process phase of correct_images
        """

        # create folder name for correction parameters based on correction method

        sub_directory_name = 'unknown_sub_directory_name'
        output_folder_name = 'unknown_output_folder_name'
        if self.correction_method == "colour_correction":
            if self.distance_metric == "none":
                sub_directory_name = "greyworld_corrected"
            elif self.distance_metric == "altitude":
                sub_directory_name = "altitude_corrected"
            elif self.distance_metric == "depth_map":
                sub_directory_name = "depth_map_corrected"

            output_folder_name = "m" + str(self.brightness) + "_std" + str(self.contrast)

            # appending bayer numpy files path with sub directory and output folder
            self.bayer_numpy_dir_path = self.bayer_numpy_dir_path / sub_directory_name / output_folder_name
            if not self.bayer_numpy_dir_path.exists():
                self.bayer_numpy_dir_path.mkdir(parents=True)

            # appending distance numpy files path with sub directory and output folder
            self.distance_matrix_numpy_folder = self.distance_matrix_numpy_folder / sub_directory_name / output_folder_name
            if not self.distance_matrix_numpy_folder.exists():
                self.distance_matrix_numpy_folder.mkdir(parents=True)

            # appending params path with sub directory and output folder
            self.attenuation_parameters_folder = self.attenuation_parameters_folder / sub_directory_name
            if phase == "parse":
                if not self.attenuation_parameters_folder.exists():
                    self.attenuation_parameters_folder.mkdir(parents=True)
                else:
                    dir_temp = self.attenuation_parameters_folder
                    file_list = list(dir_temp.glob("*.npy"))
                    if len(file_list) > 0:
                        if not self.force:
                            Console.quit(
                                "Parameters exist for current configuration.",
                                "Run parse with Force (-F flag)..."
                            )
                        else:
                            Console.warn("Code will overwrite existing parameters for current configuration...")
            elif phase == "process":
                if not self.attenuation_parameters_folder.exists():
                    Console.quit("Run parse before process for current configuration...")
                else:
                    Console.info("Found correction parameters for current configuration...")
        elif self.correction_method == "manual_balance":
            sub_directory_name = "manually_corrected"
            temp1 = str(datetime.now())
            temp2 = temp1.split(":")
            temp3 = temp2[0].split(" ")
            temp4 = temp3[1] + temp2[1]
            output_folder_name = "developed_" + temp4

        if phase == "process":
            # appending developed images path with sub directory and output folder
            self.output_images_folder = self.output_images_folder / sub_directory_name / output_folder_name
            if not self.output_images_folder.exists():
                self.output_images_folder.mkdir(parents=True)
            else:
                dir_temp = self.output_images_folder
                file_list = list(dir_temp.glob("*.*"))
                if len(file_list) > 0:
                    if not self.force:
                        Console.quit(
                            "Corrected images exist for current configuration. Run process with Force (-F flag)..."
                        )
                    else:
                        Console.warn("Code will overwrite existing corrected images for current configuration...")

    def get_camera_idx(self):
        idx = [
            i
            for i, camera_config in enumerate(self.cameraconfigs)
            if camera_config.camera_name == self.camera.name
        ]
        if len(idx) > 0:
            return idx[0]
        else:
            Console.warn("The camera", self.camera.name,
                         "could not be found in the correct_images.yaml")
            return None

    # load imagelist: output is same as camera.imagelist unless a smaller filelist is specified by the user
    def get_imagelist(self):
        """Generate list of source images"""
        if self.correction_method == "colour_correction":
            if self.distance_path == "json_renav_*":
                Console.info(
                    "Picking first JSON folder as the default path to auv_nav csv files..."
                )
                dir_ = self.path_processed
                json_list = list(dir_.glob("json_*"))
                if len(json_list) == 0:
                    Console.quit("No navigation solution could be found. Please run auv_nav parse and process first")
                self.distance_path = json_list[0]

            full_metric_path = self.path_processed / self.distance_path
            full_metric_path = full_metric_path / "csv" / "ekf"
            metric_file = "auv_ekf_" + self.camera_name + ".csv"
            self.altitude_csv_path = full_metric_path / metric_file

            # get imagelist for given camera object
            if self.user_specified_image_list == "none":
                self.camera_image_list = self.camera.image_list
            # get imagelist from user provided filelist
            else:
                path_file_list = Path(self.path_config) / self.user_specified_image_list
                trimmed_csv_file = 'trimmed_csv_' + self.camera.name + '.csv'
                self.trimmed_csv_path = Path(self.path_config) / trimmed_csv_file

                if not self.altitude_csv_path.exists():
                    message = "Path to " + metric_file + " does not exist..."
                    Console.quit(message)
                else:
                    # create trimmed csv based on user's provided list of images
                    trim_csv_files(path_file_list, self.altitude_csv_path, self.trimmed_csv_path)

                # read trimmed csv filepath
                dataframe = pd.read_csv(self.trimmed_csv_path)
                user_imagepath_list = dataframe["relative_path"]
                user_imagenames_list = [Path(image).name for image in user_imagepath_list]
                self.camera_image_list = [
                    item
                    for item in self.camera.image_list
                    for image in user_imagenames_list
                    if Path(item).name == image
                ]
        elif self.correction_method == "manual_balance":
            self.camera_image_list = self.camera.image_list

        # save a set of distance matrix numpy files

    def generate_distance_matrix(self, phase):
        """Generate distance matrix numpy files and save them"""

        # create empty distance matrix and list to store paths to the distance numpy files
        distance_matrix = np.empty((self.image_height, self.image_width))
        self.distance_matrix_numpy_filelist = []
        self.altitude_based_filtered_indices = []

        # read altitude / depth map depending on distance_metric
        if self.distance_metric == "none":
            Console.info("Null distance matrix created")

        elif self.distance_metric == "depth_map":
            path_depth = self.path_processed / 'depth_map'  # self.distance_path
            if not path_depth.exists():
                Console.quit("Depth maps not found...")
            else:
                Console.info("Path to depth maps found...")
                depth_map_list = list(path_depth.glob("*.npy"))
                depth_map_list = [
                    Path(item)
                    for item in depth_map_list
                    for image_path in self.camera_image_list
                    if Path(image_path).stem in Path(item).stem
                ]
                for idx in trange(len(depth_map_list)):
                    if idx >= len(self.camera_image_list):
                        break
                    depth_map = depth_map_list[idx]
                    depth_array = np.load(depth_map)
                    distance_matrix_size = (self.image_height, self.image_width)
                    distance_matrix = resize(depth_array, distance_matrix_size,
                                             preserve_range=True)
                    # distance_matrix = np.resize(depth_array, distance_matrix_size)

                    image_name = Path(self.camera_image_list[idx]).stem
                    distance_matrix_numpy_file = image_name + ".npy"
                    distance_matrix_numpy_file_path = (
                            self.distance_matrix_numpy_folder / distance_matrix_numpy_file
                    )
                    self.distance_matrix_numpy_filelist.append(
                        distance_matrix_numpy_file_path
                    )

                    # create the distance matrix numpy file
                    np.save(distance_matrix_numpy_file_path, distance_matrix)
                    min_depth = distance_matrix.min()
                    max_depth = distance_matrix.max()

                    if min_depth > self.altitude_max or max_depth < self.altitude_min:
                        continue
                    else:
                        self.altitude_based_filtered_indices.append(idx)
                if phase == "process":
                    if len(self.distance_matrix_numpy_filelist) < len(self.camera_image_list):
                        temp = []
                        for idx in range(len(self.distance_matrix_numpy_filelist)):
                            temp.append(self.camera_image_list[idx])
                        self.camera_image_list = temp
                        Console.warn("Image list is corrected with respect to distance list...")
                if len(self.altitude_based_filtered_indices) < 3:
                    Console.quit(
                        "Insufficient number of images to compute attenuation parameters..."
                    )

        elif self.distance_metric == "altitude":
            # check if user provides a file list
            if self.user_specified_image_list == "none":
                distance_csv_path = Path(self.altitude_csv_path)
            else:
                distance_csv_path = (
                        Path(self.path_config) / self.trimmed_csv_path
                )

            # Check if file exists
            if not distance_csv_path.exists():
                Console.quit("The navigation CSV file is not present. Run auv_nav first.")

            # read dataframe for corresponding distance csv path
            dataframe = pd.read_csv(distance_csv_path)
            distance_list = dataframe["altitude [m]"]

            if phase == "process":
                if len(distance_list) < len(self.camera_image_list):
                    temp = []
                    for idx in range(len(distance_list)):
                        temp.append(self.camera_image_list[idx])
                    self.camera_image_list = temp
                    Console.warn("Image list is corrected with respect to distance list...")

            for idx in trange(len(distance_list)):
                # TODO use actual camera geometrics to compute distance to seafloor
                distance_matrix.fill(distance_list[idx])
                if idx >= len(self.camera_image_list):
                    break
                else:
                    image_name = Path(self.camera_image_list[idx]).stem
                    distance_matrix_numpy_file = image_name + ".npy"
                    distance_matrix_numpy_file_path = (
                            self.distance_matrix_numpy_folder / distance_matrix_numpy_file
                    )
                    self.distance_matrix_numpy_filelist.append(
                        distance_matrix_numpy_file_path
                    )

                    # create the distance matrix numpy file
                    np.save(distance_matrix_numpy_file_path, distance_matrix)

                    # filter images based on altitude range
                    if self.altitude_max > distance_list[idx] > self.altitude_min:
                        self.altitude_based_filtered_indices.append(idx)

            Console.info("Distance matrix numpy files written successfully")

            Console.info(
                len(self.altitude_based_filtered_indices), '/', len(distance_list),
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

    def generate_bayer_numpyfiles(self, bayer_numpy_filelist: list):

        """Generates image numpy files
        
        Parameters
        ----------
        bayer_numpy_filelist : list
            list of paths to image numpy files
        """

        # create numpy files as per bayer_numpy_filelist
        if self.camera.extension == "tif" or self.camera.extension == "jpg":
            # write numpy files for corresponding bayer images
            for idx in trange(len(self.camera_image_list)):
                tmp_tif = imageio.imread(self.camera_image_list[idx])
                tmp_npy = np.array(tmp_tif, np.uint16)
                np.save(bayer_numpy_filelist[idx], tmp_npy)
        if self.camera.extension == "raw":
            # create numpy files as per bayer_numpy_filelist
            raw_image_for_size = np.fromfile(str(self.camera_image_list[0]), dtype=np.uint8)
            binary_data = np.zeros(
                (len(self.camera_image_list), raw_image_for_size.shape[0]),
                dtype=raw_image_for_size.dtype,
            )
            for idx in range(len(self.camera_image_list)):
                binary_data[idx] = np.fromfile(
                    str(self.camera_image_list[idx]), dtype=raw_image_for_size.dtype
                )
            Console.info("Writing RAW images to numpy...")

            for idx in trange(len(self.camera_image_list)):
                image_raw = xviii.load_xviii_bayer_from_binary(
                        binary_data[idx, :], self.image_height, self.image_width
                    )
                np.save(bayer_numpy_filelist[idx], image_raw)

        """
            image_raw = joblib.Parallel(n_jobs=-2, verbose=3, max_nbytes=1e6)(
                [
                    joblib.delayed(xviii.load_xviii_bayer_from_binary)(
                        binary_data[idx, :], self.image_height, self.image_width
                    )
                    for idx in trange(len(self.camera_image_list))
                ]
            )
            for idx in trange(len(self.camera_image_list)):
                np.save(bayer_numpy_filelist[idx], image_raw[idx])
        """

        Console.info("Image numpy files written successfully...")

    # compute correction parameters either for attenuation correction or static correction of images
    def generate_attenuation_correction_parameters(self):
        """Generates image stats and attenuation coefficients and saves the parameters for process"""

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

            image_memmap_path, image_memmap = load_memmap_from_numpyfilelist(self.memmap_folder,
                                                                             filtered_image_numpy_filelist)
            distance_memmap_path, distance_memmap = load_memmap_from_numpyfilelist(self.memmap_folder,
                                                                                   filtered_distance_numpy_filelist)

            for i in range(self.image_channels):

                if self.image_channels == 1:
                    image_memmap_per_channel = image_memmap
                else:
                    image_memmap_per_channel = image_memmap[:, :, :, i]
                # calculate mean, std for image and target_altitude
                # print(image_memmap_per_channel.shape)
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
                bin_images_sample = None
                bin_distances_sample = None

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

                        # release memory from bin_images
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
                attenuation_parameters = corrections.calculate_attenuation_parameters(
                    images_for_attenuation_calculation,
                    distances_for_attenuation_calculation,
                    self.image_height,
                    self.image_width,
                )

                self.image_attenuation_parameters[i] = attenuation_parameters

                # compute correction gains per channel
                Console.info("Computing correction gains...")
                correction_gains = corrections.calculate_correction_gains(
                    target_altitude, attenuation_parameters
                )
                self.correction_gains[i] = correction_gains

                # apply gains to images
                Console.info("Applying attenuation corrections to images...")
                image_memmap_per_channel = corrections.attenuation_correct_memmap(
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

            # delete existing memmap files
            memmap_files_path = self.memmap_folder.glob("*.map")
            for file in memmap_files_path:
                if file.exists():
                    file.unlink()

            image_memmap_path, image_memmap = load_memmap_from_numpyfilelist(
                self.memmap_folder, self.bayer_numpy_filelist
            )

            for i in range(self.image_channels):
                if self.image_channels == 1:
                    image_memmap_per_channel = image_memmap
                else:
                    image_memmap_per_channel = image_memmap[:, :, :, i]

                # calculate mean, std for image and target_altitude
                # print(image_memmap_per_channel.shape)
                Console.info(f"Memmap for channel is shape: {image_memmap_per_channel.shape}")
                Console.info(f"Memmap for channel is of type: {type(image_memmap_per_channel)}")
                raw_image_mean, raw_image_std = mean_std(image_memmap_per_channel)
                self.image_raw_mean[i] = raw_image_mean
                self.image_raw_std[i] = raw_image_std
        # print(self.attenuation_parameters_folder)
        image_raw_mean_file = self.attenuation_parameters_folder / "image_raw_mean.npy"
        image_raw_std_file = self.attenuation_parameters_folder / "image_raw_std.npy"

        np.save(image_raw_mean_file, self.image_raw_mean)
        np.save(image_raw_std_file, self.image_raw_std)

        Console.info("Correction parameters saved...")

    # execute the corrections of images using the gain values in case of attenuation correction or static color balance
    def process_correction(self, test_phase=False):
        """Execute series of corrections for a set of input images

        Parameters
        -----------
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

        Console.info("Processing images for color, distortion, gamma corrections...")

        with tqdm_joblib(tqdm(desc="Correcting images", total=len(self.bayer_numpy_filelist))) as progress_bar:
            joblib.Parallel(n_jobs=-2, verbose=3, max_nbytes=1e6)(
                joblib.delayed(self.process_image)(idx, test_phase)
                for idx in range(0, len(self.bayer_numpy_filelist))  # out of range error here
            )

        # write a filelist.csv containing image filenames which are processed
        image_files = []
        for path in self.bayer_numpy_filelist:
            image_files.append(Path(path).name)
        dataframe = pd.DataFrame(image_files)
        filelist_path = self.output_images_folder / "filelist.csv"
        dataframe.to_csv(filelist_path)
        Console.info("Processing of images is completed...")

    def process_image(self, idx, test_phase=False):
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
        image_rgb = None

        # apply corrections
        if self.correction_method == "colour_correction":
            if len(self.distance_matrix_numpy_filelist) > 0:
                distance_matrix = np.load(self.distance_matrix_numpy_filelist[idx])
            else:
                distance_matrix = None

            image = self.apply_distance_based_corrections(
                image, distance_matrix, self.brightness, self.contrast
            )
            if not self._type == "grayscale":
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
                image_rgb, self.color_gain_matrix_rgb, self.subtractors_rgb)

        # save corrected image back to numpy list for testing purposes
        if test_phase:
            np.save(self.bayer_numpy_filelist[idx], image)

        # apply distortion corrections
        if self.undistort:
            image_rgb = corrections.distortion_correct(self.camera_params_file_path, image_rgb)

        # apply gamma corrections to rgb images for colour correction
        if self.correction_method == "colour_correction":
            image_rgb = corrections.gamma_correct(image_rgb)

        # apply scaling to 8 bit and format image to unit8
        image_rgb = corrections.bytescaling(image_rgb)
        image_rgb = image_rgb.astype(np.uint8)

        # write to output files
        image_filename = Path(self.bayer_numpy_filelist[idx]).stem
        write_output_image(
            image_rgb, image_filename, self.output_images_folder, self.output_format
        )

    # apply corrections on each image using the correction parameters for targeted brightness and contrast
    def apply_distance_based_corrections(self, image, distance_matrix, brightness, contrast):
        """Apply attenuation corrections to images

        Parameters
        -----------
        image : numpy.ndarray
            image data to be debayered
        distance_matrix : numpy.ndarray
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
            if distance_matrix is not None:
                intensities = corrections.attenuation_correct(
                    intensities,
                    distance_matrix,
                    self.image_attenuation_parameters[i],
                    self.correction_gains[i],
                )
                # Achieve desired mean and std after
                # attenuation correction
                intensities = corrections.pixel_stat(
                    intensities,
                    self.image_corrected_mean[i],
                    self.image_corrected_std[i],
                    brightness,
                    contrast,
                )
            else:
                intensities = corrections.pixel_stat(
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

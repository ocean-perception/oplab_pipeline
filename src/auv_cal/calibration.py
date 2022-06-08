# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import json
from pathlib import Path
from typing import Dict

import yaml

from auv_cal.camera_calibrator import (
    CalibrationException,
    ChessboardInfo,
    MonoCalibrator,
    Patterns,
    StereoCalibrator,
)
from auv_cal.laser_calibrator import LaserCalibrator
from oplab import (
    Console,
    StereoCamera,
    check_dirs_exist,
    get_config_folder,
    get_processed_folder,
    get_raw_folder,
    get_raw_folders,
)


def collect_image_files(image_dirs, file_pattern):
    # TODO try all different formats: png, jpeg, tif
    images = []
    if isinstance(image_dirs, list):
        for d in image_dirs:
            if d.is_dir():
                images.extend(list(d.glob(file_pattern)))
            else:
                Console.warn("Directory " "'{}'" " cannot be found".format(d))
    else:
        if image_dirs.is_dir():
            images = list(image_dirs.glob(file_pattern))
        else:
            Console.warn("Directory " "'{}'" " cannot be found".format(image_dirs))
    images.sort()

    resolved_images = []
    for i in images:
        p = Path(i).resolve()
        resolved_images.append(p)
    return resolved_images


def check_pattern(config):
    if config["pattern"] == "Circles" or config["pattern"] == "circles":
        pattern = Patterns.Circles
    elif config["pattern"] == "ACircles" or config["pattern"] == "acircles":
        pattern = Patterns.ACircles
    elif config["pattern"] == "Chessboard" or config["pattern"] == "chessboard":
        pattern = Patterns.Chessboard
    else:
        Console.quit(
            "The available patterns are: Circles, Chessboard or ACircles.",
            "Please check you did not misspell the pattern type.",
        )
    return pattern


def calibrate_mono(
    name,
    filepaths,
    extension,
    config,
    output_file,
    fo,
    foa,
    target_width=None,
    target_height=None,
):
    if not check_dirs_exist(filepaths):
        filepaths = get_raw_folders(filepaths)
    Console.info(
        "Looking for {} calibration images in {}".format(extension, str(filepaths))
    )
    image_list = collect_image_files(filepaths, extension)
    Console.info("Found " + str(len(image_list)) + " images.")
    if len(image_list) < 8:
        Console.quit("Too few images. Try to get more.")

    mc = MonoCalibrator(
        boards=[ChessboardInfo(config["rows"], config["cols"], config["size"])],
        pattern=check_pattern(config),
        invert=config["invert"],
        name=name,
        target_width=target_width,
        target_height=target_height,
    )
    try:
        image_list_file = output_file.with_suffix(".json")
        if foa or not image_list_file.exists():
            mc.cal(image_list)
            with image_list_file.open("w") as f:
                Console.info("Writing JSON to " "'{}'" "".format(image_list_file))
                json.dump(mc.json, f)
        else:
            mc.cal_from_json(image_list_file, image_list)
        mc.report()
        Console.info("Writing calibration to " "'{}'" "".format(output_file))
        with output_file.open("w") as f:
            f.write(mc.yaml())
    except CalibrationException:
        Console.error("The calibration pattern was not found in the images.")


def calibrate_stereo(
    left_name,
    left_filepaths,
    left_extension,
    left_calib,
    right_name,
    right_filepaths,
    right_extension,
    right_calib,
    config,
    output_file,
    fo,
    foa,
):
    if not check_dirs_exist(left_filepaths):
        left_filepaths = get_raw_folders(left_filepaths)
    if not check_dirs_exist(right_filepaths):
        right_filepaths = get_raw_folders(right_filepaths)
    Console.info("Looking for calibration images in {}".format(str(left_filepaths)))
    left_image_list = collect_image_files(left_filepaths, left_extension)
    Console.info("Found " + str(len(left_image_list)) + " left images.")
    Console.info("Looking for calibration images in {}".format(str(right_filepaths)))
    right_image_list = collect_image_files(right_filepaths, right_extension)
    Console.info("Found " + str(len(right_image_list)) + " right images.")
    if len(left_image_list) < 8 or len(right_image_list) < 8:
        Console.quit("Too few images. Try to get more.")
    try:
        with left_calib.with_suffix(".json").open("r") as f:
            left_json = json.load(f)
        with right_calib.with_suffix(".json").open("r") as f:
            right_json = json.load(f)

        model = StereoCamera(left=left_calib, right=right_calib)
        if model.different_aspect_ratio or model.different_resolution:
            Console.warn(
                "Stereo calibration: Calibrating two cameras with different",
                "resolution.",
            )
            Console.info("  Camera:", left_name, "is", model.left.size)
            Console.info("  Camera:", right_name, "is", model.right.size)
            right_calib = right_calib.parent / ("resized_" + right_calib.name)
            if not right_calib.with_suffix(".json").exists():
                calibrate_mono(
                    right_name,
                    right_filepaths,
                    right_extension,
                    config,
                    right_calib,
                    fo,
                    foa,
                    target_width=model.left.image_width,
                    target_height=model.left.image_height,
                )
            model = StereoCamera(left=left_calib, right=right_calib)
            right_calib = right_calib.with_suffix(".json")
            with right_calib.open("r") as f:
                right_json = json.load(f)

        sc = StereoCalibrator(
            name=left_name + "-" + right_name,
            stereo_camera_model=model,
            boards=[ChessboardInfo(config["rows"], config["cols"], config["size"])],
            pattern=check_pattern(config),
            invert=config["invert"],
        )
        # sc.cal(left_image_list, right_image_list)
        sc.cal_from_json(
            left_json=left_json,
            right_json=right_json,
        )
        sc.report()
        Console.info("Writing calibration to " "'{}'" "".format(output_file))
        with output_file.open("w") as f:
            f.write(sc.yaml())
    except CalibrationException:
        Console.error("The calibration pattern was not found in the images.")


def calibrate_laser(
    laser_cam_name,
    laser_cam_filepath: Path,
    laser_cam_extension: str,
    non_laser_cam_name,
    non_laser_cam_filepath: Path,
    non_laser_cam_extension: str,
    stereo_calibration_file: Path,
    config: Dict,
    output_file: Path,
    output_file_b: Path,
    num_uncert_planes: int,
    skip_first: int = 0,
    fo: bool = False,
    foa: bool = False,
):
    Console.info("Looking for calibration images in {}".format(laser_cam_filepath))
    laser_cam_image_list = collect_image_files(laser_cam_filepath, laser_cam_extension)
    Console.info("Found " + str(len(laser_cam_image_list)) + " left images.")
    Console.info("Looking for calibration images in {}".format(non_laser_cam_filepath))
    non_laser_cam_image_list = collect_image_files(
        non_laser_cam_filepath, non_laser_cam_extension
    )
    Console.info("Found " + str(len(non_laser_cam_image_list)) + " right images.")
    if len(laser_cam_image_list) < 8 or len(non_laser_cam_image_list) < 8:
        Console.error("Too few images. Try to get more.")
        return
    try:
        model = StereoCamera(stereo_calibration_file)

        def check_prop(name, dic_to_check, default_value):
            if name in dic_to_check:
                return dic_to_check[name]
            else:
                return default_value

        lc = LaserCalibrator(model, config, num_uncert_planes, overwrite=foa)
        lc.cal(
            laser_cam_image_list[skip_first:],
            non_laser_cam_image_list[skip_first:],
            output_file.parent,
        )
        Console.info("Writing calibration to " "'{}'" "".format(output_file))
        with output_file.open("w") as f:
            f.write(lc.yaml())
        if "two_lasers" not in config["detection"]:
            return
        if config["detection"]["two_lasers"]:
            Console.info("Writing calibration to " "'{}'" "".format(output_file_b))
            with output_file_b.open("w") as f:
                f.write(lc.yaml_b())
    except CalibrationException:
        Console.error("The calibration pattern was not found in the images.")


def build_filepath(base, paths):
    filepaths = []
    if not isinstance(paths, list):
        filepaths = base / str(paths)
    else:
        for p in paths:
            filepaths.append(base / p)
    return filepaths


class Calibrator:
    def __init__(
        self,
        filepath: str,
        force_overwite: bool = False,
        overwrite_all: bool = False,
        suffix: str = "",
        num_uncert_planes: int = 300,
    ):
        filepath = Path(filepath).resolve()
        self.filepath = get_raw_folder(filepath)
        self.fo = force_overwite
        self.foa = overwrite_all
        self.suffix = suffix
        self.num_uncert_planes = num_uncert_planes

        if self.foa:
            self.fo = True

        if self.suffix:
            if self.suffix[0] != "_":
                self.suffix = "_" + self.suffix

        self.configuration_path = (
            get_config_folder(self.filepath.parent) / "calibration"
        )
        calibration_config_file = self.configuration_path / "calibration.yaml"
        if calibration_config_file.exists():
            Console.info(
                "Loading existing calibration.yaml at {}".format(
                    calibration_config_file
                )
            )
        else:
            root = Path(__file__).parents[1]
            default_file = root / "auv_cal/default_yaml" / "default_calibration.yaml"
            Console.warn(
                "Cannot find {}, generating default from {}".format(
                    calibration_config_file, default_file
                )
            )
            # save localisation yaml to processed directory
            default_file.copy(calibration_config_file)

            Console.warn("Edit the file at: \n\t" + str(calibration_config_file))
            Console.warn("Try to use relative paths to the calibration datasets")
            Console.quit("Modify the file calibration.yaml and run this code again.")

        with calibration_config_file.open("r") as stream:
            self.calibration_config = yaml.safe_load(stream)

        # Create the calibration folder at the same level as the dives
        self.output_path = get_processed_folder(self.filepath.parent) / "calibration"
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        if not self.configuration_path.exists():
            self.configuration_path.mkdir(parents=True)

    def mono(self):
        for c in self.calibration_config["cameras"]:
            cam_name = c["name"]
            # Find if the calibration file exists
            calibration_file = self.output_path / str(
                "mono_" + cam_name + self.suffix + ".yaml"
            )
            Console.info("Looking for a calibration file at " + str(calibration_file))
            if calibration_file.exists() and not self.fo:
                Console.warn(
                    "The camera "
                    + c["name"]
                    + " has already been calibrated. If you want to overwrite "
                    + "the JSON, use the -F flag."
                )
            else:
                Console.info(
                    "The camera is not calibrated, running mono calibration..."
                )
                filepaths = build_filepath(
                    get_processed_folder(self.filepath),
                    c["camera_calibration"]["path"],
                )

                if "glob_pattern" not in c["camera_calibration"]:
                    Console.error(
                        "Could not find the key glob_pattern for the camera ",
                        c["name"],
                    )
                    Console.quit("glob_pattern expected in calibration.yaml")

                calibrate_mono(
                    cam_name,
                    filepaths,
                    str(c["camera_calibration"]["glob_pattern"]),
                    self.calibration_config["camera_calibration"],
                    calibration_file,
                    self.fo,
                    self.foa,
                )

    def stereo(self):
        if len(self.calibration_config["cameras"]) > 1:
            c0 = self.calibration_config["cameras"][0]
            c1 = self.calibration_config["cameras"][1]
            calibration_file = self.output_path / str(
                "stereo_" + c0["name"] + "_" + c1["name"] + self.suffix + ".yaml"
            )
            Console.info("Looking for a calibration file at " + str(calibration_file))
            if calibration_file.exists() and not self.fo:
                Console.quit(
                    "The stereo pair "
                    + c0["name"]
                    + "_"
                    + c1["name"]
                    + " has already been calibrated. If you want to overwrite "
                    + "the calibration, use the -F flag."
                )
            Console.info(
                "The stereo camera is not calibrated, running stereo",
                "calibration...",
            )

            left_filepaths = build_filepath(
                get_processed_folder(self.filepath),
                c0["camera_calibration"]["path"],
            )
            right_filepaths = build_filepath(
                get_processed_folder(self.filepath),
                c1["camera_calibration"]["path"],
            )
            left_name = c0["name"]
            if "glob_pattern" not in c0["camera_calibration"]:
                Console.error(
                    "Could not find the key glob_pattern for the camera ",
                    c0["name"],
                )
                Console.quit("glob_pattern expected in calibration.yaml")
            left_extension = str(c0["camera_calibration"]["glob_pattern"])
            right_name = c1["name"]
            if "glob_pattern" not in c1["camera_calibration"]:
                Console.error(
                    "Could not find the key glob_pattern for the camera ",
                    c1["name"],
                )
                Console.quit("glob_pattern expected in calibration.yaml")
            right_extension = str(c1["camera_calibration"]["glob_pattern"])
            left_calibration_file = self.output_path / str(
                "mono_" + left_name + ".yaml"
            )
            right_calibration_file = self.output_path / str(
                "mono_" + right_name + ".yaml"
            )
            if (
                not left_calibration_file.exists()
                or not right_calibration_file.exists()
            ):
                if not left_calibration_file.exists():
                    Console.warn(
                        "Could not find a monocular calibration file "
                        + str(left_calibration_file)
                        + "..."
                    )
                if not right_calibration_file.exists():
                    Console.warn(
                        "Could not find a monocular calibration file "
                        + str(right_calibration_file)
                        + "..."
                    )
                if self.suffix:
                    Console.quit(
                        "Suffix is indicated, but mono calibration is missing. Automatic calling of mono calibrator is "
                        "not supported when indicating a suffix."
                    )
                else:
                    self.mono()
            if left_calibration_file.exists() and right_calibration_file.exists():
                Console.info(
                    "Loading previous monocular calibrations at \
                                \n\t * {}\n\t * {}".format(
                        str(left_calibration_file), str(right_calibration_file)
                    )
                )
            calibrate_stereo(
                left_name,
                left_filepaths,
                left_extension,
                left_calibration_file,
                right_name,
                right_filepaths,
                right_extension,
                right_calibration_file,
                self.calibration_config["camera_calibration"],
                calibration_file,
                self.fo,
                self.foa,
            )
        # Check for a second stereo pair
        if len(self.calibration_config["cameras"]) > 2:
            c0 = self.calibration_config["cameras"][0]
            c2 = self.calibration_config["cameras"][2]
            calibration_file = self.output_path / str(
                "stereo_" + c0["name"] + "_" + c2["name"] + self.suffix + ".yaml"
            )
            Console.info("Looking for a calibration file at " + str(calibration_file))
            if calibration_file.exists() and not self.fo:
                Console.quit(
                    "The stereo pair "
                    + c0["name"]
                    + "_"
                    + c2["name"]
                    + " has already been calibrated. If you want to overwrite "
                    + "the calibration, use the -F flag."
                )
            Console.info(
                "The stereo camera is not calibrated, running stereo",
                "calibration...",
            )
            left_name = c0["name"]
            left_filepaths = build_filepath(
                get_processed_folder(self.filepath),
                c0["camera_calibration"]["path"],
            )
            left_extension = str(c0["camera_calibration"]["glob_pattern"])
            right_name = c2["name"]
            right_filepaths = build_filepath(
                self.filepath, c2["camera_calibration"]["path"]
            )
            right_extension = str(c2["camera_calibration"]["glob_pattern"])
            left_calibration_file = self.output_path / str(
                "mono_" + left_name + ".yaml"
            )
            right_calibration_file = self.output_path / str(
                "mono_" + right_name + ".yaml"
            )
            if (
                not left_calibration_file.exists()
                or not right_calibration_file.exists()
            ):
                if not left_calibration_file.exists():
                    Console.warn(
                        "Could not find a monocular calibration file "
                        + str(left_calibration_file)
                        + "..."
                    )
                if not right_calibration_file.exists():
                    Console.warn(
                        "Could not find a monocular calibration file "
                        + str(right_calibration_file)
                        + "..."
                    )
                if self.suffix:
                    Console.quit(
                        "Suffix is indicated, but mono calibration is missing. Automatic calling of mono calibrator is "
                        "not supported when indicating a suffix."
                    )
                else:
                    self.mono()
            if left_calibration_file.exists() and right_calibration_file.exists():
                Console.info(
                    "Loading previous monocular calibrations at \
                                \n\t * {}\n\t * {}".format(
                        str(left_calibration_file), str(right_calibration_file)
                    )
                )
            calibrate_stereo(
                left_name,
                left_filepaths,
                left_extension,
                left_calibration_file,
                right_name,
                right_filepaths,
                right_extension,
                right_calibration_file,
                self.calibration_config["camera_calibration"],
                calibration_file,
                self.fo,
                self.foa,
            )

            calibration_file = self.output_path / str(
                "stereo_" + c2["name"] + "_" + c0["name"] + self.suffix + ".yaml"
            )

            calibrate_stereo(
                right_name,
                right_filepaths,
                right_extension,
                right_calibration_file,
                left_name,
                left_filepaths,
                left_extension,
                left_calibration_file,
                self.calibration_config["camera_calibration"],
                calibration_file,
                self.fo,
                self.foa,
            )

    def laser(self):
        if "laser_calibration" not in self.calibration_config["cameras"][0]:
            Console.quit(
                'There is no field "laser_calibration" for the first',
                "camera in the calibration.yaml",
            )
        if "laser_calibration" not in self.calibration_config["cameras"][1]:
            Console.quit(
                'There is no field "laser_calibration" for the second',
                "camera in the calibration.yaml",
            )
        c0 = self.calibration_config["cameras"][0]
        c1 = self.calibration_config["cameras"][1]
        self.laser_imp(c1, c0)
        # if len(self.calibration_config["cameras"]) > 2:
        #     c1 = self.calibration_config["cameras"][2]
        #     self.laser_imp(c0, c1)

    def laser_imp(self, c0, c1):
        main_camera_name = c1["name"]
        calibration_file = self.output_path / (
            "laser_calibration_top_" + main_camera_name + self.suffix + ".yaml"
        )
        calibration_file_b = self.output_path / (
            "laser_calibration_bottom_" + main_camera_name + self.suffix + ".yaml"
        )
        Console.info("Looking for a calibration file at " + str(calibration_file))
        if calibration_file.exists() and not self.fo:
            Console.quit(
                "The laser planes from cameras "
                + c1["name"]
                + " and "
                + c0["name"]
                + " have already been calibrated. If you want to overwite the "
                + "calibration, use the -F flag."
            )
        Console.info(
            "The laser planes are not calibrated, running laser calibration..."
        )

        # Check if the stereo pair has already been calibrated
        stereo_calibration_file = self.output_path / str(
            "stereo_" + c1["name"] + "_" + c0["name"] + ".yaml"
        )
        if not stereo_calibration_file.exists():
            Console.warn(
                "Could not find a stereo calibration file "
                + str(stereo_calibration_file)
                + "..."
            )
            if self.suffix:
                Console.quit(
                    "Suffix is indicated, but stereo calibration is missing. Automatic calling of stereo calibrator is "
                    "not supported when indicating a suffix."
                )
            else:
                self.stereo()

        non_laser_cam_name = c0["name"]
        non_laser_cam_filepath = get_raw_folder(self.filepath) / str(
            c0["laser_calibration"]["path"]
        )
        non_laser_cam_extension = str(c0["laser_calibration"]["glob_pattern"])
        laser_cam_name = c1["name"]
        laser_cam_filepath = get_raw_folder(self.filepath) / str(
            c1["laser_calibration"]["path"]
        )
        laser_cam_extension = str(c1["laser_calibration"]["glob_pattern"])
        if not non_laser_cam_filepath.exists():
            non_laser_cam_filepath = get_processed_folder(non_laser_cam_filepath)
            if not non_laser_cam_filepath.exists():
                Console.quit(
                    "Could not find stereo image folder, neither in raw nor "
                    + "in processed folder ("
                    + str(non_laser_cam_filepath)
                    + ")."
                )
        if not laser_cam_filepath.exists():
            laser_cam_filepath = get_processed_folder(laser_cam_filepath)
            if not laser_cam_filepath.exists():
                Console.quit(
                    "Could not find stereo image folder, neither in raw nor "
                    + "in processed folder ("
                    + str(laser_cam_filepath)
                    + ")."
                )
        non_laser_cam_filepath = non_laser_cam_filepath.resolve()
        laser_cam_filepath = laser_cam_filepath.resolve()
        Console.info(
            "Reading stereo images of laser line from "
            + str(non_laser_cam_filepath)
            + " and "
            + str(laser_cam_filepath)
        )
        if "skip_first" not in self.calibration_config:
            self.calibration_config["skip_first"] = 0
        calibrate_laser(
            laser_cam_name,
            laser_cam_filepath,
            laser_cam_extension,
            non_laser_cam_name,
            non_laser_cam_filepath,
            non_laser_cam_extension,
            stereo_calibration_file,
            self.calibration_config["laser_calibration"],
            calibration_file,
            calibration_file_b,
            self.num_uncert_planes,
            self.calibration_config["skip_first"],
            self.fo,
            self.foa,
        )

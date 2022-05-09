# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from pathlib import Path

import imageio
import yaml

from .console import Console
from .filename_to_date import FilenameToDate
from .folder_structure import get_raw_folder

# fmt: off
ROSBAG_IS_AVAILABLE = False
try:
    import rosbag
    from cv_bridge import CvBridge
    ROSBAG_IS_AVAILABLE = True
except ImportError:
    pass
# fmt: on


def rosbag_image_loader(img_topic, bagfile):
    """Load an image from a ROS bagfile"""
    if not ROSBAG_IS_AVAILABLE:
        Console.error("ROS bagfile support is not available.")
        Console.error("Please install the following python packages:")
        Console.error("- rosbag")
        Console.error("- roslz4")
        Console.error("- cv_bridge")
        Console.error("- sensor_msgs")
        Console.error("- geometry_msgs")
        Console.error("Example command:")
        Console.error(
            "pip install rosbag roslz4 cv_bridge sensor_msgs geometry_msgs",
            "--extra-index-url https://rospypi.github.io/simple/",
        )
        Console.quit("ROS bagfile support is not available.")
    bridge = CvBridge()
    Console.info("Loading ROS bagfile:", bagfile)
    Console.info("    Looking for topic", img_topic)
    with rosbag.Bag(str(bagfile)) as bag:
        for topic, msg, t in bag.read_messages(topics=[str(img_topic)]):
            if topic == str(img_topic):
                return bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    return None


def get_image_size(image_matrix):
    image_properties = [None, None, None]
    image_properties[0] = image_matrix.shape[0]
    image_properties[1] = image_matrix.shape[1]
    if len(image_matrix.shape) == 3:
        image_properties[2] = image_matrix.shape[2]
    else:
        image_properties[2] = 1
    return image_properties


class CameraEntry:
    """
    Camera class to read filenames and timestamps.

    Parameters
    ---------
    node
        A YAML dictionary that contains the camera information.
    raw_folder
        Path to the current dive
    """

    def __init__(self, node=None, raw_folder=None):
        self._image_list = []
        self._stamp_list = []
        self.bagfile_list = []
        self.topic = None
        self._image_properties = []
        self.raw_folder = Path.cwd()
        if raw_folder is not None:
            self.raw_folder = Path(raw_folder)

        if node is not None:
            self.name = node["name"]
            self.topic = node.get("topic", None)
            allowed_types = ["grayscale", "rgb", "bgr", "rggb", "grbg", "bggr", "gbrg"]
            self.type = node["type"]
            if self.type not in allowed_types:
                Console.error(
                    "Camera type '{}' is not allowed. Allowed types are: {}".format(
                        self.type, allowed_types
                    )
                )
                Console.quit("Camera type is not allowed.")

            self.bit_depth = node["bit_depth"]
            self.path = node["path"]
            self.extension = node["extension"]
            self.timestamp_file = node.get("timestamp_file", None)
            self.columns = node.get("columns", None)
            self.filename_to_date = node.get("filename_to_date", None)
            if self.extension == "bag":
                return
            if self.timestamp_file is None and self.filename_to_date is None:
                Console.error(
                    "The camera ",
                    self.name,
                    " is missing its timestamp format",
                )
                Console.error("You can provide it by means of filename:")
                Console.error(
                    "e.g. PR_20180811_153729_762_RC16.tif ->",
                    "xxxYYYYMMDDxhhmmssxfffxxxxx.xxx",
                )
                Console.error("or using a separate timestamp file:")
                Console.error(
                    "e.g. FileTime.csv, where separate columns z define",
                    "the date.",
                )
                Console.error("Find examples in default_yaml folder.")
                Console.quit("Missing timestamp format for a camera.")
            self.convert_filename = FilenameToDate(
                self.filename_to_date,
                self.timestamp_file,
                self.columns,
                self.raw_folder,
            )

    def write(self, node):
        pass

    @property
    def image_list(self):
        """
        Method to retrieve the list of images taken with the camera.

        Returns:
            list: a list of image paths
        """
        if self._image_list:
            return self._image_list
        raw_dir = get_raw_folder(self.raw_folder)

        split_glob = str(self.path).split("*")
        img_dir = ""
        if len(split_glob) == 2:
            pre_glob = split_glob[0] + "*"
            glob_vec = raw_dir.glob(pre_glob)
            img_dirs = [k for k in glob_vec]
            img_dir = Path(str(img_dirs[0]) + "/" + str(split_glob[1]))
            for i in img_dir.rglob("*." + self.extension):
                self._image_list.append(str(i))
        elif len(split_glob) == 3:
            pre_glob = split_glob[0] + "*"
            glob_vec = raw_dir.glob(pre_glob)
            img_dirs = [k for k in glob_vec]
            img_dir = Path(str(img_dirs[0]))
            for i in img_dir.glob("*" + split_glob[2] + "." + self.extension):
                self._image_list.append(str(i))
        elif len(split_glob) == 4:  # path/i*/*LC*
            # split = ['path/i', '/', 'LC]
            pre_glob = split_glob[0] + "*"
            glob_vec = raw_dir.glob(pre_glob)
            img_dirs = [k for k in glob_vec]
            img_dir = Path(str(img_dirs[0]))
            Console.info("Looking for images with the pattern *", split_glob[2], "*")
            for i in img_dir.glob("*" + split_glob[2] + "*." + self.extension):
                self._image_list.append(str(i))
        elif len(split_glob) == 1:
            img_dir = raw_dir / self.path
            for i in img_dir.rglob("*." + self.extension):
                self._image_list.append(str(i))
        else:
            Console.quit("Multiple globbing is not supported.")

        self._image_list.sort()
        return self._image_list

    @property
    def stamp_list(self):
        """
        Method to retrieve the list of images taken with the camera.

        Returns:
            list: a list of timestamps
        """
        if len(self._stamp_list) > 0:
            return self._stamp_list
        self._stamp_list = []
        for p in self.image_list:
            n = Path(p).name
            self._stamp_list.append(self.convert_filename(n))
        return self._stamp_list

    @property
    def image_properties(self):
        imagelist = self.image_list
        image_path = imagelist[0]

        # read tiff
        if (
            self.extension == "tif"
            or self.extension == "jpg"
            or self.extension == "JPG"
            or self.extension == "png"
            or self.extension == "PNG"
        ):
            image_matrix = imageio.imread(image_path)
            self._image_properties = get_image_size(image_matrix)
        # read raw
        elif self.extension == "raw":
            # TODO: provide a raw reader and get these properties from the file
            self._image_properties = [1024, 1280, 1]
        elif self.extension == "bag":
            for image_path in imagelist:
                image_matrix = rosbag_image_loader(self.topic, image_path)
                if image_matrix is None:
                    continue
                if image_matrix is not None:
                    self._image_properties = get_image_size(image_matrix)
                    break
        else:
            Console.quit("Extension", self.extension, "not supported")
        return self._image_properties


class CameraSystem:
    """Class to describe a camera system, for example, SeaXerocks3 or BioCam.
    Parses camera.yaml files that define camera systems mounted on ROVs or AUVs
    """

    def __init__(self, filename=None, raw_folder=None):
        """Constructor of camera system. If a filename is provided, the file
        will be parsed and its contents loaded into the class.

        Keyword Arguments:
            filename {string} -- Path to the camera.yaml file describing the
            camera system (default: {None})
        """
        self.cameras = []
        self.camera_system = None
        if filename is None and raw_folder is None:
            return
        if isinstance(filename, str):
            filename = Path(filename)
        if raw_folder is not None:
            self.raw_folder = Path(raw_folder)
        else:
            self.raw_folder = Path.cwd()
        data = ""
        try:
            with filename.open("r") as stream:
                data = yaml.safe_load(stream)
        except FileNotFoundError:
            Console.error("The file camera.yaml could not be found at ", filename)
            Console.quit("camera.yaml not provided")
        except PermissionError:
            Console.error("The file camera.yaml could not be opened at ", filename)
            Console.error(filename)
            Console.error("Please make sure you have the correct access rights.")
            Console.quit("camera.yaml not provided")
        self._parse(data)

    def __str__(self):
        msg = ""
        if self.camera_system is not None:
            msg += "CameraSystem: " + str(self.camera_system)
            if len(self.cameras) > 0:
                msg += " with cameras ["
                for c in self.cameras:
                    msg += str(c.name) + " "
                msg += "]"
            else:
                msg += " is empty"
        else:
            msg += "Empty CameraSystem"
        return msg

    def _parse(self, node):
        if "camera_system" not in node:
            Console.error("The camera.yaml file is missing the camera_system entry.")
            Console.quit("Wrong camera.yaml format or content.")
        self.camera_system = node["camera_system"]

        if "cameras" not in node:
            Console.error("The camera.yaml file is missing the cameras entry.")
            Console.quit("Wrong camera.yaml format or content.")
        for camera in node["cameras"]:
            self.cameras.append(CameraEntry(camera, self.raw_folder))

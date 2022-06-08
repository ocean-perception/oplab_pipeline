# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

# Workaround to dump OrderedDict into YAML files
from collections import OrderedDict

import yaml

from .console import Console


def represent_ordereddict(dumper, data):
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode("tag:yaml.org,2002:map", value)


yaml.add_representer(OrderedDict, represent_ordereddict)


def error_and_exit():
    Console.error(
        "If you specified a origin frame, check that they match. Otherwise,",
        "stick to default frame names:",
    )
    Console.error("sensors: dvl ins depth usbl")
    Console.error("cameras: use the same name in mission.yaml and in vehicle.yaml")
    Console.quit("Inconsistency between mission.yaml and vehicle.yaml")


class OriginEntry:
    def __init__(self):
        self.latitude = 0.0
        self.longitude = 0.0
        self.crs = ""
        self.date = ""
        self._empty = True

    def empty(self):
        return self._empty

    def load(self, node):
        self._empty = False
        self.latitude = node["latitude"]
        self.longitude = node["longitude"]
        self.crs = node["coordinate_reference_system"]
        self.date = node["date"]

    def write(self, node):
        node["latitude"] = self.latitude
        node["longitude"] = self.longitude
        node["coordinate_reference_system"] = self.crs
        node["date"] = self.date


class CameraEntry:
    def __init__(self, node=None):
        if node is not None:
            self.name = node["name"]
            self.type = node["type"]
            self.path = node["path"]
            self.origin = None
            if "origin" in node:
                self.origin = node["origin"]
                Console.info("Using camera " + self.name + " mounted at " + self.origin)
            self.topic = None
            if "topic" in node:
                self.topic = node["topic"]
            self.timeoffset = 0.0
            if "timeoffset" in node:
                self.timeoffset = node["timeoffset"]

    def write(self, node):
        node["name"] = self.name
        node["origin"] = self.origin
        node["type"] = self.type
        node["path"] = self.path
        if hasattr(self, "timeoffset"):
            node["timeoffset"] = self.timeoffset


class TimeZoneEntry:
    def __init__(self):
        self.timezone = 0
        self.timeoffset = 0
        self.timeoffset_s = 0

    def load(self, node):
        self.timezone = node["timezone"]
        # read in timezone
        if isinstance(self.timezone, str):
            if self.timezone == "utc" or self.timezone == "UTC":
                self.timezone = 0
            elif self.timezone == "jst" or self.timezone == "JST":
                self.timezone = 9
            elif self.timezone == "CET" or self.timezone == "cet":
                self.timezone = 1
            else:
                try:
                    self.timezone = float(self.timezone)
                except ValueError:
                    Console.quit(
                        "Error: timezone",
                        self.timezone,
                        "in mission.yaml not recognised,",
                        " please enter value from UTC in",
                        " hours",
                    )

        self.timeoffset = node["timeoffset"]
        self.timeoffset_s = +self.timezone * 60 * 60 + self.timeoffset

    def write(self, node):
        node["timezone"] = self.timezone
        node["timeoffset"] = self.timeoffset


class ImageEntry(TimeZoneEntry):
    def __init__(self):
        super().__init__()
        self.format = ""
        self.cameras = []
        self.calibration = None
        self._empty = True

    def empty(self):
        return self._empty

    def load(self, node, version=1):
        super().load(node)
        self.format = node["format"]
        self._empty = False
        if version == 1:
            for camera in node["cameras"]:
                self.cameras.append(CameraEntry(camera))
            if "origin" not in node["cameras"][0]:
                # Assuming that the camera names in mission.yaml corresponds
                # to the frame names in vehicle.yaml
                for i, camera in enumerate(self.cameras):
                    camera.origin = camera.name
        else:
            self.cameras.append(CameraEntry())
            self.cameras.append(CameraEntry())
            if self.format == "seaxerocks_3":
                self.cameras.append(CameraEntry())
                self.cameras[0].name = "fore"
                self.cameras[0].origin = "camera1"
                self.cameras[0].type = "bayer_rggb"
                self.cameras[0].path = node["filepath"] + node["camera1"]
                self.cameras[0].timeoffset = 0.0
                self.cameras[1].name = "aft"
                self.cameras[1].origin = "camera2"
                self.cameras[1].type = "bayer_rggb"
                self.cameras[1].path = node["filepath"] + node["camera2"]
                self.cameras[1].timeoffset = 0.0
                self.cameras[2].name = "laser"
                self.cameras[2].origin = "camera3"
                self.cameras[2].type = "grayscale"
                self.cameras[2].path = node["filepath"] + node["camera3"]
                self.cameras[2].timeoffset = 0.0
            elif self.format == "acfr_standard":
                self.cameras[0].name = node["camera1"]
                self.cameras[0].origin = "camera1"
                self.cameras[0].type = "bayer_rggb"
                self.cameras[0].path = node["filepath"]
                self.cameras[0].timeoffset = 0.0
                self.cameras[1].name = node["camera2"]
                self.cameras[1].origin = "camera2"
                self.cameras[1].type = "bayer_rggb"
                self.cameras[1].path = node["filepath"]
                self.cameras[1].timeoffset = 0.0

    def write(self, node):
        super().write(node)
        node["format"] = self.format
        node["cameras"] = []
        for c in self.cameras:
            cam_dict = OrderedDict()
            c.write(cam_dict)
            node["cameras"].append(cam_dict)
        if "calibration" in node:
            calibration_dict = OrderedDict()
            self.calibration.write(calibration_dict)
            node["calibration"].append(calibration_dict)


class DefaultEntry(TimeZoneEntry):
    def __init__(self):
        super().__init__()
        self.format = ""
        self.filepath = ""
        self.filename = ""
        self.label = 0
        self.std_factor = 0.0
        self.std_offset = 0.0
        self._empty = True
        self.topic = None

    def empty(self):
        return self._empty

    def load(self, node):
        super().load(node)
        self._empty = False
        self.format = node["format"]
        if "filepath" in node:
            self.filepath = node["filepath"]
        if "filename" in node:
            self.filename = node["filename"]
        if "label" in node:
            self.label = node["label"]
        elif "id" in node:
            self.label = node["id"]
        if "std_factor" in node:
            self.std_factor = node["std_factor"]
        if "std_offset" in node:
            self.std_offset = node["std_offset"]
        if "origin" in node:
            self.origin = node["origin"]
        if "topic" in node:
            self.topic = node["topic"]

    def write(self, node):
        super().write(node)
        node["format"] = self.format
        if "origin" in node:
            node["origin"] = self.origin
        if "topic" in node:
            node["topic"] = self.topic
        node["filepath"] = self.filepath
        node["filename"] = self.filename
        node["label"] = self.label
        node["id"] = self.label
        node["std_factor"] = self.std_factor
        node["std_offset"] = self.std_offset

    def get_offset_s(self):
        if isinstance(self.timezone, str):
            if self.timezone == "utc" or self.timezone == "UTC":
                self.timezone_offset = 0
            elif self.timezone == "jst" or self.timezone == "JST":
                self.timezone_offset = 9
        else:
            try:
                self.timezone_offset = float(self.timezone)
            except ValueError:
                print(
                    "Error: timezone",
                    self.timezone,
                    "in mission.yaml not recognised, ",
                    "please enter value from UTC in hours",
                )
                return
            timeoffset = -self.timezone_offset * 60 * 60 + self.timeoffset
            return timeoffset


class Mission:
    """Mission class that parses and writes mission.yaml"""

    def __init__(self, filename=None):
        self.version = 0
        self.origin = OriginEntry()
        self.velocity = DefaultEntry()
        self.orientation = DefaultEntry()
        self.depth = DefaultEntry()
        self.altitude = DefaultEntry()
        self.usbl = DefaultEntry()
        self.tide = DefaultEntry()
        self.dead_reckoning = DefaultEntry()
        self.image = ImageEntry()
        self.filename = None

        if filename is None:
            return

        self.filename = filename

        try:
            # Check that mission.yaml and vehicle.yaml are consistent
            vehicle_file = filename.parent / "vehicle.yaml"
            vehicle_stream = vehicle_file.open("r")
            vehicle_data = yaml.safe_load(vehicle_stream)
        except FileNotFoundError:
            Console.error("The file vehicle.yaml could not be found at the location:")
            Console.error(vehicle_file)
            Console.error(
                "In order to load a mission.yaml file, a corresponding \
                vehicle.yaml files needs to be present in the same folder."
            )
            Console.quit("vehicle.yaml not provided")
        except PermissionError:
            Console.error("The file vehicle.yaml could not be opened at the location:")
            Console.error(vehicle_file)
            Console.error("Please make sure you have the correct access rights.")
            Console.quit("vehicle.yaml not provided")

        try:
            with filename.open("r") as stream:
                data = yaml.safe_load(stream)
                if "version" in data:
                    self.version = data["version"]
                self.origin.load(data["origin"])
                if "velocity" in data:
                    self.velocity.load(data["velocity"])
                    if "origin" not in data["velocity"]:
                        self.velocity.origin = "dvl"
                    if self.velocity.origin not in vehicle_data:
                        Console.error(
                            "The velocity sensor mounted at "
                            + self.velocity.origin
                            + " does not correspond to any frame in vehicle.yaml."  # noqa
                        )
                        error_and_exit()
                if "orientation" in data:
                    self.orientation.load(data["orientation"])
                    if "origin" not in data["orientation"]:
                        self.orientation.origin = "ins"
                    if self.orientation.origin not in vehicle_data:
                        Console.error(
                            "The orientation sensor mounted at "
                            + self.orientation.origin
                            + " does not correspond to any frame in vehicle.yaml."  # noqa
                        )
                        error_and_exit()
                if "depth" in data:
                    self.depth.load(data["depth"])
                    if "origin" not in data["depth"]:
                        self.depth.origin = "depth"
                    if self.depth.origin not in vehicle_data:
                        Console.error(
                            "The depth sensor mounted at "
                            + self.depth.origin
                            + " does not correspond to any frame in vehicle.yaml."  # noqa
                        )
                        error_and_exit()
                if "altitude" in data:
                    self.altitude.load(data["altitude"])
                    if "origin" not in data["altitude"]:
                        self.altitude.origin = "dvl"
                    if self.altitude.origin not in vehicle_data:
                        Console.error(
                            "The altitude sensor mounted at "
                            + self.altitude.origin
                            + " does not correspond to any frame in vehicle.yaml."  # noqa
                        )
                        error_and_exit()
                if "usbl" in data:
                    self.usbl.load(data["usbl"])
                    if "origin" not in data["usbl"]:
                        self.usbl.origin = "usbl"
                    if self.usbl.origin not in vehicle_data:
                        Console.error(
                            "The usbl sensor mounted at "
                            + self.usbl.origin
                            + " does not correspond to any frame in vehicle.yaml."  # noqa
                        )
                        error_and_exit()

                if "tide" in data:
                    self.tide.load(data["tide"])

                if "dead_reckoning" in data:
                    self.dead_reckoning.load(data["dead_reckoning"])

                if "image" in data:
                    self.image.load(data["image"], self.version)
                    for camera in self.image.cameras:
                        if camera.origin not in vehicle_data:
                            Console.error(
                                "The camera mounted at "
                                + camera.origin
                                + " does not correspond to any frame in vehicle.yaml."  # noqa
                            )
                            error_and_exit()

        except FileNotFoundError:
            Console.error("The file mission.yaml could not be found at the location:")
            Console.error(filename)
            Console.quit("mission.yaml not provided")
        except PermissionError:
            Console.error("The file mission.yaml could not be opened at the location:")
            Console.error(filename)
            Console.error("Please make sure you have the correct access rights.")
            Console.quit("mission.yaml not provided")

    def write_metadata(self, node):
        node["username"] = Console.get_username()
        node["date"] = Console.get_date()
        node["hostname"] = Console.get_hostname()
        node["version"] = Console.get_version()

    def write(self, filename):
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        if self.filename is not None:
            self.filename.copy(filename)
            with filename.open("a") as f:
                f.write("\n\nmetadata:\n")
                f.write("    username: " + str(Console.get_username()) + "\n")
                f.write("    date: " + str(Console.get_date()) + "\n")
                f.write("    hostname: " + str(Console.get_hostname()) + "\n")
                f.write("    firmware: " + str(Console.get_version()) + "\n")
        else:
            with filename.open("w") as f:
                mission_dict = OrderedDict()
                mission_dict["version"] = 1
                mission_dict["metadata"] = OrderedDict()
                mission_dict["origin"] = OrderedDict()
                self.write_metadata(mission_dict["metadata"])
                self.origin.write(mission_dict["origin"])
                if not self.velocity.empty():
                    mission_dict["velocity"] = OrderedDict()
                    self.velocity.write(mission_dict["velocity"])
                if not self.orientation.empty():
                    mission_dict["orientation"] = OrderedDict()
                    self.orientation.write(mission_dict["orientation"])
                if not self.depth.empty():
                    mission_dict["depth"] = OrderedDict()
                    self.depth.write(mission_dict["depth"])
                if not self.altitude.empty():
                    mission_dict["altitude"] = OrderedDict()
                    self.altitude.write(mission_dict["altitude"])
                if not self.usbl.empty():
                    mission_dict["usbl"] = OrderedDict()
                    self.usbl.write(mission_dict["usbl"])
                if not self.tide.empty():
                    mission_dict["tide"] = OrderedDict()
                    self.tide.write(mission_dict["tide"])
                if not self.dead_reckoning.empty():
                    mission_dict["dead_reckoning"] = OrderedDict()
                    self.dead_reckoning.write(mission_dict["dead_reckoning"])
                if not self.image.empty():
                    mission_dict["image"] = OrderedDict()
                    self.image.write(mission_dict["image"])
                yaml.dump(
                    mission_dict,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                )

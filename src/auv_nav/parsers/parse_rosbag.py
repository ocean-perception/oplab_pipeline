# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import os
from pathlib import Path

from auv_nav.sensors import (
    Altitude,
    BodyVelocity,
    Camera,
    Category,
    Depth,
    Orientation,
    Usbl,
)
from auv_nav.tools.time_conversions import read_timezone
from oplab import Console
from oplab.folder_structure import get_raw_folder

# fmt: off
ROSBAG_IS_AVAILABLE = False
try:
    import rosbag
    ROSBAG_IS_AVAILABLE = True
except ImportError:
    pass
# fmt: on


def rosbag_topic_worker(
    bagfile_list, wanted_topic, data_object, data_list, output_format, output_dir
):
    """Process a topic from a rosbag calling a method from an object

    Parameters
    ----------
    bagfile_list : list
        list of paths to rosbags
    wanted_topic : str
        Wanted topic
    data_object : object
        Object that has the data_method implemented
    data_method : std
        Name of the method to call (e.g. data_object.data_method(msg))
    data_list : list
        List of data output
    output_format : str
        Output format
    output_dir: str
        Output directory

    Returns
    -------
    list
        Processed data list
    """
    if wanted_topic is None:
        Console.quit("data_method for bagfile is not specified for topic", wanted_topic)
    for bagfile in bagfile_list:
        bag = rosbag.Bag(bagfile, "r")
        for topic, msg, _ in bag.read_messages(topics=[wanted_topic]):
            if topic == wanted_topic:
                func = getattr(data_object, "from_ros")
                type_str = str(type(msg))
                # rosbag library does not store a clean message type,
                # so we need to make it ourselves from a dirty string
                msg_type = type_str.split(".")[1][1:-2].replace("__", "/")
                func(msg, msg_type, output_dir)
                if data_object.valid():
                    data = data_object.export(output_format)
                    data_list.append(data)
    return data_list


def parse_rosbag(mission, vehicle, category, output_format, outpath):
    """Parse rosbag files

    Parameters
    ----------
    mission : Mission
        Mission object
    vehicle : Vehicle
        Vehicle object
    category : str
        Measurement category
    output_format : str
        Output format
    outpath : str
        Output path

    Returns
    -------
    list
        Measurement data list
    """
    if not ROSBAG_IS_AVAILABLE:
        Console.error("rosbag is not available")
        Console.error("install it with:")
        Console.error(
            "pip install --extra-index-url",
            "https://rospypi.github.io/simple/ rosbag",
        )
        Console.quit("rosbag is not available and required to parse ROS bagfiles.")

    # Get your data from a file using mission paths, for example
    depth_std_factor = mission.depth.std_factor
    velocity_std_factor = mission.velocity.std_factor
    velocity_std_offset = mission.velocity.std_offset
    altitude_std_factor = mission.altitude.std_factor
    usbl_std_factor = mission.usbl.std_factor
    usbl_std_offset = mission.usbl.std_offset

    # NED origin
    origin_latitude = mission.origin.latitude
    origin_longitude = mission.origin.longitude

    ins_heading_offset = vehicle.ins.yaw
    dvl_heading_offset = vehicle.dvl.yaw

    body_velocity = BodyVelocity(
        velocity_std_factor,
        velocity_std_offset,
        dvl_heading_offset,
    )
    orientation = Orientation(ins_heading_offset)
    depth = Depth(depth_std_factor)
    altitude = Altitude(altitude_std_factor)
    usbl = Usbl(
        usbl_std_factor,
        usbl_std_offset,
        latitude_reference=origin_latitude,
        longitude_reference=origin_longitude,
    )
    camera = Camera()
    camera.sensor_string = mission.image.cameras[0].name

    body_velocity.sensor_string = "rosbag"
    orientation.sensor_string = "rosbag"
    depth.sensor_string = "rosbag"
    altitude.sensor_string = "rosbag"
    usbl.sensor_string = "rosbag"

    # Adjust timezone offsets
    body_velocity.tz_offset_s = (
        read_timezone(mission.velocity.timezone) * 60 + mission.velocity.timeoffset
    )
    orientation.tz_offset_s = (
        read_timezone(mission.orientation.timezone) * 60
        + mission.orientation.timeoffset
    )
    depth.tz_offset_s = (
        read_timezone(mission.depth.timezone) * 60 + mission.depth.timeoffset
    )
    altitude.tz_offset_s = (
        read_timezone(mission.altitude.timezone) * 60 + mission.altitude.timeoffset
    )
    usbl.tz_offset_s = (
        read_timezone(mission.usbl.timezone) * 60 + mission.usbl.timeoffset
    )
    camera.tz_offset_s = (
        read_timezone(mission.image.timezone) * 60 + mission.image.timeoffset
    )

    data_list = []

    bagfile = None
    wanted_topic = None
    data_object = None
    filepath = None

    if category == Category.ORIENTATION:
        Console.info("... parsing orientation")
        filepath = get_raw_folder(mission.orientation.filepath)
        bagfile = mission.orientation.filename
        wanted_topic = mission.orientation.topic
        data_object = orientation
    elif category == Category.VELOCITY:
        Console.info("... parsing velocity")
        filepath = get_raw_folder(mission.velocity.filepath)
        bagfile = mission.velocity.filename
        wanted_topic = mission.velocity.topic
        data_object = body_velocity
    elif category == Category.DEPTH:
        Console.info("... parsing depth")
        filepath = get_raw_folder(mission.depth.filepath)
        bagfile = mission.depth.filename
        wanted_topic = mission.depth.topic
        data_object = depth
    elif category == Category.ALTITUDE:
        Console.info("... parsing altitude")
        filepath = get_raw_folder(mission.altitude.filepath)
        bagfile = mission.altitude.filename
        wanted_topic = mission.altitude.topic
        data_object = altitude
    elif category == Category.USBL:
        Console.info("... parsing position")
        filepath = get_raw_folder(mission.usbl.filepath)
        bagfile = mission.usbl.filename
        wanted_topic = mission.usbl.topic
        data_object = usbl
    elif category == Category.IMAGES:
        Console.info("... parsing images")
        filepath = get_raw_folder(mission.image.cameras[0].path)
        bagfile = "*.bag"
        wanted_topic = mission.image.cameras[0].topic
        data_object = camera
    else:
        Console.quit("Unknown category for ROS parser", category)

    bagfile_list = list(filepath.glob(bagfile))
    outpath = Path(outpath).parent
    data_list = rosbag_topic_worker(
        bagfile_list, wanted_topic, data_object, data_list, output_format, outpath
    )
    Console.info("... parsed " + str(len(data_list)) + " " + category + " entries")
    return data_list


def parse_rosbag_extracted_images(mission, vehicle, category, ftype, outpath):
    # parser meta data
    class_string = "measurement"
    frame_string = "body"
    category = "image"
    sensor_string = "rosbag_extracted_images"
    tolerance = 0.05  # stereo pair must be within 50ms of each other
    filepath = mission.image.cameras[0].path

    Console.info("... parsing " + sensor_string + " images")

    # determine file paths
    filepath = get_raw_folder(outpath / ".." / filepath)
    all_list = os.listdir(str(filepath))

    Console.info("Looking for images at", filepath)

    camera1_filename = [
        line for line in all_list if ".txt" not in line and "._" not in line
    ]
    camera2_filename = [
        line for line in all_list if ".txt" not in line and "._" not in line
    ]

    Console.info(
        "Found",
        len(camera1_filename),
        "files for camera1 and",
        len(camera2_filename),
        "for camera2",
    )

    data_list = []
    if ftype == "acfr":
        data_list = ""

    # example filename frame1631723390.279656550.png
    epoch_timestamp_camera1 = []
    epoch_timestamp_camera2 = []

    for i in range(len(camera1_filename)):
        epoch_timestamp = camera1_filename[i][5:-5]
        epoch_timestamp_camera1.append(str(epoch_timestamp))

    for i in range(len(camera2_filename)):
        epoch_timestamp = camera2_filename[i][5:-5]
        epoch_timestamp_camera2.append(str(epoch_timestamp))

    for i in range(len(camera1_filename)):
        values = []
        for j in range(len(camera2_filename)):
            values.append(
                abs(
                    float(epoch_timestamp_camera1[i])
                    - float(epoch_timestamp_camera2[j])
                )
            )
        (sync_difference, sync_pair) = min((v, k) for k, v in enumerate(values))
        if sync_difference > tolerance:
            # Skip the pair
            continue
        if ftype == "oplab":
            data = {
                "epoch_timestamp": float(epoch_timestamp_camera1[i]),
                "class": class_string,
                "sensor": sensor_string,
                "frame": frame_string,
                "category": category,
                "camera1": [
                    {
                        "epoch_timestamp": float(epoch_timestamp_camera1[i]),
                        "filename": str(filepath) + "/" + str(camera1_filename[i]),
                    }
                ],
                "camera2": [
                    {
                        "epoch_timestamp": float(epoch_timestamp_camera2[sync_pair]),
                        "filename": str(filepath)
                        + "/"
                        + str(camera2_filename[sync_pair]),
                    }
                ],
            }
            data_list.append(data)
        if ftype == "acfr":
            data = (
                "VIS: "
                + str(float(epoch_timestamp_camera1[i]))
                + " ["
                + str(float(epoch_timestamp_camera1[i]))
                + "] "
                + str(camera1_filename[i])
                + " exp: 0\n"
            )
            data_list += data
            data = (
                "VIS: "
                + str(float(epoch_timestamp_camera2[sync_pair]))
                + " ["
                + str(float(epoch_timestamp_camera2[sync_pair]))
                + "] "
                + str(camera2_filename[sync_pair])
                + " exp: 0\n"
            )
            data_list += data

    return data_list

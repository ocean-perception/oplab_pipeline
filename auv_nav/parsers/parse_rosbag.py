# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import os

from auv_nav.sensors import (
    Altitude,
    BodyVelocity,
    Category,
    Depth,
    Orientation,
    Usbl,
)
from oplab import Console
from oplab.folder_structure import get_raw_folder

# fmt: off

ROSBAG_IS_AVAILABLE = False

try:
    import rosbag

    ROSBAG_IS_AVAILABLE = True
except ImportError:
    Console.warn("rosbag is not available")
    Console.warn("install it with:")
    Console.warn(
        "pip install --extra-index-url",
        "https://rospypi.github.io/simple/ rosbag",
    )

# fmt: on

def rosbag_topic_worker(
    bagfile, wanted_topic, data_object, data_method, data_list, output_format
):
    """Process a topic from a rosbag calling a method from an object

    Parameters
    ----------
    bag : rosbag
        python rosbag object
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

    Returns
    -------
    list
        Processed data list
    """
    if wanted_topic is None:
        Console.quit(
            "data_method for bagfile is not specified for topic", wanted_topic
        )
    bag = rosbag.Bag(bagfile, "r")
    for topic, msg, _ in bag.read_messages(topics=[wanted_topic]):
        if topic == wanted_topic:
            func = getattr(data_object, data_method)
            func(msg)
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
        Console.error("Rosbag is not available. Please install it by calling")
        Console.error(
            "pip install --extra-index-url https://rospypi.github.io/simple/ rosbag"
        )
        Console.quit("Parser rosbag not available")

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
        velocity_std_factor, velocity_std_offset, dvl_heading_offset,
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

    body_velocity.sensor_string = "rosbag"
    orientation.sensor_string = "rosbag"
    depth.sensor_string = "rosbag"
    altitude.sensor_string = "rosbag"
    usbl.sensor_string = "rosbag"

    data_list = []

    method = None
    bagfile = None
    wanted_topic = None
    data_object = None
    filepath = None

    if category == Category.ORIENTATION:
        Console.info("... parsing orientation")
        filepath = mission.orientation.filepath
        bagfile = mission.orientation.filename
        wanted_topic = mission.orientation.topic
        data_object = orientation
        if wanted_topic == "/turbot/navigator/imu_raw":
            method = "from_ros_imu"
        else:
            Console.quit("Unknown ORIENTATION topic", wanted_topic)
    elif category == Category.VELOCITY:
        Console.info("... parsing velocity")
        filepath = mission.velocity.filepath
        bagfile = mission.velocity.filename
        wanted_topic = mission.velocity.topic
        data_object = body_velocity
        if wanted_topic == "/turbot/teledyne_explorer_dvl/data":
            method = "from_ros_teledyne_explorer_dvl_data"
        else:
            Console.quit("Unknown VELOCITY topic", wanted_topic)
    elif category == Category.DEPTH:
        Console.info("... parsing depth")
        filepath = mission.depth.filepath
        bagfile = mission.depth.filename
        wanted_topic = mission.depth.topic
        data_object = depth
        if wanted_topic == "/turbot/adis_imu/depth":
            method = "from_ros_pose_with_covariance_stamped"
        else:
            Console.quit("Unknown DEPTH topic", wanted_topic)
    elif category == Category.ALTITUDE:
        Console.info("... parsing altitude")
        filepath = mission.altitude.filepath
        bagfile = mission.altitude.filename
        wanted_topic = mission.altitude.topic
        data_object = altitude
        if wanted_topic == "/turbot/teledyne_explorer_dvl/data":
            method = "from_ros_teledyne_explorer_dvl_data"
        else:
            Console.quit("Unknown ALTITUDE topic", wanted_topic)
    elif category == Category.USBL:
        Console.info("... parsing position")
        filepath = mission.usbl.filepath
        bagfile = mission.usbl.filename
        wanted_topic = mission.usbl.topic
        data_object = usbl
        if wanted_topic == "/turbot/modem_delayed":
            method = "from_ros_pose_with_covariance_stamped"
        else:
            Console.quit("Unknown POSITION topic", wanted_topic)
    else:
        Console.quit("Unknown category", category)

    bagfile = get_raw_folder(outpath / ".." / filepath / bagfile)
    data_list = rosbag_topic_worker(
        bagfile, wanted_topic, data_object, method, data_list, output_format,
    )

    Console.info(
        "... parsed " + str(len(data_list)) + " " + category + " entries"
    )

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
        (sync_difference, sync_pair) = min(
            (v, k) for k, v in enumerate(values)
        )
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
                        "filename": str(filepath)
                        + "/"
                        + str(camera1_filename[i]),
                    }
                ],
                "camera2": [
                    {
                        "epoch_timestamp": float(
                            epoch_timestamp_camera2[sync_pair]
                        ),
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

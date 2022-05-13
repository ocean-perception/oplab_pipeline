# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

# Scripts to parse sexerocks image acquisition data

# Author: Blair Thornton
# Date: 31/08/2017


from auv_nav.tools.time_conversions import date_time_to_epoch, epoch_to_day
from oplab import Console, get_raw_folder


def parse_seaxerocks_images(mission, vehicle, category, ftype, outpath):
    data_list = []
    if ftype == "acfr":
        data_list = ""

    # parser meta data
    class_string = "measurement"
    frame_string = "body"
    category_stereo = "image"
    category_laser = "laser"
    sensor_string = "seaxerocks_3"

    timezone = mission.image.timezone
    timeoffset = mission.image.timeoffset
    camera1_filepath = mission.image.cameras[0].path
    camera2_filepath = mission.image.cameras[1].path
    camera3_filepath = mission.image.cameras[2].path
    camera1_label = mission.image.cameras[0].name
    camera2_label = mission.image.cameras[1].name
    camera3_label = mission.image.cameras[2].name
    camera1_timeoffset = mission.image.cameras[0].timeoffset
    camera2_timeoffset = mission.image.cameras[1].timeoffset
    camera3_timeoffset = mission.image.cameras[2].timeoffset

    epoch_timestamp_stereo = []
    epoch_timestamp_laser = []
    epoch_timestamp_camera1 = []
    epoch_timestamp_camera2 = []
    epoch_timestamp_camera3 = []
    stereo_index = []
    laser_index = []
    camera1_index = []
    camera2_index = []
    camera3_index = []
    camera1_filename = []
    camera2_filename = []
    camera3_filename = []

    camera1_serial = list(camera1_label)
    camera2_serial = list(camera2_label)
    camera3_serial = list(camera3_label)

    for i in range(1, len(camera1_label)):
        if camera1_label[i] == "/":
            camera1_serial[i] = "_"

    for i in range(1, len(camera2_label)):
        if camera2_label[i] == "/":
            camera2_serial[i] = "_"

    for i in range(1, len(camera3_label)):
        if camera3_label[i] == "/":
            camera3_serial[i] = "_"

    camera1_serial = "".join(camera1_serial)
    camera2_serial = "".join(camera2_serial)
    camera3_serial = "".join(camera3_serial)

    i = 0
    # read in timezone
    # TODO change ALL timezones to integers
    if isinstance(timezone, str):
        if timezone == "utc" or timezone == "UTC":
            timezone_offset = 0
        elif timezone == "jst" or timezone == "JST":
            timezone_offset = 9
    else:
        try:
            timezone_offset = float(timezone)
        except ValueError:
            print(
                "Error: timezone",
                timezone,
                "in mission.cfg not recognised, \
                  please enter value from UTC in hours",
            )
            return

    # convert to seconds from utc
    # timeoffset = -timezone_offset*60*60 + timeoffset

    Console.info("  Parsing " + sensor_string + " images...")

    cam1_path = get_raw_folder(outpath / ".." / camera1_filepath / "..")
    cam1_filetime = cam1_path / "FileTime.csv"

    with cam1_filetime.open("r", encoding="utf-8", errors="ignore") as filein:
        for line in filein.readlines():
            stereo_index_timestamps = line.strip().split(",")

            index_string = stereo_index_timestamps[0]
            date_string = stereo_index_timestamps[1]
            time_string = stereo_index_timestamps[2]
            ms_time_string = stereo_index_timestamps[3]

            # read in date
            if date_string != "date":  # ignore header
                stereo_index.append(index_string)
                if len(date_string) != 8:
                    Console.warn(
                        "Date string ({}) in FileTime.csv file has "
                        "unexpected length. Expected length: 8.".format(date_string)
                    )
                yyyy = int(date_string[0:4])
                mm = int(date_string[4:6])
                dd = int(date_string[6:8])

                # read in time
                if len(time_string) != 6:
                    Console.warn(
                        "Time string ({}) in FileTime.csv file has "
                        "unexpected length. Expected length: 6.".format(time_string)
                    )
                hour = int(time_string[0:2])
                mins = int(time_string[2:4])
                secs = int(time_string[4:6])
                msec = int(ms_time_string[0:3])

                epoch_time = date_time_to_epoch(
                    yyyy, mm, dd, hour, mins, secs, timezone_offset
                )

                epoch_timestamp_stereo.append(
                    float(epoch_time + msec / 1000 + timeoffset)
                )

    camera1_list = ["{}.raw".format(i) for i in stereo_index]
    camera2_list = ["{}.raw".format(i) for i in stereo_index]

    for i in range(len(camera1_list)):
        camera1_image = camera1_list[i].split(".")
        camera2_image = camera2_list[i].split(".")

        camera1_index.append(camera1_image[0])
        camera2_index.append(camera2_image[0])

    j = 0
    for i in range(len(camera1_list)):
        # find corresponding timestamp even if some images are deleted
        if camera1_index[i] == stereo_index[j]:
            epoch_timestamp_camera1.append(
                epoch_timestamp_stereo[j] + camera1_timeoffset
            )
            epoch_timestamp_camera2.append(
                epoch_timestamp_stereo[j] + camera2_timeoffset
            )
            if ftype == "acfr":
                date1 = epoch_to_day(epoch_timestamp_stereo[0] + camera1_timeoffset)
                date2 = epoch_to_day(epoch_timestamp_stereo[0] + camera2_timeoffset)
                camera1_filename.append(
                    "sx3_"
                    + date1[2:4]
                    + date1[5:7]
                    + date1[8:10]
                    + "_image"
                    + str(camera1_index[i])
                    + "_FC.png"
                )
                camera2_filename.append(
                    "sx3_"
                    + date2[2:4]
                    + date2[5:7]
                    + date2[8:10]
                    + "_image"
                    + str(camera2_index[i])
                    + "_AC.png"
                )
            j = j + 1
        elif stereo_index[j] > camera1_index[i]:
            j = j + 1
        else:
            j = j - 1

    if ftype == "oplab":
        camera1_filename = [line for line in camera1_list]
        camera2_filename = [line for line in camera2_list]

    for i in range(len(camera1_list)):
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
                + str(float(epoch_timestamp_camera2[i]))
                + " ["
                + str(float(epoch_timestamp_camera2[i]))
                + "] "
                + str(camera2_filename[i])
                + " exp: 0\n"
            )
            data_list += data

        if ftype == "oplab":
            data = {
                "epoch_timestamp": float(epoch_timestamp_camera1[i]),
                "class": class_string,
                "sensor": sensor_string,
                "frame": frame_string,
                "category": category_stereo,
                "camera1": [
                    {
                        "epoch_timestamp": float(epoch_timestamp_camera1[i]),
                        "serial": camera1_serial,
                        "filename": str(camera1_filepath + "/" + camera1_filename[i]),
                    }
                ],
                "camera2": [
                    {
                        "epoch_timestamp": float(epoch_timestamp_camera2[i]),
                        "serial": camera2_serial,
                        "filename": str(camera2_filepath + "/" + camera2_filename[i]),
                    }
                ],
            }
            data_list.append(data)

    cam3_path = get_raw_folder(outpath / ".." / camera3_filepath)
    cam3_filetime = cam3_path / "FileTime.csv"
    with cam3_filetime.open("r", encoding="utf-8", errors="ignore") as filein:
        for line in filein.readlines():
            laser_index_timestamps = line.strip().split(",")

            if len(laser_index_timestamps) < 4:
                Console.warn("The laser FileTime.csv is apparently corrupt...")
                continue
            index_string = laser_index_timestamps[0]
            date_string = laser_index_timestamps[1]
            time_string = laser_index_timestamps[2]
            ms_time_string = laser_index_timestamps[3]

            # read in date
            if date_string != "date":  # ignore header
                laser_index.append(index_string)

                yyyy = int(date_string[0:4])
                mm = int(date_string[4:6])
                dd = int(date_string[6:8])

                # read in time
                hour = int(time_string[0:2])
                mins = int(time_string[2:4])
                secs = int(time_string[4:6])
                msec = int(ms_time_string[0:3])

                epoch_time = date_time_to_epoch(
                    yyyy, mm, dd, hour, mins, secs, timezone_offset
                )

                epoch_timestamp_laser.append(
                    float(epoch_time + msec / 1000 + timeoffset + camera3_timeoffset)
                )

    # try use pandas for all parsers, should be faster
    camera3_list = ["{}".format(i) for i in laser_index]

    # The LM165 images are saved either as jpg or as tif, and are split into
    # subfolders either at every 1000 or every 10000 images. Find out which
    # convention is used in current dataset by looking at the files.
    if len(camera3_list) > 0:
        s, extension = determine_extension_and_images_per_folder(
            cam3_path, camera3_list, camera3_label
        )

    for i in range(len(camera3_list)):
        camera3_filename.append(
            "{}/image{}.{}".format(
                camera3_list[i][s : (s + 3)],
                camera3_list[i],
                extension,  # noqa: E203
            )
        )
        camera3_index.append(camera3_list[i])

    j = 0
    # original comment: find corresponding timestamp even if some images are
    # deleted
    for i in range(len(camera3_filename)):
        if camera3_index[i] == laser_index[j]:
            epoch_timestamp_camera3.append(epoch_timestamp_laser[j])
            j = j + 1
        # Jin: incomplete? it means that laser data is missing for this image
        # file, so no epoch_timestamp data, and do what when this happens?
        elif laser_index[j] > camera3_index[i]:
            j = j + 1
        else:
            # Jin: incomplete and possibly wrong? it means that this laser
            # data is extra, with no accompanying image file, so it should be
            # j+1 till index match?
            j = j - 1

        if ftype == "oplab":
            data = {
                "epoch_timestamp": float(epoch_timestamp_camera3[i]),
                "class": class_string,
                "sensor": sensor_string,
                "frame": frame_string,
                "category": category_laser,
                "serial": camera3_serial,
                "filename": camera3_filepath + "/" + str(camera3_filename[i]),
            }
            data_list.append(data)

    Console.info("  ...done parsing " + sensor_string + " images.")

    return data_list


def determine_extension_and_images_per_folder(folder_path, image_list, label):
    """Determine filename extension and number of images per subfolder

    The number of images per subfolder dertermines how the subfolders are
    named. The subfolder name is 3 letters long and either starts from the
    first (index = 0) or the second (index = 1) digit of the image number.

    :param folder_path: Path where images are stored
    :type  folder_path: pathlib.Path
    :param image_list:  list of image 7-digit zeropadded image numbers
    :type  image_list:  list of str
    :param label:       Camera label
    :type  label:       str

    :returns:
        -index_start_of_folder_name (`int`) - Index where subfolder name starts
        -extension (`str`) - Filename extension of images ("jpg" or "tif")
    """
    Console.info(
        "    Determine filename extension and images per subfolder "
        "of camera {}...".format(label)
    )

    if build_image_path(folder_path, image_list[-1], 0, "jpg").is_file():
        index_start_of_folder_name = 0
        extension = "jpg"
        Console.info(
            '    ...Filename extension: "{}", 10000 images '
            "per subfolder.".format(extension)
        )
    elif build_image_path(folder_path, image_list[-1], 1, "jpg").is_file():
        index_start_of_folder_name = 1
        extension = "jpg"
        Console.info(
            '    ...Filename extension: "{}", 1000 images '
            "per subfolder.".format(extension)
        )
    elif build_image_path(folder_path, image_list[-1], 0, "tif").is_file():
        index_start_of_folder_name = 0
        extension = "tif"
        Console.info(
            '    ...Filename extension: "{}", 10000 images '
            "per subfolder.".format(extension)
        )
    elif build_image_path(folder_path, image_list[-1], 1, "tif").is_file():
        index_start_of_folder_name = 1
        extension = "tif"
        Console.info(
            '    ...Filename extension: "{}", 1000 images '
            "per subfolder.".format(extension)
        )
    else:
        index_start_of_folder_name = 0
        extension = "jpg"
        Console.warn(
            "    ...Did not find images from camera {} in {}. "
            'Default to using extension "{}" and 10000 images per '
            "subfolder.".format(label, folder_path, extension)
        )

    return index_start_of_folder_name, extension


def build_image_path(folder_path, image_number, s, extension):
    """Build path of image file

    Build the path to an image file given the parent folder, an image nuber,
    the index of where the subfolder name is taken from in the image number,
    and the filename extension of the image
    """
    return folder_path / "{}/image{}.{}".format(
        image_number[s : s + 3], image_number, extension  # noqa: E203
    )

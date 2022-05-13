# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

# Scripts to parse Jamstec USBL dump data.

# Author: Blair Thornton
# Date: 14/02/2018

import math

from auv_nav.tools.latlon_wgs84 import latlon_to_metres, metres_to_latlon
from auv_nav.tools.time_conversions import date_time_to_epoch, read_timezone
from oplab import Console, get_raw_folder


def parse_usbl_dump(mission, vehicle, category, ftype, outpath):
    # parser meta data
    class_string = "measurement"
    sensor_string = "jamstec_usbl"
    frame_string = "inertial"

    # gaps std models
    distance_std_factor = mission.usbl.std_factor
    distance_std_offset = mission.usbl.std_offset

    timezone = mission.usbl.timezone
    timeoffset = mission.usbl.timeoffset
    filepath = mission.usbl.filepath
    filename = mission.usbl.filename
    label = mission.usbl.label

    filepath = get_raw_folder(outpath / ".." / filepath)

    latitude_reference = mission.origin.latitude
    longitude_reference = mission.origin.longitude

    # read in timezone
    timezone_offset = read_timezone(timezone)

    # convert to seconds from utc
    # timeoffset = -timezone_offset*60*60 + timeoffset

    # extract data from files
    Console.info("... parsing usbl dump")
    data_list = []
    if ftype == "acfr":
        data_list = ""
    usbl_file = filepath / filename
    with usbl_file.open("r", encoding="utf-8", errors="ignore") as filein:

        for line in filein.readlines():
            line_split = line.strip().split(",")

            if line_split[2] == label:

                date = line_split[0].split("-")

                # read in date
                yyyy = int(date[0])
                mm = int(date[1])
                dd = int(date[2])

                timestamp = line_split[1].split(":")

                # read in time
                hour = int(timestamp[0])
                mins = int(timestamp[1])
                secs = int(timestamp[2])
                msec = 0

                epoch_time = date_time_to_epoch(
                    yyyy, mm, dd, hour, mins, secs, timezone_offset
                )
                # dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
                # time_tuple = dt_obj.timetuple()
                # epoch_time = time.mktime(time_tuple)
                epoch_timestamp = epoch_time + msec / 1000 + timeoffset

                if line_split[6] != "":

                    # get position
                    latitude_full = line_split[6].split(" ")
                    latitude_list = latitude_full[0].split("-")
                    latitude_degrees = int(latitude_list[0])
                    latitude_minutes = float(latitude_list[1])

                    if latitude_full[1] != "N":
                        latitude_degrees = latitude_degrees * -1  # southern hemisphere

                    latitude = latitude_degrees + latitude_minutes / 60

                    longitude_full = line_split[7].split(" ")
                    longitude_list = longitude_full[0].split("-")
                    longitude_degrees = int(longitude_list[0])
                    longitude_minutes = float(longitude_list[1])

                    if longitude_full[1] != "E":
                        longitude_degrees = (
                            longitude_degrees * -1
                        )  # southern hemisphere

                    longitude = longitude_degrees + longitude_minutes / 60

                    depth_full = line_split[8].split("=")
                    depth = float(depth_full[1])

                    distance_full = line_split[11].split("=")
                    distance = float(distance_full[1])

                    distance_std = distance_std_factor * distance + distance_std_offset

                    lateral_distance_full = line_split[9].split("=")
                    lateral_distance = float(lateral_distance_full[1])

                    bearing_full = line_split[10].split("=")
                    bearing = float(bearing_full[1])

                    # determine uncertainty in terms of latitude and longitude
                    latitude_offset, longitude_offset = metres_to_latlon(
                        latitude, longitude, distance_std, distance_std
                    )

                    latitude_std = latitude - latitude_offset
                    longitude_std = longitude - longitude_offset

                    # calculate in metres from reference
                    eastings_ship = 0
                    northings_ship = 0
                    latitude_ship = 0
                    longitude_ship = 0
                    heading_ship = 0

                    lateral_distance_target, bearing_target = latlon_to_metres(
                        latitude,
                        longitude,
                        latitude_reference,
                        longitude_reference,
                    )
                    eastings_target = (
                        math.sin(bearing_target * math.pi / 180.0)
                        * lateral_distance_target
                    )
                    northings_target = (
                        math.cos(bearing_target * math.pi / 180.0)
                        * lateral_distance_target
                    )

                    if ftype == "oplab":
                        data = {
                            "epoch_timestamp": float(epoch_timestamp),
                            "class": class_string,
                            "sensor": sensor_string,
                            "frame": frame_string,
                            "category": category,
                            "data_ship": [
                                {
                                    "latitude": float(latitude_ship),
                                    "longitude": float(longitude_ship),
                                },
                                {
                                    "northings": float(northings_ship),
                                    "eastings": float(eastings_ship),
                                },
                                {"heading": float(heading_ship)},
                            ],
                            "data_target": [
                                {
                                    "latitude": float(latitude),
                                    "latitude_std": float(latitude_std),
                                },
                                {
                                    "longitude": float(longitude),
                                    "longitude_std": float(longitude_std),
                                },
                                {
                                    "northings": float(northings_target),
                                    "northings_std": float(distance_std),
                                },
                                {
                                    "eastings": float(eastings_target),
                                    "eastings_std": float(distance_std),
                                },
                                {
                                    "depth": float(depth),
                                    "depth_std": float(distance_std),
                                },
                                {"distance_to_ship": float(distance)},
                            ],
                        }
                        data_list.append(data)

                    if ftype == "acfr":
                        data = (
                            "SSBL_FIX: "
                            + str(float(epoch_timestamp))
                            + " ship_x: "
                            + str(float(northings_ship))
                            + " ship_y: "
                            + str(float(eastings_ship))
                            + " target_x: "
                            + str(float(northings_target))
                            + " target_y: "
                            + str(float(eastings_target))
                            + " target_z: "
                            + str(float(depth))
                            + " target_hr: "
                            + str(float(lateral_distance))
                            + " target_sr: "
                            + str(float(distance))
                            + " target_bearing: "
                            + str(float(bearing))
                            + "\n"
                        )
                        data_list += data

    return data_list

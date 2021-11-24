# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import math
import os
from pathlib import Path

import pandas as pd

from auv_nav.tools.interpolate import interpolate
from auv_nav.tools.latlon_wgs84 import latlon_to_metres
from auv_nav.tools.time_conversions import date_time_to_epoch
from oplab import Console, get_raw_folder


class HyBisPos:
    """Class to parse and store HyBis position data."""

    def __init__(
        self,
        roll,
        pitch,
        heading,
        depth,
        altitude,
        lon,
        lat,
        date=0,
        timestr=0,
        stamp=0,
    ):
        if date != 0:
            self.epoch_timestamp = self.convert(date, timestr)
        else:
            self.epoch_timestamp = stamp
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.heading = float(heading)
        self.depth = float(depth)
        self.altitude = float(altitude)
        self.longitude = self.convert_latlon(lon)
        self.latitude = self.convert_latlon(lat)

    def convert_latlon(self, latlon):
        # 5950.89694N, 00703.39736W
        final = 0
        if len(latlon) == 11:
            final = 1
        deg = float(latlon[0 : 3 - final])  # noqa
        minutes = float(latlon[3 - final : 11 - final])  # noqa
        zone = latlon[11 - final]
        decdeg = deg + minutes / 60.0
        if zone == "W" or zone == "S":
            return -decdeg
        else:
            return decdeg

    def convert(self, date, timestr):
        yyyy = int(date[6:10])
        mm = int(date[3:5])
        dd = int(date[0:2])
        hour = int(timestr[0:2])
        mins = int(timestr[3:5])
        secs = int(timestr[6:8])
        if hour < 0:
            hour = 0
            mins = 0
            secs = 0
        epoch_time = date_time_to_epoch(yyyy, mm, dd, hour, mins, secs, 0)
        return epoch_time


class HybisParser:
    """Class to parse and write HyBis position data."""

    def __init__(self, navigation_file, image_path, reference_lat, reference_lon):
        navigation_file = get_raw_folder(navigation_file)
        image_path = Path(image_path)
        image_path = get_raw_folder(image_path)

        # extract data from files
        df = pd.read_csv(navigation_file, skipinitialspace=True)
        date = list(df["Date "])
        timestr = list(df["Time"])
        roll = list(df["Roll"])
        pitch = list(df["Pitch"])
        heading = list(df["Heading"])
        depth = list(df["Pressure"])
        altitude = list(df["Altitude"])
        lon = list(df["Hybis Long"])
        lat = list(df["Hybis Lat"])

        Console.info("Found " + str(len(df)) + " navigation records!")

        hybis_vec = []
        for i in range(len(df)):
            if len(lon[i]) < 11:
                continue
            p = HyBisPos(
                roll[i],
                pitch[i],
                heading[i],
                depth[i],
                altitude[i],
                lon[i],
                lat[i],
                date[i],
                timestr[i],
            )
            hybis_vec.append(p)

        for i in range(len(hybis_vec) - 1):
            if hybis_vec[i].altitude == 0:
                hybis_vec[i].altitude = interpolate(
                    hybis_vec[i].epoch_timestamp,
                    hybis_vec[i - 1].epoch_timestamp,
                    hybis_vec[i + 1].epoch_timestamp,
                    hybis_vec[i - 1].altitude,
                    hybis_vec[i + 1].altitude,
                )
            if hybis_vec[i].depth == 0:
                hybis_vec[i].depth = interpolate(
                    hybis_vec[i].epoch_timestamp,
                    hybis_vec[i - 1].epoch_timestamp,
                    hybis_vec[i + 1].epoch_timestamp,
                    hybis_vec[i - 1].depth,
                    hybis_vec[i + 1].depth,
                )

        if reference_lon == 0 or reference_lat == 0:
            i = 0
            while hybis_vec[i].latitude == 0:
                i += 1
            latitude_ref = hybis_vec[i].latitude
            longitude_ref = hybis_vec[i].longitude
        else:
            latitude_ref = reference_lat
            longitude_ref = reference_lon

        self.data = [
            "image_number,",
            "northing [m],",
            "easting [m],",
            "depth [m],",
            "roll [deg],",
            "pitch [deg],",
            "heading [deg],",
            "altitude [m],",
            "timestamp,",
            "latitude [deg],",
            "longitude [deg],",
            "relative_path\n",
        ]

        image_list = sorted(os.listdir(str(image_path)))
        Console.info("Found " + str(len(image_list)) + " images!")
        Console.info("Interpolating...")
        for k, filename in enumerate(image_list):
            modification_time = os.stat(str(image_path) + "/" + filename).st_mtime
            filename = str(image_path) + "/" + filename

            i = 0
            while (
                i < len(hybis_vec) - 2
                and hybis_vec[i].epoch_timestamp < modification_time
            ):
                i += 1

            latitude = interpolate(
                modification_time,
                hybis_vec[i].epoch_timestamp,
                hybis_vec[i + 1].epoch_timestamp,
                hybis_vec[i].latitude,
                hybis_vec[i + 1].latitude,
            )
            longitude = interpolate(
                modification_time,
                hybis_vec[i].epoch_timestamp,
                hybis_vec[i + 1].epoch_timestamp,
                hybis_vec[i].longitude,
                hybis_vec[i + 1].longitude,
            )
            depth = interpolate(
                modification_time,
                hybis_vec[i].epoch_timestamp,
                hybis_vec[i + 1].epoch_timestamp,
                hybis_vec[i].depth,
                hybis_vec[i + 1].depth,
            )
            roll = interpolate(
                modification_time,
                hybis_vec[i].epoch_timestamp,
                hybis_vec[i + 1].epoch_timestamp,
                hybis_vec[i].roll,
                hybis_vec[i + 1].roll,
            )
            pitch = interpolate(
                modification_time,
                hybis_vec[i].epoch_timestamp,
                hybis_vec[i + 1].epoch_timestamp,
                hybis_vec[i].pitch,
                hybis_vec[i + 1].pitch,
            )
            heading = interpolate(
                modification_time,
                hybis_vec[i].epoch_timestamp,
                hybis_vec[i + 1].epoch_timestamp,
                hybis_vec[i].heading,
                hybis_vec[i + 1].heading,
            )
            altitude = interpolate(
                modification_time,
                hybis_vec[i].epoch_timestamp,
                hybis_vec[i + 1].epoch_timestamp,
                hybis_vec[i].altitude,
                hybis_vec[i + 1].altitude,
            )

            lateral_distance, bearing = latlon_to_metres(
                latitude, longitude, latitude_ref, longitude_ref
            )
            eastings = math.sin(bearing * math.pi / 180.0) * lateral_distance
            northings = math.cos(bearing * math.pi / 180.0) * lateral_distance

            msg = (
                str(k)
                + ","
                + str(northings)
                + ","
                + str(eastings)
                + ","
                + str(depth)
                + ","
                + str(roll)
                + ","
                + str(pitch)
                + ","
                + str(heading)
                + ","
                + str(altitude)
                + ","
                + str(modification_time)
                + ","
                + str(latitude)
                + ","
                + str(longitude)
                + ","
                + str(filename)
                + "\n"
            )
            self.data.append(msg)
        self.start_epoch = hybis_vec[0].epoch_timestamp
        self.finish_epoch = hybis_vec[-1].epoch_timestamp

    def write(self, output_file):
        Console.info("Writing output to " + str(output_file))
        output_file = Path(output_file)
        with output_file.open("w", encoding="utf-8") as fileout:
            for line in self.data:
                fileout.write(str(line))

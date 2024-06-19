# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Copyright (c) 2021, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import time
from math import isnan

import pandas as pd

from auv_nav.sensors import Altitude, BodyVelocity, Category, Depth, Orientation, Usbl
from oplab import Console, get_raw_folder
from datetime import datetime

def parse_from_csv(mission, vehicle, category, ftype, outpath):
    # parser meta data
    sensor_string = "data_process_from_csv"
    category = category
    output_format = ftype

    # There are two data sources: raw data (nav/hypack) and processed data (nav/posfilter)
    # If the category is USBL then we used the processed data, otherwise we use the raw data
    if category == Category.USBL:
        filename = mission.usbl.filename
        filepath = mission.usbl.filepath
    elif category == Category.VELOCITY:
        filename = mission.velocity.filename
        filepath = mission.velocity.filepath
    elif category == Category.DEPTH:
        filename = mission.depth.filename
        filepath = mission.depth.filepath
    elif category == Category.ALTITUDE:
        filename = mission.altitude.filename
        filepath = mission.altitude.filepath
    elif category == Category.ORIENTATION:
        filename = mission.orientation.filename
        filepath = mission.orientation.filepath
    else:
        Console.quit("Unsupported Category format!")

    # ALR std models
    depth_std_factor = mission.depth.std_factor
    velocity_std_factor = mission.velocity.std_factor
    velocity_std_offset = mission.velocity.std_offset
    orientation_std_offset = mission.orientation.std_offset
    altitude_std_factor = mission.altitude.std_factor
    # usbl_std_factor = mission.usbl.std_factor
    headingoffset = vehicle.dvl.yaw

    orientation = Orientation(headingoffset, orientation_std_offset)
    depth = Depth(depth_std_factor)
    altitude = Altitude(altitude_std_factor)
    velocity = BodyVelocity(velocity_std_factor=velocity_std_factor, velocity_std_offset=velocity_std_offset)
    # usbl = Usbl(usbl_std_factor)

    orientation.sensor_string = sensor_string
    depth.sensor_string = sensor_string
    altitude.sensor_string = sensor_string
    velocity.sensor_string =sensor_string

    path = get_raw_folder(outpath / ".." / filepath/filename)

    data_list = []
    # TODO: this is the headers of csv we current process. we could correct following codes to process other csvs with different headers
    # 'Image_Name' 'path time' 'capture time' 'altitude' 'depth' 'heading' 'Lat' 'latDec' 'Lon' 'lonDec' 'pitch' 'roll' 'surge' 'sway'

    # make sure the following string is the same with csv columns
    str_epoch_timestamp = "time"
    str_roll = "roll"
    str_pitch = "pitch"
    str_yaw = "heading"
    str_depth = "depth"
    str_altitude ="altitude"
    str_surge = "surge"
    str_sway = "sway"

    # Load the data from CSV file with well-known headers
    mission_data = pd.read_csv(str(path))

    mission_data["epoch_timestamp"] = mission_data.apply(
        lambda row: datetime.strptime(row[str_epoch_timestamp], "%Y %m %d %H:%M:%S.%f").timestamp(),
        axis=1
    ) # be carefult about the time format in csv files



    if category == Category.ORIENTATION:
        Console.info("Parsing orientation...")
        previous_timestamp = 0
        for i in range(len(mission_data["epoch_timestamp"])):
            roll = mission_data[str_roll][i]
            pitch = mission_data[str_pitch][i]
            yaw = mission_data[str_yaw][i]
            if not isnan(roll) and not isnan(pitch) and not isnan(yaw):
                t = mission_data["epoch_timestamp"][i]
                orientation.from_koyo21rov(t, roll, pitch, yaw) # we could still use this function for any csv files
                data = orientation.export(output_format)
                if orientation.epoch_timestamp > previous_timestamp:
                    data_list.append(data)
                else:
                    data_list[-1] = data
                previous_timestamp = orientation.epoch_timestamp
        Console.info("...done parsing orientation")
    if category == Category.DEPTH:
        Console.info("Parsing depth...")
        previous_timestamp = 0
        for i in range(len(mission_data["epoch_timestamp"])):
            d = mission_data[str_depth][i]
            if not isnan(d):
                t = mission_data["epoch_timestamp"][i]
                depth.from_koyo21rov(t,d) # we could still use this function for any csv files
                data = depth.export(output_format)
                if t > previous_timestamp:
                    data_list.append(data)
                else:
                    data_list[-1] = data
                previous_timestamp = t
        Console.info("...done parsing depth")
    if category == Category.ALTITUDE:
        Console.info("Parsing altitude...")
        previous_timestamp = 0
        for i in range(len(mission_data["epoch_timestamp"])):
            a = mission_data[str_altitude][i]
            # The altitude should not be NaN on lines with bottom lock,
            # but check to be on the safe side:
            if not isnan(a):
                t = mission_data["epoch_timestamp"][i]
                altitude.from_koyo21rov(t, a) # we could still use this function for any csv files
                data = altitude.export(output_format)
                if altitude.epoch_timestamp > previous_timestamp:
                    data_list.append(data)
                else:
                    data_list[-1] = data
                previous_timestamp = altitude.epoch_timestamp
        Console.info("...done parsing altitude")
    # for now, we haven't corrected USBL here, just copying from other files.
    if (
        category == Category.USBL
    ):  # This one is for posfilter version of the ROV nav data (posfilter/pos_filter_1117_v2_HR02.csv)
        Console.info("Parsing koyo21-rov filtered USBL...")
        filepath = mission.usbl.filepath
        # timezone = mission.usbl.timezone
        # timeoffset = mission.usbl.timeoffset
        # timezone_offset = read_timezone(timezone)
        latitude_reference = mission.origin.latitude
        longitude_reference = mission.origin.longitude
        usbl = Usbl(
            mission.usbl.std_factor,
            mission.usbl.std_offset,
            latitude_reference,
            longitude_reference,
        )
        usbl.sensor_string = sensor_string
        previous_timestamp = 0.0

        for i in range(len(mission_data["epoch_timestamp"])):
            lat = mission_data["Lat_flt"][i]
            lon = mission_data["Lon_flt"][i]
            # read depth information, which is required to calculate the std from depth for northings/eastings
            d = mission_data["Depth(ROV)"][i]
            if not isnan(d):  # double check in case of invalid rows
                t = mission_data["epoch_timestamp"][i]
                usbl.from_koyo21rov(t, lat, lon, d)
                data = usbl.export(
                    output_format
                )  # warning: after calling export(), the entry is cleared
                if t > previous_timestamp:
                    data_list.append(data)
                else:
                    data_list[-1] = data
                previous_timestamp = t
        Console.info("...done parsing koyo21-rov USBL")

    if category == Category.VELOCITY:
        Console.info("Parsing velocity...")
        previous_timestamp = 0
        for i in range(len(mission_data["epoch_timestamp"])):
            # we only have the surge here
            surge = mission_data[str_surge][i]
            sway = mission_data[str_sway][i]
            if not isnan(surge) and not isnan(sway) :
                t = mission_data["epoch_timestamp"][i]
                # we only have the surge here
                velocity.x_velocity = surge
                velocity.y_velocity = sway
                velocity.z_velocity = 0
                velocity.x_velocity_std = velocity.get_std(surge)
                velocity.y_velocity_std = velocity.get_std(sway)
                velocity.z_velocity_std = velocity.get_std(0.0)
                velocity.epoch_timestamp = t
                velocity.epoch_timestamp_dvl = t # it seems we don't need dvl here. but just keep it

                data = velocity.export(output_format)
                if t > previous_timestamp:
                    data_list.append(data)
                else:
                    data_list[-1] = data
                previous_timestamp = t
        Console.info("...done parsing velocity")

    # if category == Category.USBL:     # This one is for RAW USBL data extracted from the hypack payload
    # Console.info("Parsing koyo21-rov hypack USBL...")
    # filepath = mission.usbl.filepath
    # timezone = mission.usbl.timezone
    # # beacon_id = mission.usbl.label
    # timeoffset = mission.usbl.timeoffset
    # timezone_offset = read_timezone(timezone)
    # latitude_reference = mission.origin.latitude
    # longitude_reference = mission.origin.longitude

    # usbl = Usbl(
    #     mission.usbl.std_factor,
    #     mission.usbl.std_offset,
    #     latitude_reference,
    #     longitude_reference,
    # )
    # usbl.sensor_string = sensor_string
    # previous_timestamp = 0.0
    # # Rename column 5 to NS, column 9 to EW
    # mission_data.rename(columns={"Unnamed: 5": "NS"}, inplace=True)
    # mission_data.rename(columns={"Unnamed: 9": "EW"}, inplace=True)
    # # Lat/Lon information is available as DMS in separate columns, we need to convert to decimal degrees
    # for i in range(len(mission_data["epoch_timestamp"])):
    #     lat = mission_data["LatD"][i] + mission_data["LatM"][i] / 60 + mission_data["LatS"][i] / 3600
    #     lon = mission_data["LonD"][i] + mission_data["LonM"][i] / 60 + mission_data["LonS"][i] / 3600
    #     # column 5 is N/S and column 9 is E/W. If it's S or W, we need to make the value negative
    #     if mission_data["NS"][i] == "S":
    #         lat = -lat
    #     if mission_data["EW"][i] == "W":
    #         lon = -lon
    #     # read depth information, which is required to calculate the std from depth for northings/eastings
    #     d = mission_data["Depth(ROV)"][i]
    #     if not isnan(d):    # double check in case of invalid rows
    #         t = mission_data["epoch_timestamp"][i]
    #         usbl.from_koyo21rov(t, lat, lon, d)
    #         data = usbl.export(output_format)       # warning: after calling export(), the entry is cleared
    #         if t > previous_timestamp:
    #             data_list.append(data)
    #         else:
    #             data_list[-1] = data
    #         previous_timestamp = t
    # Console.info("...done parsing koyo21-rov USBL")
    return data_list


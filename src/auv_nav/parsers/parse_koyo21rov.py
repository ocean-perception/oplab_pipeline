# -*- coding: utf-8 -*-
"""
Copyright (c) 2021, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import time
import pandas as pd
from math import isnan
from auv_nav.sensors import Altitude, Category, Depth, Orientation, Usbl
from oplab import Console, get_raw_folder
from auv_nav.tools.time_conversions import read_timezone


def parse_koyo21rov(mission, vehicle, category, ftype, outpath):
    # parser meta data
    sensor_string = "koyo21rov"
    category = category
    output_format = ftype
    filename = mission.orientation.filename
    filepath = mission.orientation.filepath

    # ALR std models
    depth_std_factor = mission.depth.std_factor
    # velocity_std_factor = mission.velocity.std_factor
    # velocity_std_offset = mission.velocity.std_offset
    orientation_std_offset = mission.orientation.std_offset
    altitude_std_factor = mission.altitude.std_factor
    # usbl_std_factor = mission.usbl.std_factor
    headingoffset = vehicle.dvl.yaw

    # body_velocity = BodyVelocity(
    #     velocity_std_factor, velocity_std_offset, headingoffset
    # )
    orientation = Orientation(headingoffset, orientation_std_offset)
    depth = Depth(depth_std_factor)
    altitude = Altitude(altitude_std_factor)
    # usbl = Usbl(usbl_std_factor)

    # body_velocity.sensor_string = sensor_string
    orientation.sensor_string = sensor_string
    depth.sensor_string = sensor_string
    altitude.sensor_string = sensor_string

    path = get_raw_folder(outpath / ".." / filepath / filename)

    # Load the data from CSV file with well-known headers
    mission_data = pd.read_csv(str(path))
    # print header for debugging
    # print(mission_data.columns)

    # this is a sample of the data
    #     Date	Time	LatD	LatM	LatS		LonD	LonM	LonS		North	East	Roll	Pich	Hedding	Depth(ROV)	ALT
    # 2021/11/17	19:30:16	22	51	25.5777	N	153	21	7.0884	E	2527744.54	536108.34	-0.7	1.3	94.9	1164.1	10
    # 2021/11/17	19:30:17	22	51	25.5707	N	153	21	7.1194	E	2527744.33	536109.22	-0.5	1.1	95.3	1164.2	10

    # we need to create a new column for the epoch time called epoch_timestamp from the Date and Time columns
    # to calculate the epoch time we need to convert the date and time to a string
    # then we can use the date_time_to_epoch function to convert the date and time to epoch time
    # then we can add the epoch time to the dataframe as a new column
    mission_data["epoch_timestamp"] = mission_data.apply(
        lambda row: time.mktime(time.strptime(
            str(row["Date"]) + " " + str(row["Time"]), "%Y/%m/%d %H:%M:%S"
        )
        ),
        axis=1,
    )

    data_list = []
    if category == Category.ORIENTATION:
        Console.info("Parsing koyo21-rov orientation...")
        previous_timestamp = 0
        for i in range(len(mission_data["epoch_timestamp"])):
            roll = mission_data["Roll"][i]      # roll is in the Roll column, which is ok
            pitch = mission_data["Pich"][i]     # yes, pitch is in the 'Pich' column
            yaw = mission_data["Hedding"][i]    # and yes, yaw is in the 'Hedding' column... that's how it is in the data
            if not isnan(roll) and not isnan(pitch) and not isnan(yaw):
                t = mission_data["epoch_timestamp"][i]
                orientation.from_koyo21rov(t, roll, pitch, yaw)
                data = orientation.export(output_format)
                if orientation.epoch_timestamp > previous_timestamp:
                    data_list.append(data)
                else:
                    data_list[-1] = data
                previous_timestamp = orientation.epoch_timestamp
        Console.info("...done parsing koyo21-rov orientation")
    # if category == Category.DEPTH:
    #     Console.info("Parsing koyo21-rov depth...")
    #     previous_timestamp = 0
    #     for i in range(len(mission_data["epoch_timestamp"])):
    #         d = mission_data["Depth(ROV)"][i]
    #         if not isnan(d):
    #             t = mission_data["epoch_timestamp"][i]
    #             depth.from_koyo21rov(t, d)
    #             data = depth.export(output_format)
    #             if depth.epoch_timestamp > previous_timestamp:
    #                 data_list.append(data)
    #             else:
    #                 data_list[-1] = data
    #             previous_timestamp = depth.epoch_timestamp
    #     Console.info("...done parsing koyo21-rov depth")
    if category == Category.ALTITUDE:
        Console.info("Parsing koyo21-rov altitude...")
        previous_timestamp = 0
        for i in range(len(mission_data["epoch_timestamp"])):
            a = mission_data["ALT"][i]
            # The altitude should not be NaN on lines with bottom lock,
            # but check to be on the safe side:
            if not isnan(a):
                t = mission_data["epoch_timestamp"][i]
                altitude.from_koyo21rov(t, a)
                data = altitude.export(output_format)
                if altitude.epoch_timestamp > previous_timestamp:
                    data_list.append(data)
                else:
                    data_list[-1] = data
                previous_timestamp = altitude.epoch_timestamp
        Console.info("...done parsing alr altitude")
    if category == Category.USBL:
        Console.info("Parsing koyo21-rov USBL...")

        filepath = mission.usbl.filepath
        timezone = mission.usbl.timezone
        # beacon_id = mission.usbl.label
        timeoffset = mission.usbl.timeoffset
        timezone_offset = read_timezone(timezone)
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
        # Rename column 5 to NS, column 9 to EW
        mission_data.rename(columns={"Unnamed: 5": "NS"}, inplace=True)
        mission_data.rename(columns={"Unnamed: 9": "EW"}, inplace=True)
        # Lat/Lon information is available as DMS in separate columns, we need to convert to decimal degrees
        for i in range(len(mission_data["epoch_timestamp"])):
            lat = mission_data["LatD"][i] + mission_data["LatM"][i] / 60 + mission_data["LatS"][i] / 3600
            lon = mission_data["LonD"][i] + mission_data["LonM"][i] / 60 + mission_data["LonS"][i] / 3600
            # column 5 is N/S and column 9 is E/W. If it's S or W, we need to make the value negative
            if mission_data["NS"][i] == "S":
                lat = -lat
            if mission_data["EW"][i] == "W":
                lon = -lon
            # read depth information, which is required to calculate the std from depth for northings/eastings
            d = mission_data["Depth(ROV)"][i]
            if not isnan(d):    # double check in case of invalid rows
                t = mission_data["epoch_timestamp"][i]
                usbl.from_koyo21rov(t, lat, lon, d)
                data = usbl.export(output_format)       # warning: after calling export(), the entry is cleared
                if t > previous_timestamp:
                    data_list.append(data)
                else:
                    data_list[-1] = data
                previous_timestamp = t
        Console.info("...done parsing koyo21-rov USBL")
    return data_list

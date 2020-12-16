# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

# Scripts to parse ae2000 logs

# Author: Blair Thornton
# Date: 14/02/2018

# from datetime import datetime
import pandas as pd
import math

# sys.path.append("..")
from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.time_conversions import date_time_to_epoch
from auv_nav.tools.time_conversions import read_timezone
from oplab import get_raw_folder
from oplab import Console
from oplab import Mission, Vehicle

# http://www.json.org/
data_list = []
# need to make acfr parsers


def parse_ae2000(mission: Mission, vehicle: Vehicle, category, ftype, outpath):
    # parser meta data
    class_string = "measurement"
    sensor_string = "ae20000"

    altitude_limit = 200  # value given by ae2000 when there is no bottom lock
    # read in date from filename

    timezone = mission.velocity.timezone
    timeoffset = mission.velocity.timeoffset
    filepath = mission.velocity.filepath

    if category == 'velocity' or category == 'altitude':
        filename = mission.velocity.filename
        yyyy = int(filename[3:5]) + 2000
        mm = int(filename[5:7])
        dd = int(filename[7:9])
    elif category == 'orientation':
        filename = mission.orientation.filename
        yyyy = int(filename[8:10]) + 2000
        mm = int(filename[10:12])
        dd = int(filename[12:14])
    else:
        filename = mission.depth.filename
        yyyy = int(filename[3:5]) + 2000
        mm = int(filename[5:7])
        dd = int(filename[7:9])

    timezone_offset = read_timezone(timezone)

    # convert to seconds from utc
    # timeoffset = -timezone_offset*60*60 + timeoffset

    # parse phins data
    Console.info("  Parsing ae2000 logs for " + category + "...")
    data_list = []
    if ftype == "acfr":
        data_list = ""

    filepath = get_raw_folder(outpath / ".." / filepath)
    df = pd.read_csv(filepath / filename)

    # list of time value in the first column (starting from 2nd row, not considering first row)
    time_column = df.iloc[:, 0]
    # length of this should match every other column
    for row_index in range(len(time_column)):
        timestamp = time_column[row_index].split(":")
        if len(timestamp) < 3:
            continue
        hour = int(timestamp[0])
        mins = int(timestamp[1])
        timestamp_s_ms = timestamp[2].split(".")
        secs = int(timestamp_s_ms[0])
        msec = int(timestamp_s_ms[1])

        epoch_time = date_time_to_epoch(yyyy, mm, dd, hour, mins, secs, timezone_offset)
        # dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
        # time_tuple = dt_obj.timetuple()
        # epoch_time = time.mktime(time_tuple)
        epoch_timestamp = epoch_time + msec / 1000 + timeoffset

        if "Height" in df.columns:
            if float(df["Height"][row_index]) > altitude_limit:
                continue
        
        if ftype == "oplab":
            if category == "velocity":

                frame_string = "body"

                headingoffset = vehicle.dvl.yaw

                # DVL convention is +ve aft to forward
                x_velocity = float(df["dvl_surgeVelBottom"][row_index])/1000.

                # DVL convention is +ve port to starboard
                y_velocity = float(df["dvl_swayVelBottom"][row_index])/1000.
                # DVL convention is bottom to top +ve
                z_velocity = float(df["dvl_heaveVelBottom"][row_index])/1000.

                # account for sensor rotational offset
                [x_velocity, y_velocity, z_velocity] = body_to_inertial(
                    0, 0, headingoffset, x_velocity, y_velocity, z_velocity
                )
                # print('OUT:',x_velocity, y_velocity, z_velocity)
                # y_velocity=-1*y_velocity
                # z_velocity=-1*z_velocity
                x_velocity_std = (
                    abs(x_velocity) * mission.velocity.std_factor
                    + mission.velocity.std_offset
                )
                y_velocity_std = (
                    abs(y_velocity) * mission.velocity.std_factor
                    + mission.velocity.std_offset
                )
                z_velocity_std = (
                    abs(z_velocity) * mission.velocity.std_factor
                    + mission.velocity.std_offset
                )

                # write out in the required format interlace at end
                data = {
                    "epoch_timestamp": float(epoch_timestamp),
                    "epoch_timestamp_dvl": float(epoch_timestamp),
                    "class": class_string,
                    "sensor": sensor_string,
                    "frame": frame_string,
                    "category": category,
                    "data": [
                        {
                            "x_velocity": float(x_velocity),
                            "x_velocity_std": float(x_velocity_std),
                        },
                        {
                            "y_velocity": float(y_velocity),
                            "y_velocity_std": float(y_velocity_std),
                        },
                        {
                            "z_velocity": float(z_velocity),
                            "z_velocity_std": float(z_velocity_std),
                        },
                    ],
                }
                data_list.append(data)

                """
                frame_string = "inertial"
                # phins convention is west +ve so a minus should be necessary
                east_velocity = float(df["vye"][row_index])
                north_velocity = float(df["vxe"][row_index])
                down_velocity = None

                east_velocity_std = (
                    abs(east_velocity) * mission.velocity.std_factor
                    + mission.velocity.std_offset
                )
                north_velocity_std = (
                    abs(north_velocity) * mission.velocity.std_factor
                    + mission.velocity.std_offset
                )
                down_velocity_std = None
                # write out in the required format interlace at end
                data = {
                    "epoch_timestamp": float(epoch_timestamp),
                    "class": class_string,
                    "sensor": sensor_string,
                    "frame": frame_string,
                    "category": category,
                    "data": [
                        {
                            "north_velocity": float(north_velocity),
                            "north_velocity_std": float(north_velocity_std),
                        },
                        {
                            "east_velocity": float(east_velocity),
                            "east_velocity_std": float(east_velocity_std),
                        },
                        {
                            "down_velocity": down_velocity,
                            "down_velocity_std": down_velocity_std,
                        },
                    ],
                }
                data_list.append(data)
                """

            if category == "orientation":
                frame_string = "body"

                roll = float(df["roll"][row_index])
                pitch = float(df["pitch"][row_index])
                heading = float(df["yaw"][row_index])

                heading_std = mission.orientation.std_factor*abs(heading) + mission.orientation.std_offset
                roll_std = mission.orientation.std_factor*abs(roll) + mission.orientation.std_offset
                pitch_std = mission.orientation.std_factor*abs(pitch) + mission.orientation.std_offset
                # account for sensor rotational offset
                headingoffset = vehicle.ins.yaw

                [roll, pitch, heading] = body_to_inertial(
                    0, 0, headingoffset, roll, pitch, heading
                )

                # heading=heading+headingoffset
                if heading > 360:
                    heading = heading - 360
                if heading < 0:
                    heading = heading + 360

                    # write out in the required format interlace at end
                data = {
                    "epoch_timestamp": float(epoch_timestamp),
                    "class": class_string,
                    "sensor": sensor_string,
                    "frame": frame_string,
                    "category": category,
                    "data": [
                        {
                            "heading": float(heading),
                            "heading_std": float(heading_std),
                        },
                        {"roll": float(roll), "roll_std": float(roll_std)},
                        {"pitch": float(pitch), "pitch_std": float(pitch_std)},
                    ],
                }
                data_list.append(data)

                """
                frame_string = "body"
                sub_category = "angular_rate"

                roll_rate = float(df["rrate"][row_index])
                pitch_rate = float(df["prate"][row_index])
                heading_rate = float(df["yrate"][row_index])

                heading_rate_std = None
                roll_rate_std = None
                pitch_rate_std = None

                data = {
                    "epoch_timestamp": float(epoch_timestamp),
                    "class": class_string,
                    "sensor": sensor_string,
                    "frame": frame_string,
                    "category": sub_category,
                    "data": [
                        {
                            "heading_rate": float(heading_rate),
                            "heading_rate_std": heading_rate_std,
                        },
                        {
                            "roll_rate": float(roll_rate),
                            "roll_rate_std": roll_rate_std,
                        },
                        {
                            "pitch_rate": float(pitch_rate),
                            "pitch_rate_std": pitch_rate_std,
                        },
                    ],
                }
                data_list.append(data)
                """

            if category == "depth":
                frame_string = "inertial"

                depth = float(df["Depth"][row_index])
                depth_std = (
                    abs(depth) * mission.depth.std_factor + mission.depth.std_offset
                )

                if depth <= 0 or math.isnan(depth):
                    continue

                # write out in the required format interlace at end
                data = {
                    "epoch_timestamp": float(epoch_timestamp),
                    "epoch_timestamp_depth": float(epoch_timestamp),
                    "class": class_string,
                    "sensor": sensor_string,
                    "frame": frame_string,
                    "category": category,
                    "data": [
                        {"depth": float(depth), "depth_std": float(depth_std)}
                    ],
                }
                data_list.append(data)

            if category == "altitude":
                frame_string = "body"
                altitude = float(df["dvl_rangeBottom"][row_index])
                altitude_std = (
                    altitude * mission.altitude.std_factor
                    + mission.altitude.std_offset
                )
                sound_velocity = None
                sound_velocity_correction = None

                # write out in the required format interlace at end
                data = {
                    "epoch_timestamp": float(epoch_timestamp),
                    "epoch_timestamp_dvl": float(epoch_timestamp),
                    "class": class_string,
                    "sensor": sensor_string,
                    "frame": frame_string,
                    "category": category,
                    "data": [
                        {
                            "altitude": float(altitude),
                            "altitude_std": float(altitude_std),
                        },
                        {
                            "sound_velocity": sound_velocity,
                            "sound_velocity_correction": 
                                sound_velocity_correction,
                        },
                    ],
                }
                data_list.append(data)

        if ftype == "acfr":
            if category == "velocity":

                sound_velocity = None
                sound_velocity_correction = None
                altitude = float(df["dvl_rangeBottom"][row_index])

                # DVL convention is +ve aft to forward
                xx_velocity = float(df["dvl_surgeVelBottom"][row_index])/1000.
                # DVL convention is +ve port to starboard
                yy_velocity = float(df["dvl_swayVelBottom"][row_index])/1000.
                # DVL convention is bottom to top +ve
                zz_velocity = float(df["dvl_heaveVelBottom"][row_index])/1000.

                # account for sensor offset
                # [xx_velocity,yy_velocity,zz_velocity] = body_to_inertial(0, 0, headingoffset, xx_velocity, yy_velocity, zz_velocity)
                # yy_velocity=-1*yy_velocity
                # zz_velocity=-1*zz_velocity

                roll = float(df["roll"][row_index])
                pitch = float(df["pitch"][row_index])
                heading = float(df["yaw"][row_index])

                # print(data)
                # write out in the required format interlace at end
                data = (
                    "RDI: "
                    + str(float(epoch_timestamp))
                    + " alt:"
                    + str(float(altitude))
                    + " r1:0 r2:0 r3:0 r4:0 h:"
                    + str(float(heading))
                    + " p:"
                    + str(float(pitch))
                    + " r:"
                    + str(float(roll))
                    + " vx:"
                    + str(float(xx_velocity))
                    + " vy:"
                    + str(float(yy_velocity))
                    + " vz:"
                    + str(float(zz_velocity))
                    + " nx:0 ny:0 nz:0 COG:0 SOG:0 bt_status:0 h_true:0 p_gimbal:0 sv: "
                    + str(sound_velocity)
                    + "\n"
                )
                data_list += data

            if category == "orientation":
                roll = float(df["roll"][row_index])
                pitch = float(df["pitch"][row_index])
                heading = float(df["yaw"][row_index])

                heading_std = mission.orientation.std_factor*abs(heading) + mission.orientation.std_offset
                roll_std = mission.orientation.std_factor*abs(roll) + mission.orientation.std_offset
                pitch_std = mission.orientation.std_factor*abs(pitch) + mission.orientation.std_offset

                # account for sensor rotational offset
                headingoffset = vehicle.ins.yaw

                [roll, pitch, heading] = body_to_inertial(
                    0, 0, headingoffset, roll, pitch, heading
                )
                [roll_std, pitch_std, heading_std] = body_to_inertial(
                    0, 0, headingoffset, roll_std, pitch_std, heading_std
                )

                # heading=heading+headingoffset
                if heading > 360:
                    heading = heading - 360
                if heading < 0:
                    heading = heading + 360

                    # print(data)
                    # write out in the required format interlace at end
                data = (
                    "PHINS_COMPASS: "
                    + str(float(epoch_timestamp))
                    + " r: "
                    + str(float(roll))
                    + " p: "
                    + str(float(pitch))
                    + " h: "
                    + str(float(heading))
                    + " std_r: "
                    + str(float(roll_std))
                    + " std_p: "
                    + str(float(pitch_std))
                    + " std_h: "
                    + str(float(heading_std))
                    + "\n"
                )
                data_list += data

            if category == "depth":

                depth = float(df["Depth"][row_index])
                # write out in the required format interlace at end
                data = (
                    "PAROSCI: "
                    + str(float(epoch_timestamp))
                    + " "
                    + str(float(depth))
                    + "\n"
                )
                data_list += data
            else:
                continue
        # else:
        # 	print('no bottom lock')
    Console.info("  ...done parsing ae2000 logs for " + category + ".")

    return data_list

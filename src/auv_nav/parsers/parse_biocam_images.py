# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

# Scripts to parse acfr image acquisition data

# Author: Blair Thornton
# Date: 31/08/2017

import glob
from pathlib import Path

import numpy as np
from scipy.stats import linregress

from auv_nav.tools.time_conversions import date_time_to_epoch
from oplab import Console, get_raw_folder

stamp_pc1 = []
stamp_pc2 = []
stamp_cam1 = []
stamp_cam2 = []
values = []
data_list = []
tolerance = 0.05  # 0.01 # stereo pair must be within 10ms of each other


def time_from_string(
    date_string, time_string, ms_time_string, timezone_offset, timeoffset
):
    # read in date
    yyyy = int(date_string[0:4])
    mm = int(date_string[4:6])
    dd = int(date_string[6:8])

    # read in time
    hour = int(time_string[0:2])
    mins = int(time_string[2:4])
    secs = int(time_string[4:6])
    if len(ms_time_string) == 6:
        usec = int(ms_time_string[0:6])
    elif len(ms_time_string) == 3:
        usec = int(ms_time_string[0:3]) * 1000

    if yyyy < 2000:
        return 0
    epoch_time = date_time_to_epoch(yyyy, mm, dd, hour, mins, secs, timezone_offset)
    # dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
    # time_tuple = dt_obj.timetuple()
    # epoch_time = time.mktime(time_tuple)
    epoch_timestamp = float(epoch_time + usec / 1e6 + timeoffset)
    return epoch_timestamp


def biocam_timestamp_from_filename(filename, timezone_offset, timeoffset):
    filename_split = filename.strip().split("_")

    if len(filename_split) > 4:
        date_string = filename_split[0]
        time_string = filename_split[1]
        ms_time_string = filename_split[2]
        cam_date_string = filename_split[3]
        cam_time_string = filename_split[4]
        cam_ms_time_string = filename_split[5]
    else:
        date_string = filename_split[1]
        time_string = filename_split[2]
        ms_time_string = filename_split[3]
        cam_date_string = filename_split[1]
        cam_time_string = filename_split[2]
        cam_ms_time_string = filename_split[3]

    epoch_timestamp = time_from_string(
        date_string, time_string, ms_time_string, timezone_offset, timeoffset
    )
    cam_epoch_timestamp = time_from_string(
        cam_date_string,
        cam_time_string,
        cam_ms_time_string,
        timezone_offset,
        timeoffset,
    )
    return epoch_timestamp, cam_epoch_timestamp


def pathlist_relativeto(input_pathlist, base_path):
    out_list = []
    for x in input_pathlist:
        p = Path(x)
        pr = p.relative_to(base_path)
        out_list.append(str(pr))
    return out_list


def parse_biocam_images(mission, vehicle, category, ftype, outpath):
    # parser meta data
    class_string = "measurement"
    frame_string = "body"
    category = "image"
    sensor_string = "biocam"

    timezone = mission.image.timezone
    timezone_offset = 0
    timeoffset = mission.image.timeoffset
    filepath = mission.image.cameras[0].path
    camera1_label = mission.image.cameras[0].name
    camera2_label = mission.image.cameras[1].name

    # read in timezone
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
                "in mission.cfg not recognised, please enter value from UTC",
                "in hours",
            )
            return

    # convert to seconds from utc
    # timeoffset = -timezone_offset*60*60 + timeoffset

    Console.info("... parsing " + sensor_string + " images")

    # determine file paths
    base_path = get_raw_folder(outpath / ".." / filepath)
    filepath1 = base_path / str(camera1_label + "_strobe/*.*")
    filepath1b = base_path / str(camera1_label + "_laser/**/*.*")
    filepath2 = base_path / str(camera2_label + "_strobe/*.*")
    filepath2b = base_path / str(camera2_label + "_laser/**/*.*")

    camera1_list = glob.glob(str(filepath1))
    camera1_list.extend(glob.glob(str(filepath1b), recursive=True))
    camera2_list = glob.glob(str(filepath2))
    camera2_list.extend(glob.glob(str(filepath2b), recursive=True))

    camera1_filename = [
        line for line in camera1_list if ".txt" not in line and "._" not in line
    ]
    camera2_filename = [
        line for line in camera2_list if ".txt" not in line and ".jpg" not in line
    ]
    camera3_filename = [line for line in camera2_list if ".jpg" in line]

    dive_folder = get_raw_folder(outpath / "..")
    camera1_relfilename = pathlist_relativeto(camera1_filename, dive_folder)
    camera2_relfilename = pathlist_relativeto(camera2_filename, dive_folder)
    camera3_relfilename = pathlist_relativeto(camera3_filename, dive_folder)

    Console.info(
        "Found "
        + str(len(camera2_filename) + len(camera1_filename) + len(camera3_filename))
        + " BioCam images!"
    )

    data_list = []
    if ftype == "acfr":
        data_list = ""

    for i in range(len(camera1_filename)):
        t1, tc1 = biocam_timestamp_from_filename(
            Path(camera1_filename[i]).name, timezone_offset, timeoffset
        )
        stamp_pc1.append(str(t1))
        stamp_cam1.append(str(tc1))
    for i in range(len(camera2_filename)):
        t2, tc2 = biocam_timestamp_from_filename(
            Path(camera2_filename[i]).name, timezone_offset, timeoffset
        )
        stamp_pc2.append(str(t2))
        stamp_cam2.append(str(tc2))
    for i in range(len(camera1_filename)):
        values = []
        for j in range(len(camera2_filename)):
            values.append(abs(float(stamp_pc1[i]) - float(stamp_pc2[j])))

        (sync_difference, sync_pair) = min((v, k) for k, v in enumerate(values))

        if sync_difference < tolerance:
            if ftype == "oplab":
                data = {
                    "epoch_timestamp": float(stamp_pc1[i]),
                    "class": class_string,
                    "sensor": sensor_string,
                    "frame": frame_string,
                    "category": category,
                    "camera1": [
                        {
                            "epoch_timestamp": float(stamp_pc1[i]),
                            # Duplicate for timestamp prediction purposes
                            "epoch_timestamp_cpu": float(stamp_pc1[i]),
                            "epoch_timestamp_cam": float(stamp_cam1[i]),
                            "filename": str(camera1_relfilename[i]),
                        }
                    ],
                    "camera2": [
                        {
                            "epoch_timestamp": float(stamp_pc2[sync_pair]),
                            # Duplicate for timestamp prediction purposes
                            "epoch_timestamp_cpu": float(stamp_pc2[sync_pair]),
                            "epoch_timestamp_cam": float(stamp_cam2[sync_pair]),
                            "filename": str(camera2_relfilename[sync_pair]),
                        }
                    ],
                }
                data_list.append(data)
            if ftype == "acfr":
                data = (
                    "VIS: "
                    + str(float(stamp_pc1[i]))
                    + " ["
                    + str(float(stamp_pc1[i]))
                    + "] "
                    + str(camera1_relfilename[i])
                    + " exp: 0\n"
                )
                # fileout.write(data)
                data_list += data
                data = (
                    "VIS: "
                    + str(float(stamp_pc2[sync_pair]))
                    + " ["
                    + str(float(stamp_pc2[sync_pair]))
                    + "] "
                    + str(camera2_relfilename[sync_pair])
                    + " exp: 0\n"
                )
                # fileout.write(data)
                data_list += data
    for i in range(len(camera3_filename)):
        t3, tc3 = biocam_timestamp_from_filename(
            Path(camera3_filename[i]).name, timezone_offset, timeoffset
        )
        data = {
            "epoch_timestamp": float(t3),
            "class": class_string,
            "sensor": sensor_string,
            "frame": frame_string,
            "category": "laser",
            "camera3": [
                {
                    "epoch_timestamp": float(t3),
                    # Duplicate for timestamp prediction purposes
                    "epoch_timestamp_cpu": float(t3),
                    "epoch_timestamp_cam": float(tc3),
                    "filename": str(camera3_relfilename[i]),
                }
            ],
        }
        data_list.append(data)
    return data_list


def correct_timestamps(data_list):
    cam1_cam_list = []
    cam1_offset_list = []
    cam2_cam_list = []
    cam2_offset_list = []
    cam3_cam_list = []
    cam3_offset_list = []
    for data in data_list:
        category = data["category"]
        if category == "image":
            # Grab data
            cpu1_timestamp = data["camera1"][0]["epoch_timestamp_cpu"]
            cam1_timestamp = data["camera1"][0]["epoch_timestamp_cam"]
            cpu2_timestamp = data["camera2"][0]["epoch_timestamp_cpu"]
            cam2_timestamp = data["camera2"][0]["epoch_timestamp_cam"]
            # Calculate offsets
            cam1_offset = cpu1_timestamp - cam1_timestamp
            cam2_offset = cpu2_timestamp - cam2_timestamp
            # Store
            cam1_cam_list.append(cam1_timestamp)
            cam1_offset_list.append(cam1_offset)
            cam2_cam_list.append(cam2_timestamp)
            cam2_offset_list.append(cam2_offset)
        elif category == "laser":
            # Grab data
            cpu3_timestamp = data["camera3"][0]["epoch_timestamp_cpu"]
            cam3_timestamp = data["camera3"][0]["epoch_timestamp_cam"]
            # Calculate offset
            cam3_offset = cpu3_timestamp - cam3_timestamp
            # Store
            cam3_cam_list.append(cam3_timestamp)
            cam3_offset_list.append(cam3_offset)
        else:
            continue
    # Use all data available for camera2 (camera3 defined in
    # parse_biocam_images as same camera as camera2)
    comb_cam_list = cam2_cam_list + cam3_cam_list
    comb_offset_list = cam2_offset_list + cam3_offset_list

    # Fit a straight line to the data, bounding the bottom of it

    def line_fit(x_data, y_data):
        fit = linregress(x_data, y_data)
        m = fit.slope
        c = fit.intercept
        max_diff = np.min([y_data[i] - m * x_data[i] - c for i in range(len(x_data))])
        c = c + max_diff
        return m, c

    m1, c1 = line_fit(cam1_cam_list, cam1_offset_list)
    m2, c2 = line_fit(comb_cam_list, comb_offset_list)

    def predict_cpu_time(cam_list, m, c):
        pred_list = []
        for x in cam_list:
            # offset = (cpu - cam) = m*(cam) + c
            y = (m + 1) * x + c
            pred_list.append(y)
        return pred_list

    cam1_pred_list = predict_cpu_time(cam1_cam_list, m1, c1)
    cam2_pred_list = predict_cpu_time(cam2_cam_list, m2, c2)
    cam3_pred_list = predict_cpu_time(cam3_cam_list, m2, c2)

    #    # Code to show plot of what the best fit lines look like on the data
    #    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    line1 = [
        [cam1_cam_list[1], cam1_cam_list[-2]],
        [m1 * cam1_cam_list[1] + c1, m1 * cam1_cam_list[-2] + c1],
    ]
    line2 = [
        [cam2_cam_list[1], cam2_cam_list[-2]],
        [m2 * cam2_cam_list[1] + c2, m2 * cam2_cam_list[-2] + c2],
    ]

    #    plt.figure("test preds - offset vs cam time")
    #    plt.plot(cam2_cam_list, cam2_offset_list, '.r')
    #    plt.plot(cam3_cam_list, cam3_offset_list, '.r')
    #    plt.plot(cam1_cam_list, cam1_offset_list, '.c')
    #    plt.plot(line1[0], line1[1], '-b')
    #    plt.plot(line2[0], line2[1], '-m')
    #    plt.show()

    Console.info(
        "...... Divergence over time of cpu clock to cam1 clock:",
        m1 / 100,
        "%",
    )
    Console.info("...... Initial offset of cpu clock to cam1 clock:", line1[1][0], "s")
    Console.info(
        "...... Divergence over time of cpu clock to cam2 clock:",
        m2 / 100,
        "%",
    )
    Console.info("...... Initial offset of cpu clock to cam2 clock:", line2[1][0], "s")

    i = 0
    j = 0
    for data in data_list:
        category = data["category"]
        if category == "image":
            data["epoch_timestamp"] = cam1_pred_list[i]
            data["camera1"][0]["epoch_timestamp"] = cam1_pred_list[i]
            data["camera2"][0]["epoch_timestamp"] = cam2_pred_list[i]
            i += 1

        elif category == "laser":
            data["epoch_timestamp"] = cam3_pred_list[j]
            data["camera3"][0]["epoch_timestamp"] = cam3_pred_list[j]
            j += 1
        else:
            continue

    Console.info("...done correcting timestamps.")
    return data_list

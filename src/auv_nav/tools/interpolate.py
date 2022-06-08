# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import numpy as np

from auv_nav.sensors import Camera, SyncedOrientationBodyVelocity, Usbl
from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.latlon_wgs84 import metres_to_latlon
from oplab import Console


# Scripts to interpolate values
def interpolate(x_query, x_lower, x_upper, y_lower, y_upper):
    if x_upper == x_lower:
        y_query = y_lower
    else:
        y_query = (y_upper - y_lower) / (x_upper - x_lower) * (
            x_query - x_lower
        ) + y_lower
    return y_query


def interpolate_altitude(query_timestamp, data):
    i = 1
    while i < len(data) and data[i].epoch_timestamp < query_timestamp:
        i += 1
    return interpolate(
        query_timestamp,
        data[i - 1].epoch_timestamp,
        data[i].epoch_timestamp,
        data[i - 1].altitude,
        data[i].altitude,
    )


def interpolate_dvl(query_timestamp, data_1, data_2):
    temp_data = SyncedOrientationBodyVelocity()
    temp_data.epoch_timestamp = query_timestamp
    temp_data.x_velocity = interpolate(
        query_timestamp,
        data_1.epoch_timestamp,
        data_2.epoch_timestamp,
        data_1.x_velocity,
        data_2.x_velocity,
    )
    temp_data.y_velocity = interpolate(
        query_timestamp,
        data_1.epoch_timestamp,
        data_2.epoch_timestamp,
        data_1.y_velocity,
        data_2.y_velocity,
    )
    temp_data.z_velocity = interpolate(
        query_timestamp,
        data_1.epoch_timestamp,
        data_2.epoch_timestamp,
        data_1.z_velocity,
        data_2.z_velocity,
    )
    temp_data.roll = interpolate(
        query_timestamp,
        data_1.epoch_timestamp,
        data_2.epoch_timestamp,
        data_1.roll,
        data_2.roll,
    )
    temp_data.pitch = interpolate(
        query_timestamp,
        data_1.epoch_timestamp,
        data_2.epoch_timestamp,
        data_1.pitch,
        data_2.pitch,
    )
    if abs(data_2.yaw - data_1.yaw) > 180:
        if data_2.yaw > data_1.yaw:
            temp_data.yaw = interpolate(
                query_timestamp,
                data_1.epoch_timestamp,
                data_2.epoch_timestamp,
                data_1.yaw,
                data_2.yaw - 360,
            )

        else:
            temp_data.yaw = interpolate(
                query_timestamp,
                data_1.epoch_timestamp,
                data_2.epoch_timestamp,
                data_1.yaw - 360,
                data_2.yaw,
            )

        if temp_data.yaw < 0:
            temp_data.yaw += 360

        elif temp_data.yaw > 360:
            temp_data.yaw -= 360

    else:
        temp_data.yaw = interpolate(
            query_timestamp,
            data_1.epoch_timestamp,
            data_2.epoch_timestamp,
            data_1.yaw,
            data_2.yaw,
        )
    return temp_data


def interpolate_camera(query_timestamp, camera_list, filename):
    """Interpolates a camera to the query timestamp given a camera list
    to interpolate from, and assign the filename provided to that
    timestamp

    Parameters
    ----------
    query_timestamp : float
        Query timestap
    camera_list : list(Camera)
        Populated camera list. Will be used to find matching timestamps
        to interpolate from
    filename : str
        Camera filename to assign to the queried timestamp

    Returns
    -------
    Camera
        Interpolated camera to the queried timestamp and with the
        filename provided
    """

    i = 0
    for i in range(len(camera_list)):
        if query_timestamp <= camera_list[i].epoch_timestamp:
            break
    c1 = camera_list[i]
    c2 = camera_list[i]
    if i > 1:
        c1 = camera_list[i - 1]

    c = Camera()
    c.filename = filename
    c.epoch_timestamp = query_timestamp
    c.northings = interpolate(
        query_timestamp,
        c1.epoch_timestamp,
        c2.epoch_timestamp,
        c1.northings,
        c2.northings,
    )
    c.eastings = interpolate(
        query_timestamp,
        c1.epoch_timestamp,
        c2.epoch_timestamp,
        c1.eastings,
        c2.eastings,
    )
    c.depth = interpolate(
        query_timestamp,
        c1.epoch_timestamp,
        c2.epoch_timestamp,
        c1.depth,
        c2.depth,
    )
    c.latitude = interpolate(
        query_timestamp,
        c1.epoch_timestamp,
        c2.epoch_timestamp,
        c1.latitude,
        c2.latitude,
    )
    c.longitude = interpolate(
        query_timestamp,
        c1.epoch_timestamp,
        c2.epoch_timestamp,
        c1.longitude,
        c2.longitude,
    )
    c.roll = interpolate(
        query_timestamp,
        c1.epoch_timestamp,
        c2.epoch_timestamp,
        c1.roll,
        c2.roll,
    )
    c.pitch = interpolate(
        query_timestamp,
        c1.epoch_timestamp,
        c2.epoch_timestamp,
        c1.pitch,
        c2.pitch,
    )
    c.yaw = interpolate(
        query_timestamp, c1.epoch_timestamp, c2.epoch_timestamp, c1.yaw, c2.yaw
    )
    c.x_velocity = interpolate(
        query_timestamp,
        c1.epoch_timestamp,
        c2.epoch_timestamp,
        c1.x_velocity,
        c2.x_velocity,
    )
    c.y_velocity = interpolate(
        query_timestamp,
        c1.epoch_timestamp,
        c2.epoch_timestamp,
        c1.y_velocity,
        c2.y_velocity,
    )
    c.z_velocity = interpolate(
        query_timestamp,
        c1.epoch_timestamp,
        c2.epoch_timestamp,
        c1.z_velocity,
        c2.z_velocity,
    )
    c.altitude = interpolate(
        query_timestamp,
        c1.epoch_timestamp,
        c2.epoch_timestamp,
        c1.altitude,
        c2.altitude,
    )
    return c


def interpolate_usbl(query_timestamp, data_1, data_2):
    temp_data = Usbl()
    temp_data.epoch_timestamp = query_timestamp
    temp_data.northings = interpolate(
        query_timestamp,
        data_1.epoch_timestamp,
        data_2.epoch_timestamp,
        data_1.northings,
        data_2.northings,
    )
    temp_data.eastings = interpolate(
        query_timestamp,
        data_1.epoch_timestamp,
        data_2.epoch_timestamp,
        data_1.eastings,
        data_2.eastings,
    )
    temp_data.northings_std = interpolate(
        query_timestamp,
        data_1.epoch_timestamp,
        data_2.epoch_timestamp,
        data_1.northings_std,
        data_2.northings_std,
    )
    temp_data.eastings_std = interpolate(
        query_timestamp,
        data_1.epoch_timestamp,
        data_2.epoch_timestamp,
        data_1.eastings_std,
        data_2.eastings_std,
    )
    temp_data.depth = interpolate(
        query_timestamp,
        data_1.epoch_timestamp,
        data_2.epoch_timestamp,
        data_1.depth,
        data_2.depth,
    )
    return temp_data


def eigen_sorted(a):
    eigenvalues, eigenvectors = np.linalg.eig(a)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    n = eigenvalues.size
    eigenvalues = np.eye(n, n) * eigenvalues
    return eigenvalues, eigenvectors


def interpolate_covariance(t, t0, t1, cov0, cov1):
    x = (t - t0) / (t1 - t0)
    cov = (1 - x) * cov0 + x * cov1
    return cov


def interpolate_property(centre_list, i, sensor_list, j, prop_name):
    if (
        centre_list[j - 1].__dict__[prop_name] is None
        or centre_list[j].__dict__[prop_name] is None
    ):
        return None
    else:
        return interpolate(
            sensor_list[i].epoch_timestamp,
            centre_list[j - 1].epoch_timestamp,
            centre_list[j].epoch_timestamp,
            centre_list[j - 1].__dict__[prop_name],
            centre_list[j].__dict__[prop_name],
        )


def interpolate_sensor_list(
    sensor_list,
    sensor_name,
    sensor_offsets,
    origin_offsets,
    latlon_reference,
    _centre_list,
):
    j = 0
    [origin_x_offset, origin_y_offset, origin_z_offset] = origin_offsets
    [latitude_reference, longitude_reference] = latlon_reference
    # Check if camera activates before dvl and orientation sensors.
    start_time = _centre_list[0].epoch_timestamp
    end_time = _centre_list[-1].epoch_timestamp
    if (
        sensor_list[0].epoch_timestamp > end_time
        or sensor_list[-1].epoch_timestamp < start_time
    ):
        Console.warn(
            "{} timestamps does not overlap with dead reckoning data, "
            "check timestamp_history.pdf via -v option.".format(sensor_name)
        )
    else:
        sensor_overlap_flag = 0
        for i in range(len(sensor_list)):
            if sensor_list[i].epoch_timestamp < start_time:
                sensor_overlap_flag = 1
                pass
            else:
                if i > 0:
                    Console.warn(
                        "Deleted",
                        i,
                        "entries from sensor",
                        sensor_name,
                        ". Reason: data before start of mission",
                    )
                    del sensor_list[:i]
                break
        for i in range(len(sensor_list)):
            if j >= len(_centre_list) - 1:
                ii = len(sensor_list) - i
                if ii > 0:
                    Console.warn(
                        "Deleted",
                        ii,
                        "entries from sensor",
                        sensor_name,
                        ". Reason: data after end of mission",
                    )
                    del sensor_list[i:]
                sensor_overlap_flag = 1
                break
            while _centre_list[j].epoch_timestamp < sensor_list[i].epoch_timestamp:
                if (
                    j + 1 > len(_centre_list) - 1
                    or _centre_list[j + 1].epoch_timestamp
                    > sensor_list[-1].epoch_timestamp
                ):
                    break
                j += 1
            # if j>=1: ?

            sensor_list[i].roll = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j - 1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j - 1].roll,
                _centre_list[j].roll,
            )
            sensor_list[i].pitch = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j - 1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j - 1].pitch,
                _centre_list[j].pitch,
            )
            if abs(_centre_list[j].yaw - _centre_list[j - 1].yaw) > 180:
                if _centre_list[j].yaw > _centre_list[j - 1].yaw:
                    sensor_list[i].yaw = interpolate(
                        sensor_list[i].epoch_timestamp,
                        _centre_list[j - 1].epoch_timestamp,
                        _centre_list[j].epoch_timestamp,
                        _centre_list[j - 1].yaw,
                        _centre_list[j].yaw - 360,
                    )
                else:
                    sensor_list[i].yaw = interpolate(
                        sensor_list[i].epoch_timestamp,
                        _centre_list[j - 1].epoch_timestamp,
                        _centre_list[j].epoch_timestamp,
                        _centre_list[j - 1].yaw - 360,
                        _centre_list[j].yaw,
                    )
                if sensor_list[i].yaw < 0:
                    sensor_list[i].yaw += 360
                elif sensor_list[i].yaw > 360:
                    sensor_list[i].yaw -= 360
            else:
                sensor_list[i].yaw = interpolate(
                    sensor_list[i].epoch_timestamp,
                    _centre_list[j - 1].epoch_timestamp,
                    _centre_list[j].epoch_timestamp,
                    _centre_list[j - 1].yaw,
                    _centre_list[j].yaw,
                )
            sensor_list[i].x_velocity = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j - 1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j - 1].x_velocity,
                _centre_list[j].x_velocity,
            )
            sensor_list[i].y_velocity = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j - 1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j - 1].y_velocity,
                _centre_list[j].y_velocity,
            )
            sensor_list[i].z_velocity = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j - 1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j - 1].z_velocity,
                _centre_list[j].z_velocity,
            )
            sensor_list[i].altitude = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j - 1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j - 1].altitude,
                _centre_list[j].altitude,
            )

            [x_offset, y_offset, z_offset] = body_to_inertial(
                sensor_list[i].roll,
                sensor_list[i].pitch,
                sensor_list[i].yaw,
                origin_x_offset - sensor_offsets[0],
                origin_y_offset - sensor_offsets[1],
                origin_z_offset - sensor_offsets[2],
            )

            sensor_list[i].northings = (
                interpolate(
                    sensor_list[i].epoch_timestamp,
                    _centre_list[j - 1].epoch_timestamp,
                    _centre_list[j].epoch_timestamp,
                    _centre_list[j - 1].northings,
                    _centre_list[j].northings,
                )
                - x_offset
            )
            sensor_list[i].eastings = (
                interpolate(
                    sensor_list[i].epoch_timestamp,
                    _centre_list[j - 1].epoch_timestamp,
                    _centre_list[j].epoch_timestamp,
                    _centre_list[j - 1].eastings,
                    _centre_list[j].eastings,
                )
                - y_offset
            )
            sensor_list[i].altitude = (
                interpolate(
                    sensor_list[i].epoch_timestamp,
                    _centre_list[j - 1].epoch_timestamp,
                    _centre_list[j].epoch_timestamp,
                    _centre_list[j - 1].altitude,
                    _centre_list[j].altitude,
                )
                + z_offset
            )
            sensor_list[i].depth = (
                interpolate(
                    sensor_list[i].epoch_timestamp,
                    _centre_list[j - 1].epoch_timestamp,
                    _centre_list[j].epoch_timestamp,
                    _centre_list[j - 1].depth,
                    _centre_list[j].depth,
                )
                - z_offset
            )
            [sensor_list[i].latitude, sensor_list[i].longitude] = metres_to_latlon(
                latitude_reference,
                longitude_reference,
                sensor_list[i].eastings,
                sensor_list[i].northings,
            )

            sensor_list[i].northings_std = interpolate_property(
                _centre_list, i, sensor_list, j, "northings_std"
            )
            sensor_list[i].eastings_std = interpolate_property(
                _centre_list, i, sensor_list, j, "eastings_std"
            )
            sensor_list[i].depth_std = interpolate_property(
                _centre_list, i, sensor_list, j, "depth_std"
            )
            sensor_list[i].roll_std = interpolate_property(
                _centre_list, i, sensor_list, j, "roll_std"
            )
            sensor_list[i].pitch_std = interpolate_property(
                _centre_list, i, sensor_list, j, "pitch_std"
            )
            sensor_list[i].yaw_std = interpolate_property(
                _centre_list, i, sensor_list, j, "yaw_std"
            )
            sensor_list[i].x_velocity_std = interpolate_property(
                _centre_list, i, sensor_list, j, "x_velocity_std"
            )
            sensor_list[i].y_velocity_std = interpolate_property(
                _centre_list, i, sensor_list, j, "y_velocity_std"
            )
            sensor_list[i].z_velocity_std = interpolate_property(
                _centre_list, i, sensor_list, j, "z_velocity_std"
            )
            sensor_list[i].vroll_std = interpolate_property(
                _centre_list, i, sensor_list, j, "vroll_std"
            )
            sensor_list[i].vpitch_std = interpolate_property(
                _centre_list, i, sensor_list, j, "vpitch_std"
            )
            sensor_list[i].vyaw_std = interpolate_property(
                _centre_list, i, sensor_list, j, "vyaw_std"
            )

            if _centre_list[j].covariance is not None:
                sensor_list[i].covariance = interpolate_covariance(
                    sensor_list[i].epoch_timestamp,
                    _centre_list[j - 1].epoch_timestamp,
                    _centre_list[j].epoch_timestamp,
                    _centre_list[j - 1].covariance,
                    _centre_list[j].covariance,
                )

        if sensor_overlap_flag == 1:
            Console.warn(
                "Sensor data from {} spans further than dead reckoning data."
                " Data outside DR is ignored.".format(sensor_name)
            )
        Console.info(
            "Complete interpolation and coordinate transfomations "
            "for {}".format(sensor_name)
        )

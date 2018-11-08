# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""

from auv_nav.sensors import SyncedOrientationBodyVelocity, Usbl
from auv_nav.tools.latlon_wgs84 import metres_to_latlon
from auv_nav.tools.body_to_inertial import body_to_inertial

# Scripts to interpolate values
def interpolate(x_query, x_lower, x_upper, y_lower, y_upper):
    if x_upper == x_lower:
        y_query = y_lower
    else:
        y_query = (y_upper-y_lower)/(x_upper-x_lower)*(x_query-x_lower)+y_lower
    return y_query


def interpolate_dvl(query_timestamp, data_1, data_2):
    temp_data = SyncedOrientationBodyVelocity()
    temp_data.epoch_timestamp = query_timestamp
    temp_data.x_velocity = interpolate(query_timestamp,
                                       data_1.epoch_timestamp,
                                       data_2.epoch_timestamp,
                                       data_1.x_velocity,
                                       data_2.x_velocity)
    temp_data.y_velocity = interpolate(query_timestamp,
                                       data_1.epoch_timestamp,
                                       data_2.epoch_timestamp,
                                       data_1.y_velocity,
                                       data_2.y_velocity)
    temp_data.z_velocity = interpolate(query_timestamp,
                                       data_1.epoch_timestamp,
                                       data_2.epoch_timestamp,
                                       data_1.z_velocity,
                                       data_2.z_velocity)
    temp_data.roll = interpolate(query_timestamp,
                                 data_1.epoch_timestamp,
                                 data_2.epoch_timestamp,
                                 data_1.roll,
                                 data_2.roll)
    temp_data.pitch = interpolate(query_timestamp,
                                  data_1.epoch_timestamp,
                                  data_2.epoch_timestamp,
                                  data_1.pitch,
                                  data_2.pitch)
    if abs(data_2.yaw-data_1.yaw) > 180:
        if data_2.yaw > data_1.yaw:
            temp_data.yaw = interpolate(query_timestamp,
                                        data_1.epoch_timestamp,
                                        data_2.epoch_timestamp,
                                        data_1.yaw,
                                        data_2.yaw-360)

        else:
            temp_data.yaw = interpolate(query_timestamp,
                                        data_1.epoch_timestamp,
                                        data_2.epoch_timestamp,
                                        data_1.yaw-360,
                                        data_2.yaw)

        if temp_data.yaw < 0:
            temp_data.yaw += 360

        elif temp_data.yaw > 360:
            temp_data.yaw -= 360

    else:
        temp_data.yaw = interpolate(query_timestamp,
                                    data_1.epoch_timestamp,
                                    data_2.epoch_timestamp,
                                    data_1.yaw,
                                    data_2.yaw)
    return (temp_data)


def interpolate_usbl(query_timestamp, data_1, data_2):
    temp_data = Usbl()
    temp_data.epoch_timestamp = query_timestamp
    temp_data.northings = interpolate(query_timestamp,
                                      data_1.epoch_timestamp,
                                      data_2.epoch_timestamp,
                                      data_1.northings,
                                      data_2.northings)
    temp_data.eastings = interpolate(query_timestamp,
                                     data_1.epoch_timestamp,
                                     data_2.epoch_timestamp,
                                     data_1.eastings,
                                     data_2.eastings)
    temp_data.northings_std = interpolate(query_timestamp,
                                          data_1.epoch_timestamp,
                                          data_2.epoch_timestamp,
                                          data_1.northings_std,
                                          data_2.northings_std)
    temp_data.eastings_std = interpolate(query_timestamp,
                                         data_1.epoch_timestamp,
                                         data_2.epoch_timestamp,
                                         data_1.eastings_std,
                                         data_2.eastings_std)
    temp_data.depth = interpolate(query_timestamp,
                                  data_1.epoch_timestamp,
                                  data_2.epoch_timestamp,
                                  data_1.depth,
                                  data_2.depth)
    return (temp_data)


def interpolate_sensor_list(sensor_list,
                            sensor_name,
                            sensor_offsets,
                            origin_offsets,
                            latlon_reference,
                            _centre_list):
    j = 0
    [origin_x_offset, origin_y_offset, origin_z_offset] = origin_offsets
    [latitude_reference, longitude_reference] = latlon_reference
    # Check if camera activates before dvl and orientation sensors.
    start_time = _centre_list[0].epoch_timestamp
    end_time = _centre_list[-1].epoch_timestamp
    if (sensor_list[0].epoch_timestamp > end_time
       or sensor_list[-1].epoch_timestamp < start_time):
        print('{} timestamps does not overlap with dead reckoning data, \
               check timestamp_history.pdf via -v option.'.format(sensor_name))
    else:
        sensor_overlap_flag = 0
        for i in range(len(sensor_list)):
            if sensor_list[i].epoch_timestamp < _centre_list[0].epoch_timestamp:
                sensor_overlap_flag = 1
                pass
            else:
                del sensor_list[:i]
                break
        for i in range(len(sensor_list)):
            if j >= len(_centre_list)-1:
                del sensor_list[i:]
                sensor_overlap_flag = 1
                break
            while _centre_list[j].epoch_timestamp < sensor_list[i].epoch_timestamp:
                if j+1 > len(_centre_list)-1 or _centre_list[j+1].epoch_timestamp > sensor_list[-1].epoch_timestamp:
                    break
                j += 1
            # if j>=1: ?
            sensor_list[i].roll = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j-1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j-1].roll,
                _centre_list[j].roll)
            sensor_list[i].pitch = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j-1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j-1].pitch,
                _centre_list[j].pitch)
            if abs(_centre_list[j].yaw-_centre_list[j-1].yaw) > 180:
                if _centre_list[j].yaw>_centre_list[j-1].yaw:
                    sensor_list[i].yaw = interpolate(
                        sensor_list[i].epoch_timestamp,
                        _centre_list[j-1].epoch_timestamp,
                        _centre_list[j].epoch_timestamp,
                        _centre_list[j-1].yaw,
                        _centre_list[j].yaw-360)
                else:
                    sensor_list[i].yaw = interpolate(
                        sensor_list[i].epoch_timestamp,
                        _centre_list[j-1].epoch_timestamp,
                        _centre_list[j].epoch_timestamp,
                        _centre_list[j-1].yaw-360,
                        _centre_list[j].yaw)
                if sensor_list[i].yaw < 0:
                    sensor_list[i].yaw += 360
                elif sensor_list[i].yaw > 360:
                    sensor_list[i].yaw -= 360
            else:
                sensor_list[i].yaw = interpolate(
                    sensor_list[i].epoch_timestamp,
                    _centre_list[j-1].epoch_timestamp,
                    _centre_list[j].epoch_timestamp,
                    _centre_list[j-1].yaw,
                    _centre_list[j].yaw)

            # while n<len(time_altitude)-1 and time_altitude[n+1]<time_camera1[i]:
            #     n += 1
            # camera1_altitude.append(interpolate(time_camera1[i],time_altitude[n],time_altitude[n+1],altitude[n],altitude[n+1]))
            sensor_list[i].altitude = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j-1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j-1].altitude,
                _centre_list[j].altitude)

            [x_offset, y_offset, z_offset] = body_to_inertial(
                sensor_list[i].roll,
                sensor_list[i].pitch,
                sensor_list[i].yaw,
                origin_x_offset - sensor_offsets[0],
                origin_y_offset - sensor_offsets[1],
                origin_z_offset - sensor_offsets[2])

            sensor_list[i].northings = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j-1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j-1].northings,
                _centre_list[j].northings)-x_offset
            sensor_list[i].eastings = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j-1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j-1].eastings,
                _centre_list[j].eastings)-y_offset
            sensor_list[i].depth = interpolate(
                sensor_list[i].epoch_timestamp,
                _centre_list[j-1].epoch_timestamp,
                _centre_list[j].epoch_timestamp,
                _centre_list[j-1].depth,
                _centre_list[j].depth) # -z_offset
            [sensor_list[i].latitude,
             sensor_list[i].longitude] = metres_to_latlon(
                latitude_reference,
                longitude_reference,
                sensor_list[i].eastings,
                sensor_list[i].northings)
        if sensor_overlap_flag == 1:
            print('{} data more than dead reckoning data. Only processed \
                  overlapping data and ignored the rest.'.format(sensor_name))
        print('Complete interpolation and coordinate transfomations \
               for {}'.format(sensor_name))

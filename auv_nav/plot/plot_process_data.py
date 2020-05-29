# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import plotly.graph_objs as go
from plotly import tools
import plotly.offline as py

import os
import time


def create_trace(x_list,
                 y_list,
                 trace_name,
                 trace_color,
                 visibility=True):
    trace = go.Scattergl(
        x=[float(i) for i in x_list],
        y=[float(i) for i in y_list],
        visible=visibility,  # True | False | legendonly
        name=trace_name,
        mode='lines+markers',
        marker=dict(color=trace_color),
        line=dict(color=trace_color)
    )
    return trace


def make_data(figure,
              name,
              eastings,
              northings,
              mode='lines',
              visibility=True,
              hoverinfo='x+y',
              hovertext="",
              opacity=1):
    if 'usbl' in name:
        mode = 'lines+markers'
    data_dict = {
        'x': eastings,
        'y': northings,
        'mode': '{}'.format(mode),
        'marker': {'opacity': opacity},
        'name': '{}'.format(name),
        'visible': visibility,
        'hoverinfo': hoverinfo,
        'hovertext': hovertext,
    }
    figure['data'].append(data_dict)


def make_frame(frame,
               data,
               tstamp,
               mode='lines'):
    temp_index = -1
    for i in range(len(data[1])):
        if data[1][i] <= tstamp:
            temp_index = i
        else:
            break
    eastings = data[2][:temp_index+1]
    northings = data[3][:temp_index+1]
    data_dict = {
        'x': eastings,
        'y': northings,
        'name': '{}'.format(data[0]),
        'mode': '{}'.format(mode)
    }
    frame['data'].append(data_dict)


def plot_orientation_vs_time(orientation_list,
                             plotlypath):
    # orientation
    print('...plotting orientation_vs_time...')
    orientation_time = [i.epoch_timestamp for i in orientation_list]
    yaw = [i.yaw for i in orientation_list]
    pitch = [i.pitch for i in orientation_list]
    roll = [i.roll for i in orientation_list]
    tr_yaw = create_trace(orientation_time, yaw, 'Yaw', 'red')
    tr_pitch = create_trace(orientation_time, pitch, 'Pitch', 'blue')
    tr_roll = create_trace(orientation_time, roll, 'Roll', 'green')

    layout = go.Layout(
        title='Orientation vs Time',
        hovermode='closest',
        xaxis=dict(title='Epoch time, s', tickformat='.3f'),
        yaxis=dict(title='Degrees'),
        dragmode='pan')
    config = {'scrollZoom': True}
    fig = go.Figure(data=[tr_yaw, tr_pitch, tr_roll], layout=layout)
    py.plot(fig,
            config=config,
            filename=str(plotlypath / 'orientation_vs_time.html'),
            auto_open=False)


def plot_velocity_vs_time(dead_reckoning_dvl_list,
                          velocity_inertial_list,
                          dead_reckoning_centre_list,
                          velocity_inertial_sensor_name,
                          plotlypath):
    # velocity_body (north,east,down) compared to velocity_inertial
    print('...plotting velocity_vs_time...')

    dr_time = [i.epoch_timestamp for i in dead_reckoning_dvl_list]
    dr_north_velocity = [i.north_velocity for i in dead_reckoning_dvl_list]
    dr_east_velocity = [i.east_velocity for i in dead_reckoning_dvl_list]
    dr_down_velocity = [i.down_velocity for i in dead_reckoning_dvl_list]
    dr_x_velocity = [i.x_velocity for i in dead_reckoning_dvl_list]
    dr_y_velocity = [i.y_velocity for i in dead_reckoning_dvl_list]
    dr_z_velocity = [i.z_velocity for i in dead_reckoning_dvl_list]

    tr_dr_north = create_trace(dr_time, dr_north_velocity,
                               'DVL north velocity', 'red')
    tr_dr_east = create_trace(dr_time, dr_east_velocity,
                              'DVL east velocity', 'red')
    tr_dr_down = create_trace(dr_time, dr_down_velocity,
                              'DVL down velocity', 'red')
    tr_dr_x = create_trace(dr_time, dr_x_velocity,
                           'DVL x velocity', 'red')
    tr_dr_y = create_trace(dr_time, dr_y_velocity,
                           'DVL y velocity', 'red')
    tr_dr_z = create_trace(dr_time, dr_z_velocity,
                           'DVL z velocity', 'red')

    if len(velocity_inertial_list) > 0:
        inertial_time = [i.epoch_timestamp for i in velocity_inertial_list]
        inertial_north_velocity = [i.north_velocity for i in velocity_inertial_list]
        inertial_east_velocity = [i.east_velocity for i in velocity_inertial_list]
        inertial_down_velocity = [i.down_velocity for i in velocity_inertial_list]

        tr_inertial_north = create_trace(
            inertial_time, inertial_north_velocity,
            '{} north velocity'.format(velocity_inertial_sensor_name),
            'blue')
        tr_inertial_east = create_trace(
            inertial_time, inertial_east_velocity,
            '{} east velocity'.format(velocity_inertial_sensor_name),
            'blue')
        tr_inertial_down = create_trace(
            inertial_time, inertial_down_velocity,
            '{} down velocity'.format(velocity_inertial_sensor_name),
            'blue')

    fig = tools.make_subplots(
        rows=3,
        cols=2,
        subplot_titles=('DVL vs {} - north Velocity'.format(
                            velocity_inertial_sensor_name),
                        'DVL - x velocity / surge',
                        'DVL vs {} - east Velocity'.format(
                            velocity_inertial_sensor_name),
                        'DVL - y velocity / sway',
                        'DVL vs {} - down Velocity'.format(
                            velocity_inertial_sensor_name),
                        'DVL - z velocity / heave'),
        print_grid=False)
    fig.append_trace(tr_dr_north, 1, 1)
    if len(velocity_inertial_list) > 0:
        fig.append_trace(tr_inertial_north, 1, 1)
        fig.append_trace(tr_inertial_east, 2, 1)
        fig.append_trace(tr_inertial_down, 3, 1)
    fig.append_trace(tr_dr_east, 2, 1)
    fig.append_trace(tr_dr_down, 3, 1)
    fig.append_trace(tr_dr_x, 1, 2)
    fig.append_trace(tr_dr_y, 2, 2)
    fig.append_trace(tr_dr_z, 3, 2)
    fig['layout']['xaxis1'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis2'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis3'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis4'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis5'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis6'].update(title='Epoch time, s', tickformat='.3f')

    fig['layout']['yaxis1'].update(title='Velocity, m/s')
    fig['layout']['yaxis2'].update(title='Velocity, m/s')
    fig['layout']['yaxis3'].update(title='Velocity, m/s')
    fig['layout']['yaxis4'].update(title='Velocity, m/s')
    fig['layout']['yaxis5'].update(title='Velocity, m/s')
    fig['layout']['yaxis6'].update(title='Velocity, m/s')
    fig['layout'].update(title='Velocity vs Time Plots (Left column: Inertial \
                                frame - north east down | Right column: Body \
                                frame - x y z)',
                         dragmode='pan',
                         hovermode='closest')
    config = {'scrollZoom': True}
    py.plot(fig,
            config=config,
            filename=str(plotlypath / 'velocity_vs_time.html'),
            auto_open=False)


def plot_deadreckoning_vs_time(dead_reckoning_dvl_list,
                               velocity_inertial_list,
                               usbl_list,
                               dead_reckoning_centre_list,
                               altitude_list,
                               depth_list,
                               velocity_inertial_sensor_name,
                               plotlypath):
    # time_dead_reckoning northings eastings depth vs time
    print('...plotting deadreckoning_vs_time...')

    dr_time = [i.epoch_timestamp for i in dead_reckoning_dvl_list]
    dr_northings = [i.northings for i in dead_reckoning_dvl_list]
    dr_eastings = [i.eastings for i in dead_reckoning_dvl_list]

    usbl_time = [i.epoch_timestamp for i in usbl_list]
    usbl_northings = [i.northings for i in usbl_list]
    usbl_eastings = [i.eastings for i in usbl_list]
    usbl_depth = [i.depth for i in usbl_list]

    dr_centre_time = [i.epoch_timestamp for i in dead_reckoning_centre_list]
    dr_centre_northings = [i.northings for i in dead_reckoning_centre_list]
    dr_centre_eastings = [i.eastings for i in dead_reckoning_centre_list]
    dr_centre_depth = [i.depth for i in dead_reckoning_centre_list]

    altitude_time = [i.epoch_timestamp for i in altitude_list]
    altitude_depth = [i.seafloor_depth for i in altitude_list]
    altitude_alt = [i.altitude for i in altitude_list]

    depth_time = [i.epoch_timestamp for i in depth_list]
    depth = [i.depth for i in depth_list]

    # Northing vs time
    tr_dr_northings = create_trace(dr_time, dr_northings,
                                   'Northing DVL', 'red')
    tr_usbl_northings = create_trace(usbl_time, usbl_northings,
                                     'Northing USBL', 'blue')
    tr_drc_northings = create_trace(dr_centre_time, dr_centre_northings,
                                    'Northing Centre', 'orange')

    # Easting vs time
    tr_dr_eastings = create_trace(dr_time, dr_eastings,
                                  'Easting DVL', 'red')
    tr_usbl_eastings = create_trace(usbl_time, usbl_eastings,
                                    'Easting USBL', 'blue')
    tr_drc_eastings = create_trace(dr_centre_time, dr_centre_eastings,
                                   'Easting Centre', 'orange')

    # Depth vs time
    tr_alt_depth = create_trace(altitude_time, altitude_depth,
                                'Depth  Seafloor (Depth Sensor + Altitude)',
                                'red')
    tr_depth = create_trace(depth_time, depth, 'Depth Sensor', 'purple')
    tr_usbl_depth = create_trace(usbl_time, usbl_depth, 'Depth USBL', 'blue')
    tr_dr_depth = create_trace(dr_centre_time, dr_centre_depth,
                               'Depth Centre', 'orange')

    # Altitude vs time
    tr_alt = create_trace(altitude_time, altitude_alt, 'Altitude', 'red')

    fig = tools.make_subplots(rows=2,
                              cols=2,
                              subplot_titles=('Northings',
                                              'Eastings',
                                              'Depth',
                                              'Altitude'),
                              print_grid=False)

    # Northing vs time
    fig.append_trace(tr_dr_northings, 1, 1)
    fig.append_trace(tr_usbl_northings, 1, 1)
    fig.append_trace(tr_drc_northings, 1, 1)
    # Easting vs time
    fig.append_trace(tr_dr_eastings, 1, 2)
    fig.append_trace(tr_usbl_eastings, 1, 2)
    fig.append_trace(tr_drc_eastings, 1, 2)
    # Depth vs time
    fig.append_trace(tr_alt_depth, 2, 1)
    fig.append_trace(tr_depth, 2, 1)
    fig.append_trace(tr_usbl_depth, 2, 1)
    fig.append_trace(tr_dr_depth, 2, 1)
    # Altitude vs time
    fig.append_trace(tr_alt, 2, 2)

    # Add Inertial Velocity if available
    if len(velocity_inertial_list) > 0:
        inertial_time = [i.epoch_timestamp for i in velocity_inertial_list]
        inertial_northings = [i.northings for i in velocity_inertial_list]
        inertial_eastings = [i.eastings for i in velocity_inertial_list]
        tr_vi_northings = create_trace(
            inertial_time, inertial_northings,
            'Northing {}'.format(velocity_inertial_sensor_name), 'green')
        tr_vi_eastings = create_trace(
            inertial_time, inertial_eastings,
            'Easting {}'.format(velocity_inertial_sensor_name), 'green')
        fig.append_trace(tr_vi_northings, 1, 1)
        fig.append_trace(tr_vi_eastings, 1, 2)

    fig['layout']['xaxis1'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis2'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis3'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis4'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['yaxis1'].update(title='Northing, m')
    fig['layout']['yaxis2'].update(title='Easting, m')
    fig['layout']['yaxis3'].update(title='Depth, m', autorange='reversed')
    fig['layout']['yaxis4'].update(title='Altitude, m')
    fig['layout'].update(title='Deadreckoning vs Time',
                         dragmode='pan', hovermode='closest')
    config = {'scrollZoom': True}
    py.plot(fig,
            config=config,
            filename=str(plotlypath / 'deadreckoning_vs_time.html'),
            auto_open=False)


# pf uncertainty plotly
# maybe make a slider plot for this, or a dot projection slider
def plot_pf_uncertainty(pf_fusion_dvl_list,
                        pf_northings_std,
                        pf_eastings_std,
                        pf_yaw_std,
                        plotlypath):
    print('...plotting pf_uncertainty...')
    pf_time = [i.epoch_timestamp for i in pf_fusion_dvl_list]

    tr_pf_northings_std = create_trace(pf_time, pf_northings_std,
                                       'northings_std (m)', 'red')
    tr_pf_eastings_std = create_trace(pf_time, pf_eastings_std,
                                      'eastings_std (m)', 'blue')
    tr_pf_yaw_std = create_trace(pf_time, pf_yaw_std,
                                 'yaw_std (deg)', 'green')
    layout = go.Layout(
        title='Particle Filter Uncertainty Plot',
        hovermode='closest',
        xaxis=dict(title='Epoch time, s', tickformat='.3f'),
        yaxis=dict(title='Degrees or Metres'),
        dragmode='pan',
        )
    config = {'scrollZoom': True}
    fig = go.Figure(data=[tr_pf_northings_std,
                          tr_pf_eastings_std,
                          tr_pf_yaw_std],
                    layout=layout)
    py.plot(fig,
            config=config,
            filename=str(plotlypath / 'pf_uncertainty.html'),
            auto_open=False)


def plot_uncertainty(orientation_list,
                     velocity_body_list,
                     depth_list,
                     usbl_list,
                     velocity_inertial_list,
                     velocity_inertial_sensor_name,
                     plotlypath):
    print('...plotting uncertainty...')
    # Uncertainty plotly --- https://plot.ly/python/line-charts/
    # filled-lines Something like that?
    ori_time = [i.epoch_timestamp for i in orientation_list]
    ori_roll_std = [i.roll_std for i in orientation_list]
    ori_pitch_std = [i.pitch_std for i in orientation_list]
    ori_yaw_std = [i.yaw_std for i in orientation_list]

    bv_time = [i.epoch_timestamp for i in velocity_body_list]
    bv_x_vel_std = [i.x_velocity_std for i in velocity_body_list]
    bv_y_vel_std = [i.y_velocity_std for i in velocity_body_list]
    bv_z_vel_std = [i.z_velocity_std for i in velocity_body_list]

    usbl_time = [i.epoch_timestamp for i in usbl_list]
    usbl_lat_std = [i.latitude_std for i in usbl_list]
    usbl_lon_std = [i.longitude_std for i in usbl_list]
    usbl_northing_std = [i.northings_std for i in usbl_list]
    usbl_easting_std = [i.eastings_std for i in usbl_list]

    depth_time = [i.epoch_timestamp for i in depth_list]
    depth_std = [i.depth_std for i in depth_list]

    iv_time = [i.epoch_timestamp for i in velocity_inertial_list]
    iv_n_vel_std = [i.north_velocity_std for i in velocity_inertial_list]
    iv_e_vel_std = [i.east_velocity_std for i in velocity_inertial_list]
    iv_d_vel_std = [i.down_velocity_std for i in velocity_inertial_list]

    tr_roll = create_trace(ori_time, ori_roll_std, 'Roll std', 'red')
    tr_pitch = create_trace(ori_time, ori_pitch_std, 'Pitch std', 'green')
    tr_yaw = create_trace(ori_time, ori_yaw_std, 'Yaw std', 'blue')

    tr_bv_x = create_trace(bv_time, bv_x_vel_std, 'x velocity std', 'red')
    tr_bv_y = create_trace(bv_time, bv_y_vel_std, 'y velocity std', 'green')
    tr_bv_z = create_trace(bv_time, bv_z_vel_std, 'z velocity std', 'blue')

    tr_lat = create_trace(usbl_time, usbl_lat_std, 'Lat std usbl', 'red')
    tr_lon = create_trace(usbl_time, usbl_lon_std, 'Lon std usbl', 'green')
    tr_depth = create_trace(depth_time, depth_std, 'Depth std', 'blue')

    tr_n = create_trace(usbl_time, usbl_northing_std, 'northing std usbl', 'red')
    tr_e = create_trace(usbl_time, usbl_easting_std, 'easting std usbl', 'green')

    if len(velocity_inertial_list) > 0:
        tr_iv_n = create_trace(iv_time, iv_n_vel_std, 'north velocity std inertial', 'red')
        tr_iv_e = create_trace(iv_time, iv_e_vel_std, 'east velocity std inertial', 'green')
        tr_iv_d = create_trace(iv_time, iv_d_vel_std, 'down velocity std inertial', 'blue')

    fig = tools.make_subplots(rows=2,
                              cols=3,
                              subplot_titles=('Orientation uncertainties',
                                              'DVL uncertainties',
                                              'USBL uncertainties',
                                              'Depth uncertainties',
                                              '{} uncertainties'.format(
                                                velocity_inertial_sensor_name),
                                              'USBL uncertainties'),
                              print_grid=False)
    fig.append_trace(tr_roll, 1, 1)
    fig.append_trace(tr_pitch, 1, 1)
    fig.append_trace(tr_yaw, 1, 1)

    fig.append_trace(tr_bv_x, 1, 2)
    fig.append_trace(tr_bv_y, 1, 2)
    fig.append_trace(tr_bv_z, 1, 2)

    fig.append_trace(tr_lat, 1, 3)
    fig.append_trace(tr_lon, 1, 3)
    fig.append_trace(tr_depth, 2, 1)

    if len(velocity_inertial_list) > 0:
        fig.append_trace(tr_iv_n, 2, 2)
        fig.append_trace(tr_iv_e, 2, 2)
        fig.append_trace(tr_iv_d, 2, 2)

    fig.append_trace(tr_n, 2, 3)
    fig.append_trace(tr_e, 2, 3)

    fig['layout']['xaxis1'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis2'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis3'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis4'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis5'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['xaxis6'].update(title='Epoch time, s', tickformat='.3f')
    fig['layout']['yaxis1'].update(title='Angle, degrees')
    fig['layout']['yaxis2'].update(title='Velocity, m/s')
    fig['layout']['yaxis3'].update(title='LatLong, degrees')
    fig['layout']['yaxis4'].update(title='Depth, m')
    fig['layout']['yaxis5'].update(title='Velocity, m/s')
    fig['layout']['yaxis6'].update(title='NorthEast, m')
    fig['layout'].update(title='Uncertainty Plots',
                         dragmode='pan',
                         hovermode='closest')
    config = {'scrollZoom': True}
    py.plot(fig,
            config=config,
            filename=str(plotlypath / 'uncertainties_plot.html'),
            auto_open=False)


def plot_2d_deadreckoning(camera1_list,
                          dead_reckoning_centre_list,
                          dead_reckoning_dvl_list,
                          pf_fusion_centre_list,
                          ekf_centre_list,
                          camera1_pf_list,
                          pf_fusion_dvl_list,
                          particles_time_interval,
                          pf_particles_list,
                          usbl_list,
                          plotlypath):
    # DR plotly slider *include toggle button that switches between lat long and north east
    print('...plotting auv_path...')

    # might not be robust in the future
    min_timestamp = float('inf')
    max_timestamp = float('-inf')

    plotly_list = []
    if len(camera1_list) > 1:
        plotly_list.append(['dr_camera1', camera1_list, 'legendonly'])
    if len(dead_reckoning_centre_list) > 1:
        plotly_list.append(['dr_centre', dead_reckoning_centre_list, 'legendonly'])
    if len(dead_reckoning_dvl_list) > 1:
        plotly_list.append(['dr_dvl', dead_reckoning_dvl_list, True])
    # if len(velocity_inertial_list) > 1:
    #    plotly_list.append([velocity_inertial_sensor_name, velocity_inertial_list])
    if len(usbl_list) > 1:
        plotly_list.append(['usbl', usbl_list, True])

    for i in plotly_list:
        timestamp_list = [j.epoch_timestamp for j in i[1]]
        if min(timestamp_list) < min_timestamp:
            min_timestamp = min(timestamp_list)
        if max(timestamp_list) > max_timestamp:
            max_timestamp = max(timestamp_list)

    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }

    # fill in most of layout
    figure['layout']['xaxis'] = {'title': 'Eastings,m'}
    figure['layout']['yaxis'] = {'title': 'Northings,m',
                                 'scaleanchor': 'x'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['dragmode'] = 'pan'
    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [
                        None, {
                            'frame': {'duration': 500, 'redraw': False},
                            'fromcurrent': True,
                            'transition': {
                                'duration': 300,
                                'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                }, {
                    'args': [
                        [None],
                        {'frame': {'duration': 0, 'redraw': False},
                         'mode': 'immediate',
                         'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    for i in plotly_list:
        make_data(figure, i[0],
                  [float(j.eastings) for j in i[1]],
                  [float(j.northings) for j in i[1]],
                  visibility=i[2])
    if len(ekf_centre_list) > 1:
        make_data(figure, 'ekf_centre',
                  [float(i.eastings) for i in ekf_centre_list],
                  [float(i.northings) for i in ekf_centre_list],
                  visibility='legendonly')
    if len(pf_fusion_centre_list) > 1:
        make_data(figure, 'pf_camera1',
                  [float(i.eastings) for i in camera1_pf_list],
                  [float(i.northings) for i in camera1_pf_list],
                  visibility='legendonly',
                  hoverinfo='x+y+text',
                  hovertext=[time.strftime('%H:%M:%S',
                             time.localtime(i.epoch_timestamp))
                             for i in camera1_pf_list])
        make_data(figure, 'pf_centre',
                  [float(i.eastings) for i in pf_fusion_centre_list],
                  [float(i.northings) for i in pf_fusion_centre_list],
                  visibility='legendonly')
        make_data(figure, 'pf_dvl',
                  [float(i.eastings) for i in pf_fusion_dvl_list],
                  [float(i.northings) for i in pf_fusion_dvl_list],
                  visibility=True,
                  hoverinfo='x+y+text',
                  hovertext=[time.strftime('%H:%M:%S',
                             time.localtime(i.epoch_timestamp))
                             for i in pf_fusion_dvl_list])
        pf_timestamps_interval = []
        pf_eastings_interval = []
        pf_northings_interval = []
        if particles_time_interval is not False:
            for i in pf_particles_list[0]:
                pf_timestamps_interval.append(float(
                    pf_particles_list[0][0].timestamps[0]))
                pf_eastings_interval.append(float(i.eastings[0]))
                pf_northings_interval.append(float(i.northings[0]))
            timestamp_value_tracker = pf_particles_list[0][0].timestamps[0]

            for i in range(len(pf_particles_list)):
                # timestamp_index_tracker = 0
                for j in range(len(pf_particles_list[i][0].timestamps)):
                    if (pf_particles_list[i][0].timestamps[j]
                       - timestamp_value_tracker) > particles_time_interval:
                        for k in pf_particles_list[i]:
                            pf_timestamps_interval.append(float(k.timestamps[j]))
                            pf_eastings_interval.append(float(k.eastings[j]))
                            pf_northings_interval.append(float(k.northings[j]))
                        timestamp_value_tracker = (pf_particles_list[i][0]
                                                   .timestamps[j])
            make_data(figure, 'pf_dvl_distribution',
                      pf_eastings_interval,
                      pf_northings_interval,
                      mode='markers',
                      visibility=True)
        else:
            resampling_index = 1
            for i in pf_particles_list:
                make_data(figure, 'PF_Resampling{}'.format(resampling_index),
                          [float(j.eastings) for j in i],
                          [float(j.northings) for j in i],
                          mode='markers',
                          opacity=0.5)
                # make_data(figure, 'PF_Propagation{}'.format(resampling_index),
                #           [float(j.eastings[-1]) for j in i],
                #           [float(j.northings[-1]) for j in i],
                #           mode='markers',
                #           opacity=0.5)
                resampling_index += 1

    config = {'scrollZoom': True}

    py.plot(figure,
            config=config,
            filename=str(plotlypath / 'auv_path.html'),
            auto_open=False)

    print('...plotting auv_path_slider...')

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'epoch_timestamp:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # slider plot
    # time_gap = 240
    time_gap = int((max_timestamp - min_timestamp)/40)
    epoch_timestamps_slider = list(range(int(min_timestamp),
                                         int(max_timestamp),
                                         int(time_gap)))

    # make frames
    for i in epoch_timestamps_slider:
        frame = {'data': [], 'name': str(i)}

        for j in plotly_list:
            make_frame(frame,
                       [j[0],
                        [float(k.epoch_timestamp) for k in j[1]],
                        [float(k.eastings) for k in j[1]],
                        [float(k.northings) for k in j[1]]],
                       i)
        if len(camera1_pf_list) > 1:
            make_frame(frame,
                       ['pf_camera1',
                        [float(i.epoch_timestamp) for i in camera1_pf_list],
                        [float(i.eastings) for i in camera1_pf_list],
                        [float(i.northings) for i in camera1_pf_list]],
                       i)
        if len(pf_fusion_centre_list) > 1:
            make_frame(frame,
                       ['pf_centre',
                        [float(i.epoch_timestamp) for i in pf_fusion_centre_list],
                        [float(i.eastings) for i in pf_fusion_centre_list],
                        [float(i.northings) for i in pf_fusion_centre_list]],
                       i)
        if len(ekf_centre_list) > 1:
            make_frame(frame,
                       ['ekf_centre',
                        [float(i.epoch_timestamp) for i in ekf_centre_list],
                        [float(i.eastings) for i in ekf_centre_list],
                        [float(i.northings) for i in ekf_centre_list]],
                       i)
        if len(pf_fusion_dvl_list) > 1:
            make_frame(frame,
                       ['pf_dvl',
                        [float(i.epoch_timestamp) for i in pf_fusion_dvl_list],
                        [float(i.eastings) for i in pf_fusion_dvl_list],
                        [float(i.northings) for i in pf_fusion_dvl_list]],
                       i)
        if len(pf_fusion_centre_list) > 1:
            make_frame(frame,
                       ['pf_dvl_distribution',
                        pf_timestamps_interval,
                        pf_eastings_interval,
                        pf_northings_interval],
                       i, mode='markers')

        figure['frames'].append(frame)
        slider_step = {'args': [
            [i],
            {'frame': {'duration': 300, 'redraw': False},
             'mode': 'immediate',
             'transition': {'duration': 300}}
         ],
         'label': i,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    figure['layout']['sliders'] = [sliders_dict]

    py.plot(figure,
            config=config,
            filename=str(plotlypath / 'auv_path_slider.html'),
            auto_open=False)


def plot_2d_localisation(dr_list,
                         pf_list,
                         ekf_list,
                         eks_list,
                         plotlypath):
    # DR plotly slider *include toggle button that switches between lat long and north east
    print('...plotting auv_path...')

    # might not be robust in the future
    min_timestamp = float('inf')
    max_timestamp = float('-inf')

    plotly_list = []
    if len(dr_list) > 1:
        plotly_list.append(['dr', dr_list, 'legendonly'])
    if len(pf_list) > 1:
        plotly_list.append(['pf', pf_list, 'legendonly'])
    if len(eks_list) > 1:
        plotly_list.append(['ekf', ekf_list, True])
    if len(eks_list) > 1:
        plotly_list.append(['eks', eks_list, True])

    for i in plotly_list:
        timestamp_list = [j.epoch_timestamp for j in i[1]]
        if min(timestamp_list) < min_timestamp:
            min_timestamp = min(timestamp_list)
        if max(timestamp_list) > max_timestamp:
            max_timestamp = max(timestamp_list)

    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }

    # fill in most of layout
    figure['layout']['xaxis'] = {'title': 'Eastings,m'}
    figure['layout']['yaxis'] = {'title': 'Northings,m',
                                 'scaleanchor': 'x'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['dragmode'] = 'pan'
    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [
                        None, {
                            'frame': {'duration': 500, 'redraw': False},
                            'fromcurrent': True,
                            'transition': {
                                'duration': 300,
                                'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                }, {
                    'args': [
                        [None],
                        {'frame': {'duration': 0, 'redraw': False},
                         'mode': 'immediate',
                         'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    for i in plotly_list:
        make_data(figure, i[0],
                  [float(j.eastings) for j in i[1]],
                  [float(j.northings) for j in i[1]],
                  visibility=i[2])

    config = {'scrollZoom': True}

    py.plot(figure,
            config=config,
            filename=str(plotlypath / 'auv_localisation.html'),
            auto_open=False)

    print('...plotting auv_path_slider...')

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'epoch_timestamp:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # slider plot
    # time_gap = 240
    time_gap = int((max_timestamp - min_timestamp)/40)
    epoch_timestamps_slider = list(range(int(min_timestamp),
                                         int(max_timestamp),
                                         int(time_gap)))

    # make frames
    for i in epoch_timestamps_slider:
        frame = {'data': [], 'name': str(i)}

        for j in plotly_list:
            make_frame(frame,
                       [j[0],
                        [float(k.epoch_timestamp) for k in j[1]],
                        [float(k.eastings) for k in j[1]],
                        [float(k.northings) for k in j[1]]],
                       i)
        figure['frames'].append(frame)
        slider_step = {'args': [
            [i],
            {'frame': {'duration': 300, 'redraw': False},
             'mode': 'immediate',
             'transition': {'duration': 300}}
         ],
         'label': i,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    figure['layout']['sliders'] = [sliders_dict]

    py.plot(figure,
            config=config,
            filename=str(plotlypath / 'auv_localisation_slider.html'),
            auto_open=False)

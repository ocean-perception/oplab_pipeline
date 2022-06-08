# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import math
import threading
import time
from pathlib import Path
from typing import List, Optional

import plotly.graph_objs as go
import plotly.offline as py
from plotly import subplots

from auv_nav.localisation.ekf import EkfState, Index
from auv_nav.sensors import Camera
from auv_nav.tools.time_conversions import epoch_to_utctime
from oplab import Console


def create_trace(
    x_list,
    y_list,
    trace_name,
    trace_color,
    visibility=True,
    fill="none",
    is_std_bound=False,
    is_2nd_line=False,
):
    dash = "dash" if is_2nd_line else "solid"
    line = dict(width=0) if is_std_bound else dict(color=trace_color, dash=dash)
    showlegend = False if is_std_bound else True
    mode = "lines" if is_std_bound or is_2nd_line else "lines+markers"
    trace = go.Scattergl(
        x=[float(i) for i in x_list],
        y=[float(i) for i in y_list],
        visible=visibility,  # True | False | legendonly
        name=trace_name,
        mode=mode,
        marker=dict(color=trace_color),
        line=line,
        fillcolor="rgba(200, 200, 200, 1)",
        fill=fill,
        showlegend=showlegend,
    )
    return trace


def create_trace_2(x_list, y_list, text_list, trace_name):
    trace = go.Scattergl(
        x=[float(i) for i in x_list],
        y=y_list,
        text=text_list,
        name=trace_name,
        mode="markers",
        hoverinfo="text",
    )
    return trace


def make_data(
    figure,
    name,
    eastings,
    northings,
    mode="lines",
    visibility=True,
    hoverinfo="x+y",
    hovertext="",
    opacity=1,
):
    if "usbl" in name:
        mode = "lines+markers"
    data_dict = {
        "x": eastings,
        "y": northings,
        "mode": "{}".format(mode),
        "marker": {"opacity": opacity},
        "name": "{}".format(name),
        "visible": visibility,
        "hoverinfo": hoverinfo,
        "hovertext": hovertext,
    }
    figure["data"].append(data_dict)


def make_frame(frame, data, tstamp, mode="lines"):
    temp_index = -1
    for i in range(len(data[1])):
        if data[1][i] <= tstamp:
            temp_index = i
        else:
            break
    eastings = data[2][: temp_index + 1]
    northings = data[3][: temp_index + 1]
    data_dict = {
        "x": eastings,
        "y": northings,
        "name": "{}".format(data[0]),
        "mode": "{}".format(mode),
    }
    frame["data"].append(data_dict)


def plot_orientation_vs_time(orientation_list, plotlypath):
    # orientation
    Console.info("Plotting orientation_vs_time...")
    orientation_time = [i.epoch_timestamp for i in orientation_list]
    yaw = [i.yaw for i in orientation_list]
    pitch = [i.pitch for i in orientation_list]
    roll = [i.roll for i in orientation_list]
    tr_yaw = create_trace(orientation_time, yaw, "Yaw", "red")
    tr_pitch = create_trace(orientation_time, pitch, "Pitch", "blue")
    tr_roll = create_trace(orientation_time, roll, "Roll", "green")

    layout = go.Layout(
        title="Orientation vs Time",
        hovermode="closest",
        xaxis=dict(title="Epoch time, s", tickformat=".f"),
        yaxis=dict(title="Degrees"),
        dragmode="pan",
    )
    config = {"scrollZoom": True}
    fig = go.Figure(data=[tr_yaw, tr_pitch, tr_roll], layout=layout)
    py.plot(
        fig,
        config=config,
        filename=str(plotlypath / "orientation_vs_time.html"),
        auto_open=False,
    )
    Console.info("... done plotting orientation_vs_time.")


def plot_velocity_vs_time(
    dead_reckoning_dvl_list,
    velocity_inertial_list,
    dead_reckoning_centre_list,
    velocity_inertial_sensor_name,
    plotlypath,
):
    # velocity_body (north,east,down) compared to velocity_inertial
    Console.info("Plotting velocity_vs_time...")

    dr_time = [i.epoch_timestamp for i in dead_reckoning_dvl_list]
    dr_north_velocity = [i.north_velocity for i in dead_reckoning_dvl_list]
    dr_east_velocity = [i.east_velocity for i in dead_reckoning_dvl_list]
    dr_down_velocity = [i.down_velocity for i in dead_reckoning_dvl_list]
    dr_x_velocity = [i.x_velocity for i in dead_reckoning_dvl_list]
    dr_y_velocity = [i.y_velocity for i in dead_reckoning_dvl_list]
    dr_z_velocity = [i.z_velocity for i in dead_reckoning_dvl_list]

    tr_dr_north = create_trace(dr_time, dr_north_velocity, "DVL north velocity", "red")
    tr_dr_east = create_trace(dr_time, dr_east_velocity, "DVL east velocity", "red")
    tr_dr_down = create_trace(dr_time, dr_down_velocity, "DVL down velocity", "red")
    tr_dr_x = create_trace(dr_time, dr_x_velocity, "DVL x velocity", "red")
    tr_dr_y = create_trace(dr_time, dr_y_velocity, "DVL y velocity", "red")
    tr_dr_z = create_trace(dr_time, dr_z_velocity, "DVL z velocity", "red")

    if len(velocity_inertial_list) > 0:
        inertial_time = [i.epoch_timestamp for i in velocity_inertial_list]
        inertial_north_velocity = [i.north_velocity for i in velocity_inertial_list]
        inertial_east_velocity = [i.east_velocity for i in velocity_inertial_list]
        inertial_down_velocity = [i.down_velocity for i in velocity_inertial_list]

        tr_inertial_north = create_trace(
            inertial_time,
            inertial_north_velocity,
            "{} north velocity".format(velocity_inertial_sensor_name),
            "blue",
        )
        tr_inertial_east = create_trace(
            inertial_time,
            inertial_east_velocity,
            "{} east velocity".format(velocity_inertial_sensor_name),
            "blue",
        )
        if inertial_down_velocity[0] is not None:
            tr_inertial_down = create_trace(
                inertial_time,
                inertial_down_velocity,
                "{} down velocity".format(velocity_inertial_sensor_name),
                "blue",
            )

    fig = subplots.make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "DVL vs {} - north Velocity".format(velocity_inertial_sensor_name),
            "DVL - x velocity / surge",
            "DVL vs {} - east Velocity".format(velocity_inertial_sensor_name),
            "DVL - y velocity / sway",
            "DVL vs {} - down Velocity".format(velocity_inertial_sensor_name),
            "DVL - z velocity / heave",
        ),
        print_grid=False,
    )
    fig.append_trace(tr_dr_north, 1, 1)
    if len(velocity_inertial_list) > 0:
        if velocity_inertial_list[0].north_velocity is not None:
            fig.append_trace(tr_inertial_north, 1, 1)
        if velocity_inertial_list[0].east_velocity is not None:
            fig.append_trace(tr_inertial_east, 2, 1)
        if velocity_inertial_list[0].down_velocity is not None:
            fig.append_trace(tr_inertial_down, 3, 1)
    fig.append_trace(tr_dr_east, 2, 1)
    fig.append_trace(tr_dr_down, 3, 1)
    fig.append_trace(tr_dr_x, 1, 2)
    fig.append_trace(tr_dr_y, 2, 2)
    fig.append_trace(tr_dr_z, 3, 2)
    fig["layout"]["xaxis1"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis2"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis3"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis4"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis5"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis6"].update(title="Epoch time, s", tickformat=".3f")

    fig["layout"]["yaxis1"].update(title="Velocity, m/s")
    fig["layout"]["yaxis2"].update(title="Velocity, m/s")
    fig["layout"]["yaxis3"].update(title="Velocity, m/s")
    fig["layout"]["yaxis4"].update(title="Velocity, m/s")
    fig["layout"]["yaxis5"].update(title="Velocity, m/s")
    fig["layout"]["yaxis6"].update(title="Velocity, m/s")
    fig["layout"].update(
        title="Velocity vs Time Plots (Left column: Inertial frame - north "
        "east down | Right column: Body frame - x y z)",
        dragmode="pan",
        hovermode="closest",
    )
    config = {"scrollZoom": True}
    py.plot(
        fig,
        config=config,
        filename=str(plotlypath / "velocity_vs_time.html"),
        auto_open=False,
    )

    Console.info("... done plotting velocity_vs_time.")


def plot_deadreckoning_vs_time(
    dead_reckoning_dvl_list,
    velocity_inertial_list,
    usbl_list,
    dead_reckoning_centre_list,
    altitude_list,
    depth_list,
    velocity_inertial_sensor_name,
    plotlypath,
):
    # time_dead_reckoning northings eastings depth vs time
    Console.info("Plotting deadreckoning_vs_time...")

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
    tr_dr_northings = create_trace(dr_time, dr_northings, "Northing DVL", "red")
    tr_usbl_northings = create_trace(usbl_time, usbl_northings, "Northing USBL", "blue")
    tr_drc_northings = create_trace(
        dr_centre_time, dr_centre_northings, "Northing Centre", "orange"
    )

    # Easting vs time
    tr_dr_eastings = create_trace(dr_time, dr_eastings, "Easting DVL", "red")
    tr_usbl_eastings = create_trace(usbl_time, usbl_eastings, "Easting USBL", "blue")
    tr_drc_eastings = create_trace(
        dr_centre_time, dr_centre_eastings, "Easting Centre", "orange"
    )

    # Depth vs time
    tr_alt_depth = create_trace(
        altitude_time,
        altitude_depth,
        "Depth  Seafloor (Depth Sensor + Altitude)",
        "red",
    )
    tr_depth = create_trace(depth_time, depth, "Depth Sensor", "purple")
    tr_usbl_depth = create_trace(usbl_time, usbl_depth, "Depth USBL", "blue")
    tr_dr_depth = create_trace(
        dr_centre_time, dr_centre_depth, "Depth Centre", "orange"
    )

    # Altitude vs time
    tr_alt = create_trace(altitude_time, altitude_alt, "Altitude", "red")

    fig = subplots.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Northings", "Eastings", "Depth", "Altitude"),
        print_grid=False,
    )

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
            inertial_time,
            inertial_northings,
            "Northing {}".format(velocity_inertial_sensor_name),
            "green",
        )
        tr_vi_eastings = create_trace(
            inertial_time,
            inertial_eastings,
            "Easting {}".format(velocity_inertial_sensor_name),
            "green",
        )
        fig.append_trace(tr_vi_northings, 1, 1)
        fig.append_trace(tr_vi_eastings, 1, 2)

    fig["layout"]["xaxis1"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis2"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis3"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis4"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["yaxis1"].update(title="Northing, m")
    fig["layout"]["yaxis2"].update(title="Easting, m")
    fig["layout"]["yaxis3"].update(title="Depth, m", autorange="reversed")
    fig["layout"]["yaxis4"].update(title="Altitude, m")
    fig["layout"].update(
        title="Deadreckoning vs Time", dragmode="pan", hovermode="closest"
    )
    config = {"scrollZoom": True}
    py.plot(
        fig,
        config=config,
        filename=str(plotlypath / "deadreckoning_vs_time.html"),
        auto_open=False,
    )

    Console.info("... done plotting deadreckoning_vs_time.")


# pf uncertainty plotly
# maybe make a slider plot for this, or a dot projection slider
def plot_pf_uncertainty(
    pf_fusion_dvl_list,
    pf_northings_std,
    pf_eastings_std,
    pf_yaw_std,
    plotlypath,
):
    Console.info("Plotting pf_uncertainty...")
    pf_time = [i.epoch_timestamp for i in pf_fusion_dvl_list]

    tr_pf_northings_std = create_trace(
        pf_time, pf_northings_std, "northings_std (m)", "red"
    )
    tr_pf_eastings_std = create_trace(
        pf_time, pf_eastings_std, "eastings_std (m)", "blue"
    )
    tr_pf_yaw_std = create_trace(pf_time, pf_yaw_std, "yaw_std (deg)", "green")
    layout = go.Layout(
        title="Particle Filter Uncertainty Plot",
        hovermode="closest",
        xaxis=dict(title="Epoch time, s", tickformat=".3f"),
        yaxis=dict(title="Degrees or Metres"),
        dragmode="pan",
    )
    config = {"scrollZoom": True}
    fig = go.Figure(
        data=[tr_pf_northings_std, tr_pf_eastings_std, tr_pf_yaw_std],
        layout=layout,
    )
    py.plot(
        fig,
        config=config,
        filename=str(plotlypath / "pf_uncertainty.html"),
        auto_open=False,
    )

    Console.info("... done plotting pf_uncertainty.")


#  EKF uncertainty plotly
def plot_ekf_states_and_std_vs_time(
    ekf_states: List[EkfState],
    output_folder: Path,
):
    Console.info("Plotting EKF states with std vs. time...")
    ekf_time = [i.time for i in ekf_states]
    ekf_northings = [i.state[Index.X, 0] for i in ekf_states]
    ekf_eastings = [i.state[Index.Y, 0] for i in ekf_states]
    ekf_depths = [i.state[Index.Z, 0] for i in ekf_states]
    ekf_northing_stds = [i.northing_std() for i in ekf_states]
    ekf_easting_stds = [i.easting_std() for i in ekf_states]
    ekf_depth_stds = [i.depth_std() for i in ekf_states]

    ekf_roll_deg = [180 / math.pi * i.state[Index.ROLL, 0] for i in ekf_states]
    ekf_pitch_deg = [180 / math.pi * i.state[Index.PITCH, 0] for i in ekf_states]
    ekf_yaw_deg = [180 / math.pi * i.state[Index.YAW, 0] for i in ekf_states]
    ekf_roll_std_deg = [i.roll_std_deg() for i in ekf_states]
    ekf_pitch_std_deg = [i.pitch_std_deg() for i in ekf_states]
    ekf_yaw_std_deg = [i.yaw_std_deg() for i in ekf_states]

    ekf_surge_velocities = [i.state[Index.VX, 0] for i in ekf_states]
    ekf_sway_velocities = [i.state[Index.VY, 0] for i in ekf_states]
    ekf_heave_velocities = [i.state[Index.VZ, 0] for i in ekf_states]
    ekf_surge_velocities_std = [i.surge_velocity_std() for i in ekf_states]
    ekf_sway_velocities_std = [i.sway_velocity_std() for i in ekf_states]
    ekf_heave_velocities_std = [i.heave_velocity_std() for i in ekf_states]

    threads = []
    args = [
        output_folder,
        ekf_time,
        ekf_northings,
        ekf_eastings,
        ekf_depths,
        ekf_northing_stds,
        ekf_easting_stds,
        ekf_depth_stds,
    ]
    t = threading.Thread(target=plot_position_with_std_vs_time, args=args)
    t.start()
    threads.append(t)

    args = [
        output_folder,
        ekf_time,
        ekf_roll_deg,
        ekf_pitch_deg,
        ekf_yaw_deg,
        ekf_roll_std_deg,
        ekf_pitch_std_deg,
        ekf_yaw_std_deg,
    ]
    t = threading.Thread(target=plot_orientation_with_std_vs_time, args=args)
    t.start()
    threads.append(t)

    args = [
        output_folder,
        ekf_time,
        ekf_surge_velocities,
        ekf_sway_velocities,
        ekf_heave_velocities,
        ekf_surge_velocities_std,
        ekf_sway_velocities_std,
        ekf_heave_velocities_std,
    ]
    t = threading.Thread(target=plot_velocity_with_std_vs_time, args=args)
    t.start()
    threads.append(t)

    for t in threads:
        t.join()

    Console.info("... done plotting EKF states with std vs. time.")


def plot_cameras_vs_time(cameras: List[Camera], output_folder: Path):
    Console.info("Plotting camerass vs. time...")
    timestamps = [i.epoch_timestamp for i in cameras]

    northings = [i.northings for i in cameras]
    eastings = [i.eastings for i in cameras]
    depths = [i.depth for i in cameras]
    northing_stds = [math.sqrt(max(0, i.covariance[0, 0])) for i in cameras]
    easting_stds = [math.sqrt(max(0, i.covariance[1, 1])) for i in cameras]
    depth_stds = [math.sqrt(max(0, i.covariance[2, 2])) for i in cameras]

    roll_deg = [i.roll for i in cameras]
    pitch_deg = [i.pitch for i in cameras]
    yaw_deg = [i.yaw for i in cameras]
    roll_stds_deg = [
        180 / math.pi * math.sqrt(max(0, i.covariance[3, 3])) for i in cameras
    ]
    pitch_stds_deg = [
        180 / math.pi * math.sqrt(max(0, i.covariance[4, 4])) for i in cameras
    ]
    yaw_stds_deg = [
        180 / math.pi * math.sqrt(max(0, i.covariance[5, 5])) for i in cameras
    ]

    threads = []
    args = [
        output_folder,
        timestamps,
        northings,
        eastings,
        depths,
        northing_stds,
        easting_stds,
        depth_stds,
    ]
    t = threading.Thread(target=plot_position_with_std_vs_time, args=args)
    t.start()
    threads.append(t)

    args = [
        output_folder,
        timestamps,
        roll_deg,
        pitch_deg,
        yaw_deg,
        roll_stds_deg,
        pitch_stds_deg,
        yaw_stds_deg,
    ]
    t = threading.Thread(target=plot_orientation_with_std_vs_time, args=args)
    t.start()
    threads.append(t)

    for t in threads:
        t.join()

    Console.info("... done plotting cameras vs. time.")


def plot_synced_states_and_ekf_list_and_std_from_ekf_vs_time(
    states: List[Camera], ekf_list: List[Camera], output_folder: Path
):
    Console.info("Plotting synced states and std from list of covariances vs. time...")
    states_timestamps = [i.epoch_timestamp for i in states]
    ekf_timestamps = [i.epoch_timestamp for i in ekf_list]
    assert len(states_timestamps) == len(ekf_timestamps)
    for st, et in zip(states_timestamps, ekf_timestamps):
        assert abs(st - et) < 0.01

    northings = [i.northings for i in states]
    eastings = [i.eastings for i in states]
    depths = [i.depth for i in states]
    states_northing_stds = [i.northing_std_from_cov() for i in states]
    states_easting_stds = [i.easting_std_from_cov() for i in states]
    states_depth_stds = [i.depth_std_from_cov() for i in states]
    ekf_northing_stds = [i.northings_std for i in ekf_list]
    ekf_easting_stds = [i.eastings_std for i in ekf_list]
    ekf_depth_stds = [i.depth_std for i in ekf_list]
    ekf_northings = [i.northings for i in ekf_list]
    ekf_eastings = [i.eastings for i in ekf_list]
    ekf_depths = [i.depth for i in ekf_list]

    roll_deg = [i.roll for i in states]
    pitch_deg = [i.pitch for i in states]
    yaw_deg = [i.yaw for i in states]
    states_roll_stds_deg = [i.roll_std_from_cov_deg() for i in states]
    states_pitch_stds_deg = [i.pitch_std_from_cov_deg() for i in states]
    states_yaw_stds_deg = [i.yaw_std_from_cov_deg() for i in states]
    ekf_roll_stds_deg = [i.roll_std for i in ekf_list]
    ekf_pitch_stds_deg = [i.pitch_std for i in ekf_list]
    ekf_yaw_stds_deg = [i.yaw_std for i in ekf_list]
    ekf_roll_deg = [i.roll for i in ekf_list]
    ekf_pitch_deg = [i.pitch for i in ekf_list]
    ekf_yaw_deg = [i.yaw for i in ekf_list]

    threads = []
    args = [
        output_folder,
        states_timestamps,
        northings,
        eastings,
        depths,
        ekf_northing_stds,
        ekf_easting_stds,
        ekf_depth_stds,
        ekf_northings,
        ekf_eastings,
        ekf_depths,
        states_northing_stds,
        states_easting_stds,
        states_depth_stds,
    ]
    t = threading.Thread(target=plot_position_with_std_vs_time, args=args)
    t.start()
    threads.append(t)

    args = [
        output_folder,
        states_timestamps,
        roll_deg,
        pitch_deg,
        yaw_deg,
        ekf_roll_stds_deg,
        ekf_pitch_stds_deg,
        ekf_yaw_stds_deg,
        ekf_roll_deg,
        ekf_pitch_deg,
        ekf_yaw_deg,
        states_roll_stds_deg,
        states_pitch_stds_deg,
        states_yaw_stds_deg,
    ]
    t = threading.Thread(target=plot_orientation_with_std_vs_time, args=args)
    t.start()
    threads.append(t)

    for t in threads:
        t.join()

    Console.info(
        "... done plotting synced states and std from list of covariances vs. time."
    )


def plot_position_with_std_vs_time(
    output_folder: Path,
    timestamps: List[float],
    northings: List[float],
    eastings: List[float],
    depths: List[float],
    northings_std: List[float],
    eastings_std: List[float],
    depths_std: List[float],
    northings_2: Optional[List[float]] = None,
    eastings_2: Optional[List[float]] = None,
    depths_2: Optional[List[float]] = None,
    northings_std_1: Optional[List[float]] = None,
    eastings_std_1: Optional[List[float]] = None,
    depths_std_1: Optional[List[float]] = None,
):
    assert len(timestamps) == len(northings) == len(eastings) == len(depths)
    assert len(timestamps) == len(northings_std) == len(eastings_std) == len(depths_std)
    if northings_2:
        assert len(timestamps) == len(northings_2) == len(eastings_2) == len(depths_2)
    if northings_std_1:
        assert (
            len(timestamps)
            == len(northings_std_1)
            == len(eastings_std_1)
            == len(depths_std_1)
        )

    northings_plus_sigma = [a_i + b_i for a_i, b_i in zip(northings, northings_std)]
    northings_minus_sigma = [a_i - b_i for a_i, b_i in zip(northings, northings_std)]
    eastings_plus_sigma = [a_i - b_i for a_i, b_i in zip(eastings, eastings_std)]
    eastings_minus_sigma = [a_i + b_i for a_i, b_i in zip(eastings, eastings_std)]
    depths_plus_sigma = [a_i + b_i for a_i, b_i in zip(depths, depths_std)]
    depths_minus_sigma = [a_i - b_i for a_i, b_i in zip(depths, depths_std)]

    tr_northings = create_trace(
        timestamps, northings, "northings (m)", "red", fill="tonexty"
    )
    tr_northings_upper_bound = create_trace(
        timestamps, northings_plus_sigma, "northing+sigma", "red", True, "tonexty", True
    )
    tr_northings_lower_bound = create_trace(
        timestamps,
        northings_minus_sigma,
        "northing-sigma",
        "red",
        True,
        "tonexty",
        True,
    )
    if northings_2:
        tr_northings_2 = create_trace(
            timestamps, northings_2, "northings_2", "red", is_2nd_line=True
        )

    tr_eastings = create_trace(
        timestamps, eastings, "eastings (m)", "green", fill="tonexty"
    )
    tr_eastings_lower_bound = create_trace(
        timestamps, eastings_plus_sigma, "easting-sigma", "green", True, "tonexty", True
    )
    tr_eastings_upper_bound = create_trace(
        timestamps,
        eastings_minus_sigma,
        "easting+sigma",
        "green",
        True,
        "tonexty",
        True,
    )
    if eastings_2:
        tr_eastings_2 = create_trace(
            timestamps, eastings_2, "eastings_2", "green", is_2nd_line=True
        )

    tr_depths = create_trace(timestamps, depths, "depth (m)", "blue", fill="tonexty")
    tr_depths_upper_bound = create_trace(
        timestamps, depths_plus_sigma, "depth+sigma", "blue", True, "tonexty", True
    )
    tr_depths_lower_bound = create_trace(
        timestamps, depths_minus_sigma, "depth-sigma", "blue", True, "tonexty", True
    )
    if depths_2:
        tr_depths_2 = create_trace(
            timestamps, depths_2, "depths_2", "blue", is_2nd_line=True
        )

    tr_northings_std = create_trace(
        timestamps, northings_std, "northings std ekf (m)", "red"
    )
    tr_eastings_std = create_trace(
        timestamps, eastings_std, "eastings std ekf (m)", "green"
    )
    tr_depths_std = create_trace(timestamps, depths_std, "depths std ekf (m)", "blue")

    if northings_std_1:
        tr_northings_std_1 = create_trace(
            timestamps,
            northings_std_1,
            "northings std sub (m)",
            "red",
            is_2nd_line=True,
        )
        tr_eastings_std_1 = create_trace(
            timestamps,
            eastings_std_1,
            "eastings std sub (m)",
            "green",
            is_2nd_line=True,
        )
        tr_depths_std_1 = create_trace(
            timestamps, depths_std_1, "depths std sub (m)", "blue", is_2nd_line=True
        )

    fig = subplots.make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Northing",
            "Easting",
            "Depth",
            "Northing standard deviation",
            "Easting standard deviation",
            "Depth standard deviation",
        ),
        print_grid=False,
    )
    fig.append_trace(tr_northings, 1, 1)
    fig.append_trace(tr_northings_upper_bound, 1, 1)
    fig.append_trace(tr_northings_lower_bound, 1, 1)
    if northings_2:
        fig.append_trace(tr_northings_2, 1, 1)
    fig.append_trace(tr_northings_std, 2, 1)
    if northings_std_1:
        fig.append_trace(tr_northings_std_1, 2, 1)

    fig.append_trace(tr_eastings, 1, 2)
    fig.append_trace(tr_eastings_upper_bound, 1, 2)
    fig.append_trace(tr_eastings_lower_bound, 1, 2)
    if eastings_2:
        fig.append_trace(tr_eastings_2, 1, 2)
    fig.append_trace(tr_eastings_std, 2, 2)
    if eastings_std_1:
        fig.append_trace(tr_eastings_std_1, 2, 2)

    fig.append_trace(tr_depths, 1, 3)
    fig.append_trace(tr_depths_upper_bound, 1, 3)
    fig.append_trace(tr_depths_lower_bound, 1, 3)
    if depths_2:
        fig.append_trace(tr_depths_2, 1, 3)
    fig.append_trace(tr_depths_std, 2, 3)
    if depths_std_1:
        fig.append_trace(tr_depths_std_1, 2, 3)

    fig["layout"]["xaxis1"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["xaxis2"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["xaxis3"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["xaxis4"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["xaxis5"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["xaxis6"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["yaxis1"].update(title="Northing, m")
    fig["layout"]["yaxis2"].update(title="Easting, m")
    fig["layout"]["yaxis3"].update(title="Depth, m", autorange="reversed")
    fig["layout"]["yaxis4"].update(title="Northing std, m")
    fig["layout"]["yaxis5"].update(title="Easting std, m")
    fig["layout"]["yaxis6"].update(title="Depth std, m")
    fig["layout"].update(
        title="EKF positions / position uncertainties vs Time Plots",
        dragmode="pan",
        hovermode="closest",
    )

    config = {"scrollZoom": True}
    output_folder.mkdir(parents=True, exist_ok=True)
    py.plot(
        fig,
        config=config,
        filename=str(output_folder / "positions_with_std_vs_time.html"),
        auto_open=False,
    )


def plot_orientation_with_std_vs_time(
    output_folder: Path,
    timestamps: List[float],
    roll_deg: List[float],
    pitch_deg: List[float],
    yaw_deg: List[float],
    roll_stds_deg: List[float],
    pitch_stds_deg: List[float],
    yaw_stds_deg: List[float],
    roll_deg_2: Optional[List[float]] = None,
    pitch_deg_2: Optional[List[float]] = None,
    yaw_deg_2: Optional[List[float]] = None,
    roll_std_deg_1: Optional[List[float]] = None,
    pitch_std_deg_1: Optional[List[float]] = None,
    yaw_std_deg_1: Optional[List[float]] = None,
):
    assert len(timestamps) == len(roll_deg) == len(pitch_deg) == len(pitch_deg)
    assert (
        len(timestamps)
        == len(roll_stds_deg)
        == len(pitch_stds_deg)
        == len(yaw_stds_deg)
    )
    if roll_deg_2:
        assert len(timestamps) == len(roll_deg_2) == len(pitch_deg_2) == len(yaw_deg_2)
    if roll_std_deg_1:
        assert (
            len(timestamps)
            == len(roll_std_deg_1)
            == len(pitch_std_deg_1)
            == len(yaw_std_deg_1)
        )

    roll_plus_sigma = [a_i + b_i for a_i, b_i in zip(roll_deg, roll_stds_deg)]
    roll_minus_sigma = [a_i - b_i for a_i, b_i in zip(roll_deg, roll_stds_deg)]
    pitch_plus_sigma = [a_i + b_i for a_i, b_i in zip(pitch_deg, pitch_stds_deg)]
    pitch_minus_sigma = [a_i - b_i for a_i, b_i in zip(pitch_deg, pitch_stds_deg)]
    yaw_plus_sigma = [a_i + b_i for a_i, b_i in zip(yaw_deg, yaw_stds_deg)]
    yaw_minus_sigma = [a_i - b_i for a_i, b_i in zip(yaw_deg, yaw_stds_deg)]

    tr_roll = create_trace(timestamps, roll_deg, "roll [deg]", "red", fill="tonexty")
    tr_roll_std_uppper = create_trace(
        timestamps, roll_plus_sigma, "roll+sigma", "red", True, "tonexty", True
    )
    tr_roll_std_lower = create_trace(
        timestamps, roll_minus_sigma, "roll+sigma", "red", True, "tonexty", True
    )
    tr_roll_std = create_trace(timestamps, roll_stds_deg, "roll std ekf [deg]", "red")
    if roll_deg_2:
        tr_roll_2 = create_trace(
            timestamps,
            roll_deg_2,
            "roll_2 [deg]",
            "red",
            fill="tonexty",
            is_2nd_line=True,
        )
    if roll_std_deg_1:
        tr_roll_std_1 = create_trace(
            timestamps, roll_std_deg_1, "roll std sub [deg]", "red", is_2nd_line=True
        )

    tr_pitch = create_trace(
        timestamps, pitch_deg, "pitch [deg]", "green", fill="tonexty"
    )
    tr_pitch_std_uppper = create_trace(
        timestamps, pitch_plus_sigma, "pitch+sigma", "green", True, "tonexty", True
    )
    tr_pitch_std_lower = create_trace(
        timestamps, pitch_minus_sigma, "pitch+sigma", "green", True, "tonexty", True
    )
    tr_pitch_std = create_trace(
        timestamps, pitch_stds_deg, "pitch std ekf [deg]", "green"
    )
    if pitch_deg_2:
        tr_pitch_2 = create_trace(
            timestamps,
            pitch_deg_2,
            "pitch_2 [deg]",
            "green",
            fill="tonexty",
            is_2nd_line=True,
        )
    if pitch_std_deg_1:
        tr_pitch_std_1 = create_trace(
            timestamps,
            pitch_std_deg_1,
            "pitch std sub [deg]",
            "green",
            is_2nd_line=True,
        )

    tr_yaw = create_trace(timestamps, yaw_deg, "yaw [deg]", "blue", fill="tonexty")
    tr_yaw_std_uppper = create_trace(
        timestamps, yaw_plus_sigma, "yaw+sigma", "blue", True, "tonexty", True
    )
    tr_yaw_std_lower = create_trace(
        timestamps, yaw_minus_sigma, "yaw+sigma", "blue", True, "tonexty", True
    )
    tr_yaw_std = create_trace(timestamps, yaw_stds_deg, "yaw std ekf [deg]", "blue")
    if yaw_deg_2:
        tr_yaw_2 = create_trace(
            timestamps,
            yaw_deg_2,
            "yaw_2 [deg]",
            "blue",
            fill="tonexty",
            is_2nd_line=True,
        )
    if yaw_std_deg_1:
        tr_yaw_std_1 = create_trace(
            timestamps, yaw_std_deg_1, "yaw std sub [deg]", "blue", is_2nd_line=True
        )

    fig = subplots.make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Roll",
            "Pitch",
            "Yaw",
            "Roll standard deviation",
            "Pitch standard deviation",
            "Yaw standard deviation",
        ),
        print_grid=False,
    )
    fig.append_trace(tr_roll, 1, 1)
    fig.append_trace(tr_roll_std_uppper, 1, 1)
    fig.append_trace(tr_roll_std_lower, 1, 1)
    if roll_deg_2:
        fig.append_trace(tr_roll_2, 1, 1)
    fig.append_trace(tr_roll_std, 2, 1)
    if roll_std_deg_1:
        fig.append_trace(tr_roll_std_1, 2, 1)

    fig.append_trace(tr_pitch, 1, 2)
    fig.append_trace(tr_pitch_std_uppper, 1, 2)
    fig.append_trace(tr_pitch_std_lower, 1, 2)
    if pitch_deg_2:
        fig.append_trace(tr_pitch_2, 1, 2)
    fig.append_trace(tr_pitch_std, 2, 2)
    if pitch_std_deg_1:
        fig.append_trace(tr_pitch_std_1, 2, 2)

    fig.append_trace(tr_yaw, 1, 3)
    fig.append_trace(tr_yaw_std_uppper, 1, 3)
    fig.append_trace(tr_yaw_std_lower, 1, 3)
    if yaw_deg_2:
        fig.append_trace(tr_yaw_2, 1, 3)
    fig.append_trace(tr_yaw_std, 2, 3)
    if yaw_std_deg_1:
        fig.append_trace(tr_yaw_std_1, 2, 3)

    fig["layout"]["xaxis1"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["xaxis2"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["xaxis3"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["xaxis4"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["xaxis5"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["xaxis6"].update(title="Epoch time, s", tickformat=".f")
    fig["layout"]["yaxis1"].update(title="Roll, deg")
    fig["layout"]["yaxis2"].update(title="Pitch, deg")
    fig["layout"]["yaxis3"].update(title="Yaw, deg")
    fig["layout"]["yaxis4"].update(title="Roll std, deg")
    fig["layout"]["yaxis5"].update(title="Pitch std, deg")
    fig["layout"]["yaxis6"].update(title="Yaw std, deg")
    fig["layout"].update(
        title="Orientation / orientation uncertainties vs. Time Plots",
        dragmode="pan",
        hovermode="closest",
    )

    config = {"scrollZoom": True}
    output_folder.mkdir(parents=True, exist_ok=True)
    py.plot(
        fig,
        config=config,
        filename=str(output_folder / "orientation_with_std_vs_time.html"),
        auto_open=False,
    )


def plot_velocity_with_std_vs_time(
    output_folder: Path,
    ekf_time: List[float],
    ekf_surge_speeds: List[float],
    ekf_sway_speeds: List[float],
    ekf_down_speeds: List[float],
    ekf_surge_speeds_std: List[float],
    ekf_sway_speeds_std: List[float],
    ekf_down_speeds_std: List[float],
):
    tr_ekf_surge_speeds = create_trace(
        ekf_time, ekf_surge_speeds, "surge speeds (m/s)", "red", fill="tonexty"
    )
    tr_ekf_surge_speeds_upper_bound = create_trace(
        ekf_time,
        [a_i + b_i for a_i, b_i in zip(ekf_surge_speeds, ekf_surge_speeds_std)],
        "surge_speed+sigma",
        "red",
        True,
        "tonexty",
        True,
    )
    tr_ekf_surge_speeds_lower_bound = create_trace(
        ekf_time,
        [a_i - b_i for a_i, b_i in zip(ekf_surge_speeds, ekf_surge_speeds_std)],
        "surge_speed-sigma",
        "red",
        True,
        "tonexty",
        True,
    )

    tr_ekf_sway_speeds = create_trace(
        ekf_time, ekf_sway_speeds, "sway speeds (m/s)", "green", fill="tonexty"
    )
    tr_ekf_sway_speeds_upper_bound = create_trace(
        ekf_time,
        [a_i + b_i for a_i, b_i in zip(ekf_sway_speeds, ekf_sway_speeds_std)],
        "sway_speed+sigma",
        "red",
        True,
        "tonexty",
        True,
    )
    tr_ekf_sway_speeds_lower_bound = create_trace(
        ekf_time,
        [a_i - b_i for a_i, b_i in zip(ekf_sway_speeds, ekf_sway_speeds_std)],
        "sway_speed-sigma",
        "red",
        True,
        "tonexty",
        True,
    )

    tr_ekf_down_speeds = create_trace(
        ekf_time, ekf_down_speeds, "vertical speeds (m/s)", "blue", fill="tonexty"
    )
    tr_ekf_down_speeds_upper_bound = create_trace(
        ekf_time,
        [a_i + b_i for a_i, b_i in zip(ekf_down_speeds, ekf_down_speeds_std)],
        "down_speed+sigma",
        "red",
        True,
        "tonexty",
        True,
    )
    tr_ekf_down_speeds_lower_bound = create_trace(
        ekf_time,
        [a_i - b_i for a_i, b_i in zip(ekf_down_speeds, ekf_down_speeds_std)],
        "down_speed-sigma",
        "red",
        True,
        "tonexty",
        True,
    )

    tr_ekf_surge_speeds_std = create_trace(
        ekf_time, ekf_surge_speeds_std, "surge_speeds_std (m/s)", "red"
    )
    tr_ekf_sway_speeds_std = create_trace(
        ekf_time, ekf_sway_speeds_std, "sway_speeds_std (m/s)", "green"
    )
    tr_ekf_down_speeds_std = create_trace(
        ekf_time, ekf_down_speeds_std, "down_speeds_std (m/s)", "blue"
    )

    fig2 = subplots.make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Surge speed",
            "Sway speed",
            "Down speed",
            "Surge speed standard deviation",
            "Sway speed standard deviation",
            "Down speed standard deviation",
        ),
        print_grid=False,
    )
    fig2.append_trace(tr_ekf_surge_speeds, 1, 1)
    fig2.append_trace(tr_ekf_surge_speeds_upper_bound, 1, 1)
    fig2.append_trace(tr_ekf_surge_speeds_lower_bound, 1, 1)
    fig2.append_trace(tr_ekf_surge_speeds_std, 2, 1)
    fig2.append_trace(tr_ekf_sway_speeds, 1, 2)
    fig2.append_trace(tr_ekf_sway_speeds_upper_bound, 1, 2)
    fig2.append_trace(tr_ekf_sway_speeds_lower_bound, 1, 2)
    fig2.append_trace(tr_ekf_sway_speeds_std, 2, 2)
    fig2.append_trace(tr_ekf_down_speeds, 1, 3)
    fig2.append_trace(tr_ekf_down_speeds_upper_bound, 1, 3)
    fig2.append_trace(tr_ekf_down_speeds_lower_bound, 1, 3)
    fig2.append_trace(tr_ekf_down_speeds_std, 2, 3)
    fig2["layout"]["xaxis1"].update(title="Epoch time, s", tickformat=".f")
    fig2["layout"]["xaxis2"].update(title="Epoch time, s", tickformat=".f")
    fig2["layout"]["xaxis3"].update(title="Epoch time, s", tickformat=".f")
    fig2["layout"]["xaxis4"].update(title="Epoch time, s", tickformat=".f")
    fig2["layout"]["xaxis5"].update(title="Epoch time, s", tickformat=".f")
    fig2["layout"]["xaxis6"].update(title="Epoch time, s", tickformat=".f")
    fig2["layout"]["yaxis1"].update(title="Surge speed, m/s")
    fig2["layout"]["yaxis2"].update(title="Sway speed, m/s")
    fig2["layout"]["yaxis3"].update(title="Down speed, m/s", autorange="reversed")
    fig2["layout"]["yaxis4"].update(title="Surge speed std, m/s")
    fig2["layout"]["yaxis5"].update(title="Sway speed std, m/s")
    fig2["layout"]["yaxis6"].update(title="Down speed std, m/s")
    fig2["layout"].update(
        title="EKF speeds / speed uncertainties vs Time Plots",
        dragmode="pan",
        hovermode="closest",
    )

    config = {"scrollZoom": True}
    output_folder.mkdir(parents=True, exist_ok=True)
    py.plot(
        fig2,
        config=config,
        filename=str(output_folder / "velocities_with_std_vs_time.html"),
        auto_open=False,
    )


def plot_ekf_rejected_measurements(rejected_measurements, plotlypath):
    Console.info("Plotting measurements rejected in EKF...")

    trace_list = []
    for key in rejected_measurements.keys():
        x_values = rejected_measurements.get(key)
        y_values = [key] * len(x_values)
        text_list = [
            str(t)
            + " (epoch) | "
            + time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_utctime(t))
            + " (UTC)"
            for t in x_values
        ]
        trace_list.append(create_trace_2(x_values, y_values, text_list, key))

    config = {"scrollZoom": True}
    fig = go.Figure(data=list(trace_list))
    fig.update_layout(
        title="Timestamps of sensor measurements rejected in the EKF due to exceeding the Mahalanobis distance "
        "threshold",
        xaxis=dict(exponentformat="none", title="Epoch time, s"),
    )
    py.plot(
        fig,
        config=config,
        filename=str(plotlypath / "ekf_rejected_measurements.html"),
        auto_open=False,
    )

    Console.info("... done plotting measurements rejected in EKF.")


def plot_sensor_uncertainty(
    orientation_list,
    velocity_body_list,
    depth_list,
    usbl_list,
    velocity_inertial_list,
    velocity_inertial_sensor_name,
    plotlypath,
):
    Console.info("Plotting sensor uncertainty...")
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

    tr_roll = create_trace(ori_time, ori_roll_std, "Roll std", "red")
    tr_pitch = create_trace(ori_time, ori_pitch_std, "Pitch std", "green")
    tr_yaw = create_trace(ori_time, ori_yaw_std, "Yaw std", "blue")

    tr_bv_x = create_trace(bv_time, bv_x_vel_std, "x velocity std", "red")
    tr_bv_y = create_trace(bv_time, bv_y_vel_std, "y velocity std", "green")
    tr_bv_z = create_trace(bv_time, bv_z_vel_std, "z velocity std", "blue")

    tr_lat = create_trace(usbl_time, usbl_lat_std, "Lat std usbl", "red")
    tr_lon = create_trace(usbl_time, usbl_lon_std, "Lon std usbl", "green")
    tr_depth = create_trace(depth_time, depth_std, "Depth std", "blue")

    tr_n = create_trace(usbl_time, usbl_northing_std, "northing std usbl", "red")
    tr_e = create_trace(usbl_time, usbl_easting_std, "easting std usbl", "green")

    if len(velocity_inertial_list) > 0:
        if iv_n_vel_std[0] is not None:
            tr_iv_n = create_trace(
                iv_time, iv_n_vel_std, "north velocity std inertial", "red"
            )
        if iv_e_vel_std[0] is not None:
            tr_iv_e = create_trace(
                iv_time, iv_e_vel_std, "east velocity std inertial", "green"
            )
        if iv_d_vel_std[0] is not None:
            tr_iv_d = create_trace(
                iv_time, iv_d_vel_std, "down velocity std inertial", "blue"
            )

    fig = subplots.make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Orientation uncertainties",
            "DVL uncertainties",
            "USBL uncertainties",
            "Depth uncertainties",
            "{} uncertainties".format(velocity_inertial_sensor_name),
            "USBL uncertainties",
        ),
        print_grid=False,
    )
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
        if iv_n_vel_std[0] is not None:
            fig.append_trace(tr_iv_n, 2, 2)
        if iv_e_vel_std[0] is not None:
            fig.append_trace(tr_iv_e, 2, 2)
        if iv_d_vel_std[0] is not None:
            fig.append_trace(tr_iv_d, 2, 2)

    fig.append_trace(tr_n, 2, 3)
    fig.append_trace(tr_e, 2, 3)

    fig["layout"]["xaxis1"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis2"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis3"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis4"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis5"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["xaxis6"].update(title="Epoch time, s", tickformat=".3f")
    fig["layout"]["yaxis1"].update(title="Angle, degrees")
    fig["layout"]["yaxis2"].update(title="Velocity, m/s")
    fig["layout"]["yaxis3"].update(title="LatLong, degrees")
    fig["layout"]["yaxis4"].update(title="Depth, m")
    fig["layout"]["yaxis5"].update(title="Velocity, m/s")
    fig["layout"]["yaxis6"].update(title="NorthEast, m")
    fig["layout"].update(title="Uncertainty Plots", dragmode="pan", hovermode="closest")
    config = {"scrollZoom": True}
    py.plot(
        fig,
        config=config,
        filename=str(plotlypath / "sensor_uncertainties_plot.html"),
        auto_open=False,
    )

    Console.info("... done plotting sensor uncertainty.")


def plot_2d_deadreckoning(
    camera1_list,
    camera1_ekf_list,
    dead_reckoning_centre_list,
    dead_reckoning_dvl_list,
    pf_fusion_centre_list,
    ekf_centre_list,
    camera1_pf_list,
    pf_fusion_dvl_list,
    particles_time_interval,
    pf_particles_list,
    usbl_list_no_dist_filter,
    usbl_list,
    plotlypath,
):
    # DR plotly slider *include toggle button that switches between lat lon
    # and north east
    Console.info("Plotting auv_path...")

    plotly_list_static = []
    plotly_list_slider = []
    if len(camera1_list) > 1:
        plotly_list_static.append(["dr_camera1", camera1_list, True])
        plotly_list_slider.append(["dr_camera1", camera1_list, True])
    if len(camera1_ekf_list) > 1:
        plotly_list_static.append(["ekf_camera1", camera1_ekf_list, True])
        plotly_list_slider.append(["ekf_camera1", camera1_ekf_list, True])
    if len(dead_reckoning_centre_list) > 1:
        plotly_list_static.append(
            ["dr_centre", dead_reckoning_centre_list, "legendonly"]
        )
        # centre and dvl lists contain very large number of points -> plot in static plot but not in slider plot
    if len(dead_reckoning_dvl_list) > 1:
        plotly_list_static.append(["dr_dvl", dead_reckoning_dvl_list, "legendonly"])
    if len(ekf_centre_list) > 1:
        plotly_list_static.append(["ekf_dvl", ekf_centre_list, "legendonly"])

    if len(usbl_list_no_dist_filter) > 1:
        plotly_list_static.append(
            ["usbl_without_distance_filter", usbl_list_no_dist_filter, "legendonly"]
        )
    if len(usbl_list) > 1:
        plotly_list_static.append(["usbl", usbl_list, True])
        plotly_list_slider.append(["usbl", usbl_list, True])

    figure = {"data": [], "layout": {}, "frames": []}

    # fill in most of layout
    figure["layout"]["xaxis"] = {"title": "Eastings,m"}
    figure["layout"]["yaxis"] = {"title": "Northings,m", "scaleanchor": "x"}
    figure["layout"]["hovermode"] = "closest"
    figure["layout"]["dragmode"] = "pan"

    for i in plotly_list_static:
        print("Processing ", i[0])
        try:
            make_data(
                figure,
                i[0],
                [float(j.eastings) for j in i[1]],
                [float(j.northings) for j in i[1]],
                visibility=i[2],
            )
        except TypeError:
            Console.error("TypeError in plotting ", i[0])

    if len(pf_fusion_centre_list) > 1:
        make_data(
            figure,
            "pf_camera1",
            [float(i.eastings) for i in camera1_pf_list],
            [float(i.northings) for i in camera1_pf_list],
            visibility="legendonly",
            hoverinfo="x+y+text",
            hovertext=[
                time.strftime("%H:%M:%S", time.localtime(i.epoch_timestamp))
                for i in camera1_pf_list
            ],
        )
        make_data(
            figure,
            "pf_centre",
            [float(i.eastings) for i in pf_fusion_centre_list],
            [float(i.northings) for i in pf_fusion_centre_list],
            visibility="legendonly",
        )
        make_data(
            figure,
            "pf_dvl",
            [float(i.eastings) for i in pf_fusion_dvl_list],
            [float(i.northings) for i in pf_fusion_dvl_list],
            visibility=True,
            hoverinfo="x+y+text",
            hovertext=[
                time.strftime("%H:%M:%S", time.localtime(i.epoch_timestamp))
                for i in pf_fusion_dvl_list
            ],
        )
        pf_timestamps_interval = []
        pf_eastings_interval = []
        pf_northings_interval = []
        if particles_time_interval is not False:
            for i in pf_particles_list[0]:
                pf_timestamps_interval.append(
                    float(pf_particles_list[0][0].timestamps[0])
                )
                pf_eastings_interval.append(float(i.eastings[0]))
                pf_northings_interval.append(float(i.northings[0]))
            timestamp_value_tracker = pf_particles_list[0][0].timestamps[0]

            for i in range(len(pf_particles_list)):
                # timestamp_index_tracker = 0
                for j in range(len(pf_particles_list[i][0].timestamps)):
                    if (
                        pf_particles_list[i][0].timestamps[j] - timestamp_value_tracker
                    ) > particles_time_interval:
                        for k in pf_particles_list[i]:
                            pf_timestamps_interval.append(float(k.timestamps[j]))
                            pf_eastings_interval.append(float(k.eastings[j]))
                            pf_northings_interval.append(float(k.northings[j]))
                        timestamp_value_tracker = pf_particles_list[i][0].timestamps[j]
            make_data(
                figure,
                "pf_dvl_distribution",
                pf_eastings_interval,
                pf_northings_interval,
                mode="markers",
                visibility=True,
            )
        else:
            resampling_index = 1
            for i in pf_particles_list:
                make_data(
                    figure,
                    "PF_Resampling{}".format(resampling_index),
                    [float(j.eastings) for j in i],
                    [float(j.northings) for j in i],
                    mode="markers",
                    opacity=0.5,
                )
                resampling_index += 1

    config = {"scrollZoom": True}

    py.plot(
        figure,
        config=config,
        filename=str(plotlypath / "auv_path.html"),
        auto_open=False,
    )

    Console.info("... done plotting auv_path.html")
    Console.info("Plotting auv_path_slider.html ...")

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "epoch_timestamp:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    # slider plot
    # might not be robust in the future
    min_timestamp = float("inf")
    max_timestamp = float("-inf")
    for i in plotly_list_slider:
        timestamp_list = [j.epoch_timestamp for j in i[1]]
        if min(timestamp_list) < min_timestamp:
            min_timestamp = min(timestamp_list)
        if max(timestamp_list) > max_timestamp:
            max_timestamp = max(timestamp_list)
    # time_gap = 240
    time_gap = int((max_timestamp - min_timestamp) / 40)
    epoch_timestamps_slider = list(
        range(int(min_timestamp), int(max_timestamp), int(time_gap))
    )

    # make frames
    for i in epoch_timestamps_slider:
        frame = {"data": [], "name": str(i)}

        for j in plotly_list_slider:
            make_frame(
                frame,
                [
                    j[0],
                    [float(k.epoch_timestamp) for k in j[1]],
                    [float(k.eastings) for k in j[1]],
                    [float(k.northings) for k in j[1]],
                ],
                i,
            )
        if len(camera1_pf_list) > 1:
            make_frame(
                frame,
                [
                    "pf_camera1",
                    [float(i.epoch_timestamp) for i in camera1_pf_list],
                    [float(i.eastings) for i in camera1_pf_list],
                    [float(i.northings) for i in camera1_pf_list],
                ],
                i,
            )

        figure["frames"].append(frame)
        slider_step = {
            "args": [
                [i],
                {
                    "frame": {"duration": 300, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            "label": i,
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)

    figure["layout"]["sliders"] = [sliders_dict]
    figure["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500, "redraw": False},
                            "fromcurrent": True,
                            "transition": {
                                "duration": 300,
                                "easing": "quadratic-in-out",
                            },
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    py.plot(
        figure,
        config=config,
        filename=str(plotlypath / "auv_path_slider.html"),
        auto_open=False,
    )

    Console.info("... done plotting auv_path_slider.html.")


def plot_2d_localisation(dr_list, pf_list, ekf_list, eks_list, plotlypath):
    # DR plotly slider *include toggle button that switches between lat long
    # and north east
    Console.info("Plotting auv_path (localisation)...")

    # might not be robust in the future
    min_timestamp = float("inf")
    max_timestamp = float("-inf")

    plotly_list = []
    if len(dr_list) > 1:
        plotly_list.append(["dr", dr_list, "legendonly"])
    if len(pf_list) > 1:
        plotly_list.append(["pf", pf_list, "legendonly"])
    if len(eks_list) > 1:
        plotly_list.append(["ekf", ekf_list, True])
    if len(eks_list) > 1:
        plotly_list.append(["eks", eks_list, True])

    for i in plotly_list:
        timestamp_list = [j.epoch_timestamp for j in i[1]]
        if min(timestamp_list) < min_timestamp:
            min_timestamp = min(timestamp_list)
        if max(timestamp_list) > max_timestamp:
            max_timestamp = max(timestamp_list)

    figure = {"data": [], "layout": {}, "frames": []}

    # fill in most of layout
    figure["layout"]["xaxis"] = {"title": "Eastings,m"}
    figure["layout"]["yaxis"] = {"title": "Northings,m", "scaleanchor": "x"}
    figure["layout"]["hovermode"] = "closest"
    figure["layout"]["dragmode"] = "pan"
    figure["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500, "redraw": False},
                            "fromcurrent": True,
                            "transition": {
                                "duration": 300,
                                "easing": "quadratic-in-out",
                            },
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    for i in plotly_list:
        make_data(
            figure,
            i[0],
            [float(j.eastings) for j in i[1]],
            [float(j.northings) for j in i[1]],
            visibility=i[2],
        )

    config = {"scrollZoom": True}

    py.plot(
        figure,
        config=config,
        filename=str(plotlypath / "auv_localisation.html"),
        auto_open=False,
    )

    Console.info("...plotting auv_path_slider (localisation)...")

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "epoch_timestamp:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    # slider plot
    # time_gap = 240
    time_gap = int((max_timestamp - min_timestamp) / 40)
    epoch_timestamps_slider = list(
        range(int(min_timestamp), int(max_timestamp), int(time_gap))
    )

    # make frames
    for i in epoch_timestamps_slider:
        frame = {"data": [], "name": str(i)}

        for j in plotly_list:
            make_frame(
                frame,
                [
                    j[0],
                    [float(k.epoch_timestamp) for k in j[1]],
                    [float(k.eastings) for k in j[1]],
                    [float(k.northings) for k in j[1]],
                ],
                i,
            )
        figure["frames"].append(frame)
        slider_step = {
            "args": [
                [i],
                {
                    "frame": {"duration": 300, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            "label": i,
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)

    figure["layout"]["sliders"] = [sliders_dict]

    py.plot(
        figure,
        config=config,
        filename=str(plotlypath / "auv_localisation_slider.html"),
        auto_open=False,
    )

    Console.info("... done plotting auv_path (localisation).")

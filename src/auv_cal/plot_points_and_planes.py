# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plyfile import PlyData

import pandas as pd
import matplotlib.pyplot as plt

def read_ply(filename):
    filename = Path(filename)
    f = filename.open("rb")
    plydata = PlyData.read(f)
    # Read vertices
    nVerts = plydata["vertex"].count
    verts = np.zeros((nVerts, 3))
    verts[:, 0] = np.array(plydata["vertex"].data["x"])
    verts[:, 1] = np.array(plydata["vertex"].data["y"])
    verts[:, 2] = np.array(plydata["vertex"].data["z"])
    return verts


def plane_to_rectuangular_grid(plane, ymin, ymax, zmin, zmax):
    """Calculate the x,y,z values of the corners of a rectangular plane

    Drawing::

        .zmin__ _________
        .      |         |
        .      |  plane  |
        .zmax__|_________|
        .      |         |
        .      ymin      ymax

    Parameters
    ----------
    plane : :obj:`list` of :obj:`double`
        Parameters of the plane parametrisation ax+by+cz+d = 0.
    ymin : double
        y-coordinate of top corners
    ymax : double
        y-coordinate of bottom corners
    zmin : double
        z-coordinate of top corners
    zmax : double
        z-coordinate of bottom corners

    Returns
    -------
    np.ndarray
        x, y and z of the 4 corners of the plane as matrices for plotting
    """

    a, b, c, d = plane.tolist()
    yy = np.array([[ymin, ymin], [ymax, ymax]])
    zz = np.array([[zmin, zmax], [zmin, zmax]])
    return (-d - c * zz - b * yy) / a, yy, zz


def plane_to_isosceles_trapezoid_grid(plane, ydistmin, ydistmax, zmin1, zmax1):
    r"""Calculate the x,y,z values of the corners of a plane in the shape of
        the sheet laser

    Drawing::

        .           y=0 ydistmin
        .zmin___   __|__|
        .         /     \
        .        / plane \
        .zmax__ /_________\
        .                 |
        .                 ydistmax



    Parameters
    ----------
    plane : :obj:`list` of :obj:`double`
        Parameters of the plane parametrisation ax+by+cz+d = 0.
    ydistmin : double
        Absolute values of y-coordinates of top corners
    ydistmax : double
        Absolute values y-coordinates of bottom corners
    zmin1 : double
        z-coordinate of top corners
    zmax1 : double
        z-coordinate of bottom corners

    Returns
    -------
    np.ndarray
        x, y and z of the 4 corners of the plane as matrices for plotting
    """

    a, b, c, d = plane.tolist()
    yy = np.array([[-ydistmin, -ydistmax], [ydistmin, ydistmax]])
    zz = np.array([[zmin1, zmax1], [zmin1, zmax1]])
    return (-d - c * zz - b * yy) / a, yy, zz

def plot_pointcloud_and_planes(pointclouds, planes, plot_path=None):
    """Plots list of pointclouds and planes with plotly

    Parameters
    ----------
    pointclouds : list of ndarray
        List of (n x 3) ndarrays with coordintes of points to be plotted
    planes : list of ndarray
        List of ndarrays (vectors of length 4) containing plane parameters
    plot_path: String or None, optional
        Filename for storing plot as html. If None, plot is displayed but not
        saved. Default: None.

    Returns
    -------
        None
    """

    fig = go.Figure()

    for i, pc in enumerate(pointclouds):
        sample_size = min(10000, pc.shape[0])
        indices = np.random.choice(pc.shape[0], sample_size, replace=False)
        pc_rs = pc[indices]
        marker_size = 1
        if pc_rs.shape[0] < 10:
            marker_size = 5
        fig.add_trace(
            go.Scatter3d(
                x=pc_rs.T[0],
                y=pc_rs.T[1],
                z=pc_rs.T[2],
                mode="markers",
                marker=dict(size=marker_size),
                showlegend=True,
                name="pointcloud" + str(i),
            )
        )
    cmin = 0
    cmax = len(planes) - 1
    colorscale = "rainbow"
    for i, plane in enumerate(planes):
        # x, y, z = plane_to_rectuangular_grid(plane, -6, 6, 4, 14)
        x, y, z = plane_to_isosceles_trapezoid_grid(plane, 2, 8, 4, 14)
        surfacecolor = i * np.ones(shape=z.shape)
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                name="plane" + str(i),
                surfacecolor=surfacecolor,
                cmin=cmin,
                cmax=cmax,
                colorscale=colorscale,
                showscale=False,
                showlegend=True,
            )
        )
    now = datetime.now()
    fig.update_layout(title="Pointcloud and plane(s) generated on " + str(now))
    if plot_path is None:
        fig.show()
    else:
        fig.write_html(plot_path, auto_open=True)

def plot_pointcloud_and_lines(pointclouds, lines, plot_path=None):
    """Plots list of pointclouds and lines with plotly

    Parameters
    ----------
    pointclouds : list of ndarray
        List of (n x 3) ndarrays with coordintes of points to be plotted
    lines : list of ndarray
        List of ndarrays (vectors of length 6) containing Line parameters (direction vector, line point coordinate)
    plot_path: String or None, optional
        Filename for storing plot as html. If None, plot is displayed but not
        saved. Default: None.

    Returns
    -------
        None
    """
    

    fig = go.Figure()

    # cmin = 0
    # cmax = len(lines) - 1
    # colorscale = "rainbow"

    # print("kaboom", len(lines))

    scalar = 1   #t value
    X1, Y1, Z1 = lines[0][3:6]
    X2, Y2, Z2 = scalar*lines[0][0] , scalar*lines[0][1], scalar*lines[0][2]  

    X_list = np.array([X1, X2])
    Y_list = np.array([Y1, Y2])
    Z_list = np.array([Z1, Z2])
    df_line = pd.DataFrame({"x": X_list, "y":Y_list, "z":Z_list})

    fig = px.line_3d(df_line, x="x", y="y", z="z" )

    for i, pc in enumerate(pointclouds):
        sample_size = min(10000, pc.shape[0])
        indices = np.random.choice(pc.shape[0], sample_size, replace=False)
        pc_rs = pc[indices]
        marker_size = 1
        if pc_rs.shape[0] < 10:
            marker_size = 5
        fig.add_trace(
            go.Scatter3d(
                x=pc_rs.T[0],
                y=pc_rs.T[1],
                z=pc_rs.T[2],
                mode="markers",
                marker=dict(size=marker_size),
                showlegend=True,
                name="pointcloud" + str(i),
            )
        )
    
    now = datetime.now()
    fig.update_layout(title="Pointcloud and Line(s) generated on " + str(now))
    if plot_path is None:
        fig.show()
    else:
        fig.write_html(plot_path, auto_open=True)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_file", help="Path to a PLY file containing a point cloud")
    parser.add_argument(
        "-l",
        "--line",
        dest="line",
        type=str,
        help="Line coefficients separated by spaces",
    )

    args = parser.parse_args()

    cloud = read_ply(args.ply_file)
    coeffs = args.line.split(" ")
    coeffs = np.array([float(x) for x in coeffs])
    plot_pointcloud_and_lines([cloud], [coeffs])

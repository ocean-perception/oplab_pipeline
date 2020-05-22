# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
"""

import numpy as np
import plotly.graph_objects as go
from datetime import datetime


def plane_to_rectuangular_grid(plane, ymin, ymax, zmin, zmax):
    """Calculate the x,y,z values of the corners of a rectangular plane

    zmin_ _________
         |         |
         |  plane  |
    zmax_|_________|
         |         |
         ymin      ymax

    Returns:
        x, y and z of the 4 corners of the plane as matrices for plotting
    """
    a, b, c, d = plane.tolist()
    yy = np.array([[ymin, ymin], [ymax, ymax]])
    zz = np.array([[zmin, zmax], [zmin, zmax]])
    return (-d - c * zz - b * yy) / a, yy, zz


def plane_to_isosceles_trapezoid_grid(plane, ydistmin, ydistmax, zmin, zmax):
    """Calculate the x,y,z values of the corners of a plane in the shape of the sheet laser

              y=0 ydistmin  
    zmin__   __|__|
            /     \
           / plane \
    zmax_ /_________\
                    |
                    ydistmax

    Returns:
        x, y and z of the 4 corners of the plane as matrices for plotting
    """
    a, b, c, d = plane.tolist()
    yy = np.array([[-ydistmin, -ydistmax], [ydistmin, ydistmax]])
    zz = np.array([[     zmin,      zmax], [    zmin,     zmax]])
    return (-d - c * zz - b * yy) / a, yy, zz


def plot_pointcloud_and_planes(pointcloud, planes, plot_path=None):
    """Plots the pointcloud and each of theplanes in the list of planes with plotly

    :param pointcloud: Pointcloud to be plotted
    :type  pointcloud: (n x 3) numpy array
    :param planes:     List of planes, parametrized as (a,b,c,d)
    :type  planes:     List of numpy arrays (vectors with 4 elements)
    :param plot_path:  (optional) Filename for storing plot as html. If None, plot is not saved.
    :type  plot_path:  String or None
                                                   
    """
    fig = go.Figure()

    for i, pc in enumerate(pointcloud):
        fig.add_trace(go.Scatter3d(x=pc.T[0], y=pc.T[1], z=pc.T[2],
                                    mode='markers',
                                    marker=dict(size=1),
                                    showlegend=True,
                                    name='pointcloud'+str(i)))
    cmin = 0
    cmax = len(planes)-1
    colorscale = 'rainbow'
    for i, plane in enumerate(planes):
        #x, y, z = plane_to_rectuangular_grid(plane, -6, 6, 4, 14)
        x, y, z = plane_to_isosceles_trapezoid_grid(plane, 2, 8, 4, 14)
        surfacecolor = i*np.ones(shape=z.shape)
        fig.add_trace(go.Surface(x=x, y=y, z=z,
                                 name='plane'+str(i),
                                 surfacecolor=surfacecolor, 
                                 cmin=cmin,
                                 cmax=cmax,
                                 colorscale=colorscale,
                                 showscale=False,
                                 showlegend=True))
    now = datetime.now()
    fig.update_layout(title='Pointcloud and plane(s) generated on ' + str(now))
    if plot_path is None:
        fig.show()
    else:
        fig.write_html(plot_path, auto_open=True)

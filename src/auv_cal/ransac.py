# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import random

import numpy as np

from oplab import Console


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))                  #converts a 3x1 matrix to a 4x1, by adding a 1 value at the end.
    axyz[:, :3] = xyzs                              #used for following functions as the matrices need to be same size
    return axyz


def fit_plane(xyzs):                                #accepts 3D points
    axyz = augment(xyzs[:3])                        #converts them to a 4x1 matrix
    m = np.linalg.svd(axyz)[-1][-1, :]              #solves points to find plane parameters
    if m[0] < 0:                                    #ensures that the a coefficient is positive
        m = m * (-1)
    return m                                        #returns plane coefficients


def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold          #calculates distance from plane 'coeffs' to the point 'xyz', returns true or false


def fit_line(xyzs, debug=False):  
    x_list = []
    y_list = []
    z_list = []
    for x,y,z in xyzs:
        x_list.append(x)                      #split into seperate lists so that np.polyfit can be used
        y_list.append(y)                      #there is likely to be a better way to do this but this was the method chosen
        z_list.append(z)
        
    t = np.arange(len(z_list))  #use of 4th dimension 't' to give a common axis for x,y and z
    dir_x, px = np.polyfit(t, x_list, 1)
    dir_y, py = np.polyfit(t, y_list, 1)
    dir_z, pz = np.polyfit(t, z_list, 1)

    m = np.array([dir_x, dir_y, dir_z, px, py, pz], dtype = np.float64)  
    if debug==True:
        print(m)
    return m  #return line vector equation [direction vector,point on line]

def is_inlier_line(coeffs, xyz, threshold):
    PointOnLine = coeffs[3:]
    direction = coeffs[:3]
    selfpoint2point = xyz - PointOnLine
    d_numerator = np.abs(np.cross(selfpoint2point,direction))
    d_denominator = np.abs(direction)
    d = d_numerator/d_denominator
    return d < threshold

def run_ransac(
    data,
    estimate,
    is_inlier,
    sample_size,
    goal_inliers,
    max_iterations,
    stop_at_goal=True,
    random_seed=None,
):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        inliers = []
        s = random.sample(data, int(sample_size))      #pulls out a random set of points from the data set
        m = estimate(s)                                #run a function to estimate a plane to fit to the points that were sampled (line directly above)
        ic = 0                                         # ic = inlier count?
        for j in range(len(data)):                     #checks if data point is an inlier (within acceptable distance from the plane)
            if is_inlier(m, data[j]):                  # m is plane coefficients, no threshold value???
                ic += 1
                inliers.append(data[j])

        # print(s)
        # print('estimate:', m,)
        # print('# inliers:', ic)

        if ic > best_ic:                               # checks if current iteration has a higher inlier count than previous iteration
            best_ic = ic                               # if so, set current as the current best (to test later iterations against)
            best_model = m                             
            if ic > goal_inliers and stop_at_goal:     # checks if current amount of inliers is within the accepted number
                break                                  # if so stops iterations and returns best plane estimate
    # estimate final model using all inliers
    best_model = estimate(inliers)
    return best_model, inliers, i


def bounding_box(iterable):
    print(iterable.shape)
    min_x, min_y = np.min(iterable, axis=0)
    max_x, max_y = np.max(iterable, axis=0)
    return min_x, max_x, min_y, max_y


def plot_plane(a, b, c, d):
    yy, zz = np.mgrid[-6:7, 4:15]
    return (-d - c * zz - b * yy) / a, yy, zz

def plane_fitting_ransac(
    cloud_xyz,
    min_distance_threshold,
    sample_size,
    goal_inliers,
    max_iterations,
    plot=False,
):
    model, inliers, iterations = run_ransac(
        cloud_xyz,
        fit_plane,
        lambda x, y: is_inlier(x, y, min_distance_threshold),
        sample_size,
        goal_inliers,
        max_iterations,
    )
    a, b, c, d = model
    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = Axes3D(fig)                                             # <------- this line of code is bad, it did not work on ipynb or spyder, possibly needs to be changed to plotly or otherwise
        # min_x, max_x, min_y, max_y = bounding_box(cloud_xyz[:, 0:2])
        xx, yy, zz = plot_plane(a, b, c, d)  # , min_x, max_x, min_y, max_y)
        ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))
        ax.scatter3D(cloud_xyz.T[0], cloud_xyz.T[1], cloud_xyz.T[2], s=1)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")
        ax.set_zlabel("$Z$")
        plt.show()
    Console.info(
        "RANSAC took",
        iterations + 1,
        " iterations. Best model:",
        model,
        "explains:",
        len(inliers),
    )
    return (a, b, c, d), inliers

def plot_line(a, b, c, x0, y0, z0):
    #x are values on the line already, trying to find the y & z value of line
    direction = np.array([a,b,c], dtype = np.float64)
    point = np.array([x0,y0,z0], dtype = np.float64)
    x_list = [i for i in range(-5,15)]
    y_list  = []
    z_list = []
    for  x in x_list:
        t = (x - point[0]) / coeffs[0]
        y = point[1] + t*direction[1]
        z = point[2] + t*direction[2]
        y_list.append(y)
        z_list.append(z)
    return x_list, y_list, z_list


def line_fitting_ransac(
    cloud_xyz,
    min_distance_threshold,
    sample_size,
    goal_inliers,
    max_iterations,
    plot=False,
):
    model, inliers, iterations = run_ransac(
        cloud_xyz,
        fit_plane,
        lambda x, y: is_inlier_line(x, y, min_distance_threshold),
        sample_size,
        goal_inliers,
        max_iterations,
    )
    a, b, c, x0, y0, z0 = model                                    #attributes the direction vector coeffs to a,b and c and point on the line to x0,y0 and z0
    
    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = Axes3D(fig)               # <------- this line of code is bad, it did not work on ipynb or spyder, possibly needs to be changed to plotly or otherwise
        xx, yy, zz = plot_line( a, b, c, x0, y0, z0)
        ax.plot3D(xx, yy, zz)
        ax.scatter3D(cloud_xyz.T[0], cloud_xyz.T[1], cloud_xyz.T[2], s=1)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")
        ax.set_zlabel("$Z$")
        plt.show()
    Console.info(
        "RANSAC took",
        iterations + 1,
        " iterations. Best model:",
        model,
        "explains:",
        len(inliers),
    )
    return (a, b, c, x0, y0, z0), inliers


if __name__ == "__main__":
    N_POINTS = 100
    GOAL_INLIERS = N_POINTS * 0.8
    MAX_ITERATIONS = 100

    TARGET_A = 1.0
    TARGET_B = 0.000001
    TARGET_C = 0.000001
    TARGET_D = -1.5
    EXTENTS = 10.0
    NOISE = 0.1

    # create random data
    xyzs = np.zeros((N_POINTS, 3))
    xyzs[:, 0] = [np.random.uniform(2 * EXTENTS) - EXTENTS for i in range(N_POINTS)]
    xyzs[:, 1] = [np.random.uniform(2 * EXTENTS) - EXTENTS for i in range(N_POINTS)]
    for i in range(N_POINTS):
        xyzs[i, 2] = (
            -TARGET_D - xyzs[i, 0] * TARGET_A - xyzs[i, 1] * TARGET_B
        ) / TARGET_C + np.random.normal(scale=NOISE)

    # RANSAC
    m, inliers = line_fitting_ransac(
        xyzs, 0.01, 3, GOAL_INLIERS, MAX_ITERATIONS, plot=True
    )
    scale = TARGET_D / m[3]
    print(np.array(m) * scale)
    print("Inliers: ", len(inliers))

    # [a, b, c, d, plane_angle, pitch_angle, yaw_angle] = fit_plane(xyzs)
    # print(a, b, c, d)
    # print(plane_angle, pitch_angle, yaw_angle)

    # mean_x = np.mean(xyzs[:, 0])
    # mean_y = np.mean(xyzs[:, 1])
    # mean_z = np.mean(xyzs[:, 2])
    # mean_xyz = np.array([mean_x, mean_y, mean_z])

    # plane, normal, offset = build_plane(pitch_angle, yaw_angle, mean_xyz)

    # print(plane)
    # print(normal)
    # print(offset)

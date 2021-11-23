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
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def fit_plane(xyzs):
    axyz = augment(xyzs[:3])
    m = np.linalg.svd(axyz)[-1][-1, :]
    if m[0] < 0:
        m = m * (-1)
    return m


def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold


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
        s = random.sample(data, int(sample_size))
        m = fit_plane(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1
                inliers.append(data[j])

        # print(s)
        # print('estimate:', m,)
        # print('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    # estimate final model using all inliers
    best_model = fit_plane(inliers)
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
        ax = Axes3D(fig)
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
    m, inliers = plane_fitting_ransac(
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


"""
05/08/2018
Michael Leat

Program to process the laser line images, remap them for lens distortion based
on the calibration parameters, build a point cloud representing the sheet laser,
and then performs PCA on the point cloud, finding the pitch and yaw error angles.


Inputs (via .yaml file):
1. Laser line images for fore camera - path and file name list in .txt
2. Laser line images for aft camera - path and file name list in .txt
3. k value. Number of interpolation data points, k = 5 is sufficient
4. MinimumGValue. The minimum greenness value specified for laser line extraction. MinimumGValue = 15 is sufficient
5. image_sample_size. Number of images to sample.
6. no_columns. No of pixel-wide vertical columns to sample from each image.
7. no_iterations. Number of iterations for point cloud generation and sheet laser plane analysis
8. continuous_interpolation. A flag to indicate whether to use discrete or continuous laser line extraction. 1 = continuous.


The function returns a .yaml file called uncertainty_parameters.yaml containing
the pitch and yaw error angles, the pitch and yaw error sheet laser planes
and the offsets required for these planes. All of which must then be input into
the laser bathymetry code by Bodenmann et al., to generate the bathymetry ply
files.

"""


import cv2
import sys
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import yaml
import math
import random
from matplotlib.mlab import PCA
import queue as Queue
import threading
import multiprocessing
import subprocess
from datetime import datetime


class LaserCalibrator():
    def __init__(self):
        self.data = []
        # TODO read camera calibration
        self.left = []
        self.right = []

        left_maps = cv2.initUndistortRectifyMap(
            mtx_1, dist_1, R1, P1, (1280, 1024), cv2.CV_16SC2)
        right_maps = cv2.initUndistortRectifyMap(
            mtx_2, dist_2, R2, P2, (1280, 1024), cv2.CV_16SC2)

    def build_plane(pitch, yaw, point):
        a = math.cos(pitch) * math.cos(yaw)
        b = math.cos(pitch) * math.sin(yaw)
        c = math.sin(pitch)
        normal = np.array([a, b, c])
        d = np.dot(normal, point)
        plane = np.array([a, b, c, d])
        offset = (d - point[1]*b)/a
        return plane, normal, offset

    def cal(self, limages, rimages):
        for img1, img2 in zip(limages, rimages):
            peaks1 = detect_laser_px(img1)
            peaks2 = detect_laser_px(img2)

        point_cloud = []
        for f1, r1, c1, f2, r2, c2 in zip(peaks1, peaks2):
            if f1 and f2:
                p = disparity_px_to_point(r1, r2, c1, f1, cx1)
                point_cloud.extend(p)

        n_points = len(point_cloud)
        num_iterations = 100
        # Get a sample size of 1000 points or a 5% of the dataset.
        sample_size = min(1000, int(n_points*0.05))
        planes = []
        for i in range(0, num_iterations):
            indexes = random.sample(point_cloud, sample_size)
            plane = fit_plane(point_cloud, indexes)
            planes.append(plane)

        planes = np.array(planes)
        planes = planes.reshape(-1, 8)

        point_cloud = np.array(point_cloud)
        point_cloud = point_cloud.reshape(-1, 3)
        total_no_points = len(point_cloud)

        plane_angle_std = np.std(planes[:, 4])
        plane_angle_mean = np.mean(planes[:, 4])
        plane_angle_median = np.median(planes[:, 4])
        pitch_angle_std = np.std(planes[:, 5])
        pitch_angle_mean = np.mean(planes[:, 5])
        yaw_angle_std = np.std(planes[:, 6])
        yaw_angle_mean = np.mean(planes[:, 6])

        print('Total Number of Points:', total_no_points)
        print('Plane Standard deviation:\n', plane_angle_std)
        print('Plane Mean:\n', plane_angle_mean)
        print('Plane Median:\n', plane_angle_median)
        print('Pitch Standard deviation:\n', pitch_angle_std)
        print('Pitch Mean:\n', pitch_angle_mean)
        print('Yaw Standard deviation:\n', yaw_angle_std)
        print('Yaw Mean:\n', yaw_angle_mean)

        mean_a = np.mean(planes[:, 0])
        mean_b = np.mean(planes[:, 1])
        mean_c = np.mean(planes[:, 2])
        mean_d = np.mean(planes[:, 3])

        mean_x = np.mean(point_cloud[:, 0])
        mean_y = np.mean(point_cloud[:, 1])
        mean_z = np.mean(point_cloud[:, 2])
        mean_xyz = np.array([mean_x, mean_y, mean_z])

        msg = ['minus_2sigma', 'mean', 'plus_2sigma']
        t = ['_pitch_', '_yaw_', '_offset']
        k = 1
        for i in range(0, 3):
            for j in range(0, 3):
                a = pitch_angle_mean + (1-i)*2*pitch_angle_std
                b = yaw_angle_mean + (1-j)*2*yaw_angle_std
                c = mean_xyz
                plane, normal, offset = build_plane(a, b, c)
                d = msg[i] + t[0] + msg[j] + t[1] + msg[k] + t[2]
                self.data.append([plane, normal, offset, d])

    def yaml(self):
        msg = yaml.dump(
            {'mean_pitch_mean_yaw_plane': mean_pitch_mean_yaw_plane.tolist(),
             'mean_pitch_mean_yaw_plane_d': mean_pitch_mean_yaw_plane_d,
             'mean_pitch_mean_yaw_plane_offset': mean_pitch_mean_yaw_plane_offset,
             'mean_pitch_plus_2sigma_yaw_plane': mean_pitch_plus_2sigma_yaw_plane.tolist(),
             'mean_pitch_plus_2sigma_yaw_plane_d': mean_pitch_plus_2sigma_yaw_plane_d,
             'mean_pitch_plus_2sigma_yaw_plane_offset': mean_pitch_plus_2sigma_yaw_plane_offset,
             'mean_pitch_minus_2sigma_yaw_plane': mean_pitch_minus_2sigma_yaw_plane.tolist(),
             'mean_pitch_minus_2sigma_yaw_plane_d': mean_pitch_minus_2sigma_yaw_plane_d,
             'mean_pitch_minus_2sigma_yaw_plane_offset': mean_pitch_minus_2sigma_yaw_plane_offset,
             'plus_2sigma_pitch_mean_yaw_plane': plus_2sigma_pitch_mean_yaw_plane.tolist(),
             'plus_2sigma_pitch_mean_yaw_plane_d': plus_2sigma_pitch_mean_yaw_plane_d,
             'plus_2sigma_pitch_mean_yaw_plane_offset': plus_2sigma_pitch_mean_yaw_plane_offset,
             'plus_2sigma_pitch_plus_2sigma_yaw_plane': plus_2sigma_pitch_plus_2sigma_yaw_plane.tolist(),
             'plus_2sigma_pitch_plus_2sigma_yaw_plane_d': plus_2sigma_pitch_plus_2sigma_yaw_plane_d,
             'plus_2sigma_pitch_plus_2sigma_yaw_plane_offset': plus_2sigma_pitch_plus_2sigma_yaw_plane_offset,
             'plus_2sigma_pitch_minus_2sigma_yaw_plane': plus_2sigma_pitch_minus_2sigma_yaw_plane.tolist(),
             'plus_2sigma_pitch_minus_2sigma_yaw_plane_d': plus_2sigma_pitch_minus_2sigma_yaw_plane_d,
             'plus_2sigma_pitch_minus_2sigma_yaw_plane_offset': plus_2sigma_pitch_minus_2sigma_yaw_plane_offset,
             'minus_2sigma_pitch_mean_yaw_plane': minus_2sigma_pitch_mean_yaw_plane.tolist(),
             'minus_2sigma_pitch_mean_yaw_plane_d': minus_2sigma_pitch_mean_yaw_plane_d,
             'minus_2sigma_pitch_mean_yaw_plane_offset': minus_2sigma_pitch_mean_yaw_plane_offset,
             'minus_2sigma_pitch_plus_2sigma_yaw_plane': minus_2sigma_pitch_plus_2sigma_yaw_plane.tolist(),
             'minus_2sigma_pitch_plus_2sigma_yaw_plane_d': minus_2sigma_pitch_plus_2sigma_yaw_plane_d,
             'minus_2sigma_pitch_plus_2sigma_yaw_plane_offset': minus_2sigma_pitch_plus_2sigma_yaw_plane_offset,
             'minus_2sigma_pitch_minus_2sigma_yaw_plane': minus_2sigma_pitch_minus_2sigma_yaw_plane.tolist(),
             'minus_2sigma_pitch_minus_2sigma_yaw_plane_d': minus_2sigma_pitch_minus_2sigma_yaw_plane_d,
             'minus_2sigma_pitch_minus_2sigma_yaw_plane_offset': minus_2sigma_pitch_minus_2sigma_yaw_plane_offset,
             'mean_xyz': mean_xyz.tolist(),
             'planes': planes.tolist(),
             'plane_angle_std': plane_angle_std,
             'plane_angle_mean': plane_angle_mean,
             'plane_angle_median': plane_angle_median,
             'pitch_angle_std': pitch_angle_std,
             'pitch_angle_mean': pitch_angle_mean,
             'yaw_angle_std': yaw_angle_std,
             'yaw_angle_mean': yaw_angle_mean,
             'no_points': no_points,
             'total_no_points': total_no_points})
        return msg


def detect_laser_px(img, min_green_val, k):
    # k is the +-number of pixels from the maximum G pixel i.e. k=1 means that
    # maxgreen_1-1,maxgreen_1 and maxgreen_1+1 are considered for interpolation
    image_rgb_px = []
    for py in range(0, width):
        for px in range(0, height):
            bgr = img[px][py]
            image_rgb_px.extend(
                [bgr[2], bgr[1], bgr[0], px+1, py+1])

    image_rgb_px_1_shaped = np.array(image_rgb_px_1)
    image_rgb_px_1_shaped = np.reshape(
        image_rgb_px_1_shaped, (width*height, 5))

    image_rgb_px_1_sectioned = []
    maxgreen_1 = []

    peaks = []

    width_array = np.array(range(0, width))
    columns = [width_array[i] for i in sorted(
        random.sample(range(len(width_array)), no_columns))]
    for i in columns:
        x_1 = []
        y_1 = []
        lower = height*i
        upper = height+(height*i)
        image_rgb_px_1_sectioned = image_rgb_px_1_shaped[lower:upper, :]
        maxgreen_1 = max(image_rgb_px_1_sectioned,
                         key=lambda item: item[1])
        if maxgreen_1[1] > min_green_val:
            maxgreen_1x = maxgreen_1[3]
            for m in range(-k, k+1):
                x_1.append(maxgreen_1x+m)
            for n in range(-k-1, k):
                p = maxgreen_1x+n
                y_1.append(image_rgb_px_1_sectioned[p, 1])

            if continuous_interpolation == 1:
                f1 = interpolate.interp1d(x_1, y_1, kind='cubic')
                xnew_1 = np.arange(
                    maxgreen_1x-k, maxgreen_1x+k, 0.001)
                ynew_1 = f1(xnew_1)
                max_ind_1 = np.argmax(ynew_1)
                max_height_1 = xnew_1[max_ind_1]
            else:
                max_ind_1 = np.argmax(y_1)
                max_height_1 = x_1[max_ind_1]
            peaks.append([True, max_height_1, i])
        else:
            peaks.append([False, -1, -1])
    return peaks


def disparity_px_to_point(r1, r2, c, f1, cx1):
    z = (f1 * DistanceBetweenCameras) / (r2 - r1)
    y = ((c+1) * z)/f1
    x = ((cx1 - r1) * z)/f1
    return [x, y, z]


def fit_plane(point_cloud_local):
    # MATLIB PCA
    results = PCA(point_cloud_local, standardize=False)
    e3 = np.transpose(results.Wt[2, :])

    a = e3[0]
    b = e3[1]
    c = e3[2]
    d = np.dot(e3, results.mu)
    expected_laser_plane = np.array([1, 0, 0])
    plane_angle = math.degrees(math.acos(abs(np.dot(
        e3, expected_laser_plane))/((np.linalg.norm(e3))*(np.linalg.norm(expected_laser_plane)))))
    expected_pitch_plane = np.array([1, 0])
    e3_pitch = np.array([e3[0], e3[2]])
    pitch_angle = math.degrees(math.acos(abs(np.dot(e3_pitch, expected_pitch_plane))/(
        (np.linalg.norm(e3_pitch))*(np.linalg.norm(expected_pitch_plane)))))
    expected_yaw_plane = np.array([1, 0])
    e3_yaw = np.array([e3[0], e3[1]])
    yaw_angle = math.degrees(math.acos(abs(np.dot(e3_yaw, expected_yaw_plane))/(
        (np.linalg.norm(e3_yaw))*(np.linalg.norm(expected_yaw_plane)))))

    print('Plane Equation:\n', a, 'x + ', b, 'y + ', c, 'z = ', d)
    print('Plane angle:\n', plane_angle, 'degrees')
    print('Pitch angle:\n', pitch_angle, 'degrees')
    print('Yaw angle:\n', yaw_angle, 'degrees')
    return [a, b, c, d, plane_angle, pitch_angle, yaw_angle, no_points]


def detect_all_points(folder_path_1, file_name):
    filelist = []
    with open(os.path.join(folder_path_1, file_name), "r") as ifile:
        filelist = [line.rstrip() for line in ifile]
    for image_file in filelist:
        # Load image
        print("Processing image {0}".format(str(image_file)))
        img_1 = cv2.imread(os.path.join(
            folder_path_1, image_file), cv2.IMREAD_COLOR)
        img_2 = cv2.imread(os.path.join(
            folder_path_2, image_file), cv2.IMREAD_COLOR)
        height, width, channels = img_1.shape

        # Remap images
        if remap == 1:
            img_1 = cv2.remap(img_1, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
            img_2 = cv2.remap(img_2, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)

        point_cloud_local = detect_laser(img_1, img_2)

    point_cloud_local = np.array(point_cloud_local)
    point_cloud_local = point_cloud_local.reshape(-1, 3)
    no_points = len(point_cloud_local)
    print("No_points:", no_points)
    return point_cloud_local


def point_cloud_pca():
    point_cloud = detect_all_points(folder_path_1, file_name)
    planes = []
    for i in iterations:
        # TODO perform iteration over random sampled points
        print("Iteration")
        point_cloud_local = random_sample(point_cloud)

        if OutputCSV == 1:
            print("Saving CSV")
            np.savetxt(os.path.join(outpath_1, file_name_format[0] + "_image_Step" + str(
                image_Step) + ".csv"), point_cloud, delimiter=",")
        plane = fit_plane(point_cloud_local)
        planes.extend(plane)
    return planes, point_cloud

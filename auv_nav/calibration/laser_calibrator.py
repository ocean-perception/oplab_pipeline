# Encoding: utf-8

import cv2
import numpy as np
from scipy import interpolate
import yaml
import math
import random
from auv_nav.calibration.ransac import plane_fitting_ransac
from auv_nav.tools.console import Console


class LaserPoint:
    def __init__(self, found, row, col):
        self.found = found
        self.row = row
        self.col = col

    def __str__(self):
        return 'Found: ' + str(self.found) + ' (' + str(self.row) + ', ' + str(self.col) + ')'


def build_plane(pitch, yaw, point):
        a = math.cos(pitch) * math.cos(yaw)
        b = math.cos(pitch) * math.sin(yaw)
        c = math.sin(pitch)
        normal = np.array([a, b, c])
        d = np.dot(normal, point)
        plane = np.array([a, b, c, d])
        offset = (d - point[1]*b)/a
        return plane, normal, offset


def detect_laser_px(img, min_green_val, k, num_columns, continuous_interpolation, prior=None):
    # k is the +-number of pixels from the maximum G pixel i.e. k=1 means that
    # maxgreen_1-1,maxgreen_1 and maxgreen_1+1 are considered for interpolation
    height, width, channels = img.shape
    peaks = []
    show_img = img.copy()
    width_array = np.array(range(20, width-20))

    if prior is None:
        columns = [width_array[i] for i in sorted(
            random.sample(range(len(width_array)), num_columns))]
    else:
        columns = [i.col for i in prior]

    for i in columns:
        stripe = img[:, i, 1]
        max_ind = stripe.argmax()
        x = []
        y = []
        if stripe[max_ind] > min_green_val:
            maxgreen_x = max_ind
            for m in range(-k, k+1):
                x.append(maxgreen_x+m)
            for n in range(-k-1, k):
                p = maxgreen_x+n
                y.append(img[p, i, 1])

            if continuous_interpolation:
                f = interpolate.interp1d(x, y, kind='cubic')
                xnew = np.arange(
                    maxgreen_x-k, maxgreen_x+k, 0.001)
                ynew = f(xnew)
                max_ind = np.argmax(ynew)
                max_height = xnew[max_ind]
            else:
                max_ind = np.argmax(y)
                max_height = x[max_ind]
            cv2.circle(show_img, (i, int(max_height)), 3, (0, 0, 255))
            peaks.append(LaserPoint(True, max_height, i))
        else:
            peaks.append(LaserPoint(False, -1, -1))
    cv2.namedWindow('Laser detections', 0)
    cv2.imshow('Laser detections', show_img)
    cv2.waitKey(3)
    return peaks


def triangulate_dlt(p1, p2, P1, P2):
    """Find 3D coordinate using all data given

    Implements a linear triangulation method to find a 3D
    point. For example, see Hartley & Zisserman section 12.2
    (p.312).
    """
    # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see also p. 587)
    A = []
    A.append(float(p1.col)*P1[2, :] - P1[0, :])
    A.append(float(p1.row)*P1[2, :] - P1[1, :])
    A.append(float(p2.col)*P2[2, :] - P2[0, :])
    A.append(float(p2.row)*P2[2, :] - P2[1, :])
    A = np.array(A)
    u, d, vt = np.linalg.svd(A)
    X = vt[-1, 0:3]/vt[-1, 3]  # normalize
    return X


def fit_plane(xyz):
    # 1. Calculate centroid of points and make points relative to it
    centroid = xyz.mean(axis=0)
    xyzT = np.transpose(xyz)
    xyzR = xyz - centroid  # points relative to centroid
    xyzRT = np.transpose(xyzR)

    # 2. Calculate the singular value decomposition of the xyzT matrix
    #    and get the normal as the last column of u matrix
    u, v, sd = np.linalg.svd(xyzRT)

    # 3. Get d coefficient to plane for display
    d = normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2]

    axyz = np.ones((len(point_cloud_local), 4))
    axyz[:, :3] = point_cloud_local
    a, b, c, d = np.linalg.svd(axyz)[-1][-1, :]
    e3 = np.array([a, b, c])

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

    return [a, b, c, d, plane_angle, pitch_angle, yaw_angle]


class LaserCalibrator():
    def __init__(self,
                 stereo_camera_model,
                 k=5,
                 min_greenness_value=15,
                 image_step=1,
                 image_sample_size=200,
                 num_iterations=30,
                 num_columns=50,
                 remap=True,
                 continuous_interpolation=True):
        self.data = []

        self.sc = stereo_camera_model
        self.k = k
        self.min_greenness_value = min_greenness_value
        self.image_step = image_step
        self.image_sample_size = image_sample_size
        self.num_iterations = num_iterations
        self.num_columns = num_columns
        self.remap = remap
        self.continuous_interpolation = continuous_interpolation

        self.left_maps = cv2.initUndistortRectifyMap(
            self.sc.left.K,
            self.sc.left.d,
            self.sc.left.R,
            self.sc.left.P,
            (self.sc.left.image_width, self.sc.left.image_height),
            cv2.CV_16SC2)
        self.right_maps = cv2.initUndistortRectifyMap(
            self.sc.right.K,
            self.sc.right.d,
            self.sc.right.R,
            self.sc.right.P,
            (self.sc.right.image_width, self.sc.right.image_height),
            cv2.CV_16SC2)

    def cal(self, limages, rimages):
        i = 0
        num_images = len(limages)
        peaks1 = []
        peaks2 = []
        for img_path1, img_path2 in zip(limages, rimages):
            # Load image

            img1 = cv2.imread(str(img_path1))
            img2 = cv2.imread(str(img_path2))

            # Remap images
            if self.remap:
                img1 = cv2.remap(img1, self.left_maps[0], self.left_maps[1], cv2.INTER_LANCZOS4)
                img2 = cv2.remap(img2, self.right_maps[0], self.right_maps[1], cv2.INTER_LANCZOS4)

            p1 = detect_laser_px(img1, self.min_greenness_value, self.k, self.num_columns, self.continuous_interpolation)
            p2 = detect_laser_px(img2, self.min_greenness_value, self.k, self.num_columns, self.continuous_interpolation, prior=p1)
            peaks1.extend(p1)
            peaks2.extend(p2)
            #Console.progress(i, num_images, prefix='Detecting laser points')
            i += 1
            if i > 5:
                break
        point_cloud = []
        i = 0

        fx1 = self.sc.left.P[0, 0]
        cx1 = self.sc.left.P[0, 2]
        baseline = - self.sc.right.P[1, 3] / fx1
        for p1, p2 in zip(peaks1, peaks2):
            if p1.found and p2.found:
                if p1.row - p2.row > 1.0:
                    p = triangulate_dlt(p1, p2, self.sc.left.P, self.sc.right.P)
                    print(p1)
                    print(p2)
                    print(p)
                    point_cloud.append(p)
            #Console.progress(i, self.num_iterations, prefix='Triangulating points')
            i += 1

        # Get a sample size of 1000 points or a 5% of the dataset.
        planes = []
        for i in range(0, self.num_iterations):
            point_cloud_local = random.sample(point_cloud, self.image_sample_size)
            plane = fit_plane(point_cloud_local)
            planes.append(plane)
            Console.progress(i, self.num_iterations, prefix='Iterating planes')

        planes = np.array(planes)
        planes = planes.reshape(-1, 7)

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

        self.yaml_msg = (
            'mean_xyz: ' + str(mean_xyz.tolist()) + '\n'
            + 'plane_angle_std: ' + str(plane_angle_std) + '\n'
            + 'plane_angle_mean: ' + str(plane_angle_mean) + '\n'
            + 'plane_angle_median: ' + str(plane_angle_median) + '\n'
            + 'pitch_angle_std: ' + str(pitch_angle_std) + '\n'
            + 'pitch_angle_mean: ' + str(pitch_angle_mean) + '\n'
            + 'yaw_angle_std: ' + str(yaw_angle_std) + '\n'
            + 'yaw_angle_mean: ' + str(yaw_angle_mean) + '\n'
            + 'num_iterations: ' + str(self.num_iterations) + '\n'
            + 'total_no_points: ' + str(total_no_points) + '\n')

        msg = ['minus_2sigma', 'mean', 'plus_2sigma']
        t = ['_pitch_', '_yaw_', '_offset_']
        msg_type = ['plane', 'normal', 'offset']

        for i in range(0, 3):
            for j in range(0, 3):
                a = pitch_angle_mean + (1-i)*2*pitch_angle_std
                b = yaw_angle_mean + (1-j)*2*yaw_angle_std
                c = mean_xyz
                plane, normal, offset = build_plane(a, b, c)
                d = msg[i] + t[0] + msg[j] + t[1] + msg[1] + t[2]
                self.yaml_msg += d + msg_type[0] + ': ' + str(plane) + '\n'
                self.yaml_msg += d + msg_type[1] + ': ' + str(normal) + '\n'
                self.yaml_msg += d + msg_type[2] + ': ' + str(offset) + '\n'
                self.data.append([plane, normal, offset, d])

    def yaml(self):
        return self.yaml_msg

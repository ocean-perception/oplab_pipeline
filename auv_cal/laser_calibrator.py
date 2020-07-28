# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import cv2
import numpy as np
import math
import random
from auv_cal.plane_fitting import Plane
from auv_cal.plot_points_and_planes import plot_pointcloud_and_planes
from oplab import Console
from oplab import get_processed_folder
from auv_nav.parsers.parse_biocam_images import biocam_timestamp_from_filename
from .euler_angles_from_rotation_matrix import euler_angles_from_rotation_matrix
import joblib
import yaml
import time


def build_plane(pitch, yaw, point):
    """Compute plane parametrisation given a point and 2 angles

    Parameters
    ----------
    pitch : float
        Pitch in degrees
    yaw : float
        Yaw in degrees
    point : np.ndarray
        (vector of length 3) x, y and z values of point

    Returns
    -------
    np.ndarray
        (vector of length 4) Parametrisation of plane defined by ax+by+cz+d=0
    np.ndarray
        (vector of length 3) Normal vector (normalised to length 1)
    float
        offset
    """
    a = 1.0
    b = math.tan(math.radians(yaw))
    c = math.tan(math.radians(pitch))
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, point)
    plane = np.array([a, b, c, d])
    offset = (-d - point[1] * b) / a
    return plane, normal, offset


def opencv_to_ned(xyz):
    new_point = np.zeros((3, 1), dtype=np.float32)
    new_point[0] = xyz[1]
    new_point[1] = -xyz[0]
    new_point[2] = xyz[2]
    return new_point


def get_angle(normal, reference=[1, 0, 0]):
    normal = np.array(normal)
    reference = np.array(reference)
    unit_normal = normal / np.linalg.norm(normal)

    # Ensure normal points towards positive X axis
    if unit_normal[0] < 0:
        unit_normal = unit_normal * (-1)
    unit_reference = reference / np.linalg.norm(reference)
    cosang = np.dot(unit_normal, unit_reference)
    sinang = np.linalg.norm(np.cross(unit_normal, unit_reference))
    angle = np.arctan2(sinang, cosang)
    return math.degrees(angle)


def get_angles(normal):
    # Convert to numpy array and normalise
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)

    plane_angle = get_angle(normal)
    pitch_angle = math.degrees(math.atan2(normal[2], normal[0]))
    yaw_angle = math.degrees(math.atan2(normal[1], normal[0]))

    if pitch_angle > 90:
        pitch_angle -= 180.0
    if pitch_angle < -90:
        pitch_angle += 180.0

    if yaw_angle > 90:
        yaw_angle -= 180.0
    if yaw_angle < -90:
        yaw_angle += 180.0

    return plane_angle, pitch_angle, yaw_angle


def findLaserInImage(
    img, min_green_ratio, k, num_columns, start_row=0, end_row=-1, prior=None
):
    """Find laser line projection in image

    For each column of the image, find the coordinate (y-value) where the laser
    line passes, provided the laser is visible in that column. The laser
    position is calculated with sub-pixel resolution (-> y value is not an
    integer).

    Returns
    -------
    np.ndarray
        (n x 2) array of y and x values of detected laser pixels
    """


    img_max_value = 0
    if img.dtype.type is np.float32 or img.dtype.type is np.float64:
        img_max_value = 1.0
    elif img.dtype.type is np.uint8:
        img_max_value = 255
    elif img.dtype.type is np.uint16:
        img_max_value = 65535
    elif img.dtype.type is np.uint32:
        img_max_value = 4294967295
    else:
        Console.quit("Image bit depth not supported")

    height, width = img.shape
    peaks = []
    width_array = np.array(range(50, width - 50))

    if end_row == -1:
        end_row = height

    if prior is None:
        if num_columns > 0:
            incr = int(len(width_array) / num_columns) - 1
            incr = max(incr, 1)
            columns = [width_array[i] for i in range(0, len(width_array), incr)]
        else:
            columns = width_array
    else:
        columns = [i[1] for i in prior]

    for u in columns:
        gmax = 0
        vw = start_row
        while vw < end_row - k:
            gt = 0
            gt_m = 0
            gt_mv = 0
            v = vw
            while v < vw + k:
                weight = 1 - 2 * abs(vw + float(k - 1) / 2.0 - v) / k
                intensity = img[v, u]
                gt += weight * intensity
                gt_m += intensity
                gt_mv += (v - vw) * intensity
                v += 1
            if gt > gmax:
                gmax = gt  # gmax:  highest integrated green value
                vgmax = vw + (
                    gt_mv / gt_m
                )  # vgmax: v value in image, where gmax occurrs
            vw += 1
        if (
            gmax > min_green_ratio*img_max_value
        ):  # If `true`, there is a point in the current column, which presumably belongs to the laser line
            peaks.append([vgmax, u])
    return np.array(peaks)


def triangulate_lst(x1, x2, P1, P2):
    """Point pair triangulation from least squares solution"""

    M = np.zeros((6, 6))
    M[:3, :4] = P1
    M[3:, :4] = P2
    M[:3, 4] = -np.array([x1[1], x1[0], 1.0])
    M[3:, 5] = -np.array([x2[1], x2[0], 1.0])
    U, S, V = np.linalg.svd(M)
    X = V[-1, :3]
    return X / V[-1, 3]


def triangulate_dlt(p1, p2, P1, P2):
    """Find 3D coordinate using all data given

    Implements a linear triangulation method to find a 3D
    point. For example, see Hartley & Zisserman section 12.2
    (p.312).
    """

    # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see also p. 587)
    A = []
    A.append(float(p1[1]) * P1[2, :] - P1[0, :])
    A.append(float(p1[0]) * P1[2, :] - P1[1, :])
    A.append(float(p2[1]) * P2[2, :] - P2[0, :])
    A.append(float(p2[0]) * P2[2, :] - P2[1, :])
    A = np.array(A)
    u, s, vt = np.linalg.svd(A)
    X = vt[-1, 0:3] / vt[-1, 3]  # normalize
    return X


def draw_laser(
    left_image_name,
    right_image_name,
    left_maps,
    right_maps,
    left_roi, 
    right_roi,
    top_left,
    top_right,
    bottom_left,
    bottom_right,
    remap,
):
    """Draw identified laser positions on top of laser line images

    Returns
    -------
        None
    """

    lfilename = left_image_name.name
    rfilename = right_image_name.name
    lprocessed_folder = get_processed_folder(left_image_name.parent)
    rprocessed_folder = get_processed_folder(right_image_name.parent)
    lsaving_folder = lprocessed_folder / "laser_detection"
    rsaving_folder = rprocessed_folder / "laser_detection"
    if not lsaving_folder.exists():
        lsaving_folder.mkdir(parents=True, exist_ok=True)
    if not rsaving_folder.exists():
        rsaving_folder.mkdir(parents=True, exist_ok=True)
    lfilename = lsaving_folder / lfilename
    rfilename = rsaving_folder / rfilename
    if not lfilename.exists() or not rfilename.exists():
        limg = cv2.imread(str(left_image_name), cv2.IMREAD_ANYDEPTH)
        rimg = cv2.imread(str(right_image_name), cv2.IMREAD_ANYDEPTH)
        limg = limg[left_roi[0]:left_roi[1], left_roi[2]:left_roi[3]]
        rimg = rimg[right_roi[0]:right_roi[1], right_roi[2]:right_roi[3]]

        channels = 1
        if limg.ndim == 3:
            channels = limg.shape[-1]

        if remap:
            limg_remap = cv2.remap(limg, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
            rimg_remap = cv2.remap(
                rimg, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4
            )
            limg_colour = np.zeros(
                (limg_remap.shape[0], limg_remap.shape[1], 3), dtype=np.uint8
            )
            rimg_colour = np.zeros(
                (rimg_remap.shape[0], rimg_remap.shape[1], 3), dtype=np.uint8
            )
            limg_colour[:, :, 1] = limg_remap
            rimg_colour[:, :, 1] = rimg_remap
        elif channels == 1:
            limg_colour = np.zeros((limg.shape[0], limg.shape[1], 3), dtype=np.uint8)
            rimg_colour = np.zeros((rimg.shape[0], rimg.shape[1], 3), dtype=np.uint8)
            limg_colour[:, :, 1] = limg
            rimg_colour[:, :, 1] = rimg
        else:
            limg_colour = limg
            rimg_colour = rimg
        for p in top_left:
            cv2.circle(limg_colour, (int(p[1]), int(p[0])), 1, (0, 0, 255), -1)
        for p in top_right:
            cv2.circle(rimg_colour, (int(p[1]), int(p[0])), 1, (255, 0, 0), -1)
        for p in bottom_left:
            cv2.circle(limg_colour, (int(p[1]), int(p[0])), 1, (255, 0, 127), -1)
        for p in bottom_right:
            cv2.circle(rimg_colour, (int(p[1]), int(p[0])), 1, (0, 255, 127), -1)
        cv2.imwrite(str(lfilename), limg_colour)
        cv2.imwrite(str(rfilename), rimg_colour)
        Console.info("Saved " + str(lfilename) + " and " + str(rfilename))


def get_laser_pixels_in_image_pair(
    left_image_name,
    left_maps,
    left_roi,
    right_image_name,
    right_maps,
    right_roi,
    min_greenness_ratio,
    k,
    num_columns,
    start_row,
    end_row,
    start_row_b,
    end_row_b,
    two_lasers,
    remap,
    overwrite,
):
    """Get pixel positions of laser line(s) in images

    Returns
    -------
    np.ndarray
        (n x 2) array of y and x values of top laser line in left_image
    np.ndarray
        (n x 2) array of y and x values of top laser line in right_image
    np.ndarray
        (n x 2) array of y and x values of bottom laser line in left_image
        (empty array if `two_lasers` is `False`)
    np.ndarray
        (n x 2) array of y and x values of bottom laser line in right_image
        (empty array if `two_lasers` is `False`)
    """

    def write_file(
        filename,
        image_name,
        maps,
        roi,
        min_greenness_ratio,
        k,
        num_columns,
        start_row,
        end_row,
        start_row_b,
        end_row_b,
        two_lasers,
        remap,
    ):
        """Find laser line(s) in image, write result to file and return values

        Returns
        -------
        np.ndarray
            (n x 2) array of y and x values of top laser line
        np.ndarray
            (n x 2) array of y and x values of bottom laser line (empty array
            if `two_lasers` is `False`)
        """

        p1b = []
        img1 = cv2.imread(str(image_name), cv2.IMREAD_ANYDEPTH)
        # get ROI
        img1 = img1[roi[0]:roi[1], roi[2]:roi[3]]
        if remap:
            img1 = cv2.remap(img1, maps[0], maps[1], cv2.INTER_LANCZOS4)
        p1 = findLaserInImage(
            img1,
            min_greenness_ratio,
            k,
            num_columns,
            start_row=start_row,
            end_row=end_row,
        )
        write_str = "top: \n- " + "- ".join(
            ["[" + str(p1[i][0]) + ", " + str(p1[i][1]) + "]\n" for i in range(len(p1))]
        )
        if two_lasers:
            p1b = findLaserInImage(
                img1,
                min_greenness_ratio,
                k,
                num_columns,
                start_row=start_row_b,
                end_row=end_row_b,
                debug=True,
            )
            write_str += "bottom: \n- " + "- ".join(
                [
                    "[" + str(p1b[i][0]) + ", " + str(p1b[i][1]) + "]\n"
                    for i in range(len(p1b))
                ]
            )
        with filename.open("w") as f:
            f.write(write_str)
        p1 = np.array(p1, dtype=np.float32)
        p1b = np.array(p1b, dtype=np.float32)
        return p1, p1b

    def get_laser_pixels(
        image_name,
        maps,
        roi,
        min_greenness_ratio,
        k,
        num_columns,
        start_row,
        end_row,
        start_row_b,
        end_row_b,
        two_lasers,
        remap,
        overwrite,
    ):
        """Get pixel positions of laser line(s) in image

        If laser detector has been run previously, read laser pixel positions
        from file. Otherwis, or if `overwrite` is `True`, run laser detector.

        Returns
        -------
        np.ndarray
            (n x 2) array of y and x values of top laser line
        np.ndarray
            (n x 2) array of y and x values of bottom laser line (empty array
            if `two_lasers` is `False`)
        """

        # Load image
        points = []
        points_b = []

        output_path = get_processed_folder(image_name.parent)
        # print(str(image_name))
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        fstem = str(image_name.stem) + ".txt"
        filename = output_path / fstem
        if overwrite or not filename.exists():
            points, points_b = write_file(
                filename,
                image_name,
                maps,
                roi,
                min_greenness_ratio,
                k,
                num_columns,
                start_row,
                end_row,
                start_row_b,
                end_row_b,
                two_lasers,
                remap,
            )
        else:
            # print('Opening ' + filename.name)
            with filename.open("r") as f:
                r = yaml.safe_load(f)
            if r is not None:
                a1 = r["top"]
                if "bottom" in r:
                    a1b = r["bottom"]
                else:
                    a1b = []
                i = 0
                for i in range(len(a1)):
                    points.append([a1[i][0], a1[i][1]])
                points = np.array(points, dtype=np.float32)
                if two_lasers:
                    for i in range(len(a1b)):
                        points_b.append([a1b[i][0], a1b[i][1]])
                    points_b = np.array(points_b, dtype=np.float32)
            else:
                points, points_b = write_file(
                    filename,
                    image_name,
                    maps,
                    roi,
                    min_greenness_ratio,
                    k,
                    num_columns,
                    start_row,
                    end_row,
                    start_row_b,
                    end_row_b,
                    two_lasers,
                    remap,
                )
        return points, points_b

    # print('PAIR: ' + left_image_name.stem + ' - ' + right_image_name.stem)
    p1, p1b = get_laser_pixels(
        left_image_name,
        left_maps,
        left_roi,
        min_greenness_ratio,
        k,
        num_columns,
        start_row,
        end_row,
        start_row_b,
        end_row_b,
        two_lasers,
        remap,
        overwrite,
    )
    p2, p2b = get_laser_pixels(
        right_image_name,
        right_maps,
        right_roi,
        min_greenness_ratio,
        k,
        num_columns,
        start_row,
        end_row,
        start_row_b,
        end_row_b,
        two_lasers,
        remap,
        overwrite,
    )
    draw_laser(
        left_image_name,
        right_image_name,
        left_maps,
        right_maps,
        left_roi,
        right_roi,
        p1,
        p2,
        p1b,
        p2b,
        remap,
    )
    return p1, p2, p1b, p2b


def save_cloud(filename, cloud):
    """Write list of 3D points to ply file

    Returns
    -------
        None
    """

    cloud_size = len(cloud)

    header_msg = "ply\n\
                  format ascii 1.0\n\
                  element vertex {0}\n\
                  property float x\n\
                  property float y\n\
                  property float z\n\
                  end_header\n".format(
        cloud_size
    )

    Console.info("Saving cloud to " + str(filename))

    with filename.open("w") as f:
        f.write(header_msg)
        for p in cloud:
            f.write("{0:.5f} {1:.5f} {2:.5f}\n".format(p[0][0], p[1][0], p[2][0]))


class LaserCalibrator:
    def __init__(self, stereo_camera_model, config, overwrite=False):
        self.data = []

        self.sc = stereo_camera_model
        self.config = config
        self.overwrite = overwrite

        detection = config.get("detection", {})
        filtering = config.get("filter", {})
        ransac = config.get("ransac", {})
        uncertainty_generation = config.get("uncertainty_generation", {})

        self.k = detection.get("window_size", 5)
        self.min_greenness_ratio = detection.get("min_greenness_ratio", 0.01)
        self.num_columns = detection.get("num_columns", 1024)
        self.remap = detection.get("remap", True)
        self.start_row = detection.get("start_row", 0)
        self.end_row = detection.get("end_row", -1)
        self.start_row_b = detection.get("start_row_b", 0)
        self.end_row_b = detection.get("end_row_b", -1)
        self.two_lasers = detection.get("two_lasers", False)

        self.filter_max_range = filtering.get("max_range_m", 20.0)
        self.filter_min_range = filtering.get("min_range_m", 3.0)
        self.filter_rss_bin_size = filtering.get("rss_bin_size_m", 0.5)
        self.filter_max_bin_elements = filtering.get("max_bin_elements", 300)

        self.max_point_cloud_size = ransac.get("max_cloud_size", 10000)
        self.mdt = ransac.get("min_distance_threshold", 0.002)
        self.ssp = ransac.get("sample_size_ratio", 0.8)
        self.gip = ransac.get("goal_inliers_ratio", 0.999)
        self.max_iterations = ransac.get("max_iterations", 5000)
        self.css = uncertainty_generation.get("cloud_sample_size", 1000)
        self.num_iterations = uncertainty_generation.get("iterations", 100)

        a = -1

        wl, hl = self.sc.left.size
        wr, hr = self.sc.right.size

        print("Left size is:", wl, hl, "with ratio:", wl/hl)
        print("Right size is:", wr, hr, "with ratio:", wr/hr)
        # Select the smallest common size.
        self.new_size = (min([wl, wr]), min([hl, hr]))
        new_w, new_h = self.new_size

        # Left ROI
        w_shift = 0
        h_shift = 0
        if new_w < wl:
            w_shift = int((wl - new_w) / 2)
        if new_h < wl:
            h_shift = int((hl - new_h) / 2)
        self.left_roi =  w_shift, w_shift+new_w, h_shift, h_shift+new_h
        # Shift CX and CY in camera intrinsics
        self.sc.left.K[0, 2] -= w_shift 
        self.sc.left.K[1, 2] -= h_shift
        print("Shifting left by", w_shift, h_shift)
        print("Left ROI is", self.left_roi)

        # Right ROI
        w_shift = 0
        h_shift = 0
        if new_w < wr:
            w_shift = int((wr - new_w) / 2)
        if new_h < wr:
            h_shift = int((hr - new_h) / 2)
        self.right_roi =  w_shift, w_shift+new_w, h_shift, h_shift+new_h
        # Shift CX and CY in camera intrinsics
        self.sc.right.K[0, 2] -= w_shift 
        self.sc.right.K[1, 2] -= h_shift
        print("Shifting right by", w_shift, h_shift)
        print("Right ROI is", self.right_roi)

        print("Rectifying stereo...")
        self.sc.left.R, self.sc.right.R, self.sc.left.P, self.sc.right.P = cv2.stereoRectify(
            self.sc.left.K,
            self.sc.left.d,
            self.sc.right.K,
            self.sc.right.d,
            self.new_size,
            self.sc.R,
            self.sc.t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=a,
        )[0:4]

        self.left_maps = cv2.initUndistortRectifyMap(
            self.sc.left.K,
            self.sc.left.d,
            self.sc.left.R,
            self.sc.left.P,
            self.new_size,
            cv2.CV_32FC1,
        )
        self.right_maps = cv2.initUndistortRectifyMap(
            self.sc.right.K,
            self.sc.right.d,
            self.sc.right.R,
            self.sc.right.P,
            self.new_size,
            cv2.CV_32FC1,
        )

    def range_stratified_sampling(self, cloud):
        """Perform range stratified sampling following the parameters 
            filter_rss_bin_size and filter_max_bin_elements over 
            the range filter_min_range and filter_max_range

        Parameters
        ----------
        cloud : list
            Input list of points

        Returns
        -------
        sampled cloud
            Output list of points sampled 
        """
        num_bins = math.ceil((self.filter_max_range - self.filter_min_range) / self.filter_rss_bin_size)
        bin_den = (self.filter_max_range - self.filter_min_range)/num_bins
        
        max_bin_elements = self.filter_max_bin_elements
        bins = [None]*num_bins

        # Shuffle the list in-place
        random.shuffle(cloud)

        output_cloud = []
        for p in cloud:
            r = math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
            if r <= self.filter_max_range or r >= self.filter_min_range:
                corresp_bin = round((r - self.filter_min_range)/bin_den)
                if corresp_bin < len(bins):
                    if bins[corresp_bin] is None:
                        bins[corresp_bin] = [p]
                        output_cloud.append(p)
                    elif len(bins[corresp_bin]) < max_bin_elements:
                        bins[corresp_bin].append(p)
                        output_cloud.append(p)
        print('Summary for bins:')
        for i in range(num_bins):
            if bins[i] is not None:
                print(' * bin', i, ':', len(bins[i]))
            else:
                print(' * bin', i, ': 0')

        return output_cloud
        

    def pointcloud_from_peaks(self, pk1, pk2):
        """Triangulate a point cloud from two frames.
        Two peaks are a valid pair if they share the same column and have a difference in row
        index of 1.0.

        Parameters
        ----------
        pk1 : list of lists
            List of lists of (row, col) of detected laser peaks for the first frame
        pk2 : list of lists
            List of lists of (row, col) of detected laser peaks for the second frame

        Returns
        -------
        list, int, int
            Returns the triangulated pointcloud using LST and the number of valid/invalid points.
        """
        cloud = []
        count = 0
        i = 0
        count_inversed = 0
        i1 = 0
        i2 = 0
        for i1 in range(len(pk1)):
            while i2 < len(pk2):
                if pk2[i2][1] >= pk1[i1][1]:
                    break
                i2 += 1
            if i2 == len(pk2):
                continue
            if pk2[i2][1] - pk1[i1][1] < 1.0:
                # Triangulate using Least Squares
                p = triangulate_lst(pk1[i1], pk2[i2], self.sc.left.P, self.sc.right.P)

                # print(pk1[i1], pk2[i2], p)

                # Remove rectification rotation
                if self.remap:
                    p_unrec = self.sc.left.R.T @ p
                else:
                    p_unrec = p

                # Check that the point is not behind the camera
                if p[2] < 0 or p is None:
                    count_inversed += 1
                else:
                    # Store it in the cloud
                    count += 1
                    cloud.append(p_unrec)
                cloud.append(p_unrec)
            i += 1
        return cloud, count, count_inversed

    def fit_and_save(self, cloud):
        """Fit mean plane and uncertainty bounding planes to point cloud

        Parameters
        ----------
        cloud : ndarray of shape (nx3)
            Point cloud

        Returns
        -------
        String
            Plane parameters of the mean plane and a set of uncertainty
            bounding planes of the point cloud in yaml-file format.
        """

        total_no_points = len(cloud)

        # Fit mean plane
        Console.info("Fitting a plane to", total_no_points, "points...")
        p = Plane([1, 0, 0, 1.5])
        mean_plane, inliers_cloud = p.fit(cloud, self.mdt)
        # p.plot(cloud=cloud)

        filename = time.strftime("pointclouds_and_best_model_%Y%m%d_%H%M%S.html")
        plot_pointcloud_and_planes(
            [np.array(cloud), np.array(inliers_cloud)], [np.array(mean_plane)], filename
        )  #'pointclouds_and_best_model.html')

        scale = 1.0 / mean_plane[0]
        mean_plane = np.array(mean_plane) * scale
        mean_plane = mean_plane.tolist()

        inliers_cloud_list = list(inliers_cloud)

        Console.info("Least squares found", len(inliers_cloud_list), "inliers")

        if len(inliers_cloud_list) < 0.5 * len(cloud) * self.gip:
            Console.warn("The number of inliers found are off from what you expected.")
            Console.warn(" * Expected inliers:", len(cloud) * self.gip)
            Console.warn(" * Found inliers:", len(inliers_cloud_list))
            Console.warn(
                "Check the output cloud to see if the found plane makes sense."
            )
            Console.warn("Try to increase your distance threshold.")

        # Determine uncertainty bounding planes
        cloud_sample_size = int(self.css)
        if cloud_sample_size > len(inliers_cloud_list):
            cloud_sample_size = len(inliers_cloud_list)
        Console.info("Randomly sampling with", cloud_sample_size, "points...")

        planes = []
        for i in range(0, self.num_iterations):
            point_cloud_local = random.sample(inliers_cloud_list, cloud_sample_size)
            total_no_points = len(point_cloud_local)
            p = Plane([1, 0, 0, 1.5])
            m = p.fit_non_robust(point_cloud_local)
            # m, _ = p.fit(cloud, self.mdt, verbose=False, output_inliers=False)
            angle, pitch, yaw = get_angles(m[0:3])
            planes.append([angle, pitch, yaw])
            Console.progress(i, self.num_iterations, prefix="Iterating planes")

        planes = np.array(planes)
        planes = planes.reshape(-1, 3)

        plane_angle_std = np.std(planes[:, 0])
        plane_angle_mean = np.mean(planes[:, 0])
        plane_angle_median = np.median(planes[:, 0])
        pitch_angle_std = np.std(planes[:, 1])
        pitch_angle_mean = np.mean(planes[:, 1])
        yaw_angle_std = np.std(planes[:, 2])
        yaw_angle_mean = np.mean(planes[:, 2])

        Console.info("Total Number of Points:", total_no_points)
        Console.info("Plane Standard deviation:\n", plane_angle_std)
        Console.info("Mean angle:\n", plane_angle_mean)
        Console.info("Median angle:\n", plane_angle_median)
        Console.info("Pitch Standard deviation:\n", pitch_angle_std)
        Console.info("Pitch Mean:\n", pitch_angle_mean)
        Console.info("Yaw Standard deviation:\n", yaw_angle_std)
        Console.info("Yaw Mean:\n", yaw_angle_mean)

        inliers_cloud = np.array(inliers_cloud)
        mean_x = np.mean(inliers_cloud[:, 0])
        mean_y = np.mean(inliers_cloud[:, 1])
        mean_z = np.mean(inliers_cloud[:, 2])
        mean_xyz = np.array([mean_x, mean_y, mean_z])

        yaml_msg = ""

        yaml_msg = (
            "mean_xyz_m: "
            + str(mean_xyz.tolist())
            + "\n"
            + "mean_plane: "
            + str(mean_plane)
            + "\n"
            + "plane_angle_std_deg: "
            + str(plane_angle_std)
            + "\n"
            + "plane_angle_mean_deg: "
            + str(plane_angle_mean)
            + "\n"
            + "plane_angle_median_deg: "
            + str(plane_angle_median)
            + "\n"
            + "pitch_angle_std_deg: "
            + str(pitch_angle_std)
            + "\n"
            + "pitch_angle_mean_deg: "
            + str(pitch_angle_mean)
            + "\n"
            + "yaw_angle_std_deg: "
            + str(yaw_angle_std)
            + "\n"
            + "yaw_angle_mean_deg: "
            + str(yaw_angle_mean)
            + "\n"
            + "num_iterations: "
            + str(self.num_iterations)
            + "\n"
            + "total_no_points: "
            + str(total_no_points)
            + "\n"
        )

        msg = ["minus_2sigma", "mean", "plus_2sigma"]
        t = ["_pitch_", "_yaw_", "_offset_"]
        msg_type = ["plane", "normal", "offset_m"]

        for i in range(0, 3):
            for j in range(0, 3):
                a = pitch_angle_mean + (1 - i) * 2 * pitch_angle_std
                b = yaw_angle_mean + (1 - j) * 2 * yaw_angle_std
                c = mean_xyz
                plane, normal, offset = build_plane(a, b, c)
                d = msg[i] + t[0] + msg[j] + t[1] + msg[1] + t[2]
                yaml_msg += d + msg_type[0] + ": " + str(plane.tolist()) + "\n"
                yaml_msg += d + msg_type[1] + ": " + str(normal.tolist()) + "\n"
                yaml_msg += d + msg_type[2] + ": " + str(offset) + "\n"
                self.data.append([plane, normal, offset, d])

        uncertainty_planes = [item[0] for item in self.data]
        filename = time.strftime("pointclouds_and_uncertainty_%Y%m%d_%H%M%S.html")
        plot_pointcloud_and_planes(
            [np.array(cloud), inliers_cloud], uncertainty_planes, filename
        )  # 'pointclouds_and_uncertainty_planes.html')
        # np.save('inliers_cloud.npy', inliers_cloud)    # uncomment to save for debugging
        # for i, plane in enumerate(uncertainty_planes):
        #     np.save('plane' + str(i) + '.npy', plane)

        yaml_msg += (
            'date: "'
            + Console.get_date()
            + '" \n'
            + 'user: "'
            + Console.get_username()
            + '" \n'
            + 'host: "'
            + Console.get_hostname()
            + '" \n'
            + 'version: "'
            + Console.get_version()
            + '" \n'
        )

        return yaml_msg

    def cal(self, limages, rimages):
        """Main function that is called by the code using the LaserCalibrator
            class to trigger the computation of laser plane parameters

        Parameters
        ----------
        limages : list of Path
            Paths of images from the first camera
        rimages : list of Path
            Paths of images from the second camera

        Returns
        -------
        None
        """
        # Synchronise images
        limages_sync = []
        rimages_sync = []

        if len(limages[0].stem) < 26:
            for i, lname in enumerate(limages):
                for j, rname in enumerate(rimages):
                    name = lname.stem
                    if 'image' in name:
                        name = name[5:]
                    if name == rname.stem:
                        limages_sync.append(lname)
                        rimages_sync.append(rname)
        else:
            stamp_pc1 = []
            stamp_cam1 = []
            stamp_pc2 = []
            stamp_cam2 = []
            for i in range(len(limages)):
                t1, tc1 = biocam_timestamp_from_filename(limages[i].stem, 0, 0)
                stamp_pc1.append(float(t1))
                stamp_cam1.append(float(tc1))
            for i in range(len(rimages)):
                t1, tc1 = biocam_timestamp_from_filename(rimages[i].stem, 0, 0)
                stamp_pc2.append(float(t1))
                stamp_cam2.append(float(tc1))

            tolerance = 0.05  # stereo pair must be within 50ms of each other

            for i in range(len(limages)):
                values = []
                for j in range(len(rimages)):
                    values.append(abs(stamp_pc1[i] - stamp_pc2[j]))

                (sync_difference, sync_pair) = min((v, k) for k, v in enumerate(values))
                if sync_difference < tolerance:
                    # print(limages[i].stem + ' syncs with ' + rimages[sync_pair].stem + ' with dif ' + str(sync_difference))
                    limages_sync.append(limages[i])
                    rimages_sync.append(rimages[sync_pair])

        peaks1 = []
        peaks2 = []
        peaks1b = []
        peaks2b = []

        processed_folder = get_processed_folder(limages[0].parent)

        limages_rs = []
        rimages_rs = []

        rs_size = 500
        if len(limages) > rs_size:
            rs_size = int((len(limages_sync) / rs_size) - 1)
            i = 0
            while i < len(limages_sync):
                limages_rs.append(limages_sync[i])
                rimages_rs.append(rimages_sync[i])
                i += rs_size
        else:
            limages_rs = limages
            rimages_rs = rimages

        Console.info("Processing ", str(len(limages_sync)), " synchronised images...")
        result = joblib.Parallel(n_jobs=-1)(
            [
                joblib.delayed(get_laser_pixels_in_image_pair)(
                    i,
                    self.left_maps,
                    self.left_roi,
                    j,
                    self.right_maps,
                    self.right_roi,
                    self.min_greenness_ratio,
                    self.k,
                    self.num_columns,
                    self.start_row,
                    self.end_row,
                    self.start_row_b,
                    self.end_row_b,
                    self.two_lasers,
                    self.remap,
                    self.overwrite,
                )
                for i, j in zip(limages_rs, rimages_rs)
            ]
        )

        count1l = 0
        count2l = 0
        count1lb = 0
        count2lb = 0
        for p1l, p2l, p1bl, p2bl in result:
            if p1l is None or p2l is None:
                continue
            peaks1.append(p1l)
            count1l += len(p1l)
            peaks2.append(p2l)
            count2l += len(p2l)
            if self.two_lasers:
                if p1bl is None or p2bl is None:
                    continue
                peaks1b.append(p1bl)
                count1lb += len(p1bl)
                peaks2b.append(p2bl)
                count2lb += len(p2bl)
            # histogram_bins[disp_idx] += 1

        Console.info("Found {} top peaks in camera 1!".format(str(count1l)))
        Console.info("Found {} top peaks in camera 2!".format(str(count2l)))
        if self.two_lasers:
            Console.info("Found {} bottom peaks in camera 1!".format(str(count1lb)))
            Console.info("Found {} bottom peaks in camera 2!".format(str(count2lb)))

        point_cloud = []
        point_cloud_b = []

        result = joblib.Parallel(n_jobs=-1)(
            [
                joblib.delayed(self.pointcloud_from_peaks)(f1, f2)
                for f1, f2 in zip(peaks1, peaks2)
            ]
        )
        count_good = 0
        count_bad = 0
        for c, cg, cb in result:
            point_cloud.extend(c)
            count_good += cg
            count_bad += cb
        Console.info("Found {} potential TOP points".format(count_good))
        Console.info("Found {} wrong TOP points".format(count_bad))
        Console.info("Found " + str(len(point_cloud)) + " TOP triangulated points!")
        if self.two_lasers:
            result = joblib.Parallel(n_jobs=-1)(
                [
                    joblib.delayed(self.pointcloud_from_peaks)(f1b, f2b)
                    for f1b, f2b in zip(peaks1b, peaks2b)
                ]
            )
            count_good = 0
            count_bad = 0
            for c, cg, cb in result:
                point_cloud_b.extend(c)
                count_good += cg
                count_bad += cb
            Console.info("Found {} potential BOTTOM points".format(count_good))
            Console.info("Found {} wrong BOTTOM points".format(count_bad))
            Console.info(
                "Found " + str(len(point_cloud)) + " BOTTOM triangulated points!"
            )

        # Change coordinate system
        point_cloud_ned = joblib.Parallel(n_jobs=-1)(
            [joblib.delayed(opencv_to_ned)(i) for i in point_cloud]
        )

        if self.two_lasers:
            point_cloud_ned_b = joblib.Parallel(n_jobs=-1)(
                [joblib.delayed(opencv_to_ned)(i) for i in point_cloud_b]
            )

        save_cloud(processed_folder / "../points.ply", point_cloud_ned)
        if self.two_lasers:
            save_cloud(processed_folder / "../points_b.ply", point_cloud_ned_b)

        rss_before = len(point_cloud_ned)
        #point_cloud_filt = self.filter_cloud(point_cloud_ned)
        point_cloud_filt = self.range_stratified_sampling(point_cloud_ned)
        rss_after = len(point_cloud_filt)
        Console.info(
            "Points after filtering: " + str(rss_after) + "/" + str(rss_before)
        )
        rs_size = min(rss_after, self.max_point_cloud_size)
        point_cloud_rs = random.sample(point_cloud_filt, rs_size)
        save_cloud(processed_folder / "../points_rs.ply", point_cloud_rs)
        point_cloud_rs = np.array(point_cloud_rs)
        point_cloud_rs = point_cloud_rs.reshape(-1, 3)
        point_cloud_filt = np.array(point_cloud_filt)
        point_cloud_filt = point_cloud_filt.reshape(-1, 3)
        # self.yaml_msg = self.fit_and_save(point_cloud_rs)
        self.yaml_msg = self.fit_and_save(point_cloud_filt)

        if self.two_lasers:
            Console.info("Fitting a plane to second line...")
            rss_before = len(point_cloud_ned_b)
            #point_cloud_b_filt = self.filter_cloud(point_cloud_ned_b)
            point_cloud_b_filt = self.range_stratified_sampling(point_cloud_ned_b)
            rss_after = len(point_cloud_b_filt)
            Console.info(
                "Points after filtering: " + str(rss_after) + "/" + str(rss_before)
            )
            rs_size = min(rss_after, self.max_point_cloud_size)
            point_cloud_b_rs = random.sample(point_cloud_b_filt, rs_size)
            save_cloud(processed_folder / "../points_b_rs.ply", point_cloud_b_rs)
            point_cloud_b_rs = np.array(point_cloud_b_rs)
            point_cloud_b_rs = point_cloud_b_rs.reshape(-1, 3)
            point_cloud_b_filt = np.array(point_cloud_b_filt)
            point_cloud_b_filt = point_cloud_b_filt.reshape(-1, 3)
            # self.yaml_msg_b = self.fit_and_save(point_cloud_b_rs)
            self.yaml_msg_b = self.fit_and_save(point_cloud_b_filt)

    def yaml(self):
        return self.yaml_msg

    def yaml_b(self):
        return self.yaml_msg_b

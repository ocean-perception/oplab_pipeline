# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import math
import random
import time
from datetime import timedelta
from typing import Dict

import cv2
import joblib
import numpy as np
import yaml

from auv_cal.camera_calibrator import resize_with_padding
from auv_cal.plane_fitting import Plane
from auv_cal.plot_points_and_planes import plot_pointcloud_and_planes
from auv_nav.parsers.parse_biocam_images import biocam_timestamp_from_filename
from oplab import Console, StereoCamera, get_processed_folder


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


def plane_through_3_points(points):
    """Compute plane passing through 3 points

    Parameters
    ----------
    points : list np.ndarray
        (list of length 3 of ndarray vectors of length 3) coordinates of points

    Returns
    -------
    np.ndarray
        (vector of length 4) Parametrisation of plane defined by ax+by+cz+d=0
    """

    assert len(points) == 3

    p0p1 = points[1] - points[0]
    p0p2 = points[2] - points[0]
    n = np.cross(p0p1, p0p2)
    d = -np.dot(n, points[0])
    plane_parametrization = np.concatenate([n, np.array([d])])
    if plane_parametrization[0] != 0:
        plane_parametrization /= plane_parametrization[0]

    return plane_parametrization


def opencv_to_ned(xyz):
    new_point = np.zeros((3, 1), dtype=np.float32)
    new_point[0] = -xyz[1]
    new_point[1] = xyz[0]
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
    width_array = np.array(range(80, width - 80))

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
            gmax > min_green_ratio * img_max_value
        ):  # If `true`, there is a point in the current column, which
            # presumably belongs to the laser line
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
    top_left,
    top_right,
    bottom_left,
    bottom_right,
    remap,
    camera_name,
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
    lsaving_folder = lprocessed_folder / ("laser_detection_" + camera_name)
    rsaving_folder = rprocessed_folder / ("laser_detection_" + camera_name)
    if not lsaving_folder.exists():
        lsaving_folder.mkdir(parents=True, exist_ok=True)
    if not rsaving_folder.exists():
        rsaving_folder.mkdir(parents=True, exist_ok=True)
    lfilename = lsaving_folder / lfilename
    rfilename = rsaving_folder / rfilename
    if not lfilename.exists() or not rfilename.exists():
        limg = cv2.imread(str(left_image_name), cv2.IMREAD_ANYDEPTH)
        rimg = cv2.imread(str(right_image_name), cv2.IMREAD_ANYDEPTH)

        height, width = limg.shape[0], limg.shape[1]
        map_height, map_width = left_maps[0].shape[0], left_maps[0].shape[1]
        if height != map_height or width != map_width:
            limg = resize_with_padding(limg, (map_width, map_height))

        height, width = rimg.shape[0], rimg.shape[1]
        map_height, map_width = right_maps[0].shape[0], right_maps[0].shape[1]
        if height != map_height or width != map_width:
            rimg = resize_with_padding(rimg, (map_width, map_height))

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
    right_image_name,
    right_maps,
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
    camera_name,
):
    """Get pixel positions of laser line(s) in images

    Returns
    -------
    np.ndarray
        (n x 2) array of y and x values of top laser line in left_image
    np.ndarray
        (n x 2) array of y and x values of top laser line in right_image
    np.ndarray
        (m x 2) array of y and x values of bottom laser line in left_image
        (empty array if `two_lasers` is `False`)
    np.ndarray
        (m x 2) array of y and x values of bottom laser line in right_image
        (empty array if `two_lasers` is `False`)
    """

    def find_laser_write_file(
        filename,
        image_name,
        maps,
        min_greenness_ratio,
        k,
        num_columns,
        start_row,
        end_row,
        start_row_b,
        end_row_b,
        two_lasers,
        remap,
        camera_name,
    ):
        """Find laser line(s) in image, write result to file and return values

        Finds the laser line (both, if there are two) in an image and writes
        the corrdinates to file(s) as well as returning them to the calling
        function. The map passed as argument to this function is used to
        rectify the image first, so the returned pixel coordinates are
        rectified ones.

        Returns
        -------
        np.ndarray
            (n x 2) array of y and x values of top laser line
        np.ndarray
            (m x 2) array of y and x values of bottom laser line (empty array
            if `two_lasers` is `False`)
        """

        p1b = []
        img1 = cv2.imread(str(image_name), cv2.IMREAD_ANYDEPTH)

        height, width = img1.shape[0], img1.shape[1]
        map_height, map_width = maps[0].shape[0], maps[0].shape[1]
        if height != map_height or width != map_width:
            img1 = resize_with_padding(img1, (map_width, map_height))

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
        camera_name,
    ):
        """Get pixel positions of laser line(s) in image

        If laser detector has been run previously, read laser pixel positions
        from file. Otherwis, or if `overwrite` is `True`, run laser detector.

        Returns
        -------
        np.ndarray
            (n x 2) array of y and x values of top laser line
        np.ndarray
            (m x 2) array of y and x values of bottom laser line (empty array
            if `two_lasers` is `False`)
        """

        # Load image
        points = []
        points_b = []

        output_path = get_processed_folder(image_name.parent)
        # print(str(image_name))
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        fstem = str(image_name.stem) + "_" + camera_name + ".txt"
        filename = output_path / fstem
        if overwrite or not filename.exists():
            points, points_b = find_laser_write_file(
                filename,
                image_name,
                maps,
                min_greenness_ratio,
                k,
                num_columns,
                start_row,
                end_row,
                start_row_b,
                end_row_b,
                two_lasers,
                remap,
                camera_name,
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
                points, points_b = find_laser_write_file(
                    filename,
                    image_name,
                    maps,
                    min_greenness_ratio,
                    k,
                    num_columns,
                    start_row,
                    end_row,
                    start_row_b,
                    end_row_b,
                    two_lasers,
                    remap,
                    camera_name,
                )
        return points, points_b

    # print('PAIR: ' + left_image_name.stem + ' - ' + right_image_name.stem)
    p1, p1b = get_laser_pixels(
        left_image_name,
        left_maps,
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
        camera_name,
    )
    p2, p2b = get_laser_pixels(
        right_image_name,
        right_maps,
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
        camera_name,
    )
    draw_laser(
        left_image_name,
        right_image_name,
        left_maps,
        right_maps,
        p1,
        p2,
        p1b,
        p2b,
        remap,
        camera_name,
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
    def __init__(
        self,
        stereo_camera_model: StereoCamera,
        config: Dict,
        num_uncert_planes: int,
        overwrite: bool = False,
    ):
        self.data = []

        self.sc = stereo_camera_model
        self.camera_name = self.sc.left.name
        self.config = config
        self.num_uncert_planes = num_uncert_planes
        self.overwrite = overwrite

        if config.get("filter") is not None:
            Console.quit(
                "The node 'filter' is no longer used in calibration.yaml. "
                "'stratify' is used instead."
            )
        detection = config.get("detection", {})
        stratify = config.get("stratify", {})
        ransac = config.get("ransac", {})
        # uncertainty_generation = config.get("uncertainty_generation", {})

        self.k = detection.get("window_size", 5)
        self.min_greenness_ratio = detection.get("min_greenness_ratio", 0.01)
        self.num_columns = detection.get("num_columns", 1024)
        self.remap = detection.get("remap", True)
        self.start_row = detection.get("start_row", 0)
        self.end_row = detection.get("end_row", -1)
        self.start_row_b = detection.get("start_row_b", 0)
        self.end_row_b = detection.get("end_row_b", -1)
        self.two_lasers = detection.get("two_lasers", False)

        self.min_z_m = stratify.get("min_z_m", 1)
        self.max_z_m = stratify.get("max_z_m", 20)
        self.number_of_bins = stratify.get("number_of_bins", 5)
        self.max_points_per_bin = stratify.get("max_bin_elements", 300)

        self.max_point_cloud_size = ransac.get("max_cloud_size", 10000)
        self.mdt = ransac.get("min_distance_threshold", 0.002)
        self.ssp = ransac.get("sample_size_ratio", 0.8)
        self.gip = ransac.get("goal_inliers_ratio", 0.999)
        self.max_iterations = ransac.get("max_iterations", 5000)

        self.left_maps = cv2.initUndistortRectifyMap(
            self.sc.left.K,
            self.sc.left.d,
            self.sc.left.R,
            self.sc.left.P,
            (self.sc.left.image_width, self.sc.left.image_height),
            cv2.CV_32FC1,
        )
        self.right_maps = cv2.initUndistortRectifyMap(
            self.sc.right.K,
            self.sc.right.d,
            self.sc.right.R,
            self.sc.right.P,
            (self.sc.right.image_width, self.sc.right.image_height),
            cv2.CV_32FC1,
        )

        self.inliers_1 = None
        self.triples = []
        self.uncertainty_planes = []
        self.in_front_or_behind = []

    def z_stratified_sampling(self, cloud):
        """Perform vertical (w.r.t. mapping device) stratified sampling

        Stratify point cloud between min_z_m and max_z_m using number_of_bins
        bins and a maximum number of max_points_per_bin points per bin.

        Parameters
        ----------
        cloud : list
            Input list of points

        Returns
        -------
        sampled cloud
            Output list of points sampled
        """

        Console.info("Stratify point cloud")
        Console.info("min_z_m:            ", self.min_z_m)
        Console.info("max_z_m:            ", self.max_z_m)
        Console.info("number_of_bins:     ", self.number_of_bins)
        Console.info("max_points_per_bin: ", self.max_points_per_bin)

        bin_size_m = (self.max_z_m - self.min_z_m) / self.number_of_bins
        bins = [None] * self.number_of_bins

        # Shuffle the list in-place
        random.shuffle(cloud)

        output_cloud = []
        for p in cloud:
            z = p[2]
            if self.min_z_m <= z and z < self.max_z_m:
                corresp_bin = math.floor((z - self.min_z_m) / bin_size_m)
                if corresp_bin < len(bins):
                    if bins[corresp_bin] is None:
                        bins[corresp_bin] = [p]
                        output_cloud.append(p)
                    elif len(bins[corresp_bin]) < self.max_points_per_bin:
                        bins[corresp_bin].append(p)
                        output_cloud.append(p)
                else:
                    raise IndexError("List index corresp_bin out of range")
        Console.info("Summary for bins:")
        for i in range(self.number_of_bins):
            if bins[i] is not None:
                Console.info(" * bin", i, ":", len(bins[i]))
            else:
                Console.info(" * bin", i, ": 0")

        return output_cloud

    def pointcloud_from_peaks(self, pk1, pk2):
        """Triangulate a point cloud from two frames.
        Two peaks are a valid pair if they share the same column and have a
        difference in row index of 1.0.

        Parameters
        ----------
        pk1 : list of lists
            List of lists of (row, col) of detected laser peaks for the first
            frame
        pk2 : list of lists
            List of lists of (row, col) of detected laser peaks for the second
            frame

        Returns
        -------
        list, int, int
            Returns the triangulated pointcloud using LST and the number of
            valid/invalid points.
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
            if abs(pk2[i2][1] - pk1[i1][1]) < 1.0:
                # Triangulate using Least Squares
                p = triangulate_lst(pk1[i1], pk2[i2], self.sc.left.P, self.sc.right.P)

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
            i += 1
        return cloud, count, count_inversed

    def fit_and_save(self, cloud, processed_folder):
        """Fit mean plane and uncertainty bounding planes to point cloud

        Parameters
        ----------
        cloud : ndarray of shape (nx3)
            Point cloud
        processed_folder : Path
            Path of the processed folder where outputs are written

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
        mean_plane, self.inliers_cloud_list = p.fit(cloud, self.mdt)
        # p.plot(cloud=cloud)

        filename = time.strftime("pointclouds_and_best_model_%Y%m%d_%H%M%S.html")
        plot_pointcloud_and_planes(
            [np.array(cloud), np.array(self.inliers_cloud_list)],
            [np.array(mean_plane)],
            str(processed_folder / filename),
        )

        scale = 1.0 / mean_plane[0]
        mean_plane = np.array(mean_plane) * scale
        mean_plane = mean_plane.tolist()

        Console.info("Least squares found", len(self.inliers_cloud_list), "inliers")

        if len(self.inliers_cloud_list) < 0.5 * len(cloud) * self.gip:
            Console.warn("The number of inliers found are off from what you expected.")
            Console.warn(" * Expected inliers:", len(cloud) * self.gip)
            Console.warn(" * Found inliers:", len(self.inliers_cloud_list))
            Console.warn(
                "Check the output cloud to see if the found plane makes sense."
            )
            Console.warn("Try to increase your distance threshold.")

        inliers_cloud = np.array(self.inliers_cloud_list)
        mean_x = np.mean(inliers_cloud[:, 0])
        mean_y = np.mean(inliers_cloud[:, 1])
        mean_z = np.mean(inliers_cloud[:, 2])
        mean_xyz = np.array([mean_x, mean_y, mean_z])

        # Determine minimum distance between points as function of inlier
        # point cloud size
        std_y = np.std(inliers_cloud[:, 1])
        std_z = np.std(inliers_cloud[:, 2])
        # print("Min y: " + str(np.min(inliers_cloud[:, 1])))
        # print("Max y: " + str(np.max(inliers_cloud[:, 1])))
        # print("Std y: " + str(std_y))
        # print("Min z: " + str(np.min(inliers_cloud[:, 2])))
        # print("Max z: " + str(np.max(inliers_cloud[:, 2])))
        # print("Std z: " + str(std_z))
        min_dist = 2 * math.sqrt(std_y**2 + std_z**2)
        Console.info("Minimum distance for poisson disc sampling: {}".format(min_dist))
        min_sin_angle = 0.866  # = sin(60°)

        # Append 1 to the points, so they can be multiplied (dot product) with
        # plane paramters to find out if they are in front, behind or on a
        # plane.
        self.inliers_1 = np.concatenate(
            [inliers_cloud, np.ones((inliers_cloud.shape[0], 1))], axis=1
        )

        Console.info("Generating", self.num_uncert_planes, "uncertainty planes...")
        generate_planes_start = time.time()
        tries = 0
        failed_distance = 0
        failed_angle = 0
        while len(self.uncertainty_planes) < self.num_uncert_planes:
            tries += 1
            point_cloud_local = random.sample(self.inliers_cloud_list, 3)

            # Check if the points are sufficiently far apart and not aligned
            p0p1 = point_cloud_local[1][1:3] - point_cloud_local[0][1:3]
            p0p2 = point_cloud_local[2][1:3] - point_cloud_local[0][1:3]
            p1p2 = point_cloud_local[2][1:3] - point_cloud_local[1][1:3]
            p0p1_norm = np.linalg.norm(p0p1)
            p0p2_norm = np.linalg.norm(p0p2)
            p1p2_norm = np.linalg.norm(p1p2)

            # Poisson disc sampling: reject points that are too close together
            if p0p1_norm < min_dist or p0p2_norm < min_dist or p1p2_norm < min_dist:
                failed_distance += 1
                if failed_distance % 100000 == 0:
                    Console.info_verbose(
                        "Combinations rejected due to distance criterion",
                        "(Poisson disk sampling):",
                        failed_distance,
                        "times,",
                        "due to angle criterion:",
                        failed_angle,
                        "times",
                    )
                continue

            # Reject points that are too closely aligned
            if abs(np.cross(p0p1, p0p2)) / (p0p1_norm * p0p2_norm) < min_sin_angle:
                failed_angle += 1
                if failed_angle % 100000 == 0:
                    Console.info_verbose(
                        "Combinations rejected due to distance criterion",
                        "(Poisson disk sampling):",
                        failed_distance,
                        "times,",
                        "due to angle criterion:",
                        failed_angle,
                        "times",
                    )
                continue

            # Compute plane through the 3 points and append to list
            self.triples.append(np.array(point_cloud_local))
            self.uncertainty_planes.append(plane_through_3_points(point_cloud_local))
            Console.info_verbose(
                "Number of planes: ",
                len(self.uncertainty_planes),
                ", " "Number of tries so far: ",
                tries,
                ".",
                "Combinations rejected due to distance criterion",
                "(Poisson disk sampling):",
                failed_distance,
                "times,",
                "due to angle criterion:",
                failed_angle,
                "times",
            )

        elapsed = time.time() - generate_planes_start
        elapsed_formatted = timedelta(seconds=elapsed)
        Console.info(
            f"... finished generating {len(self.uncertainty_planes)} uncertainty planes in {elapsed_formatted}"
        )

        filename = time.strftime(
            "pointclouds_and_uncertainty_planes_all_" "%Y%m%d_%H%M%S.html"
        )
        plot_pointcloud_and_planes(
            self.triples + [np.array(cloud), inliers_cloud],
            self.uncertainty_planes,
            str(processed_folder / filename),
        )
        # uncomment to save for debugging
        # np.save('inliers_cloud.npy', inliers_cloud)
        # for i, plane in enumerate(self.uncertainty_planes):
        #     np.save('plane' + str(i) + '.npy', plane)

        filename = time.strftime(
            "pointclouds_and_uncertainty_planes_%Y%m%d_" "%H%M%S.html"
        )
        plot_pointcloud_and_planes(
            self.triples + [np.array(cloud), inliers_cloud],
            self.uncertainty_planes,
            str(processed_folder / filename),
        )

        yaml_msg = (
            "mean_xyz_m: "
            + str(mean_xyz.tolist())
            + "\n"
            + "mean_plane: "
            + str(mean_plane)
            + "\n"
        )

        if len(self.uncertainty_planes) > 0:
            uncertainty_planes_str = "uncertainty_planes:\n"
            for i, up in enumerate(self.uncertainty_planes):
                uncertainty_planes_str += "  - " + str(up.tolist()) + "\n"
            yaml_msg += uncertainty_planes_str

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

    def cal(self, limages, rimages, processed_folder):
        """Main function that is called by the code using the LaserCalibrator
            class to trigger the computation of laser plane parameters

        Parameters
        ----------
        limages : list of Path
            Paths of images from the first camera
        rimages : list of Path
            Paths of images from the second camera
        processed_folder : Path
            Path of the processed folder where outputs are written

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
                    if "image" in name:
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
                    # print(limages[i].stem + ' syncs with '
                    # + rimages[sync_pair].stem + ' with dif '
                    # + str(sync_difference))
                    limages_sync.append(limages[i])
                    rimages_sync.append(rimages[sync_pair])

        limages_rs = []
        rimages_rs = []
        rs_size = 500
        if len(limages) > rs_size:
            f = lambda m, n: [  # noqa
                i * n // m + n // (2 * m) for i in range(m)
            ]  # See https://stackoverflow.com/a/9873804/707946
            selection = f(rs_size, len(limages))
            for s in selection:
                limages_rs.append(limages_sync[s])
                rimages_rs.append(rimages_sync[s])
        else:
            limages_rs = limages
            rimages_rs = rimages

        Console.info("Processing", str(len(limages_sync)), "synchronised images...")
        result = joblib.Parallel(n_jobs=-1)(
            [
                joblib.delayed(get_laser_pixels_in_image_pair)(
                    i,
                    self.left_maps,
                    j,
                    self.right_maps,
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
                    self.camera_name,
                )
                for i, j in zip(limages_rs, rimages_rs)
            ]
        )

        # Lists of (rectified) coordinates of laser detections stored in
        # ndarrays, where each entry contains the detections of one image.
        peaks1 = []  # Laser detections in images of top laser in left camera
        peaks2 = []  # Laser detections in images of top laser in right camera
        peaks1b = []  # Laser detections in images of bottom laser in left camera
        peaks2b = []  # Laser detections in images of bottom laser in right camera
        count1l = 0
        count2l = 0
        count1lb = 0
        count2lb = 0
        for p1l, p2l, p1bl, p2bl in result:
            if p1l is not None and p2l is not None:
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
        Console.info("Converted points to NED")

        save_cloud(
            processed_folder / ("points_" + self.camera_name + ".ply"),
            point_cloud_ned,
        )
        if self.two_lasers:
            save_cloud(
                processed_folder / ("points_b_" + self.camera_name + ".ply"),
                point_cloud_ned_b,
            )
        Console.info("Saved cloud to ply")

        rss_before = len(point_cloud_ned)
        # point_cloud_filt = self.filter_cloud(point_cloud_ned)
        point_cloud_filt = self.z_stratified_sampling(point_cloud_ned)
        rss_after = len(point_cloud_filt)
        Console.info(
            "Points after stratifying: " + str(rss_after) + "/" + str(rss_before)
        )
        rs_size = min(rss_after, self.max_point_cloud_size)
        point_cloud_rs = random.sample(point_cloud_filt, rs_size)
        save_cloud(
            processed_folder / ("points_rs_" + self.camera_name + ".ply"),
            point_cloud_rs,
        )
        point_cloud_rs = np.array(point_cloud_rs)
        point_cloud_rs = point_cloud_rs.reshape(-1, 3)
        point_cloud_filt = np.array(point_cloud_filt)
        point_cloud_filt = point_cloud_filt.reshape(-1, 3)
        # self.yaml_msg = self.fit_and_save(point_cloud_rs)
        self.yaml_msg = self.fit_and_save(point_cloud_filt, processed_folder)

        if self.two_lasers:
            Console.info("Fitting a plane to second line...")
            rss_before = len(point_cloud_ned_b)
            # point_cloud_b_filt = self.filter_cloud(point_cloud_ned_b)
            point_cloud_b_filt = self.z_stratified_sampling(point_cloud_ned_b)
            rss_after = len(point_cloud_b_filt)
            Console.info(
                "Points after stratifying: " + str(rss_after) + "/" + str(rss_before)
            )
            rs_size = min(rss_after, self.max_point_cloud_size)
            point_cloud_b_rs = random.sample(point_cloud_b_filt, rs_size)
            save_cloud(
                processed_folder / ("points_b_rs_" + self.camera_name + ".ply"),
                point_cloud_b_rs,
            )
            point_cloud_b_rs = np.array(point_cloud_b_rs)
            point_cloud_b_rs = point_cloud_b_rs.reshape(-1, 3)
            point_cloud_b_filt = np.array(point_cloud_b_filt)
            point_cloud_b_filt = point_cloud_b_filt.reshape(-1, 3)
            # self.yaml_msg_b = self.fit_and_save(point_cloud_b_rs)
            self.yaml_msg_b = self.fit_and_save(point_cloud_b_filt, processed_folder)

    def yaml(self):
        return self.yaml_msg

    def yaml_b(self):
        return self.yaml_msg_b

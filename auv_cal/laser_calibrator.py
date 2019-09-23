# Encoding: utf-8

import cv2
import numpy as np
import math
import random
from auv_cal.ransac import plane_fitting_ransac
from auv_nav.tools.console import Console
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.parsers.parse_biocam_images import biocam_timestamp_from_filename
import joblib
import yaml


class LaserPoint:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __str__(self):
        return ' (' + str(self.row) + ', ' + str(self.col) + ')'


def gaussian_window(length):
    d = []
    for i in range(length):
        d.append(1 + (1 - 2*abs((length-1)/2 - i))/length)
    return d


def build_plane(pitch, yaw, point):
        a = math.cos(math.radians(pitch)) * math.cos(math.radians(yaw))
        b = math.cos(math.radians(pitch)) * math.sin(math.radians(yaw))
        c = math.sin(math.radians(pitch))
        normal = np.array([a, b, c])
        d = np.dot(normal, point)
        plane = np.array([a, b, c, d])
        offset = (d - point[1]*b)/a
        return plane, normal, offset


def detect_laser_px(img, min_green_val, k, min_area, num_columns, start_row=0, end_row=-1, prior=None, debug=False):
    # k is the +-number of pixels from the maximum G pixel i.e. k=1 means that
    # maxgreen_1-1,maxgreen_1 and maxgreen_1+1 are considered for interpolation
    height, width = img.shape
    peaks = []
    width_array = np.array(range(50, width-50))

    if end_row == -1:
        end_row = height

    if prior is None:
        if num_columns > 0:
            incr = int(len(width_array) / num_columns) - 1
            columns = [width_array[i] for i in range(0, len(width_array), incr)]
        else:
            columns = width_array
    else:
        columns = [i.col for i in prior]

    window = gaussian_window(k)

    for i in columns:
        stripe = img[start_row:end_row, i]
        max_ind = stripe.argmax()
        # print('max_ind: ' + str(max_ind))
        if stripe[max_ind] > min_green_val:
            result = np.convolve(stripe, window, 'same')
            max_ind = result.argmax()
            # print('max_ind (conv): ' + str(max_ind))
            if result[max_ind] > min_area and end_row - start_row - k > max_ind:
                weight_sum = 0
                acc_sum = 0
                for ci in range(k):
                    ind = int(ci - (k-1)/2)
                    acc_sum += result[max_ind - ind]
                    weight_sum += result[max_ind - ind] * ind
                cog = weight_sum / acc_sum
                # print('cog: ' + str(cog))
                # print('final: ' + str(start_row + max_ind + cog) + ' start_row: ' + str(start_row))
                # TODO compute center of mass
                peaks.append(LaserPoint(start_row + max_ind + cog, i))
    return peaks


def triangulate_lst(x1, x2, P1, P2):
    """ Point pair triangulation from
    least squares solution. """
    M = np.zeros((6, 6))
    M[:3, :4] = P1
    M[3:, :4] = P2
    M[:3, 4] = -np.array([x1.col, x1.row, 1.0])
    M[3:, 5] = -np.array([x2.col, x2.row, 1.0])
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
    A.append(float(p1.col)*P1[2, :] - P1[0, :])
    A.append(float(p1.row)*P1[2, :] - P1[1, :])
    A.append(float(p2.col)*P2[2, :] - P2[0, :])
    A.append(float(p2.row)*P2[2, :] - P2[1, :])
    A = np.array(A)
    u, s, vt = np.linalg.svd(A)
    X = vt[-1, 0:3]/vt[-1, 3]  # normalize
    return X


def opencv_to_ned(xyz):
    new_point = np.zeros((3, 1), dtype=np.float32)
    new_point[0] = -xyz[1]
    new_point[1] = xyz[0]
    new_point[2] = xyz[2]
    return new_point


def fit_plane(xyz):
    # 1. Calculate centroid of points and make points relative to it
    centroid = xyz.mean(axis=0)
    xyzT = np.transpose(xyz)
    xyzR = xyz - centroid  # points relative to centroid
    xyzRT = np.transpose(xyzR)

    # 2. Calculate the singular value decomposition of the xyzT matrix
    #    and get the normal as the last column of u matrix
    normal = np.linalg.svd(xyzRT)[0][:,-1]

    a = normal[0]
    b = normal[1]
    c = normal[2]
    # 3. Get d coefficient to plane for display
    d = normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2]
    e3 = normal

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


def draw_laser(left_image_name, left_maps, top_left, top_right, bottom_left, bottom_right):
    filename = left_image_name.name
    processed_folder = get_processed_folder(left_image_name.parent)
    saving_folder = processed_folder / 'calibration/laser_detection'
    if not saving_folder.exists():
        saving_folder.mkdir(parents=True, exist_ok=True)
    filename = saving_folder / filename
    if not filename.exists():
        img = cv2.imread(str(left_image_name), cv2.IMREAD_ANYDEPTH)
        img = cv2.remap(img, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
        img_colour = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        img_colour[:, :, 1] = img
        for p in top_left:
            cv2.circle(img_colour, (p.col, int(p.row)), 1, (0, 0, 255), -1)
        for p in top_right:
            cv2.circle(img_colour, (p.col, int(p.row)), 1, (255, 0, 255), -1)
        for p in bottom_left:
            cv2.circle(img_colour, (p.col, int(p.row)), 1, (255, 0, 127), -1)
        for p in bottom_right:
            cv2.circle(img_colour, (p.col, int(p.row)), 1, (0, 255, 127), -1)
        cv2.imwrite(str(filename), img_colour)
        print('Saved ' + str(filename))


def thread_detect(left_image_name, left_maps, right_image_name, right_maps, min_greenness_value, k, min_area, num_columns, start_row, end_row, start_row_b, end_row_b, two_lasers):
        def write_file(filename, image_name, maps, min_greenness_value, k, min_area, num_columns, start_row, end_row, start_row_b, end_row_b, two_lasers):
            p1b = []
            img1 = cv2.imread(str(image_name), cv2.IMREAD_ANYDEPTH)
            img1 = cv2.remap(img1, maps[0], maps[1], cv2.INTER_LANCZOS4)

            # lh, lw = img1.shape
            # final_image = np.zeros((lh, lw, 3), dtype=np.uint8)
            # final_image[:lh, :lw, 1] = img1
            # cv2.namedWindow('Laser distorted', 0)
            # cv2.imshow('Laser distorted', final_image)
            # cv2.waitKey(0)

            # cv2.namedWindow('Laser remap', 0)
            # final_image[:lh, :lw, 1] = img1
            # cv2.imshow('Laser remap', final_image)
            # cv2.waitKey(0)
            p1 = detect_laser_px(img1, min_greenness_value, k, min_area, num_columns, start_row=start_row, end_row=end_row)
            if two_lasers:
                p1b = detect_laser_px(img1, min_greenness_value, k, min_area, num_columns, start_row=start_row_b, end_row=end_row_b, debug=True)

            write_str = "top: \n- " + "- ".join(["[" + str(i.row) + ", " + str(i.col) + ']\n'for i in p1])
            write_str += "bottom: \n- " + "- ".join(["[" + str(i.row) + ", " + str(i.col) + ']\n'for i in p1b])
            with filename.open('w') as f:
                f.write(write_str)
            return p1, p1b

        def do_image(image_name, maps, min_greenness_value, k, min_area, num_columns, start_row, end_row, start_row_b, end_row_b, two_lasers):
            # Load image
            points = []
            points_b = []

            output_path = get_processed_folder(image_name.parent)
            # print(str(image_name))
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            fstem = str(image_name.stem) + '.txt'
            filename = output_path / fstem
            if filename.exists():
                # print('Opening ' + filename.name)
                with filename.open('r') as f:
                    r = yaml.safe_load(f)
                if r is not None:
                    a1 = r['top']
                    a1b = r['bottom']
                    i = 0
                    for i in range(len(a1)):
                        points.append(LaserPoint(a1[i][0], a1[i][1]))
                    for i in range(len(a1b)):
                        points_b.append(LaserPoint(a1b[i][0], a1b[i][1]))
                else:
                    points, points_b = write_file(filename, image_name, maps, min_greenness_value, k, min_area, num_columns, start_row, end_row, start_row_b, end_row_b, two_lasers)
            else:
                points, points_b = write_file(filename, image_name, maps, min_greenness_value, k, min_area, num_columns, start_row, end_row, start_row_b, end_row_b, two_lasers)
            return points, points_b
        # print('PAIR: ' + left_image_name.stem + ' - ' + right_image_name.stem)
        p1, p1b = do_image(
            left_image_name,
            left_maps,
            min_greenness_value,
            k,
            min_area,
            num_columns,
            start_row,
            end_row,
            start_row_b,
            end_row_b,
            two_lasers)
        p2, p2b = do_image(
            right_image_name,
            right_maps,
            min_greenness_value,
            k,
            min_area,
            num_columns,
            start_row,
            end_row,
            start_row_b,
            end_row_b,
            two_lasers)
        draw_laser(left_image_name, left_maps, p1, p2, p1b, p2b)
        return p1, p2, p1b, p2b


def save_cloud(filename, cloud):
    cloud_size = len(cloud)

    header_msg = 'ply\n\
                  format ascii 1.0\n\
                  element vertex {0}\n\
                  property float x\n\
                  property float y\n\
                  property float z\n\
                  end_header\n'.format(cloud_size)

    print('Saving cloud to ' + str(filename))

    with filename.open('w') as f:
        f.write(header_msg)
        for p in cloud:
            f.write('{0:.5f} {1:.5f} {2:.5f}\n'.format(p[0][0], p[1][0], p[2][0]))


class LaserCalibrator():
    def __init__(self,
                 stereo_camera_model,
                 k,
                 min_greenness_value,
                 image_step,
                 image_sample_size,
                 num_iterations,
                 num_columns,
                 remap,
                 continuous_interpolation,
                 start_row=0,
                 end_row=-1,
                 start_row_b=0,
                 end_row_b=-1,
                 two_lasers=False):
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
        self.start_row = start_row
        self.end_row = end_row
        self.start_row_b = start_row_b
        self.end_row_b = end_row_b
        self.two_lasers = two_lasers

        self.min_area = 5

        self.left_maps = cv2.initUndistortRectifyMap(
            self.sc.left.K,
            self.sc.left.d,
            self.sc.left.R,
            self.sc.left.P,
            (self.sc.left.image_width, self.sc.left.image_height),
            cv2.CV_32FC1)
        self.right_maps = cv2.initUndistortRectifyMap(
            self.sc.right.K,
            self.sc.right.d,
            self.sc.right.R,
            self.sc.right.P,
            (self.sc.right.image_width, self.sc.right.image_height),
            cv2.CV_32FC1)

    def cal(self, limages, rimages):

        # Synchronise images

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

        limages_sync = []
        rimages_sync = []

        for i in range(len(limages)):
            values = []
            for j in range(len(rimages)):
                values.append(abs(stamp_pc1[i]-stamp_pc2[j]))

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
        rs_size = int((len(limages_sync)/rs_size) - 1)
        i = 0
        while i < len(limages_sync):
            limages_rs.append(limages_sync[i])
            rimages_rs.append(rimages_sync[i])
            i += rs_size

        print('Processing images...')
        result = joblib.Parallel(n_jobs=-1)([
            joblib.delayed(thread_detect)(i, self.left_maps, j, self.right_maps, self.min_greenness_value, self.k, self.min_area, self.num_columns, self.start_row, self.end_row, self.start_row_b, self.end_row_b, self.two_lasers)
            for i, j in zip(limages_rs, rimages_rs)])

        count1l = 0
        count2l = 0
        count1lb = 0
        count2lb = 0
        # 13 histogram bins from 70 to 200
        # bin 0: from 70 to 80
        # bin 1: from 80 to 90
        # bin 2: from 90 to 100
        # ...
        # bin 12: from 190 to 200
        # histogram_bins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for p1l, p2l, p1bl, p2bl in result:
            if p1l is None or p2l is None:
                continue
            # p1l = np.array(p1l)
            # p2l = np.array(p2l)
            # if len(p2l) != len(p1l):
            #     continue
            # disp = p2l - p1l
            # disp_mean = int(np.mean(disp))

            # disp_idx = int((disp_mean - 70)/10)

            # if histogram_bins[disp_idx] > 10:
            #     continue

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

        def pointcloud_from_peaks(pk1, pk2):
            cloud = []
            count = 0
            i = 0
            count_inversed = 0
            i1 = 0
            i2 = 0
            for i1 in range(len(pk1)):
                while i2 < len(pk2):
                    if pk2[i2].col >= pk1[i1].col:
                        break
                    i2 += 1
                if i2 == len(pk2):
                    continue
                if pk2[i2].col == pk1[i1].col:
                    if ((pk1[i1].row - pk2[i2].row > 70 and pk1[i1].row - pk2[i2].row < 200)
                        or (pk2[i2].row - pk1[i1].row > 70 and pk2[i2].row - pk1[i1].row < 200)):
                        p = triangulate_lst(pk1[i1], pk2[i2], self.sc.left.P, self.sc.right.P)
                        if p[2] < 0 or p is None:
                            count_inversed += 1
                        else:
                            count += 1
                            cloud.append(p)
                i += 1
            return cloud, count, count_inversed

        point_cloud = []
        point_cloud_b = []

        result = joblib.Parallel(n_jobs=-1)([
            joblib.delayed(pointcloud_from_peaks)(f1, f2)
            for f1, f2 in zip(peaks1, peaks2)])
        count_good = 0
        count_bad = 0
        for c, cg, cb in result:
            point_cloud.extend(c)
            count_good += cg
            count_bad += cb
        Console.info("Found {} potential TOP points".format(count_good))
        Console.info("Found {} wrong TOP points".format(count_bad))
        print('Found ' + str(len(point_cloud)) + ' TOP triangulated points!')
        if self.two_lasers:
            result = joblib.Parallel(n_jobs=-1)([
                joblib.delayed(pointcloud_from_peaks)(f1b, f2b)
                for f1b, f2b in zip(peaks1b, peaks2b)])
            count_good = 0
            count_bad = 0
            for c, cg, cb in result:
                point_cloud_b.extend(c)
                count_good += cg
                count_bad += cb
            Console.info("Found {} potential BOTTOM points".format(count_good))
            Console.info("Found {} wrong BOTTOM points".format(count_bad))
            print('Found ' + str(len(point_cloud)) + ' BOTTOM triangulated points!')

        # Change coordinate system
        point_cloud_ned = joblib.Parallel(n_jobs=-1)([
            joblib.delayed(opencv_to_ned)(i)
            for i in point_cloud])

        point_cloud_ned_b = joblib.Parallel(n_jobs=-1)([
            joblib.delayed(opencv_to_ned)(i)
            for i in point_cloud_b])

        def fit_and_save(cloud):
            total_no_points = len(cloud)

            mean_plane, inliers_cloud = plane_fitting_ransac(
                cloud,
                min_distance_threshold=0.0005,
                sample_size=200,
                goal_inliers=len(cloud)*0.85,
                max_iterations=5000,
                plot=True)

            inliers_cloud = list(inliers_cloud)

            print('RANSAC plane with all points: {}'.format(mean_plane))

            planes = []
            for i in range(0, self.num_iterations):
                cloud_sample_size = int(0.5 * len(inliers_cloud))
                point_cloud_local = np.array(random.sample(inliers_cloud, cloud_sample_size))
                plane = fit_plane(point_cloud_local)
                print(plane)
                planes.append(plane)
                Console.progress(i, self.num_iterations, prefix='Iterating planes')

            planes = np.array(planes)
            planes = planes.reshape(-1, 7)

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

            # mean_a = np.mean(planes[:, 0])
            # mean_b = np.mean(planes[:, 1])
            # mean_c = np.mean(planes[:, 2])
            # mean_d = np.mean(planes[:, 3])

            mean_x = np.mean(cloud[:, 0])
            mean_y = np.mean(cloud[:, 1])
            mean_z = np.mean(cloud[:, 2])
            mean_xyz = np.array([mean_x, mean_y, mean_z])

            yaml_msg = ''

            yaml_msg = (
                'mean_xyz: ' + str(mean_xyz.tolist()) + '\n'
                + 'mean_plane: ' + str(list(mean_plane)) + '\n'
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
                    yaml_msg += d + msg_type[0] + ': ' + str(plane) + '\n'
                    yaml_msg += d + msg_type[1] + ': ' + str(normal) + '\n'
                    yaml_msg += d + msg_type[2] + ': ' + str(offset) + '\n'
                    self.data.append([plane, normal, offset, d])

            return yaml_msg

        def filter_cloud(cloud):
            def valid(p):
                first = (p[0] > -10.0) and (p[0] < 10.0)
                second = (p[1] > -10.0) and (p[1] < 10.0)
                third = (p[2] > 0.0) and (p[2] < 10.0)
                return first and second and third
            return [p for p in cloud if valid(p)]

        print('Saving clouds')
        save_cloud(processed_folder / '../points.ply', point_cloud_ned)
        save_cloud(processed_folder / '../points_b.ply', point_cloud_ned_b)

        print('Fitting a plane...')
        rs_size = min(len(point_cloud_ned), 10000)
        point_cloud_rs = random.sample(point_cloud_ned, rs_size)
        rss_before = len(point_cloud_rs)
        point_cloud_rs = filter_cloud(point_cloud_rs)
        rss_after = len(point_cloud_rs)
        print('Points after filtering: ' + str(rss_after) + '/' + str(rss_before))
        point_cloud_rs = np.array(point_cloud_rs)
        point_cloud_rs = point_cloud_rs.reshape(-1, 3)
        self.yaml_msg = fit_and_save(point_cloud_rs)
        if self.two_lasers:
            print('Fitting a plane to second line...')
            rs_size = min(len(point_cloud_ned_b), 10000)
            point_cloud_b_rs = random.sample(point_cloud_ned_b, rs_size)
            rss_before = len(point_cloud_b_rs)
            point_cloud_b_rs = filter_cloud(point_cloud_b_rs)
            rss_after = len(point_cloud_b_rs)
            print('Points after filtering: ' + str(rss_after) + '/' + str(rss_before))
            point_cloud_b_rs = np.array(point_cloud_b_rs)
            point_cloud_b_rs = point_cloud_b_rs.reshape(-1, 3)
            self.yaml_msg_b = fit_and_save(point_cloud_b_rs)

    def yaml(self):
        return self.yaml_msg

    def yaml_b(self):
        return self.yaml_msg_b


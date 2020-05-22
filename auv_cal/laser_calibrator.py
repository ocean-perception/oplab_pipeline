# Encoding: utf-8

import cv2
import numpy as np
import math
import random
from auv_cal.plane_fitting import Plane
from auv_cal.plot_points_and_planes import plot_pointcloud_and_planes
from oplab import Console
from oplab import get_processed_folder
from auv_nav.parsers.parse_biocam_images import biocam_timestamp_from_filename
import joblib
import yaml
import time


def build_plane(pitch, yaw, point):
    a = 1.0
    b = math.tan(math.radians(yaw))
    c = math.tan(math.radians(pitch))
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    d = - np.dot(normal, point)
    plane = np.array([a, b, c, d])
    offset = (- d - point[1]*b)/a
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


def findLaserInImage(img, min_green_val, k, num_columns, start_row=0, end_row=-1, prior=None, debug=False):
    height, width = img.shape
    peaks = []
    width_array = np.array(range(50, width-50))

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
                weight = 1-2*abs(vw+float(k-1)/2.0-v)/k
                intensity = img[v, u]
                gt += weight*intensity
                gt_m += intensity
                gt_mv += (v-vw)*intensity
                v += 1
            if gt>gmax:
                gmax = gt                  # gmax:  highest integrated green value
                vgmax = vw+(gt_mv/gt_m)    # vgmax: v value in image, where gmax occurrs
            vw += 1
        if gmax > min_green_val:    # If `true`, there is a point in the current column, which presumably belongs to the laser line
            peaks.append([vgmax, u])
    return np.array(peaks)


def triangulate_lst(x1, x2, P1, P2):
    """ Point pair triangulation from
    least squares solution. """
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
    A.append(float(p1[1])*P1[2, :] - P1[0, :])
    A.append(float(p1[0])*P1[2, :] - P1[1, :])
    A.append(float(p2[1])*P2[2, :] - P2[0, :])
    A.append(float(p2[0])*P2[2, :] - P2[1, :])
    A = np.array(A)
    u, s, vt = np.linalg.svd(A)
    X = vt[-1, 0:3]/vt[-1, 3]  # normalize
    return X


def draw_laser(left_image_name, right_image_name, left_maps, right_maps, top_left, top_right, bottom_left, bottom_right, remap):
    lfilename = left_image_name.name
    rfilename = right_image_name.name
    lprocessed_folder = get_processed_folder(left_image_name.parent)
    rprocessed_folder = get_processed_folder(right_image_name.parent)
    lsaving_folder = lprocessed_folder / 'laser_detection'
    rsaving_folder = rprocessed_folder / 'laser_detection'
    if not lsaving_folder.exists():
        lsaving_folder.mkdir(parents=True, exist_ok=True)
    if not rsaving_folder.exists():
        rsaving_folder.mkdir(parents=True, exist_ok=True)
    lfilename = lsaving_folder / lfilename
    rfilename = rsaving_folder / rfilename
    if not lfilename.exists() or not rfilename.exists():
        limg = cv2.imread(str(left_image_name), cv2.IMREAD_ANYDEPTH)
        rimg = cv2.imread(str(right_image_name), cv2.IMREAD_ANYDEPTH)

        channels = 1
        if limg.ndim == 3:
            channels = limg.shape[-1]

        if remap:
            limg_remap = cv2.remap(limg, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
            rimg_remap = cv2.remap(rimg, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
            limg_colour = np.zeros((limg_remap.shape[0], limg_remap.shape[1], 3), dtype=np.uint8)
            rimg_colour = np.zeros((rimg_remap.shape[0], rimg_remap.shape[1], 3), dtype=np.uint8)
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
        Console.info('Saved ' + str(lfilename) + ' and ' + str(rfilename))


def thread_detect(left_image_name, left_maps, right_image_name, right_maps, min_greenness_value, k, num_columns, start_row, end_row, start_row_b, end_row_b, two_lasers, remap, overwrite):
    def write_file(filename, image_name, maps, min_greenness_value, k, num_columns, start_row, end_row, start_row_b, end_row_b, two_lasers, remap):
        p1b = []
        img1 = cv2.imread(str(image_name), cv2.IMREAD_ANYDEPTH)
        if remap:
            img1 = cv2.remap(img1, maps[0], maps[1], cv2.INTER_LANCZOS4)
        p1 = findLaserInImage(img1, min_greenness_value, k, num_columns, start_row=start_row, end_row=end_row)
        write_str = "top: \n- " + "- ".join(["[" + str(p1[i][0]) + ", " + str(p1[i][1]) + ']\n'for i in range(len(p1))])
        if two_lasers:
            p1b = findLaserInImage(img1, min_greenness_value, k, num_columns, start_row=start_row_b, end_row=end_row_b, debug=True)
            write_str += "bottom: \n- " + "- ".join(["[" + str(p1b[i][0]) + ", " + str(p1b[i][1]) + ']\n'for i in range(len(p1b))])
        with filename.open('w') as f:
            f.write(write_str)
        p1 = np.array(p1, dtype=np.float32)
        p1b = np.array(p1b, dtype=np.float32)
        return p1, p1b

    def do_image(image_name, maps, min_greenness_value, k, num_columns, start_row, end_row, start_row_b, end_row_b, two_lasers, remap, overwrite):
        # Load image
        points = []
        points_b = []

        output_path = get_processed_folder(image_name.parent)
        # print(str(image_name))
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        fstem = str(image_name.stem) + '.txt'
        filename = output_path / fstem
        if overwrite or not filename.exists():
            points, points_b = write_file(filename, image_name, maps, min_greenness_value, k, num_columns, start_row, end_row, start_row_b, end_row_b, two_lasers, remap)
        else:
            # print('Opening ' + filename.name)
            with filename.open('r') as f:
                r = yaml.safe_load(f)
            if r is not None:
                a1 = r['top']
                if 'bottom' in r:
                    a1b = r['bottom']
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
                points, points_b = write_file(filename, image_name, maps, min_greenness_value, k, num_columns, start_row, end_row, start_row_b, end_row_b, two_lasers, remap)
        return points, points_b
    # print('PAIR: ' + left_image_name.stem + ' - ' + right_image_name.stem)
    p1, p1b = do_image(
        left_image_name,
        left_maps,
        min_greenness_value,
        k,
        num_columns,
        start_row,
        end_row,
        start_row_b,
        end_row_b,
        two_lasers, remap, overwrite)
    p2, p2b = do_image(
        right_image_name,
        right_maps,
        min_greenness_value,
        k,
        num_columns,
        start_row,
        end_row,
        start_row_b,
        end_row_b,
        two_lasers, remap, overwrite)
    draw_laser(left_image_name, right_image_name, left_maps, right_maps, p1, p2, p1b, p2b, remap)
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

    Console.info('Saving cloud to ' + str(filename))

    with filename.open('w') as f:
        f.write(header_msg)
        for p in cloud:
            f.write('{0:.5f} {1:.5f} {2:.5f}\n'.format(p[0][0], p[1][0], p[2][0]))


class LaserCalibrator():
    def __init__(self,
                 stereo_camera_model,
                 config,
                 overwrite=False):
        self.data = []

        self.sc = stereo_camera_model
        self.config = config
        self.overwrite = overwrite

        detection = config.get('detection', {})
        filtering = config.get('filter', {})
        ransac = config.get('ransac', {})
        uncertainty_generation = config.get('uncertainty_generation', {})

        self.k = detection.get('window_size', 5)
        self.min_greenness_value = detection.get('min_greenness_value', 15)
        self.num_columns = detection.get('num_columns', 1024)
        self.remap = detection.get('remap', True)
        self.start_row = detection.get('start_row', 0)
        self.end_row = detection.get('end_row', -1)
        self.start_row_b = detection.get('start_row_b', 0)
        self.end_row_b = detection.get('end_row_b', -1)
        self.two_lasers = detection.get('two_lasers', False)

        self.filter_xy = filtering.get('cloud_xy', 30.0)
        self.filter_z_min = filtering.get('cloud_z_min', 0.0)
        self.filter_z_max = filtering.get('cloud_z_max', 15.0)

        self.max_point_cloud_size = ransac.get('max_cloud_size', 10000)
        self.mdt = ransac.get('min_distance_threshold', 0.002)
        self.ssp = ransac.get('sample_size_ratio', 0.8)
        self.gip = ransac.get('goal_inliers_ratio', 0.999)
        self.max_iterations = ransac.get('max_iterations', 5000)
        self.css = uncertainty_generation.get('cloud_sample_size', 1000)
        self.num_iterations = uncertainty_generation.get('iterations', 100)       

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

    def valid(self, p):
        first = (p[0] > -self.filter_xy) and (p[0] < self.filter_xy)
        second = (p[1] > -self.filter_xy) and (p[1] < self.filter_xy)
        third = (p[2] > self.filter_z_min) and (p[2] < self.filter_z_max)
        return first and second and third

    def filter_cloud(self, cloud):
        return [p for p in cloud if self.valid(p)]

    def pointcloud_from_peaks(self, pk1, pk2):
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

    def fit_and_save(self, cloud):
        total_no_points = len(cloud)

        Console.info('Fitting a plane to', total_no_points, 'points...')
        p = Plane([1, 0, 0, 1.5])
        mean_plane, inliers_cloud = p.fit(cloud, self.mdt)
        #p.plot(cloud=cloud)

        filename = time.strftime("pointclouds_and_best_model_%Y%m%d_%H%M%S.html")
        plot_pointcloud_and_planes([np.array(cloud), np.array(inliers_cloud)], [np.array(mean_plane)], filename) #'pointclouds_and_best_model.html')


        scale = 1.0/mean_plane[0]
        mean_plane = np.array(mean_plane)*scale
        mean_plane = mean_plane.tolist()

        inliers_cloud_list = list(inliers_cloud)

        Console.info('Least squares found', len(inliers_cloud_list), 'inliers')

        if len(inliers_cloud_list) < 0.5*len(cloud)*self.gip:
            Console.warn('The number of inliers found are off from what you expected.')
            Console.warn(' * Expected inliers:', len(cloud)*self.gip)
            Console.warn(' * Found inliers:', len(inliers_cloud_list))
            Console.warn('Check the output cloud to see if the found plane makes sense.')
            Console.warn('Try to increase your distance threshold.')

        cloud_sample_size = int(self.css)
        if cloud_sample_size > len(inliers_cloud_list):
            cloud_sample_size = len(inliers_cloud_list)
        Console.info('Randomly sampling with', cloud_sample_size, 'points...')

        planes = []
        for i in range(0, self.num_iterations):
            point_cloud_local = random.sample(inliers_cloud_list, cloud_sample_size)
            total_no_points = len(point_cloud_local)
            p = Plane([1, 0, 0, 1.5])
            m = p.fit_non_robust(point_cloud_local)
            # m, _ = p.fit(cloud, self.mdt, verbose=False, output_inliers=False)
            angle, pitch, yaw = get_angles(m[0:3])
            planes.append([angle, pitch, yaw])
            Console.progress(i, self.num_iterations, prefix='Iterating planes')

        planes = np.array(planes)
        planes = planes.reshape(-1, 3)

        plane_angle_std = np.std(planes[:, 0])
        plane_angle_mean = np.mean(planes[:, 0])
        plane_angle_median = np.median(planes[:, 0])
        pitch_angle_std = np.std(planes[:, 1])
        pitch_angle_mean = np.mean(planes[:, 1])
        yaw_angle_std = np.std(planes[:, 2])
        yaw_angle_mean = np.mean(planes[:, 2])

        Console.info('Total Number of Points:', total_no_points)
        Console.info('Plane Standard deviation:\n', plane_angle_std)
        Console.info('Mean angle:\n', plane_angle_mean)
        Console.info('Median angle:\n', plane_angle_median)
        Console.info('Pitch Standard deviation:\n', pitch_angle_std)
        Console.info('Pitch Mean:\n', pitch_angle_mean)
        Console.info('Yaw Standard deviation:\n', yaw_angle_std)
        Console.info('Yaw Mean:\n', yaw_angle_mean)

        inliers_cloud = np.array(inliers_cloud)
        mean_x = np.mean(inliers_cloud[:, 0])
        mean_y = np.mean(inliers_cloud[:, 1])
        mean_z = np.mean(inliers_cloud[:, 2])
        mean_xyz = np.array([mean_x, mean_y, mean_z])

        yaml_msg = ''

        yaml_msg = (
            'mean_xyz_m: ' + str(mean_xyz.tolist()) + '\n'
            + 'mean_plane: ' + str(mean_plane) + '\n'
            + 'plane_angle_std_deg: ' + str(plane_angle_std) + '\n'
            + 'plane_angle_mean_deg: ' + str(plane_angle_mean) + '\n'
            + 'plane_angle_median_deg: ' + str(plane_angle_median) + '\n'
            + 'pitch_angle_std_deg: ' + str(pitch_angle_std) + '\n'
            + 'pitch_angle_mean_deg: ' + str(pitch_angle_mean) + '\n'
            + 'yaw_angle_std_deg: ' + str(yaw_angle_std) + '\n'
            + 'yaw_angle_mean_deg: ' + str(yaw_angle_mean) + '\n'
            + 'num_iterations: ' + str(self.num_iterations) + '\n'
            + 'total_no_points: ' + str(total_no_points) + '\n')

        msg = ['minus_2sigma', 'mean', 'plus_2sigma']
        t = ['_pitch_', '_yaw_', '_offset_']
        msg_type = ['plane', 'normal', 'offset_m']

        for i in range(0, 3):
            for j in range(0, 3):
                a = pitch_angle_mean + (1-i)*2*pitch_angle_std
                b = yaw_angle_mean + (1-j)*2*yaw_angle_std
                c = mean_xyz
                plane, normal, offset = build_plane(a, b, c)
                d = msg[i] + t[0] + msg[j] + t[1] + msg[1] + t[2]
                yaml_msg += d + msg_type[0] + ': ' + str(plane.tolist()) + '\n'
                yaml_msg += d + msg_type[1] + ': ' + str(normal.tolist()) + '\n'
                yaml_msg += d + msg_type[2] + ': ' + str(offset) + '\n'
                self.data.append([plane, normal, offset, d])

        uncertainty_planes = [item[0] for item in self.data]
        filename = time.strftime("pointclouds_and_uncertainty_%Y%m%d_%H%M%S.html")
        plot_pointcloud_and_planes([np.array(cloud), inliers_cloud], uncertainty_planes, filename)# 'pointclouds_and_uncertainty_planes.html')
        # np.save('inliers_cloud.npy', inliers_cloud)    # uncomment to save for debugging
        # for i, plane in enumerate(uncertainty_planes):
        #     np.save('plane' + str(i) + '.npy', plane)

        yaml_msg += ('date: \"' + Console.get_date() + "\" \n"
                        + 'user: \"' + Console.get_username() + "\" \n"
                        + 'host: \"' + Console.get_hostname() + "\" \n"
                        + 'version: \"' + Console.get_version() + "\" \n")

        return yaml_msg

    def cal(self, limages, rimages):
        # Synchronise images
        limages_sync = []
        rimages_sync = []

        if len(limages[0].stem) < 26:
            for i, lname in enumerate(limages):
                for j, rname in enumerate(rimages):
                    if lname.stem == rname.stem:
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
        if len(limages) > rs_size:
            rs_size = int((len(limages_sync)/rs_size) - 1)
            i = 0
            while i < len(limages_sync):
                limages_rs.append(limages_sync[i])
                rimages_rs.append(rimages_sync[i])
                i += rs_size
        else:
            limages_rs = limages
            rimages_rs = rimages

        Console.info('Processing ', str(len(limages_sync)) , ' synchronised images...')
        result = joblib.Parallel(n_jobs=-1)([
            joblib.delayed(thread_detect)(i, self.left_maps, j, self.right_maps, self.min_greenness_value, self.k, self.num_columns, self.start_row, self.end_row, self.start_row_b, self.end_row_b, self.two_lasers, self.remap, self.overwrite)
            for i, j in zip(limages_rs, rimages_rs)])

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
            Console.info(
                "Found {} bottom peaks in camera 1!".format(str(count1lb)))
            Console.info(
                "Found {} bottom peaks in camera 2!".format(str(count2lb)))

        point_cloud = []
        point_cloud_b = []

        result = joblib.Parallel(n_jobs=-1)([
            joblib.delayed(self.pointcloud_from_peaks)(f1, f2)
            for f1, f2 in zip(peaks1, peaks2)])
        count_good = 0
        count_bad = 0
        for c, cg, cb in result:
            point_cloud.extend(c)
            count_good += cg
            count_bad += cb
        Console.info("Found {} potential TOP points".format(count_good))
        Console.info("Found {} wrong TOP points".format(count_bad))
        Console.info('Found ' + str(len(point_cloud)) + ' TOP triangulated points!')
        if self.two_lasers:
            result = joblib.Parallel(n_jobs=-1)([
                joblib.delayed(self.pointcloud_from_peaks)(f1b, f2b)
                for f1b, f2b in zip(peaks1b, peaks2b)])
            count_good = 0
            count_bad = 0
            for c, cg, cb in result:
                point_cloud_b.extend(c)
                count_good += cg
                count_bad += cb
            Console.info("Found {} potential BOTTOM points".format(count_good))
            Console.info("Found {} wrong BOTTOM points".format(count_bad))
            Console.info('Found ' + str(len(point_cloud)) + ' BOTTOM triangulated points!')

        # Change coordinate system
        point_cloud_ned = joblib.Parallel(n_jobs=-1)([
            joblib.delayed(opencv_to_ned)(i)
            for i in point_cloud])
        
        if self.two_lasers:
            point_cloud_ned_b = joblib.Parallel(n_jobs=-1)([
                joblib.delayed(opencv_to_ned)(i)
                for i in point_cloud_b])

        save_cloud(processed_folder / '../points.ply', point_cloud_ned)
        if self.two_lasers:
            save_cloud(processed_folder / '../points_b.ply', point_cloud_ned_b)

        rss_before = len(point_cloud_ned)
        point_cloud_filt = self.filter_cloud(point_cloud_ned)
        rss_after = len(point_cloud_filt)
        Console.info('Points after filtering: ' + str(rss_after) + '/' + str(rss_before))
        rs_size = min(rss_after, self.max_point_cloud_size)
        point_cloud_rs = random.sample(point_cloud_filt, rs_size)
        save_cloud(processed_folder / '../points_rs.ply', point_cloud_rs)
        point_cloud_rs = np.array(point_cloud_rs)
        point_cloud_rs = point_cloud_rs.reshape(-1, 3)
        point_cloud_filt = np.array(point_cloud_filt)
        point_cloud_filt = point_cloud_filt.reshape(-1, 3)
        # self.yaml_msg = self.fit_and_save(point_cloud_rs)
        self.yaml_msg = self.fit_and_save(point_cloud_filt)
        
        if self.two_lasers:
            Console.info('Fitting a plane to second line...')
            rss_before = len(point_cloud_ned_b)
            point_cloud_b_filt = self.filter_cloud(point_cloud_ned_b)
            rss_after = len(point_cloud_b_filt)
            Console.info('Points after filtering: ' + str(rss_after) + '/' + str(rss_before))
            rs_size = min(rss_after, self.max_point_cloud_size)
            point_cloud_b_rs = random.sample(point_cloud_b_filt, rs_size)
            save_cloud(processed_folder / '../points_b_rs.ply', point_cloud_b_rs)
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

